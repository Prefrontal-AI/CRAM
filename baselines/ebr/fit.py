"""EBR (Energy-Based Reasoner): one timed cold fit on one GPU (github.com/wwillchen/ebr at 8c5271c, unmodified, imported
from its source tree).

Runs EBR's own training pieces from pretrain.py (create_dataloader, _build_train_state, load_training_state, Run,
_train_loop) with the recipe config/train/<recipe>.yaml and the authors' numerics (bf16 forward, torch.compile, and the
fused Triton attention kernels where the recipe turns them on). Differences from pretrain.py's launch, all outside the
training computation:
- no in-loop evaluation (`eval_interval_steps=null`). The authors' recipes evaluate the EMA weights every few thousand
  steps with `eval_rng: coupled`, which draws from the training RNG; without it the training draws differ from a run
  with evaluations, as a different seed would. The final EMA model is scored afterwards (ebr/score.sbatch);
- the run folder is <out>/run (not outputs/<run id>/<date>/<time>), and W&B is off;
- the authors' checkpoints (every 5,000 steps) are timed and excluded, and only the latest is kept. A requeued job
  resumes from it with pretrain.py's own resume path; that fit is flagged `resumed` and is not a cold fit.
Clock: baselines/clock.py. It starts when the training split is in host memory (the memory-mapped arrays are read into RAM
before the start) and stops when the final EMA model exists.

Run from EBR's source tree (ebr/fit.sbatch does this):
  python <repository>/baselines/ebr/fit.py <out_dir> <recipe> <hydra overrides...>
Writes <out_dir>/fit.json (the fit record) and <out_dir>/run/ (EBR's run folder with the final checkpoint).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path.cwd()
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clock import FitClock, cuda_context_seconds, write_json  # noqa: E402

UPSTREAM = 'wwillchen/ebr 8c5271c05cf35f816e36db41ff324b3081175c64'


def latest_checkpoint(ckpt_dir):
    """Path of the step_<n>.pth with the largest n in `ckpt_dir`, or None when the folder has none."""
    found = sorted(ckpt_dir.glob('step_*.pth'), key=lambda p: int(p.name.split('_')[1].split('.')[0]))
    return found[-1] if found else None


def main(out, recipe, overrides):
    """Train `recipe` (a name under EBR's config/train/) with the Hydra `overrides` (list of str) and write <out>/fit.json."""
    out = Path(out).resolve()
    if (out / 'fit.json').exists():
        print(f'{out}/fit.json exists: nothing to do', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    ctx = cuda_context_seconds()
    import pretrain
    from dataset.puzzle_dataset import create_dataloader
    from utils.checkpoint import load_training_state, save_checkpoint
    from utils.config import compose_config
    from utils.runtime import seed_everything

    run_dir = out / 'run'
    ckpt_dir = run_dir / 'checkpoints'
    resume = latest_checkpoint(ckpt_dir)
    overrides = list(overrides) + ['eval_interval_steps=null', 'wandb_mode=disabled']
    if resume is not None:
        overrides.append(f'load_checkpoint={resume}')
    config, raw = compose_config(f'train/{recipe}', overrides)
    if config.run_name is None:
        config.run_name = raw['run_name'] = f'ebr_{recipe}_s{config.seed}'
    assert config.eval_interval_steps is None

    # pretrain._train, up to the loop, with the clock around it.
    device = torch.device('cuda')
    seed_everything(config.seed)
    train_loader, train_meta = create_dataloader(config, 'train', test_set_mode=False,
                                                 epochs_per_iter=config.train_epochs_per_iter or config.epochs,
                                                 global_batch_size=config.global_batch_size)
    ds = train_loader.dataset
    ds._lazy_load()
    ds._data = {s: {f: np.array(a) for f, a in fields.items()} for s, fields in ds._data.items()}  # into host memory

    clock = FitClock()
    clock.start()
    state = pretrain._build_train_state(config, train_meta, device, 0, 1)
    clock_file = out / 'clock.json'  # fit seconds at each saved step, for a resume
    resume_info = None
    if resume is not None:
        with clock.exclude('checkpoint_load'):
            resume_info = load_training_state(config, state, train_loader)
        saved = json.loads(clock_file.read_text())
        clock.resumed, clock.prior_fit_seconds = True, saved[str(state.step)]
        print(f'RESUME step {state.step} from {resume}: not a cold fit', flush=True)

    run_id = f'{recipe}-s{config.seed}'
    logger = pretrain._start_run_dir(config, raw, state, run_id, str(run_dir), resumed=resume is not None, rank=0, world=1)
    run = pretrain.Run(config, state, train_loader, None, run_id, str(run_dir), logger, device, 0, 1)

    def checkpoint():
        fit_so_far = clock.elapsed()
        with clock.exclude('checkpoint_save'):
            path = save_checkpoint(config, state, train_loader, run_id=run_id)
            times = json.loads(clock_file.read_text()) if clock_file.exists() else {}
            times[str(state.step)] = fit_so_far
            write_json(clock_file, times)
            for old in ckpt_dir.glob('step_*.pth'):
                if old.resolve() != Path(path).resolve():
                    old.unlink()
        print(f'saved checkpoint {path} at fit {fit_so_far:.1f} s', flush=True)
    run.checkpoint = checkpoint

    stop = pretrain.StopSignal()
    import signal
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, stop.request_stop)
    run.write_status('running', None)
    # _train_loop checkpoints the last step itself (excluded, through run.checkpoint); the EMA is final there.
    stop_reason = pretrain._train_loop(run, stop, rng_fix=(resume_info or {}).get('rng_fix'))
    if state.step < state.total_steps:
        run.write_status('stopped', stop_reason)
        print(f'stopped at {state.step}: {stop_reason}', flush=True)
        sys.exit(1)
    final = state.ema_helper.ema_copy(state.model)  # the inference-ready model
    fit_seconds = clock.stop()
    run.write_status('completed', 'target_reached')
    if logger:
        logger.finish()
    rec = clock.record(method='EBR', upstream=UPSTREAM, recipe=recipe, overrides=overrides, steps=state.step,
                       total_steps=state.total_steps, cuda_context_seconds=ctx,
                       numerics='bf16 forward (authors default), torch.compile and fused kernels as the recipe',
                       gpus=1, global_batch_size=config.global_batch_size, seed=config.seed,
                       final_checkpoint=str(latest_checkpoint(ckpt_dir)),
                       params=sum(p.numel() for p in final.parameters()),
                       peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30)
    write_json(out / 'fit.json', rec)
    print(f'FIT {fit_seconds:.1f} s, {state.step} updates', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
