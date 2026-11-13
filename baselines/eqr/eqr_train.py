"""EqR's training loop (pretrain.py's launch) on one GPU, with a rolling checkpoint and automatic resume.

The loop is pretrain.py's: its create_dataloader, init_train_state, train_batch (the ACT segment step with the
truncated gradient, AdamATan2 and the puzzle-embedding SignSGD, the warmup-then-constant learning rate), its EMA, and
torch.compile as pretrain.py applies it, all imported from the EqR clone unmodified (see eqr_common.py for the two
stand-ins). What this file adds, so that a requeued job continues the same run:

- one rolling checkpoint, <run>/ckpt.pt, every --ckpt-minutes of training, written to a temporary file and renamed. It
  holds the model, both optimizers, the EMA shadow, the step (the learning-rate schedule follows from it), the ACT carry
  of the in-flight batch, the torch, CUDA, numpy and Python RNG states, the data position (iteration, batches done in
  it) and the best validation record.
- RNG-neutral loaders: each loader's worker base seed comes from its own torch.Generator (see main).
- the data order of pretrain.py on resume: iteration i of the loader draws from Philox(seed + i + 1)
  (dataset/puzzle_dataset.py), so the dataset's counter is preset to i and the batches trained before are drawn and
  skipped. (EqR's own resume restores the sampler from a snapshot the loader worker publishes, which runs ahead of the
  trainer by the prefetched batches; this file does not use it.)
- validation every eval_interval_steps (5,000) on <data>/dev with EqR's evaluate (the EMA weights, depth 16, breadth
  1, as pretrain.py evaluates); the best EMA weights by validation exact accuracy are kept as best.pt.
- at the end: model.pt (the final EMA weights, in EqR's checkpoint format, so evaluate.py and eqr_score.py read it),
  best.pt, final.json and log.jsonl; ckpt.pt is deleted. A run whose final.json exists exits at once.

Run with EqR's Python:  python eqr_train.py --run-dir <dir> --preset sudoku --data <dataset dir> --seed 0
"""

import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro

import eqr_common as C

KILL_AT = int(os.environ.get('EQR_KILL_AT_UPDATE', -1))  # resume test only: die without cleanup after this update


@dataclass
class Args:
    run_dir: Path  # the run's folder: ckpt.pt, log.jsonl, model.pt, best.pt, final.json
    preset: Literal['sudoku', 'maze-paper', 'maze-repo'] = 'sudoku'  # the recipe (eqr_common.PRESETS)
    data: Optional[Path] = None  # EqR-format dataset with train/ and dev/ (default: runs/eqr/data/sudoku-s<seed> or maze-unique)
    seed: int = 0  # training seed; for Sudoku also the FRM training split
    max_updates: Optional[int] = None  # stop after this many updates (smoke tests); the schedule still uses the recipe's total
    ckpt_minutes: float = 10.0  # minutes of training between rolling checkpoints
    eval_every: Optional[int] = None  # updates between validations (default: the recipe's eval_interval_steps)
    log_every: int = 100  # updates between training-metric lines in log.jsonl
    overrides: tuple[str, ...] = ()  # extra hydra overrides for EqR's config, e.g. lr=2e-4


CUDA = torch.cuda.is_available()  # the CPU path exists only for dry runs of this file at a tiny batch


def sync():
    if CUDA:
        torch.cuda.synchronize()


def carry_to_dict(c):
    if c is None:
        return None
    return dict(z_H=c.inner_carry.z_H, z_L=c.inner_carry.z_L, steps=c.steps, halted=c.halted,
                current_data=dict(c.current_data), global_step=c.global_step)


def carry_from_dict(d):
    if d is None:
        return None
    from models.eqr import LatentCarry, ModelCarry
    return ModelCarry(LatentCarry(d['z_H'], d['z_L']), d['steps'], d['halted'], d['current_data'], d['global_step'])


def strip(sd):
    return {k.replace('_orig_mod.', ''): v for k, v in sd.items()}


def eqr_checkpoint(cfg, raw, ema, step):
    """EqR's checkpoint format (utils/checkpoint.save_checkpoint): model, EMA {mu, shadow}, config, step."""
    sd = {k: v.detach().cpu() for k, v in raw.state_dict().items() if not k.endswith(('local_weights', 'local_ids'))}
    shadow = {k: v.detach().cpu() for k, v in strip(ema.shadow).items()}
    return dict(model=sd, ema=dict(mu=ema.mu, shadow=shadow), config=cfg.model_dump(), step=step)


def atomic_save(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(a: Args):
    run = a.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'final.json').exists():
        print('final.json exists: nothing to do', flush=True)
        return
    data = a.data or C.DATA / ('sudoku-s%d' % a.seed if a.preset == 'sudoku' else 'maze-unique')
    C.install_stand_ins()
    import pretrain
    from evaluators.eval_fn import evaluate
    from utils.training import set_deterministic_mode

    cfg = C.load_config(a.preset, data, a.seed, a.overrides)
    cfg.checkpoint_path = None
    set_deterministic_mode(cfg, rank=0)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = os.environ.get('EQR_INDUCTOR_CACHE', str(run / '.inductor'))
    epi = cfg.train_epochs_per_iter or cfg.epochs
    assert cfg.epochs % epi == 0
    total_iters = cfg.epochs // epi
    train_loader, train_meta = pretrain.create_dataloader(cfg, 'train', rank=0, world_size=1, test_set_mode=False,
                                                          epochs_per_iter=epi, global_batch_size=cfg.global_batch_size,
                                                          enable_shared_sampler_state=False)
    dev_loader, dev_meta = pretrain.create_dataloader(cfg, 'dev', rank=0, world_size=1, test_set_mode=True,
                                                      epochs_per_iter=1, global_batch_size=cfg.global_batch_size,
                                                      enable_shared_sampler_state=False)
    # Each loader draws its workers' base seed from its own generator, not from the global RNG, so that a restart (which
    # creates the loader iterators anew) leaves the global RNG where the checkpoint put it. (pretrain.py lets the first
    # iterator draw one number from the global RNG; the training is otherwise the same.)
    train_loader.generator = torch.Generator().manual_seed(cfg.seed)
    dev_loader.generator = torch.Generator().manual_seed(cfg.seed + 1)
    torch.manual_seed(cfg.seed)
    if CUDA:
        torch.cuda.manual_seed_all(cfg.seed)
    ts = pretrain.init_train_state(cfg, train_meta, rank=0, world_size=1)
    raw = ts.model
    ema = ts.ema_helper
    stop = min(ts.total_steps, a.max_updates or ts.total_steps)
    every = a.eval_every or cfg.eval_interval_steps
    ckpt, log_path = run / 'ckpt.pt', run / 'log.jsonl'

    start_iter, skip, train_s, best, dts, wall_prev = 0, 0, 0.0, None, [], 0.0
    if ckpt.exists():
        st = torch.load(ckpt, map_location='cuda' if CUDA else 'cpu', weights_only=False)  # our own file
        raw.load_state_dict(st['model'])
        for opt, sd in zip(ts.optimizers, st['optimizers']):
            opt.load_state_dict(sd)
        ema.shadow = {k: v.to(next(raw.parameters()).device) for k, v in st['ema'].items()}
        ts.step = st['step']
        ts.carry = carry_from_dict(st['carry'])
        torch.set_rng_state(st['rng_torch'].cpu())
        if CUDA:
            torch.cuda.set_rng_state(st['rng_cuda'].cpu())
        np.random.set_state(st['rng_numpy'])
        random.setstate(st['rng_python'])
        start_iter, skip, train_s, best, dts = st['iter'], st['batches_done'], st['train_s'], st['best'], st['dts']
        wall_prev = st['wall_s']
        lines = [l for l in log_path.read_text().splitlines() if json.loads(l).get('step', 0) <= ts.step] if log_path.exists() else []
        log_path.write_text(''.join(l + '\n' for l in lines))
        print(f'RESUME step {ts.step} iteration {start_iter} batches done {skip} training so far {train_s:.0f} s', flush=True)
        resumed = True
    else:
        log_path.write_text('')
        resumed = False
    logf = open(log_path, 'a')

    def log(rec):
        logf.write(json.dumps(rec) + '\n')
        logf.flush()

    if resumed:
        log({'resume': ts.step})
    else:
        (run / 'config.json').write_text(json.dumps(dict(args={k: str(v) for k, v in vars(a).items()},
                                                         eqr_commit=C.EQR_COMMIT, config=cfg.model_dump()), indent=1, default=str))
        log(dict(step=0, num_params=sum(p.numel() for p in raw.parameters()), total_updates=ts.total_steps, stop=stop))

    if 'DISABLE_COMPILE' not in os.environ:  # as pretrain.py: compile, then re-register the EMA on the compiled names
        torch._dynamo.config.cache_size_limit = 32
        ts.model = torch.compile(ts.model, dynamic=False, mode='default')
        ema.register(ts.model)

    def save(it, batches_done):
        t = time.perf_counter()
        atomic_save(dict(model=raw.state_dict(), optimizers=[o.state_dict() for o in ts.optimizers],
                         ema=strip(ema.shadow), step=ts.step, carry=carry_to_dict(ts.carry),
                         rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state() if CUDA else None,
                         rng_numpy=np.random.get_state(), rng_python=random.getstate(), iter=it,
                         batches_done=batches_done, train_s=train_s, best=best, dts=dts,
                         wall_s=wall_prev + time.perf_counter() - wall0), ckpt)
        print(f'CHECKPOINT step {ts.step} iteration {it} batches {batches_done} ({time.perf_counter() - t:.1f} s)', flush=True)

    def validate():
        nonlocal best
        ev = ts.ema_helper.ema_copy(ts.model)
        from dataclasses import replace
        ev_state = replace(ts, model=ev)
        ev.eval()
        t = time.perf_counter()
        m, _ = evaluate(cfg, ev_state, dev_loader, dev_meta, [], rank=0, world_size=1, cpu_group=None)
        m = m['all']
        rec = dict(step=ts.step, dev_exact=m['exact_accuracy'], dev_token=m['accuracy'], dev_s=time.perf_counter() - t)
        log(rec)
        print('VALIDATION', json.dumps(rec), flush=True)
        if best is None or rec['dev_exact'] > best['dev_exact']:
            best = rec
            atomic_save(eqr_checkpoint(cfg, raw, ema, ts.step), run / 'best.pt')
        del ev, ev_state

    train_loader.dataset._iters = start_iter  # pretrain.py seeds iteration i with seed + i + 1
    wall0 = time.perf_counter()  # the loop's wall clock, validation and checkpoint writes included (summed over restarts)
    done = ts.step >= stop
    for it in range(start_iter, total_iters):
        if done:
            break
        ts.model.train()
        sync()
        t0 = last_ckpt = prev = time.perf_counter()
        for b, (_, batch, gbs) in enumerate(train_loader):
            if it == start_iter and b < skip:
                continue  # trained before the restart; drawn only to keep the order
            metrics = pretrain.train_batch(cfg, ts, batch, gbs, rank=0, world_size=1)  # syncs: metrics go to the CPU
            ema.update(ts.model)
            if ts.step == KILL_AT:
                print(f'KILL (test of resume) after update {ts.step}', flush=True)
                os._exit(9)  # as a preemption: no checkpoint, no cleanup
            sync()
            now = time.perf_counter()
            dts.append(now - prev)
            prev = now
            if metrics is not None and ts.step % a.log_every == 0:
                log(dict(step=ts.step, s_per_update=dts[-1], **{k.removeprefix('train/'): float(v) for k, v in metrics.items()
                                                                 if isinstance(v, (int, float, np.floating, np.integer))}))
            done = ts.step >= stop
            pause = ts.step % every == 0 or done or now - last_ckpt > 60 * a.ckpt_minutes
            if pause:
                train_s += now - t0
                if ts.step % every == 0 or done:
                    validate()
                    ts.model.train()
                if done:
                    break
                if time.perf_counter() - last_ckpt > 60 * a.ckpt_minutes:
                    save(it, b + 1)
                    last_ckpt = time.perf_counter()
                sync()
                t0 = prev = time.perf_counter()
        else:
            train_s += time.perf_counter() - t0
            save(it + 1, 0)
            continue
        break

    atomic_save(eqr_checkpoint(cfg, raw, ema, ts.step), run / 'model.pt')
    timed = dts[1:] if len(dts) > 1 else dts
    final = dict(step=ts.step, total_updates=ts.total_steps, train_seconds_loop=train_s,
                 wall_seconds_loop_incl_validation=wall_prev + time.perf_counter() - wall0, restarts=sum('resume' in l for l in log_path.read_text().splitlines()),
                 s_per_update_median=statistics.median(timed), s_per_update_median_after_200=statistics.median(dts[200:]) if len(dts) > 400 else None,
                 projected_hours_full=statistics.median(dts[200:] if len(dts) > 400 else timed) * ts.total_steps / 3600,
                 max_memory_gb=torch.cuda.max_memory_allocated() / 2**30 if CUDA else None, gpu=torch.cuda.get_device_name() if CUDA else 'cpu',
                 num_params=sum(p.numel() for p in raw.parameters()), best=best, preset=a.preset, seed=a.seed,
                 data=str(data), eqr_commit=C.EQR_COMMIT)
    log(dict(step=ts.step, final=True))
    atomic_save_json = run / 'final.json.tmp'
    atomic_save_json.write_text(json.dumps(final, indent=1))
    os.replace(atomic_save_json, run / 'final.json')
    if ckpt.exists():
        ckpt.unlink()
    print('FINAL', json.dumps(final), flush=True)


if __name__ == '__main__':
    main(tyro.cli(Args))
