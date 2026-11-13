"""HRM: one timed cold fit on one GPU (sapientinc/HRM ac15626, unmodified, imported from its source tree).

Runs pretrain.py's own training pieces (create_dataloader, init_train_state, train_batch, save_train_state) with the
authors' defaults: bf16 forward (`forward_dtype: bfloat16` in the model config), torch.compile(dynamic=False) unless
DISABLE_COMPILE is set (pretrain.py's switch; the choice is recorded in fit.json), FlashAttention 3 when installed
(models/layers.py imports flash_attn_interface first, as the README asks for Hopper GPUs). HRM keeps no EMA: the
inference-ready model is the trained model.
Differences from pretrain.py's launch, all outside the training computation:
- no in-run evaluation (it does not change the training state); the final model is scored afterwards (score.py);
- pretrain.py's checkpoint after each eval interval (checkpoint_every_eval: True) is written as it writes it, timed
  and excluded, so the intermediate models exist as in the authors' run; they are not used for the table;
- a rolling full checkpoint every TIMING_CKPT_EVERY seconds (default 1800), timed and excluded, for crash recovery;
  a run that resumes from it is flagged `resumed` and is not a cold fit;
- adam-atan2 0.0.3 calls Optimizer._cuda_graph_capture_health_check, which torch 2.11 renamed to
  _accelerator_graph_capture_health_check; the old name is aliased to the new one (a check that acts only during CUDA
  graph capture). The optimizer kernel is the authors' CUDA package, unchanged.
Clock: baselines/clock.py. It starts after the data loaders are built and stops when the final model exists.

Run from the HRM source tree:
  python baselines/hrm/fit.py <out_dir> <hydra overrides...>
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

SRC = Path.cwd()
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clock import FitClock, cuda_context_seconds, write_json  # noqa: E402


def alias_adam_atan2():
    """adam-atan2 0.0.3 on torch >= 2.10: give its step() the health check under the name it calls."""
    from adam_atan2 import AdamATan2
    import adam_atan2_backend  # noqa: F401  (the CUDA kernel; fail here if it is missing)
    if not hasattr(AdamATan2, '_cuda_graph_capture_health_check'):
        AdamATan2._cuda_graph_capture_health_check = torch.optim.Optimizer._accelerator_graph_capture_health_check


def attention_backend():
    try:
        import flash_attn_interface  # noqa: F401
        from importlib.metadata import version
        return f'FlashAttention 3 (flash_attn_3 {version("flash_attn_3")})'
    except ImportError:
        from importlib.metadata import version
        return f'FlashAttention 2 (flash_attn {version("flash_attn")})'


def carry_to_dict(c):
    if c is None:
        return None
    return dict(z_H=c.inner_carry.z_H, z_L=c.inner_carry.z_L, steps=c.steps, halted=c.halted,
                current_data=dict(c.current_data))


def carry_from_dict(d):
    if d is None:
        return None
    from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1Carry as C, \
        HierarchicalReasoningModel_ACTV1InnerCarry as IC
    return C(IC(z_H=d['z_H'], z_L=d['z_L']), d['steps'], d['halted'], d['current_data'])


def main(out, overrides, every, log_every):
    assert 'LOCAL_RANK' not in os.environ, 'one GPU'
    compiled = 'DISABLE_COMPILE' not in os.environ
    out = Path(out)
    if (out / 'fit.json').exists():
        print(f'{out}/fit.json exists: nothing to do', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    ctx = cuda_context_seconds()
    import pretrain
    alias_adam_atan2()
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(SRC / 'config'), version_base=None):
        hcfg = compose(config_name='cfg_pretrain', overrides=list(overrides))
    config = pretrain.load_synced_config(hcfg, rank=0, world_size=1)
    config.checkpoint_path = str(out)
    torch.random.manual_seed(config.seed)
    epi = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // epi
    assert config.epochs % epi == 0
    train_loader, train_meta = pretrain.create_dataloader(config, 'train', test_set_mode=False, epochs_per_iter=epi,
                                                          global_batch_size=config.global_batch_size, rank=0,
                                                          world_size=1)
    (out / 'config.json').write_text(json.dumps(config.model_dump(), indent=1, default=str))
    clock = FitClock()
    clock.start()
    train_state = pretrain.init_train_state(config, train_meta, world_size=1)

    full = out / 'full.pt'
    start_iter, skip = 0, 0
    if full.exists():
        with clock.exclude('checkpoint_load'):
            st = torch.load(full, map_location='cuda', weights_only=False)
        train_state.model.load_state_dict(st['model'])
        for opt, sd in zip(train_state.optimizers, st['optimizers']):
            opt.load_state_dict(sd)
        train_state.step = st['step']
        train_state.carry = carry_from_dict(st['carry'])
        torch.set_rng_state(st['rng_cpu'].cpu())
        torch.cuda.set_rng_state(st['rng_cuda'].cpu())
        start_iter, skip = st['iter'], st['batches_done']
        clock.resumed, clock.prior_fit_seconds = True, st['fit_seconds']
        print(f'RESUME step {train_state.step} iteration {start_iter} batches {skip}: not a cold fit', flush=True)
    train_loader.dataset._iters = start_iter  # puzzle_dataset.py seeds iteration i with seed + i + 1

    def save(it, batches_done):
        fit_so_far = clock.elapsed()
        with clock.exclude('checkpoint_save'):
            st = dict(model=train_state.model.state_dict(), optimizers=[o.state_dict() for o in train_state.optimizers],
                      step=train_state.step, carry=carry_to_dict(train_state.carry), rng_cpu=torch.get_rng_state(),
                      rng_cuda=torch.cuda.get_rng_state(), iter=it, batches_done=batches_done,
                      fit_seconds=fit_so_far)
            tmp = full.with_suffix('.pt.tmp')
            torch.save(st, tmp)
            os.replace(tmp, full)

    last = time.perf_counter()
    for it in range(start_iter, total_iters):
        train_state.model.train()
        for b, (set_name, batch, gbs) in enumerate(train_loader):
            if it == start_iter and b < skip:
                continue
            metrics = pretrain.train_batch(config, train_state, batch, gbs, rank=0, world_size=1)
            if train_state.step % log_every == 0 and metrics is not None:
                print('STEP ' + json.dumps(dict(step=train_state.step, fit_seconds=round(clock.elapsed(), 2),
                                                peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                                                **{k: float(v) for k, v in metrics.items()})), flush=True)
            if time.perf_counter() - last > every:
                save(it, b + 1)
                last = time.perf_counter()
        if config.checkpoint_every_eval and it < total_iters - 1:  # pretrain.py's per-eval checkpoint
            with clock.exclude('eval_checkpoint_save'):
                pretrain.save_train_state(config, train_state)
    final = train_state.model  # no EMA in HRM
    fit_seconds = clock.stop()

    t = time.perf_counter()
    pretrain.save_train_state(config, train_state)
    final_save_seconds = time.perf_counter() - t  # after the clock stopped; listed for completeness
    from importlib.metadata import version
    rec = clock.record(final_save_seconds=final_save_seconds, method='HRM', upstream='sapientinc/HRM ac15626',
                       overrides=list(overrides), steps=train_state.step, total_steps=train_state.total_steps,
                       global_batch_size=config.global_batch_size, cuda_context_seconds=ctx,
                       numerics='bf16 forward (authors default), torch.compile ' + ('on' if compiled else 'off'),
                       compiled=compiled, attention=attention_backend(),
                       optimizer=f'adam-atan2 {version("adam_atan2")} CUDA package (authors)',
                       peak_memory_gb=torch.cuda.max_memory_allocated() / 2**30,
                       final_checkpoint=f'step_{train_state.step}',
                       params=sum(p.numel() for p in final.parameters()))
    write_json(out / 'fit.json', rec)
    full.unlink(missing_ok=True)
    print(f'FIT {fit_seconds:.1f} s, {train_state.step} updates', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:], float(os.environ.get('TIMING_CKPT_EVERY', 1800)),
         int(os.environ.get('TIMING_LOG_EVERY', 500)))
