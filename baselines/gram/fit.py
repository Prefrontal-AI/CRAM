"""GRAM: one timed cold fit on one GPU (ahn-ml/gram-release f2e32b9, unmodified, imported from its source tree).

A copy of baselines/trm/fit.py (GRAM is a fork of TinyRecursiveModels): the same clock, the same rolling checkpoint and
the same final EMA model. Differences, all GRAM's own: its carry classes, its seeding (pretrain.py main(): random,
numpy, torch and CUDA seeds, cuDNN deterministic) and its checkpoint format (step_<n>/checkpoint.pt with the config).

Every reported GRAM fit runs on one H200 (Maze-Unique at global batch 384). The data-parallel path below is not used
by any reported fit (it ran Maze-Unique at batch 768 over two GPUs). Under torchrun the script initializes NCCL and
a gloo group as GRAM's pretrain.py main() does, seeds with seed + rank, and passes rank and world size to GRAM's
loader, model and train_batch (which all-reduces the gradients); the global batch is unchanged. Rank 0 writes the
records and the final model; the clock stops after a barrier, so it covers the slowest rank. Each rank keeps its own
rolling checkpoint (carry and RNG differ per rank), every TIMING_CKPT_STEPS updates, so all ranks save at one step.

Runs pretrain.py's own training pieces (create_dataloader, init_train_state, train_batch, EMA, save_train_state) with
the authors' defaults: bf16 forward (`forward_dtype: bfloat16`) and torch.compile (DISABLE_COMPILE must be unset).
Differences from pretrain.py's launch, all outside the training computation:
- no in-run evaluation (it does not change the training state); the final EMA model is scored afterwards;
- a rolling full checkpoint every TIMING_CKPT_EVERY seconds (default 1800), timed and excluded, for crash recovery;
  a run that resumes from it is flagged `resumed` and is not a cold fit.
Clock: baselines/clock.py. It starts after the data loaders are built and stops when the final EMA model exists.

Run from the GRAM source tree with experiments/runs/gram/pydeps (a folder holding the adam-atan2-pytorch package) on PYTHONPATH:
  python baselines/gram/fit.py <out_dir> <hydra overrides...>
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

SRC = Path.cwd()
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clock import FitClock, cuda_context_seconds, write_json  # noqa: E402


def carry_to_dict(c):
    if c is None:
        return None
    return dict(z_H=c.inner_carry.z_H, z_L=c.inner_carry.z_L, steps=c.steps, halted=c.halted,
                current_data=dict(c.current_data), module=type(c).__module__, cls=type(c).__name__,
                inner_cls=type(c.inner_carry).__name__)


def carry_from_dict(d):
    if d is None:
        return None
    # the carry classes of the trained architecture (GRAM, or the TRM and HRM baselines in GRAM's tree); records
    # written before the class was saved are GRAM's
    import importlib
    mod = importlib.import_module(d.get('module', 'models.recursive_reasoning.gram'))
    C = getattr(mod, d.get('cls', 'GenerativeRecursiveReasoningModelCarry'))
    IC = getattr(mod, d.get('inner_cls', 'GenerativeRecursiveReasoningModelInnerCarry'))
    return C(IC(z_H=d['z_H'], z_L=d['z_L']), d['steps'], d['halted'], d['current_data'])


def main(out, overrides, every, every_steps):
    assert 'DISABLE_COMPILE' not in os.environ, 'the authors train with torch.compile'
    rank, world = 0, 1
    if 'LOCAL_RANK' in os.environ:  # GRAM's pretrain.py main()
        dist.init_process_group(backend='nccl')
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.new_group(backend='gloo')
    out = Path(out)
    if (out / 'fit.json').exists():
        print(f'{out}/fit.json exists: nothing to do', flush=True)
        return
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    ctx = cuda_context_seconds()
    import pretrain
    from hydra import compose, initialize_config_dir
    from models.ema import EMAHelper
    with initialize_config_dir(config_dir=str(SRC / 'config'), version_base=None):
        hcfg = compose(config_name='cfg_pretrain', overrides=list(overrides))
    config = pretrain.load_synced_config(hcfg, rank=rank, world_size=world)
    config.checkpoint_path = str(out)
    random.seed(config.seed + rank)  # GRAM's pretrain.py main()
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)
    torch.cuda.manual_seed_all(config.seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    epi = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // epi
    assert config.epochs % epi == 0
    train_loader, train_meta = pretrain.create_dataloader(config, 'train', test_set_mode=False, epochs_per_iter=epi,
                                                          global_batch_size=config.global_batch_size, rank=rank,
                                                          world_size=world)
    clock = FitClock()
    clock.start()
    train_state = pretrain.init_train_state(config, train_meta, rank=rank, world_size=world)
    ema = EMAHelper(mu=config.ema_rate) if config.ema else None
    if ema is not None:
        ema.register(train_state.model)

    full = out / ('full.pt' if world == 1 else f'full-r{rank}.pt')
    start_iter, skip = 0, 0
    if full.exists():
        with clock.exclude('checkpoint_load'):
            st = torch.load(full, map_location='cuda', weights_only=False)
        train_state.model.load_state_dict(st['model'])
        for opt, sd in zip(train_state.optimizers, st['optimizers']):
            opt.load_state_dict(sd)
        if ema is not None:
            ema.shadow = st['ema']
        train_state.step = st['step']
        train_state.carry = carry_from_dict(st['carry'])
        torch.set_rng_state(st['rng_cpu'].cpu())
        torch.cuda.set_rng_state(st['rng_cuda'].cpu())
        start_iter, skip = st['iter'], st['batches_done']
        clock.resumed, clock.prior_fit_seconds = True, st['fit_seconds']
        print(f'RESUME step {train_state.step} iteration {start_iter} batches {skip}: not a cold fit', flush=True)
    train_loader.dataset._iters = start_iter  # pretrain.py seeds iteration i with seed + i + 1

    def save(it, batches_done):
        fit_so_far = clock.elapsed()
        with clock.exclude('checkpoint_save'):
            st = dict(model=train_state.model.state_dict(), optimizers=[o.state_dict() for o in train_state.optimizers],
                      ema=ema.shadow if ema is not None else None, step=train_state.step,
                      carry=carry_to_dict(train_state.carry), rng_cpu=torch.get_rng_state(),
                      rng_cuda=torch.cuda.get_rng_state(), iter=it, batches_done=batches_done,
                      fit_seconds=fit_so_far)
            tmp = full.with_suffix('.pt.tmp')
            torch.save(st, tmp)
            os.replace(tmp, full)

    last = time.perf_counter()
    log_every = 500
    for it in range(start_iter, total_iters):
        train_state.model.train()
        for b, (set_name, batch, gbs) in enumerate(train_loader):
            if it == start_iter and b < skip:
                continue
            metrics = pretrain.train_batch(config, train_state, batch, gbs, rank=rank, world_size=world)
            if ema is not None:
                ema.update(train_state.model)
            if train_state.step % log_every == 0 and metrics is not None:
                print('STEP ' + json.dumps(dict(step=train_state.step, fit_seconds=round(clock.elapsed(), 1),
                                                **{k: float(v) for k, v in metrics.items()})), flush=True)
            if (world == 1 and time.perf_counter() - last > every) or (world > 1 and train_state.step % every_steps == 0):
                save(it, b + 1)
                last = time.perf_counter()
    final = ema.ema_copy(train_state.model) if ema is not None else train_state.model
    if world > 1:
        dist.barrier()  # the fit ends when the slowest rank ends
    fit_seconds = clock.stop()
    if rank > 0:
        full.unlink(missing_ok=True)
        dist.destroy_process_group()
        return

    t = time.perf_counter()
    train_state.model = final
    pretrain.save_train_state(config, train_state)
    final_save_seconds = time.perf_counter() - t  # after the clock stopped; listed for completeness
    rec = clock.record(final_save_seconds=final_save_seconds, method=next((o.split('=', 1)[1].upper() for o in overrides if o.startswith('arch=')), 'GRAM'), upstream='ahn-ml/gram-release f2e32b9', overrides=list(overrides),
                       steps=train_state.step, total_steps=train_state.total_steps, cuda_context_seconds=ctx,
                       numerics='bf16 forward (authors default), torch.compile on', world_size=world,
                       gpus=world, global_batch_size=config.global_batch_size,
                       params=sum(p.numel() for p in final.parameters()))
    write_json(out / 'fit.json', rec)
    full.unlink(missing_ok=True)
    print(f'FIT {fit_seconds:.1f} s, {train_state.step} updates, {world} GPU(s)', flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:], float(os.environ.get('TIMING_CKPT_EVERY', 1800)),
         int(os.environ.get('TIMING_CKPT_STEPS', 10000)))
