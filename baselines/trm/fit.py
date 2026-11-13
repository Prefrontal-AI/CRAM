"""TRM: one timed cold fit on one GPU (TinyRecursiveModels c011037, unmodified, imported from its source tree).

Runs pretrain.py's own training pieces (create_dataloader, init_train_state, train_batch, EMA, save_train_state) with
the authors' defaults: bf16 forward (`forward_dtype: bfloat16`) and torch.compile (DISABLE_COMPILE must be unset).
Differences from pretrain.py's launch, all outside the training computation:
- no in-run evaluation (it does not change the training state); the final EMA model is scored afterwards;
- a rolling full checkpoint every TIMING_CKPT_EVERY seconds (default 1800), timed and excluded, for crash recovery;
  a run that resumes from it is flagged `resumed` and is not a cold fit.
Clock: baselines/clock.py. It starts after the data loaders are built and stops when the final EMA model exists.

Run from the TRM source tree with the AdamATan2 shim on PYTHONPATH:
  python baselines/trm/fit.py <out_dir> <hydra overrides...>
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


def carry_to_dict(c):
    if c is None:
        return None
    return dict(z_H=c.inner_carry.z_H, z_L=c.inner_carry.z_L, steps=c.steps, halted=c.halted,
                current_data=dict(c.current_data))


def carry_from_dict(d):
    if d is None:
        return None
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1Carry as C, \
        TinyRecursiveReasoningModel_ACTV1InnerCarry as IC
    return C(IC(z_H=d['z_H'], z_L=d['z_L']), d['steps'], d['halted'], d['current_data'])


def main(out, overrides, every):
    assert 'DISABLE_COMPILE' not in os.environ, 'the authors train with torch.compile'
    assert 'LOCAL_RANK' not in os.environ, 'one GPU'
    out = Path(out)
    if (out / 'fit.json').exists():
        print(f'{out}/fit.json exists: nothing to do', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    ctx = cuda_context_seconds()
    import pretrain
    from hydra import compose, initialize_config_dir
    from models.ema import EMAHelper
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
    clock = FitClock()
    clock.start()
    train_state = pretrain.init_train_state(config, train_meta, rank=0, world_size=1)
    ema = EMAHelper(mu=config.ema_rate) if config.ema else None
    if ema is not None:
        ema.register(train_state.model)

    full = out / 'full.pt'
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
            metrics = pretrain.train_batch(config, train_state, batch, gbs, rank=0, world_size=1)
            if ema is not None:
                ema.update(train_state.model)
            if train_state.step % log_every == 0 and metrics is not None:
                print('STEP ' + json.dumps(dict(step=train_state.step, fit_seconds=round(clock.elapsed(), 1),
                                                **{k: float(v) for k, v in metrics.items()})), flush=True)
            if time.perf_counter() - last > every:
                save(it, b + 1)
                last = time.perf_counter()
    final = ema.ema_copy(train_state.model) if ema is not None else train_state.model
    fit_seconds = clock.stop()

    t = time.perf_counter()
    train_state.model = final
    pretrain.save_train_state(config, train_state)
    final_save_seconds = time.perf_counter() - t  # after the clock stopped; listed for completeness
    rec = clock.record(final_save_seconds=final_save_seconds, method='TRM', upstream='TinyRecursiveModels c011037', overrides=list(overrides),
                       steps=train_state.step, total_steps=train_state.total_steps, cuda_context_seconds=ctx,
                       numerics='bf16 forward (authors default), torch.compile on',
                       params=sum(p.numel() for p in final.parameters()))
    write_json(out / 'fit.json', rec)
    full.unlink(missing_ok=True)
    print(f'FIT {fit_seconds:.1f} s, {train_state.step} updates', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:], float(os.environ.get('TIMING_CKPT_EVERY', 1800)))
