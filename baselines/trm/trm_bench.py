"""TRM's cost and score, measured with its own code (SamsungSAILMontreal/TinyRecursiveModels at commit c011037).

Run with TRM's Python from the TRM clone ($THIRD_PARTY/trm), with trm/trm_shim on PYTHONPATH (as trm/fit.sbatch does):

  python trm_bench.py time  <out.json> <hydra overrides...>   # median time per training update, one GPU
  python trm_bench.py flops <out.json> <hydra overrides...>   # params, flop per puzzle (batch-1 eval, 16 ACT steps)
  python trm_bench.py flops-full <out.json> <hydra overrides...>   # the same, all arithmetic (bench_flops_full)
  python trm_bench.py score <out.json> <checkpoint> <hydra overrides...>   # per-puzzle exact solve on the test split

`time` runs pretrain.py's own train_batch and EMA update on its own data loader; the update count it reports is
pretrain.py's total_steps for the given epochs. `flops` counts one eval-mode ACT step of the model under
FlopCounterMode with the math SDPA kernel (so the attention matmuls count), times halt_max_steps: evaluation runs every
ACT step (trm.py halts early only in training). `score` loads a checkpoint as pretrain.py does and runs its eval loop.
ACT = adaptive computation time: one ACT step is one outer (deep-supervision) step of the model.
gram/flops.py and gram/score.py import bench_flops, bench_flops_full and build_eval_model for GRAM, whose tree has the same layout.
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path.cwd()
sys.path.insert(0, str(SRC))


def load_config(overrides):
    """TRM's config for the given hydra overrides, as pretrain.py builds it for one process."""
    from hydra import compose, initialize_config_dir
    import pretrain
    with initialize_config_dir(config_dir=str(SRC / 'config'), version_base=None):
        hcfg = compose(config_name='cfg_pretrain', overrides=list(overrides))
    cfg = pretrain.load_synced_config(hcfg, rank=0, world_size=1)
    cfg.checkpoint_path = None  # never write checkpoints from here
    return cfg


def bench_time(out, overrides, warmup=30, timed=100):
    """Median seconds per training update over `timed` updates after `warmup`, with pretrain.py's train_batch and EMA
    update. A sizing aid for job requests; the reported fit times come from trm/fit.py."""
    import pretrain
    from models.ema import EMAHelper
    cfg = load_config(overrides)
    epi = cfg.eval_interval if cfg.eval_interval is not None else cfg.epochs
    loader, meta = pretrain.create_dataloader(cfg, 'train', test_set_mode=False, epochs_per_iter=epi,
                                              global_batch_size=cfg.global_batch_size, rank=0, world_size=1)
    state = pretrain.init_train_state(cfg, meta, rank=0, world_size=1)
    ema = None
    if cfg.ema:
        ema = EMAHelper(mu=cfg.ema_rate)
        ema.register(state.model)
    state.model.train()
    ticks = []
    t_first = time.perf_counter()
    for i, (set_name, batch, gbs) in enumerate(loader):
        m = pretrain.train_batch(cfg, state, batch, gbs, rank=0, world_size=1)  # syncs: its metrics go to the CPU
        if ema is not None:
            ema.update(state.model)
        torch.cuda.synchronize()
        ticks.append(time.perf_counter())
        if i + 1 == warmup + timed:
            break
    dts = np.diff(ticks[warmup - 1:])
    res = dict(overrides=list(overrides), global_batch_size=cfg.global_batch_size, total_steps=state.total_steps,
               epochs=cfg.epochs, warmup_updates=warmup, timed_updates=len(dts),
               s_per_update_median=float(np.median(dts)), s_per_update_mean=float(np.mean(dts)),
               s_first_updates_incl_compile=float(ticks[warmup - 1] - t_first),
               train_hours_one_gpu=float(np.median(dts)) * state.total_steps / 3600,
               max_memory_gb=torch.cuda.max_memory_allocated() / 2**30, gpu=torch.cuda.get_device_name(),
               last_train_metrics={k: float(v) for k, v in (m or {}).items()})
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def build_eval_model(cfg, meta, checkpoint=None, compile=False):
    """pretrain.py's model; compiled as pretrain.py compiles it when `compile` (scores then match its own evaluation)."""
    import pretrain
    if compile:
        os.environ.pop('DISABLE_COMPILE', None)
    else:
        os.environ['DISABLE_COMPILE'] = '1'
    cfg.load_checkpoint = None
    model, _, _ = pretrain.create_model(cfg, meta, rank=0, world_size=1)
    if checkpoint is not None:
        sd = torch.load(checkpoint, map_location='cuda')
        sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
        getattr(model, '_orig_mod', model).load_state_dict(sd, assign=True)
    model.eval()
    return model


def bench_flops(out, overrides):
    """Matmul-only operation count of one eval ACT step (torch FlopCounterMode), times halt_max_steps. bench_flops_full
    recounts it as a cross-check and adds every other operation class."""
    import pretrain
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode
    cfg = load_config(overrides)
    cfg.global_batch_size = 1
    loader, meta = pretrain.create_dataloader(cfg, 'test', test_set_mode=True, epochs_per_iter=1, global_batch_size=1,
                                              rank=0, world_size=1)
    model = build_eval_model(cfg, meta)
    _, batch, _ = next(iter(loader))
    batch = {k: v.cuda() for k, v in batch.items()}
    with torch.inference_mode(), torch.device('cuda'):
        carry = model.initial_carry(batch)
        with sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as fc:
            carry, *_ = model(carry=carry, batch=batch, return_keys=[])
    per_step = fc.get_total_flops()
    act = model.model.config.halt_max_steps
    inner = model.model.config
    res = dict(overrides=list(overrides), params=sum(p.numel() for p in model.parameters()),
               flop_per_act_step=per_step, act_steps=act, flop_per_puzzle=per_step * act,
               H_cycles=inner.H_cycles, L_cycles=inner.L_cycles, L_layers=inner.L_layers, hidden=inner.hidden_size,
               seq_len_incl_puzzle_emb=meta.seq_len + model.model.inner.puzzle_emb_len, mlp_t=inner.mlp_t,
               convention='FlopCounterMode, math SDPA, batch 1, eval mode; multiply-add = 2')
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


SUSPECT_KERNEL = ('triton', 'flash', 'fmha', 'attention', 'liger', 'apex')  # kernels FullCounter cannot see


def combine(parts):
    """Sum of FullCounter.summary() records, each with a multiplicity: [(summary, times), ...]."""
    by_op, classes = {}, {}
    for s, n in parts:
        for op, d in s['by_op'].items():
            for cls, v in d.items():
                by_op.setdefault(op, {}).setdefault(cls, 0)
                by_op[op][cls] += n * v
        for cls, v in s['classes'].items():
            classes[cls] = classes.get(cls, 0) + n * v
    return dict(total=sum(n * s['total'] for s, n in parts), total_with_compare=sum(n * s['total_with_compare'] for s, n in parts), classes=classes,
                matmul_only_flopcountermode=sum(n * s['matmul_only_flopcountermode'] for s, n in parts),
                by_op=dict(sorted(by_op.items())))


def cuda_kernels(prof):
    """Distinct CUDA kernel names launched under a torch.profiler run, and those that may bypass aten."""
    names = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
    return names, [k for k in names if any(s in k.lower() for s in SUSPECT_KERNEL)]


RNG_OPS = {'randn', 'randn_like', 'rand', 'rand_like', 'normal', 'uniform', 'randint', 'randint_like', 'multinomial', 'bernoulli'}


def check_rng_ops(summaries):
    """Only random-number generators may be booked as draws (an undecomposed composite such as SDPA would be too)."""
    bad = {op for s in summaries for op, d in s['by_op'].items() if 'rng' in d and op not in RNG_OPS}
    assert not bad, f'operators booked as random draws: {bad}'


def bench_flops_full(out, overrides, finish=None):
    """`flops` under the paper's single convention: all arithmetic (baselines/flops/fullcount.py FullCounter).
    Same model (random initialization: the count has no weight- or
    data-dependent control flow), batch-1 test puzzle, eval mode, math SDPA and no compilation as `flops`; counted under
    no_grad instead of inference_mode, so that composite operators (SDPA) decompose before they reach the counter.
    Counted: the prediction pipeline only, i.e. the ACT model (loss head's `.model`) for every ACT step until it halts
    (eval mode halts at halt_max_steps), with the halted test, and the argmax decode of the final logits. Not counted:
    the loss head's losses and metrics, which need the labels. The old component (one ACT step of the loss head under
    FlopCounterMode, as `flops`) is recounted in the same process; FullCounter's matmul class must equal it. The CUDA
    kernels of the counted region are listed (torch.profiler) to confirm that no kernel bypasses aten.
    `finish(rec, outputs, batch, model)` may add components (GRAM's sample selection) before the record is written."""
    import hashlib
    import pretrain
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.profiler import ProfilerActivity, profile
    from torch.utils.flop_counter import FlopCounterMode
    sys.path.append(str(Path(__file__).resolve().parents[2] / 'baselines' / 'flops'))
    from fullcount import CONVENTION, FullCounter
    cfg = load_config(overrides)
    cfg.global_batch_size = 1
    loader, meta = pretrain.create_dataloader(cfg, 'test', test_set_mode=True, epochs_per_iter=1, global_batch_size=1,
                                              rank=0, world_size=1)
    model = build_eval_model(cfg, meta)
    assert not hasattr(model, '_orig_mod') and not model.training
    _, batch, _ = next(iter(loader))
    batch = {k: v.cuda() for k, v in batch.items()}
    act = model.model.config.halt_max_steps
    steps = []
    with torch.inference_mode(), torch.device('cuda'), sdpa_kernel(SDPBackend.MATH):
        carry = model.initial_carry(batch)
        with FlopCounterMode(display=False) as fc:  # the old component, as bench_flops
            model(carry=carry, batch=batch, return_keys=[])
        old_step = int(fc.get_total_flops())
    # Counted under no_grad, not inference_mode: under inference_mode composite operators (scaled_dot_product_attention,
    # linear) reach FullCounter undecomposed, and SDPA (tagged nondeterministic_seeded) would be booked as random draws.
    with torch.no_grad(), torch.device('cuda'), sdpa_kernel(SDPBackend.MATH):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            with FullCounter() as c:
                carry = model.model.initial_carry(batch)
            setup = c.summary()
            while True:
                with FullCounter() as c:
                    carry, outputs = model.model(carry=carry, batch=batch)
                    done = bool(carry.halted.all())
                steps.append(c.summary())
                if done:
                    break
            with FullCounter() as c:
                preds = torch.argmax(outputs['logits'], dim=-1)
            decode = c.summary()
            torch.cuda.synchronize()
    kernels, suspect = cuda_kernels(prof)
    assert not suspect, f'kernels invisible to FullCounter: {suspect}'
    check_rng_ops([setup, decode] + steps)
    assert len(steps) == act, (len(steps), act)
    step = steps[0]
    assert all(s['total'] == step['total'] and s['by_op'] == step['by_op'] for s in steps), [s['total'] for s in steps]
    trajectory = combine([(setup, 1), (step, act), (decode, 1)])
    inner = model.model.config
    here = Path(__file__).resolve()
    rec = dict(overrides=list(overrides), cwd=str(Path.cwd()), command=' '.join([sys.executable] + sys.argv),
               slurm_job=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
               checkpoint=None, checkpoint_note='random initialization, as the old count (trm_bench.py flops): no weight- or data-dependent control flow in eval mode',
               params=sum(p.numel() for p in model.parameters()), act_steps=act, act_steps_source='halt_max_steps (eval mode halts only at the last step); the counted loop ran until carry.halted.all()',
               H_cycles=inner.H_cycles, L_cycles=inner.L_cycles, L_layers=inner.L_layers, hidden=inner.hidden_size,
               seq_len_incl_puzzle_emb=meta.seq_len + model.model.inner.puzzle_emb_len, mlp_t=inner.mlp_t,
               components=dict(
                   act_step=dict(full=step, old_matmul_only=old_step, ratio=step['total'] / old_step,
                                 matmul_class_equals_old=step['classes']['matmul'] == old_step,
                                 note='one ACT step of the ACT model (loss head excluded), plus the halted test; every step identical'),
                   decode=dict(full=decode, note='argmax over the vocabulary of the final logits, once per trajectory'),
                   setup=dict(full=setup, note='initial carry (allocation only)'),
                   trajectory=dict(full=trajectory, formula=f'setup + {act} x act_step + decode', old_matmul_only=old_step * act,
                                   ratio=trajectory['total'] / (old_step * act))),
               per_step_totals=[s['total'] for s in steps], decoded_tokens=int(preds.numel()),
               cuda_kernels=kernels, cuda_kernels_suspect=suspect,
               convention=CONVENTION + '; math SDPA, batch 1, eval mode, uncompiled',
               script=str(here), script_sha256=hashlib.sha256(here.read_bytes()).hexdigest(),
               fullcount_sha256=hashlib.sha256((Path(sys.modules['fullcount'].__file__)).read_bytes()).hexdigest(),
               created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    rec['flop_per_puzzle'] = trajectory['total']
    rec['flop_per_puzzle_with_compare'] = trajectory['total_with_compare']
    rec['flop_per_puzzle_old'] = old_step * act
    rec['flop_per_puzzle_formula'] = f'{act} x act_step + decode (+ setup = 0); old: {act} x act_step (matmul only)'
    if finish is not None:
        finish(rec, outputs, batch, model)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('components', 'cuda_kernels')}, indent=1))
    assert rec['components']['act_step']['matmul_class_equals_old'], 'FullCounter matmul class differs from FlopCounterMode'
    return rec


def score(out, checkpoint, overrides):
    """Per-puzzle exact solve of `checkpoint` on the test split, with pretrain.py's eval loop (compiled as pretrain.py
    compiles it). Writes <out> (JSON) and <out stem>.right.npy, one solved flag per test row in loader order."""
    import pretrain
    from models.losses import IGNORE_LABEL_ID
    cfg = load_config(overrides)
    bs = cfg.global_batch_size
    loader, meta = pretrain.create_dataloader(cfg, 'test', test_set_mode=True, epochs_per_iter=1, global_batch_size=bs,
                                              rank=0, world_size=1)
    model = build_eval_model(cfg, meta, checkpoint, compile=True)
    right, token_acc, steps_used, trm_sums = [], [], 0, {}
    with torch.inference_mode(), torch.device('cuda'):
        for _, batch, _ in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            carry = model.initial_carry(batch)
            while True:
                carry, _, metrics, preds, done = model(carry=carry, batch=batch, return_keys=['preds'])
                steps_used += 1
                if done:
                    break
            labels = batch['labels']
            mask = labels != IGNORE_LABEL_ID  # the loader maps ignored labels (0) to -100 and pads the last batch
            real = mask.any(-1)               # padding rows have no label at all
            hit = (preds['preds'] == labels) & mask
            right.append((hit | ~mask).all(-1)[real].cpu().numpy())
            token_acc.append((hit.sum(-1) / mask.sum(-1).clamp_min(1))[real].cpu().numpy())  # pretrain.py's accuracy
            for k in ('count', 'exact_accuracy', 'accuracy'):  # TRM's own metric on the same predictions, as a check
                trm_sums[k] = trm_sums.get(k, 0.0) + float(metrics[k])
    right = np.concatenate(right)
    res = dict(checkpoint=str(checkpoint), overrides=list(overrides), n=int(right.size),
               exact=float(right.mean()), solved=int(right.sum()),
               token_accuracy=float(np.concatenate(token_acc).mean()), act_steps_per_batch=steps_used,
               trm_metric=dict(exact=trm_sums['exact_accuracy'] / trm_sums['count'],
                               token_accuracy=trm_sums['accuracy'] / trm_sums['count'], count=trm_sums['count']))
    assert abs(res['trm_metric']['exact'] - res['exact']) < 1e-9 and res['trm_metric']['count'] == res['n'], res
    np.save(Path(out).with_suffix('.right.npy'), right)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    mode, out = sys.argv[1], sys.argv[2]
    if mode == 'time':
        bench_time(out, sys.argv[3:])
    elif mode == 'flops':
        bench_flops(out, sys.argv[3:])
    elif mode == 'flops-full':
        bench_flops_full(out, sys.argv[3:])
    elif mode == 'score':
        score(out, sys.argv[3], sys.argv[4:])
    else:
        raise SystemExit(__doc__)
