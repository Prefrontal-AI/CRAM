"""Inference flop per puzzle of HRM, with the authors' model code (sapientinc/HRM), as trm_bench.py `flops` counts TRM.

FlopCounterMode (multiply-add = 2; it counts matmuls and attention) on HRM's eval-mode ACT loop for one batch-1 test
puzzle, run until HRM reports every sequence halted, with no compilation. FlashAttention 3 is a custom op that the
counter does not see, so for counting only models.layers.flash_attn_func is replaced by SDPA with the math kernel
(the same attention arithmetic); its output is compared with FA3's on the same step. The count is checked against an
analytic matmul count from the configuration.

Run from the HRM source tree with HRM's Python (one GPU):
  DISABLE_COMPILE=1 python <repo>/baselines/flops/hrm_count.py <out.json> <checkpoint> <hydra overrides...>
  DISABLE_COMPILE=1 python <repo>/baselines/flops/hrm_count.py --full <out.json> <checkpoint> <hydra overrides...>

--full (added for the all-arithmetic recount; the default is unchanged; `load` was factored out of `main` for it):
FullCounter (baselines/flops/fullcount.py) on the prediction pipeline of the first batch-1 test puzzle, with the same
SDPA-math substitution: the ACT model (the loss head's `.model`) for every ACT step until it reports all halted (16 in
eval mode), with the halted test, and the argmax decode of the final logits. The loss head's losses and metrics (they
need the labels) are not counted. The old component (the loss head's loop under FlopCounterMode, as `main`) is
recounted in the same process; FullCounter's matmul class must equal it. Counted under no_grad, not inference_mode,
so that composite operators (SDPA) decompose before they reach the counter. torch.profiler lists the CUDA kernels of the
counted region, to confirm that none bypasses aten (FA3 must not run there).
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

SRC = Path.cwd()
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hrm'))


def sdpa_math(q, k, v, causal=False):
    """flash_attn_func's interface ([b, s, h, d] in and out) through SDPA's math kernel."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
    return out.transpose(1, 2).contiguous()  # FA3 returns [b, s, h, d] contiguous; layers.py views it


def run_puzzle(model, batch):
    """HRM's eval loop (as evaluate.py / baselines/hrm/score.py) on one batch: (ACT steps, final logits)."""
    with torch.device('cuda'):
        carry = model.initial_carry(batch)
    steps = 0
    while True:
        carry, _, _, preds, done = model(carry=carry, batch=batch, return_keys=['logits'])
        steps += 1
        if done:
            return steps, preds['logits']


def analytic(cfg, seq, vocab):
    """Dense matmul flop of one ACT step (2 per multiply-add) from the configuration."""
    h, heads = cfg['hidden_size'], cfg['num_heads']
    d = h // heads
    inter = -(round(cfg['expansion'] * h * 2 / 3) // -256) * 256
    per_token = 2 * (h * 3 * h + h * h + h * 2 * inter + inter * h)  # qkv, o, gate_up, down (no grouped KV)
    attention = 2 * 2 * heads * seq * seq * d  # Q K^T and A V
    H, L = cfg['H_cycles'], cfg['L_cycles']
    calls = H * L + H  # L_level H*L times, H_level H times per ACT step (no-grad part plus the 1-step part)
    blocks = calls * cfg['H_layers'] if cfg['H_layers'] == cfg['L_layers'] else None
    assert blocks is not None
    heads_flop = 2 * seq * h * vocab + 2 * h * 2  # lm_head on all positions (then sliced), q_head on position 0
    return dict(blocks_per_step=blocks, per_block_linear=per_token * seq, per_block_attention=attention,
                heads=heads_flop, per_step=blocks * (per_token * seq + attention) + heads_flop)


def load(checkpoint, overrides):
    """HRM's eval-mode model (the loss head around the ACT model) with the checkpoint, and the batch-1 test loader."""
    assert 'DISABLE_COMPILE' in os.environ, 'count the uncompiled model'
    import pretrain
    from fit import alias_adam_atan2
    from hydra import compose, initialize_config_dir
    alias_adam_atan2()
    with initialize_config_dir(config_dir=str(SRC / 'config'), version_base=None):
        hcfg = compose(config_name='cfg_pretrain', overrides=list(overrides))
    config = pretrain.load_synced_config(hcfg, rank=0, world_size=1)
    config.checkpoint_path = None
    config.global_batch_size = 1
    _, train_meta = pretrain.create_dataloader(config, 'train', test_set_mode=False, epochs_per_iter=1,
                                               global_batch_size=1, rank=0, world_size=1)
    loader, meta = pretrain.create_dataloader(config, 'test', test_set_mode=True, epochs_per_iter=1,
                                              global_batch_size=1, rank=0, world_size=1)
    state = pretrain.init_train_state(config, train_meta, world_size=1)
    model = state.model
    sd = torch.load(checkpoint, map_location='cuda')
    sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
    # the per-batch sparse puzzle-embedding buffer is sized by the batch (768 in the checkpoint, 1 here); not a weight
    own = model.state_dict()
    skipped = [k for k in sd if k in own and sd[k].shape != own[k].shape]
    model.load_state_dict({k: v for k, v in sd.items() if k not in skipped}, strict=False, assign=True)
    model.eval()
    return config, loader, meta, model, skipped


def main(out, checkpoint, overrides):
    import models.layers as layers
    from fit import attention_backend
    from torch.utils.flop_counter import FlopCounterMode
    config, loader, meta, model, skipped = load(checkpoint, overrides)
    arch = config.arch.__pydantic_extra__
    results = []
    with torch.inference_mode():
        for index, (_, batch, _) in enumerate(loader):
            if index == 3:
                break
            batch = {k: v.cuda() for k, v in batch.items()}
            fa3_steps, fa3_logits = run_puzzle(model, batch)
            original = layers.flash_attn_func
            layers.flash_attn_func = sdpa_math
            try:
                with FlopCounterMode(display=False) as fc:
                    steps, logits = run_puzzle(model, batch)
            finally:
                layers.flash_attn_func = original
            with FlopCounterMode(display=False) as fc_fa3:
                run_puzzle(model, batch)
            by_op = {str(k): int(v) for k, v in fc.get_flop_counts()['Global'].items()}
            results.append(dict(test_row=index, act_steps=steps, act_steps_fa3=fa3_steps,
                                flop_per_puzzle=int(fc.get_total_flops()), by_op=by_op,
                                flop_seen_with_fa3=int(fc_fa3.get_total_flops()),
                                logits_max_abs_diff_sdpa_vs_fa3=float((logits.float() - fa3_logits.float()).abs().max()),
                                argmax_equal=bool((logits.argmax(-1) == fa3_logits.argmax(-1)).all())))
    seq = meta.seq_len + model.model.inner.puzzle_emb_len
    a = analytic(arch, seq, meta.vocab_size)
    r0 = results[0]
    assert all(r['flop_per_puzzle'] == r0['flop_per_puzzle'] and r['act_steps'] == arch['halt_max_steps'] for r in results)
    per_step = r0['flop_per_puzzle'] // r0['act_steps']
    assert per_step * r0['act_steps'] == r0['flop_per_puzzle']
    res = dict(method='HRM', checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
               skipped_buffers=skipped, overrides=list(overrides),
               parameters_all=sum(p.numel() for p in model.parameters()), seq_len_incl_puzzle_emb=seq,
               halt_max_steps=arch['halt_max_steps'], act_steps=r0['act_steps'], flop_per_act_step=per_step,
               flop_per_puzzle=r0['flop_per_puzzle'], analytic=a, analytic_matches=a['per_step'] == per_step,
               puzzles_checked=results, attention_backend_scored=attention_backend(),
               attention_counted='SDPA math kernel substituted for flash_attn_func during counting only',
               convention='FlopCounterMode, batch 1, eval mode, uncompiled; multiply-add = 2; matmuls and attention only',
               gpu=torch.cuda.get_device_name(), slurm_job=os.environ.get('SLURM_JOB_ID'),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1), flush=True)
    assert res['analytic_matches'], 'FlopCounterMode differs from the analytic count'


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


def main_full(out, checkpoint, overrides):
    import models.layers as layers
    from torch.profiler import ProfilerActivity, profile
    from torch.utils.flop_counter import FlopCounterMode
    sys.path.append(str(Path(__file__).resolve().parent))
    from fullcount import CONVENTION, FullCounter
    config, loader, meta, model, skipped = load(checkpoint, overrides)
    arch = config.arch.__pydantic_extra__
    act = arch['halt_max_steps']
    _, batch, _ = next(iter(loader))
    batch = {k: v.cuda() for k, v in batch.items()}
    original = layers.flash_attn_func
    layers.flash_attn_func = sdpa_math
    steps = []
    try:
        with torch.inference_mode():
            with FlopCounterMode(display=False) as fc:  # the old component, as main
                old_steps, old_logits = run_puzzle(model, batch)
            old_total = int(fc.get_total_flops())
        # Counted under no_grad, not inference_mode: under inference_mode composite operators (scaled_dot_product_attention,
        # linear) reach FullCounter undecomposed, and SDPA (tagged nondeterministic_seeded) would be booked as random draws.
        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                with FullCounter() as c, torch.device('cuda'):
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
    finally:
        layers.flash_attn_func = original
    kernels = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
    suspect = [k for k in kernels if any(t in k.lower() for t in ('triton', 'flash', 'fmha', 'attention', 'liger', 'apex'))]
    assert not suspect, f'kernels invisible to FullCounter: {suspect}'
    rng_ops = {op for s in [setup, decode] + steps for op, d in s['by_op'].items() if 'rng' in d}
    assert not rng_ops, f'operators booked as random draws (HRM draws none in eval): {rng_ops}'
    assert len(steps) == act == old_steps, (len(steps), act, old_steps)
    step = steps[0]
    assert all(s['total'] == step['total'] and s['by_op'] == step['by_op'] for s in steps), [s['total'] for s in steps]
    old_step = old_total // act
    assert old_step * act == old_total
    puzzle = combine([(setup, 1), (step, act), (decode, 1)])
    here = Path(__file__).resolve()
    rec = dict(method='HRM', overrides=list(overrides), cwd=str(Path.cwd()), command=' '.join([sys.executable] + sys.argv),
               slurm_job=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
               checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
               skipped_buffers=skipped, params=sum(p.numel() for p in model.parameters()),
               seq_len_incl_puzzle_emb=meta.seq_len + model.model.inner.puzzle_emb_len, act_steps=act,
               act_steps_source='halt_max_steps: HRM eval mode halts only at the last step (hrm_act_v1.py); the counted loop ran until carry.halted.all()',
               components=dict(
                   act_step=dict(full=step, old_matmul_only=old_step, ratio=step['total'] / old_step,
                                 matmul_class_equals_old=step['classes']['matmul'] == old_step,
                                 note='one ACT step of the ACT model (loss head excluded), plus the halted test; every step identical'),
                   decode=dict(full=decode, note='argmax over the vocabulary of the final logits, once per puzzle'),
                   setup=dict(full=setup, note='initial carry (allocation only)'),
                   puzzle=dict(full=puzzle, formula=f'setup + {act} x act_step + decode', old_matmul_only=old_total,
                               ratio=puzzle['total'] / old_total,
                               matmul_class_equals_old=puzzle['classes']['matmul'] == old_total)),
               per_step_totals=[s['total'] for s in steps],
               argmax_equal_to_loss_head_run=bool((preds == old_logits.argmax(-1)).all()),
               flop_per_puzzle=puzzle['total'], flop_per_puzzle_with_compare=puzzle['total_with_compare'], flop_per_puzzle_old=old_total,
               flop_per_puzzle_formula=f'{act} x act_step + decode (+ setup = 0); old: {act} x act_step (matmul only)',
               cuda_kernels=kernels, cuda_kernels_suspect=suspect,
               attention_counted='SDPA math kernel substituted for flash_attn_func during counting only',
               convention=CONVENTION + '; batch 1, eval mode, uncompiled',
               script=str(here), script_sha256=hashlib.sha256(here.read_bytes()).hexdigest(),
               fullcount_sha256=hashlib.sha256((Path(sys.modules['fullcount'].__file__)).read_bytes()).hexdigest(),
               created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('components', 'cuda_kernels')}, indent=1), flush=True)
    assert rec['components']['act_step']['matmul_class_equals_old'], 'FullCounter matmul class differs from FlopCounterMode'


if __name__ == '__main__':
    if sys.argv[1] == '--full':
        main_full(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3:])
