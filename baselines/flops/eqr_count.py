"""EqR's inference operation count under the all-arithmetic convention (fullcount.py), beside the old matmul-only count.

The old count is FlopCounterMode on the first batch-1 network function evaluation (NFE, one EqRModel.forward) of the
released checkpoint's config, times D x B; its recorded value per task is in TASKS. This script counts the
prediction pipeline of eqr_score.py at the paper's headline setting (D 64 outer iterations, B 128 trajectories, path
noise 0.5, random latent reset std 1.0, top1_conv selection over a window of 3), on the released checkpoints, with
FullCounter:
- first: the first NFE of a trajectory (includes the random latent reset: a truncated-normal draw and its scaling);
- next: every later NFE (no reset; the same block stack with its noise draws);
- decode: the argmax over the vocabulary of the last logits (once per trajectory; the loss head also takes it at every
  NFE, but only the last is the answer);
- select: the top1_conv selection over B trajectories (float cast of the last 4 logits, differences, L2 norm over the
  vocabulary, mean over the window and the cells, argmin over B).
Per puzzle (full) = B x (first + (D - 1) x next + decode) + select. Per puzzle (old) = old first x D x B.
Not counted: the loss head's losses and metrics (they read the labels), the majority vote (not the headline selection).

Totals follow fullcount.py: `total` counts matmul, arith and transc; `total_with_compare` adds the compare class.
Every count runs on the CPU, uncompiled, under torch.no_grad() (not inference_mode, so that composite operators reach
the counters decomposed) and the math SDPA (scaled dot-product attention) backend: on the CPU EqR's Attention takes
the torch SDPA path (its flash_attn_func branch needs CUDA tensors), so no fused or custom kernel runs; the script
asserts that the flash stand-in was never called. EqR's own Python; run from anywhere:
  python baselines/flops/eqr_count.py --task sudoku|maze --out baselines/runs/flops-full/eqr-<benchmark>.json

Dimension names: b trajectories, l cells (+ puzzle-embedding positions), v vocabulary.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'baselines/eqr'))
import eqr_common as C  # noqa: E402  (puts EqR's clone on sys.path)
import safe_load  # noqa: E402
from fullcount import FullCounter  # noqa: E402

RELEASED = Path(os.environ.get('EQR_RELEASED', Path(__file__).resolve().parents[2] / 'third_party/EqR-model'))  # locuslab/EqR-model at 89069c8c
TASKS = {'sudoku': ('sudoku-extreme', 81, 11, 25_632_306_176), 'maze': ('maze-unique', 900, 6, 13_649_050_112)}
D, B, WINDOW = 64, 128, 3


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(task, out):
    import pretrain
    from config.schema import PretrainConfig
    from dataset.common import PuzzleDatasetMetadata
    from evaluate import _apply_runtime_arch_overrides, _setup_random_reset_carry
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode
    import models.layers
    torch.set_num_threads(8)
    C.install_stand_ins()
    flash_calls = [0]
    stand_in = models.layers.flash_attn_func

    def guarded(*a, **k):
        flash_calls[0] += 1
        return stand_in(*a, **k)
    models.layers.flash_attn_func = guarded
    bench, seq_len, vocab, old_recorded = TASKS[task]
    ckpt_path = RELEASED / bench / 'eqr.pth'
    ck = safe_load.load(ckpt_path, map_location='cpu')
    cfg = PretrainConfig(**ck['config'])
    cfg.global_batch_size = 1
    meta = PuzzleDatasetMetadata(seq_len=seq_len, vocab_size=vocab, pad_id=0, ignore_label_id=0, blank_identifier_id=0,
                                 num_puzzle_identifiers=1, total_groups=1, mean_puzzle_examples=1, sets=['all'])
    model, _, _ = pretrain.create_model(cfg, meta, rank=0, world_size=1)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ck['model'].items()}
    res = model.load_state_dict(sd, strict=False)
    assert all(k.endswith(('local_weights', 'local_ids', 'puzzle_emb.weights')) for k in res.missing_keys) and not res.unexpected_keys, res
    shadow = {k.replace('_orig_mod.', ''): v for k, v in ck['ema']['shadow'].items()}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in shadow:
                p.copy_(shadow[name])  # EqR evaluates the EMA weights
    model.eval()
    eqr = model.model
    eqr.config.halt_max_steps = D
    _apply_runtime_arch_overrides(model, {'arch.halt_max_steps': D, 'arch.noise_scale': C.EVAL_NOISE,
                                          'arch.H_init_std': C.EVAL_INIT_STD, 'arch.L_init_std': C.EVAL_INIT_STD}, 0)
    _setup_random_reset_carry(model, C.EVAL_INIT_STD, 0)  # B > 1: eqr_score.py's random latent reset
    batch = dict(inputs=torch.ones(1, seq_len, dtype=torch.int32), labels=torch.ones(1, seq_len, dtype=torch.int32),
                 puzzle_identifiers=torch.zeros(1, dtype=torch.int32))

    def summ(fn):
        with FullCounter() as fc:
            r = fn()
        return fc.summary(), r

    torch.manual_seed(0)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):  # not inference_mode: SDPA would reach the counters undecomposed
        carry0 = eqr.initial_carry(batch)
        with FlopCounterMode(display=False) as fcm:
            eqr(carry=carry0, batch=batch)
        old_first = int(fcm.get_total_flops())
        carry0 = eqr.initial_carry(batch)
        first, (carry, outs) = summ(lambda: eqr(carry=carry0, batch=batch))
        assert not bool(carry.halted.any())
        with FlopCounterMode(display=False) as fcm:
            eqr(carry=carry, batch=batch)
        old_next = int(fcm.get_total_flops())
        nxt, (carry2, outs2) = summ(lambda: eqr(carry=carry, batch=batch))
        decode, _ = summ(lambda: torch.argmax(outs2['logits'], dim=-1))
        logits = [torch.randn(B, seq_len, vocab, dtype=outs2['logits'].dtype) for _ in range(WINDOW + 1)]

        def select():  # eqr_score.py: last.append(logits.float()); score; argmin
            lg = torch.stack([x.float() for x in logits])
            score = (lg[1:] - lg[:-1]).norm(dim=-1).mean(dim=(0, -1)).view(1, B)
            return score.argmin(1)
        sel, _ = summ(select)
    assert flash_calls[0] == 0, 'the flash stand-in ran: a kernel outside the count'
    for name, s, old in (('first', first, old_first), ('next', nxt, old_next)):
        assert s['classes']['matmul'] == s['matmul_only_flopcountermode'] == old, (name, s['classes']['matmul'], old)
    per_puzzle_full = B * (first['total'] + (D - 1) * nxt['total'] + decode['total']) + sel['total']
    per_puzzle_full_wc = B * (first['total_with_compare'] + (D - 1) * nxt['total_with_compare'] + decode['total_with_compare']) + sel['total_with_compare']
    per_puzzle_old = old_recorded * D * B
    rec = dict(method='EqR', benchmark=bench, setting=f'D {D}, B {B}, noise {C.EVAL_NOISE}, init std {C.EVAL_INIT_STD}, top1_conv window {WINDOW}',
               components=dict(first_nfe=first, next_nfe=nxt, decode=decode, select_over_B=sel),
               old=dict(flop_per_nfe_recorded=old_recorded, flop_first_nfe_flopcountermode=old_first, flop_next_nfe_flopcountermode=old_next,
                        source='FlopCounterMode on the first NFE (recorded value in TASKS), x D x B', per_puzzle=per_puzzle_old),
               ratio_first_nfe=first['total'] / old_first, ratio_next_nfe=nxt['total'] / old_next,
               check_matmul_class_equals_flopcountermode=dict(first=first['classes']['matmul'] == old_first == old_recorded,
                                                              next=nxt['classes']['matmul'] == old_next == old_recorded),
               per_puzzle_full=per_puzzle_full, per_puzzle_ratio=per_puzzle_full / per_puzzle_old,
               per_puzzle_full_with_compare=per_puzzle_full_wc, per_puzzle_ratio_with_compare=per_puzzle_full_wc / per_puzzle_old,
               formula='full = B x (first + (D - 1) x next + decode) + select; old = old flop per NFE x D x B; D = 64 and B = 128 from the paper '
                       '(Table 4 headline); every trajectory runs all D NFEs at evaluation (no ACT halting)',
               checks=dict(flash_stand_in_calls=flash_calls[0], device='cpu', compiled=False, sdpa='math',
                           note='EqR Attention takes torch SDPA on CPU tensors (flash_attn_func only on CUDA); no Triton or compiled code in the counted path'),
               checkpoint=str(ckpt_path), checkpoint_sha256=sha(ckpt_path), script=str(Path(__file__).resolve()), script_sha256=sha(__file__),
               command=' '.join([sys.executable, *sys.argv]), slurm_job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename,
               torch=torch.__version__, eqr_commit=C.EQR_COMMIT, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k != 'components'}, indent=1))
    for k, s in rec['components'].items():
        print(k, s['total'], s['classes'])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=TASKS, required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    main(a.task, a.out)
