"""Correctness checks of trmp/fit.py's update before any timed fit (run on a GPU node, from baselines/).

1. reference: fit.make_update in fp32 eager equals a plain reference update (torch.optim.AdamW foreach=False and
   clip_grad_norm_; reference_run below) over K updates on the same boards: losses, norms and parameters agree to float rounding.
   Planted faults (no weight decay; rejection threshold below every norm, so the reference must still move) must fail.
2. numerics: bf16 autocast + torch.compile against fp32 eager from the same weights and boards: per-update loss
   and the relative parameter change after K updates.

  python trmp/check.py --task sudoku-extreme
"""
from __future__ import annotations

import copy
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Literal

import torch
import tyro
from torch import nn
from torch.func import functional_call, grad_and_value, vmap

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit  # noqa: E402
from model import DORMANT, RECIPES, Trainable  # noqa: E402


@dataclasses.dataclass
class Args:
    task: Literal['sudoku-extreme', 'maze-unique'] = 'sudoku-extreme'  # benchmark
    k: int = 5  # updates compared
    slots: int = 0  # slots (0: the recipe's)
    out: str = ''  # optional JSON output path


def boards(task, slots):
    cfg = fit.Config(task=task)
    host_x, host_y = fit.encode_train(cfg, fit.read_raw(cfg))
    torch.manual_seed(123)
    x, y = host_x.cuda(), host_y.cuda()
    return fit.sampled_batch(task, x, y, slots)


def reference_run(model, bx, by, k, lr=fit.LR):
    """The reference update, written with torch.optim.AdamW: fixed boards, carried state, constant learning rate."""
    parameters = dict(model.named_parameters())
    decay = [p for n, p in parameters.items() if p.ndim >= 2 and not n.startswith(DORMANT)]
    rest = [p for n, p in parameters.items() if p.ndim < 2 and not n.startswith(DORMANT)]
    optimizer = torch.optim.AdamW([{'params': decay, 'weight_decay': .04821}, {'params': rest, 'weight_decay': 0}],
                                  lr=lr, betas=(.9, .95), foreach=False)

    def loss_one(params, xx, yy, ss):
        return functional_call(model, params, (xx.unsqueeze(0), yy.unsqueeze(0), tuple(v.unsqueeze(0) for v in ss)))

    derivative = vmap(grad_and_value(loss_one, has_aux=True), in_dims=(None, 0, 0, 0), randomness='error')
    state, ref, accepted, out = model.initial(bx), torch.zeros((), device='cuda'), 0, []
    for _ in range(k):
        grads, (losses, (new_state, halt, ce)) = derivative(parameters, bx, by, state)
        norms = sum(g.flatten(1).square().sum(1) for g in grads.values()).sqrt()
        ratios = norms / losses.clamp_min(1e-12)
        median = ratios.median()
        if accepted == 0:
            ref = median
        scale = (50 * ref / ratios.clamp_min(1e-12)).clamp(max=1)
        for name, p in parameters.items():
            p.grad = None if name.startswith(DORMANT) else (grads[name] * scale.reshape(-1, *([1] * p.ndim))).mean(0)
        norm = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.grad is not None], .5)
        ok = bool(torch.isfinite(norm) & (norm <= 50))
        if ok:
            optimizer.step()
            accepted += 1
            ref = (.99 * ref + .01 * median).clamp(min=ref * .986, max=ref * 1.014)
        state = tuple(v.squeeze(1).float() for v in new_state)
        out.append(dict(loss=float(losses.mean()), norm=float(norm), accepted=ok))
    return out


def fit_run(model, bx, by, k, amp, compiled, lr=fit.LR):
    names = [n for n, _ in model.named_parameters() if not n.startswith(DORMANT)]
    P = [model.get_parameter(n) for n in names]
    M, V, E = ([torch.zeros_like(p) for p in P] for _ in range(3))
    acc, rejected = (torch.zeros((), dtype=torch.long, device='cuda') for _ in range(2))
    ref, lr_t = torch.zeros((), device='cuda'), torch.full((), lr, device='cuda')
    update = fit.make_update(model, names, amp)
    if compiled:
        update = torch.compile(update)
    state, out = model.initial(bx), []
    for i in range(k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        state, halt, loss, ce, norm = update(P, M, V, E, acc, ref, rejected, bx, by, state, lr_t)
        state = tuple(v.contiguous() for v in state)
        torch.cuda.synchronize()
        out.append(dict(loss=float(loss), norm=float(norm), accepted=int(acc), seconds=time.perf_counter() - t))
    return out


def max_param_diff(a, b):
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    return max(float((pa[n] - pb[n]).abs().max()) for n in pa)


def rel_change_diff(init, a, b):
    """|(a - init) - (b - init)| / |a - init|, over all trained parameters."""
    pi, pa, pb = dict(init.named_parameters()), dict(a.named_parameters()), dict(b.named_parameters())
    num = sum(float(((pa[n] - pb[n]) ** 2).sum()) for n in pa) ** .5
    den = sum(float(((pa[n] - pi[n]) ** 2).sum()) for n in pa) ** .5
    return num / den


def main(args: Args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    slots = args.slots or RECIPES[args.task]['slots']
    bx, by = boards(args.task, slots)
    torch.manual_seed(0)
    init = Trainable(args.task).cuda()
    for p in init.parameters():
        p.requires_grad_(False)
    report = dict(task=args.task, slots=slots, k=args.k)

    # 1. reference equality (fp32 eager), then planted faults.
    ref_model, ours = copy.deepcopy(init), copy.deepcopy(init)
    ref_log = reference_run(ref_model, bx, by, args.k)
    our_log = fit_run(ours, bx, by, args.k, amp=False, compiled=False)
    report['reference'] = dict(ref=ref_log, ours=our_log, max_param_diff=max_param_diff(ref_model, ours),
                               max_loss_diff=max(abs(a['loss'] - b['loss']) for a, b in zip(ref_log, our_log)))
    passed = report['reference']['max_param_diff'] < 1e-5 and report['reference']['max_loss_diff'] < 1e-4
    report['reference']['passed'] = passed
    planted = {}
    for name, patch in (('no_weight_decay', dict(WD=0.0)), ('reject_everything', dict(REJECT=-1.0))):
        saved = {k: getattr(fit, k) for k in patch}
        for kk, vv in patch.items():
            setattr(fit, kk, vv)
        faulty = copy.deepcopy(init)
        fit_run(faulty, bx, by, args.k, amp=False, compiled=False)
        for kk, vv in saved.items():
            setattr(fit, kk, vv)
        d = max_param_diff(ref_model, faulty)
        planted[name] = dict(max_param_diff=d, detected=d >= 1e-5)
    report['planted'] = planted
    print(json.dumps(report, indent=1), flush=True)

    # 2. bf16 + compile against fp32 eager (both from init, same boards).
    fast = copy.deepcopy(init)
    t = time.perf_counter()
    fast_log = fit_run(fast, bx, by, args.k, amp=True, compiled=True)
    report['bf16_compile'] = dict(log=fast_log, wall_s=time.perf_counter() - t,
                                  loss_rel_diff=[abs(a['loss'] - b['loss']) / abs(b['loss']) for a, b in zip(fast_log, our_log)],
                                  norm_rel_diff=[abs(a['norm'] - b['norm']) / abs(b['norm']) for a, b in zip(fast_log, our_log)],
                                  param_change_rel_diff=rel_change_diff(init, ours, fast),
                                  peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
    bf16_eager = copy.deepcopy(init)
    report['bf16_eager_vs_compile_param_change_rel_diff'] = None
    bf16_log = fit_run(bf16_eager, bx, by, args.k, amp=True, compiled=False)
    report['bf16_eager'] = dict(log=bf16_log, param_change_rel_diff_vs_fp32=rel_change_diff(init, ours, bf16_eager))
    report['bf16_eager_vs_compile_param_change_rel_diff'] = rel_change_diff(init, bf16_eager, fast)
    ok = passed and all(v['detected'] for v in planted.values())
    report['all_reference_checks_passed'] = ok
    print(json.dumps({k: report[k] for k in ('bf16_compile', 'bf16_eager', 'bf16_eager_vs_compile_param_change_rel_diff',
                                             'all_reference_checks_passed')}, indent=1), flush=True)
    if args.out:
        fit.write_json(args.out, report)
    assert ok, 'reference check failed'


if __name__ == '__main__':
    main(tyro.cli(Args))
