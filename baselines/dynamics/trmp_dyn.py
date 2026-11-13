"""Latent dynamics and q-head halting of TRMP run past the trained depth.

TRMP has no public code; its model is the reimplementation in baselines/trmp/ (model.py, fit.py, score.py). Same model, weights, inputs, bf16 autocast, state update and answer rule as baselines/trmp/infer.best_of, but one
rollout per puzzle and no path noise (sigma = 0), so the run is the deterministic map of the trained model.
No rollout halts: every puzzle runs --steps supervision steps. After every step t this records, per puzzle:
the squared norms of the cell memory h, the answer state a and the carried messages (outgoing and unit replies),
and of their change since t - 1; whether the masked-argmax answer changed since t - 1; whether it is exact; and the
board halting confidence q = -logsumexp(-q_cell) (best_of halts a rollout at the first step with q > 0).
Training refills a slot after 32 supervision steps (fit.py PHASES), so the trained depth is 32.

Run from baselines/ on one GPU:
  ../.venv/bin/python dynamics/trmp_dyn.py --run runs/trmp/sudoku-extreme-s0 --steps 1024 --stop 1000 --out <prefix>
Writes <prefix>.npz and, last, <prefix>.json; exits at once when <prefix>.json exists.
Arrays in <prefix>.npz are (n, t), n puzzles and t supervision steps (1-based): z{h,a,m}2, dz{h,a,m}2, changed, exact, q.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tyro

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'trmp'))
import fit  # noqa: E402
import score  # noqa: E402
from model import RECIPES, Trainable  # noqa: E402


@dataclasses.dataclass
class Args:
    run: str  # fit folder (fit.json, model.pt)
    out: str  # output prefix
    steps: int = 256  # supervision steps per puzzle
    stop: int = 1000  # the first `stop` test puzzles
    batch: int = 250  # puzzles per batch (one rollout each)
    trained_depth: int = 32  # supervision steps per slot in training (fit.py PHASES)
    fp32: bool = False  # float32 instead of the bf16 autocast of score.py (separates convergence from bf16 rounding)


def sq(x):
    """Squared L2 norm per puzzle: x (n, ...) -> (n,), computed in float32."""
    return x.float().square().flatten(1).sum(1)


def main(a: Args):
    """Run the first a.stop test puzzles for a.steps supervision steps, one rollout each, and write the per-step records."""
    run, out = Path(a.run), Path(a.out)
    if out.with_suffix('.json').exists():
        print(f'{out}.json exists', flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    rec_fit = json.loads((run / 'fit.json').read_text())
    task = rec_fit['task']
    model = Trainable(task).cuda().eval()
    model.load_state_dict(torch.load(run / 'model.pt', map_location='cuda', weights_only=True))
    sha = hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest()
    assert sha == rec_fit['model_sha256']
    x, y, idx, desc = score.load_split(task, 'test')
    x, y, idx = x[:a.stop], y[:a.stop], idx[:a.stop]
    names = ('h', 'a', 'm')
    acc = {k: [] for k in [f'z{c}2' for c in names] + [f'dz{c}2' for c in names] + ['changed', 'exact', 'q']}
    t0 = time.perf_counter()
    with torch.no_grad():
        for lo in range(0, len(x), a.batch):
            xb, yb = x[lo:lo + a.batch].cuda(), y[lo:lo + a.batch].cuda()
            per = {k: [] for k in acc}
            with fit.amp_context(not a.fp32):
                state = model.initial(xb)
                clues = model.embedding(xb)
                prev = None
                for t in range(a.steps):
                    state = model.step(state, clues)
                    q = -torch.logsumexp(-model.halt_logits(state[1]).float(), dim=-1)
                    logits = model.symbol(state[1]).float()
                    p = logits.masked_fill(xb == 0, -torch.inf).argmax(-1)
                    parts = {'h': state[0], 'a': state[1], 'm': torch.cat((state[2].float().flatten(1), state[3].float().flatten(1)), 1)}
                    for c in names:
                        per[f'z{c}2'].append(sq(parts[c]))
                        per[f'dz{c}2'].append(torch.full_like(per[f'z{c}2'][-1], float('nan')) if prev is None else sq(parts[c].float() - prev[c].float()))
                    per['changed'].append(torch.zeros(len(xb), dtype=torch.bool, device=xb.device) if prev is None else (p != prev['p']).any(-1))
                    per['exact'].append((p == yb).all(-1))
                    per['q'].append(q)
                    prev = dict(parts, p=p)
            for k in acc:
                acc[k].append(torch.stack(per[k], 1).cpu().numpy())
            ex = np.concatenate(acc['exact'])
            print(f'{lo + len(xb)}/{len(x)}  exact@32 {ex[:, min(31, a.steps - 1)].mean():.4f} @last {ex[:, -1].mean():.4f}  {time.perf_counter() - t0:.0f} s', flush=True)
    torch.cuda.synchronize()
    rec = {k: np.concatenate(v) for k, v in acc.items()}
    res = dict(method='TRMP', task=task, run=str(run), model_sha256=sha, split=desc, n=int(len(x)), steps=a.steps, trained_depth=a.trained_depth,
               table1_step_cap=RECIPES[task]['inference_steps'], depth_unit='supervision step (cycles x rounds message passing)',
               halt_rule='q > 0 (board q = -logsumexp(-q_cell))', rollouts=1, sigma=0.0, numerics='float32' if a.fp32 else 'bf16 autocast',
               state='concatenation of cell memory h, answer state a and carried messages (fp32 copies of the state)',
               solved_per_step=rec['exact'].sum(0).tolist(), seconds=time.perf_counter() - t0, gpu=torch.cuda.get_device_name(),
               slurm_job=os.environ.get('SLURM_JOB_ID'), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    np.savez_compressed(out.with_suffix('.npz'), t=np.arange(1, a.steps + 1), dataset_index=idx, **rec)
    tmp = out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(res, indent=1) + '\n')
    os.replace(tmp, out.with_suffix('.json'))
    print(json.dumps({k: v for k, v in res.items() if k != 'solved_per_step'}, indent=1), flush=True)


if __name__ == '__main__':
    main(tyro.cli(Args))
