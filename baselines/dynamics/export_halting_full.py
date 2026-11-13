"""Halting of the looped models on the full Sudoku-Extreme test set: writes results/halting-full.json.

This script calls no model code. It reads the passes of dynamics/submit_halting_full.sh
(runs/looped-dynamics/halting-full/<method>/sudoku-extreme-s<k>/, written by act_dyn.py, eqr_dyn.py and trmp_dyn.py
with the authors' models), each one rollout per puzzle for the trained depth on all 412,786 test puzzles (full-all rows
10,000 and up), and results/ebr-halting.json (subset `unseen`, EBR's first sample, tau 0.2, cap 64). Definitions as
dynamics/export.py: halted = the stopping rule fired at a step <= the cap; wrong halt = halted at an incorrect board;
solved = the board at the halt, or at the cap when the rule never fired, is exact.

Run from baselines/ (CPU):
  ../.venv/bin/python dynamics/export_halting_full.py
A method appears in `rows` only when all three seeds are present; `missing` lists the passes that are not.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from export import halting  # noqa: E402

CODE = HERE.parents[1]
ROOT = CODE / 'baselines/runs/looped-dynamics/halting-full'
N = 412786
PASSES = {'HRM': ('hrm', 'act16', 16), 'TRM': ('trm', 'act16', 16), 'EqR': ('eqr', 'd16-noise0p01', 16), 'TRMP': ('trmp', 'steps32', 32)}


def one(method, seed):
    """Halting counts of one method and training seed at its trained depth, or None when the pass does not exist."""
    folder, name, cap = PASSES[method]
    js, npz = (ROOT / folder / f'sudoku-extreme-s{seed}' / f'{name}.{e}' for e in ('json', 'npz'))
    if not js.exists():
        return None
    meta, d = json.loads(js.read_text()), np.load(npz)
    assert meta['n'] == N == len(d['exact']) and d['exact'].shape[1] == cap, (method, seed, meta['n'], d['exact'].shape)
    if method == 'HRM':
        qm = d['q_halt'] - d['q_continue']
    elif method == 'TRMP':
        qm = d['q']
    else:
        qm = d['q_halt']
    h = halting(qm, d['exact'], cap)
    return dict(seed=seed, halt_rule=meta['halt_rule'], file=str(js.relative_to(CODE)), slurm_job=meta.get('slurm_job'), seconds=meta.get('seconds'),
                check=meta.get('check'), npz_sha256=hashlib.sha256(npz.read_bytes()).hexdigest(), **h)


def pooled(seeds):
    """Sum the per-seed counts and give the halted, wrong-halt and solved percentages."""
    n, h, w, s = (sum(x[k] for x in seeds) for k in ('n', 'halted', 'halted_wrong', 'solved'))
    return dict(n=n, halted=h, halted_wrong=w, solved=s, halted_pct=100 * h / n, wrong_halt_pct=100 * w / max(h, 1), solved_pct=100 * s / n)


def main():
    rows, missing = {}, []
    for method in PASSES:
        seeds = [one(method, k) for k in range(3)]
        missing += [f'{method} s{k}' for k, x in enumerate(seeds) if x is None]
        if all(seeds):
            rows[method] = dict(per_seed=seeds, pooled=pooled(seeds), cap=PASSES[method][2], mode='one rollout per puzzle, cap at the trained depth')
    ebr = json.loads((CODE / 'results/ebr-halting.json').read_text())
    tau = str(ebr['main_tau'])
    if all('unseen' in ebr['sudoku'][f's{k}'] for k in range(3)):
        v = [ebr['sudoku'][f's{k}']['unseen'][tau]['single'] for k in range(3)]
        assert all(x['n'] == N for x in v)
        seeds = [dict(seed=k, n=x['n'], halted=x['halted'], halted_wrong=x['wrong_halts'], solved=round(x['solved_pct'] * x['n'] / 100)) for k, x in enumerate(v)]
        rows['EBR'] = dict(per_seed=seeds, pooled=pooled(seeds), cap=64, mode=f'first sample, gradient norm below {tau} or 64 segments (results/ebr-halting.json, unseen)')
    else:
        missing.append('EBR unseen')
    out = dict(description='Halting of looped models on the full Sudoku-Extreme test set',
               puzzles='Sudoku-Extreme: all 412,786 test puzzles (full-all rows 10,000-422,785)', n_per_seed=N, rows=rows, missing=missing,
               complete=not missing)
    (CODE / 'results/halting-full.json').write_text(json.dumps(out, indent=1) + '\n')
    for m, r in rows.items():
        p = r['pooled']
        print(f"{m:5} halted {p['halted_pct']:.2f}%  wrong halts {p['halted_wrong']}/{p['halted']} = {p['wrong_halt_pct']:.3f}%  solved {p['solved_pct']:.2f}%")
    print('missing:', missing)


if __name__ == '__main__':
    main()
