"""Collect the looped-dynamics passes into results/looped-dynamics.json.

The passes run each looped baseline (HRM, TRM, EqR, TRMP, FRM) past its trained depth with its authors' model code
(act_dyn.py, eqr_dyn.py, trmp_dyn.py, frm_dyn.py; submitted by submit.sh). This script calls no model code. It reads
baselines/runs/looped-dynamics/<method>/<task>-s<k>/*.json|npz and the job list jobs.txt, and writes two lists, one
entry per method x task x seed x variant:
- `e2` (state change): trained depth, the iteration grid, and at each t of the grid the median and 90th percentile over
  puzzles of the relative state change ||z_t - z_(t-1)|| / ||z_(t-1)||, the fraction of puzzles whose answer changed
  since t - 1, and the solve rate; plus the convergence verdict of the criterion in `converged`.
- `e4` (halting; models with a halting head, called the q-head): n, halted (the q-head fired at a step <= the trained
  depth), halted_wrong, wrong_halt_rate, solved (the board at the halt, or at the cap when the head never fired, is
  exact), mean halt step; also the same with the cap at the last logged step.
Array names: rel, changed, exact and the halting logits are (n, t) with n puzzles and t logged steps.

Run from baselines/ (CPU):
  ../.venv/bin/python dynamics/export.py [--root runs/looped-dynamics] [--out ../results/looped-dynamics.json]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASELINES = HERE.parent
REPO = BASELINES.parent


def grid(K, trained):
    """Sorted iteration counts to report for a run of K steps: 1, 2, 3, the trained depth, K, the powers of two and 24
    log-spaced values, all within 1..K."""
    g = {1, 2, 3, trained, K} | {2 ** i for i in range(31) if 2 ** i <= K}
    g |= set(np.unique(np.round(np.geomspace(1, K, 24)).astype(int)).tolist())
    return sorted(x for x in g if 1 <= x <= K)


def pct(x, q):
    """Percentile q of the finite entries of x (shape (n,)), or None when there is none."""
    x = x[np.isfinite(x)]
    return float(np.percentile(x, q)) if len(x) else None


def converged(t, med, trained, K):
    """The convergence criterion: median relative change at 8 x trained depth (or the last t) below 1e-3, or 10 x
    smaller than at the trained depth.

    t: sorted iteration counts; med: median relative change at each t; returns (verdict or None, median at
    8 x trained depth, median at the trained depth)."""
    i_end = int(np.searchsorted(t, min(8 * trained, K)))
    i_tr = int(np.searchsorted(t, trained))
    end, tr = med[i_end], med[i_tr]
    if end is None or tr is None:
        return None, end, tr
    return bool(end < 1e-3 or end <= tr / 10), end, tr


def halting(q_margin, exact, cap):
    """Apply the halting rule: a puzzle halts at the first step (1-based) with q_margin > 0 among steps 1..cap.

    q_margin: (n, t) halting margin per puzzle and step; exact: (n, t) bool, the board at that step is right.
    Returns the counts of one `e4` entry: halted, halted at a wrong board, solved at the halt or at the cap."""
    fire = q_margin[:, :cap] > 0
    halted = fire.any(1)
    step = np.where(halted, fire.argmax(1) + 1, cap)
    at = exact[np.arange(len(exact)), step - 1]
    n = len(exact)
    hw = int((halted & ~at).sum())
    return dict(cap=int(cap), n=int(n), halted=int(halted.sum()), halted_wrong=int(hw),
                wrong_halt_rate=(hw / int(halted.sum())) if halted.any() else None, solved=int(at.sum()),
                solve_rate=float(at.mean()), mean_halt_step=float(step.mean()),
                mean_halt_step_of_halted=float(step[halted].mean()) if halted.any() else None,
                capped_wrong=int((~halted & ~at).sum()))


def entry(js, npz, method, task, seed, variant):
    """One pass (its summary js and arrays npz) as an `e2` entry and, for a model with a q-head, an `e4` entry (else None)."""
    d = np.load(npz)
    meta = json.loads(js.read_text())
    trained = int(meta['trained_depth'])
    if method == 'FRM':
        t = d['t']
        rel = d['rel']
        changed, exact = d['changed'], d['exact']
        if 'readout_exact' in d:  # sudoku: the answer of a t-iteration run is the readout
            exact = d['readout_exact']
        sel = np.arange(len(t))
    else:
        comps = sorted(k[2:-1] for k in d.files if k.startswith('dz') and k.endswith('2'))
        num = sum(d[f'dz{c}2'] for c in comps)
        den = sum(d[f'z{c}2'] for c in comps)
        prev = np.concatenate([np.full((len(den), 1), np.nan), den[:, :-1]], 1)
        rel = np.sqrt(num) / np.sqrt(prev)
        changed, exact = d['changed'], d['exact']
        t_all = d['t']
        K = int(t_all[-1])
        t = np.array(grid(K, trained))
        sel = t - 1
    med = [pct(rel[:, i], 50) for i in sel]
    p90 = [pct(rel[:, i], 90) for i in sel]
    K = int(t[-1])
    conv, end, tr = converged(t, med, trained, K)
    e = dict(method=method, task=task, seed=seed, variant=variant, n=int(len(exact)), trained_depth=trained,
             depth_unit=meta.get('depth_unit'), state=meta.get('state'), max_iteration=K, iterations=t.tolist(),
             rel_change_median=med, rel_change_p90=p90,
             answer_change_fraction=[float(changed[:, i].mean()) if t[j] > 1 else None for j, i in enumerate(sel)],
             solve_rate=[float(exact[:, i].mean()) for i in sel],
             converged=conv, rel_change_median_at_trained_depth=tr, rel_change_median_at_8x=end,
             noise=meta.get('noise', meta.get('sigma', meta.get('renoise'))), slurm_job=meta.get('slurm_job'),
             file=str(js.relative_to(REPO)), check=meta.get('check'))
    if method == 'FRM':
        e['note'] = ('sudoku: solve rate = readout answer of a t-iteration run' if task == 'sudoku-extreme'
                     else 'maze: calls 1-256 integrate the sde_selfcond sampler, 257 is the Table-2 readout, >257 iterate the readout map at tau = 1')
    e4 = None
    if method in ('HRM', 'TRM', 'EqR', 'TRMP'):
        if method == 'HRM':
            qm = d['q_halt'] - d['q_continue']
        elif method == 'TRMP':
            qm = d['q']
        else:
            qm = d['q_halt']
        e4 = dict(method=method, task=task, seed=seed, variant=variant, halt_rule=meta['halt_rule'],
                  at_trained_cap=halting(qm, d['exact'], trained), at_max_iteration=halting(qm, d['exact'], int(d['t'][-1])),
                  slurm_job=meta.get('slurm_job'), file=str(js.relative_to(REPO)))
        if method == 'TRMP':
            e4['at_table1_cap'] = halting(qm, d['exact'], int(meta['table1_step_cap']))
            e4['note'] = 'one rollout, sigma 0; Table-2 TRMP uses 64 noisy rollouts and picks the most confident'
    return e, e4


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=BASELINES / 'runs/looped-dynamics')
    p.add_argument('--out', type=Path, default=REPO / 'results/looped-dynamics.json')
    a = p.parse_args()
    a.root = a.root.resolve()
    names = {'hrm': 'HRM', 'trm': 'TRM', 'eqr': 'EqR', 'trmp': 'TRMP', 'frm': 'FRM'}
    e2, e4 = [], []
    for m, M in names.items():
        for js in sorted((a.root / m).glob('*-s[0-9]/*.json')):
            npz = js.with_suffix('.npz')
            if js.name.endswith('.tmp') or not npz.exists():
                continue
            task, seed = js.parent.name.rsplit('-s', 1)
            x, y = entry(js, npz, M, task, int(seed), js.stem)
            e2.append(x)
            if y:
                e4.append(y)
    jobs = (a.root / 'jobs.txt').read_text().splitlines() if (a.root / 'jobs.txt').exists() else []
    doc = dict(
        description='Looped baselines run past their trained depth (e2) and their q-head halting (e4)',
        puzzles='Sudoku-Extreme: FRM 1,000 test puzzles (full-all rows 10,000-10,999); Maze-Unique: all 1,000 test mazes',
        definitions=dict(rel_change='||z_t - z_(t-1)|| / ||z_(t-1)|| per puzzle; median and 90th percentile over puzzles',
                         answer_change_fraction='fraction of puzzles whose decoded answer at t differs from t - 1',
                         wrong_halt_rate='puzzles halted by the q-head at an incorrect board / puzzles halted by the q-head',
                         converged='median rel_change at 8 x trained depth < 1e-3 or <= 1/10 of its value at the trained depth'),
        cram_comparison='CRAM energy stopping test: 0 test puzzles stopped at an incorrect board on either benchmark',
        source=dict(runs=str(a.root.relative_to(REPO)), scripts='baselines/dynamics/', jobs=[j for j in jobs if 'bench' not in j.split()[1]],
                    export_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        e2=e2, e4=e4)
    a.out.write_text(json.dumps(doc, indent=1) + '\n')
    print(f'{a.out}: {len(e2)} e2 entries, {len(e4)} e4 entries')
    for x in e2:
        print(f"{x['method']:5} {x['task']:15} s{x['seed']} {x['variant']:22} conv={x['converged']}  med@trained={x['rel_change_median_at_trained_depth']}  med@8x={x['rel_change_median_at_8x']}  solve@end={x['solve_rate'][-1]:.3f}")
    for y in e4:
        h = y['at_trained_cap']
        print(f"{y['method']:5} {y['task']:15} s{y['seed']} {y['variant']:22} halted {h['halted']}/{h['n']} wrong {h['halted_wrong']} rate {h['wrong_halt_rate']} solved {h['solved']} mean step {h['mean_halt_step']:.2f}")


if __name__ == '__main__':
    main()
