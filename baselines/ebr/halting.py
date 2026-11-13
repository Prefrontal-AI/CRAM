"""The stopping rule of EBR (Energy-Based Reasoner) as a halting test, from its saved per-segment records.

EBR stops a sample at its first segment end with |grad_M E| < tau (report.py `settings`). This script asks of that rule
the question the halting table asks of each looped model: how often does a sample stop before the cap, and how often is
its board wrong when it stops?
Nothing is re-run: EBR's evaluators saved, for every sample and segment end, whether the decoded board is exact and
the tangential gradient norm (baselines/runs/ebr/<task>-s<k>/score).

Definitions (as the paper's halting table): halted = the sample stopped before the cap; wrong halt = halted and its board at the stop
is incorrect (percent of halted); solved = the board at the stop, or at the cap, is correct. "single" uses sample 0
only, EBR's single-trajectory mode and the analogue of TRMP's single rollout; "pooled" counts every sample.
Subsets: Sudoku-Extreme frm1k = FRM's 1,000 test puzzles, sweep10k = the compute sweep's 10,000,
unseen = all 412,786 test puzzles (full-all rows 10,000 and up; the row of the paper's halting table);
Maze-Unique all 1,000 test mazes. Caps: 64 segments on Sudoku-Extreme (the compute sweep's most accurate setting,
gtan0.2-cap64-N8), 16 on Maze-Unique (the scored depth). Dimensions: d = segments, p = puzzles, b = samples.

  python baselines/ebr/halting.py      (CPU only, about 8 GB of memory; reads 7 GB of Sudoku shards)
Reads baselines/runs/ebr/{sudoku,mazeunique}-s<k>/score (written by ebr/score.sbatch). Writes results/ebr-halting.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import report  # noqa: E402

CODE = HERE.parents[1]
RUNS = CODE / 'baselines/runs/ebr'
TAUS = (0.1, 0.2, 0.3, 0.5)
MAIN_TAU = 0.2


def halting(exact, gtan, cap, tau):
    """Halting statistics of the rule |grad_M E| < tau with at most `cap` segments.

    exact [d, p, b] bool: the decoded board is correct at that segment end. gtan [d, p, b]: |grad_M E| there.
    Returns {'single': stats of sample 0, 'pooled': stats over all samples}."""
    hit = gtan[:cap] < tau
    halted = hit.any(0)  # [p, b]
    stop = np.where(halted, hit.argmax(0), cap - 1)
    pi, bi = np.meshgrid(np.arange(exact.shape[1]), np.arange(exact.shape[2]), indexing='ij')
    ok = exact[stop, pi, bi]
    out = {}
    for name, sl in (('single', np.s_[:, :1]), ('pooled', np.s_[:, :])):
        h, o = halted[sl], ok[sl]
        out[name] = dict(n=int(h.size), halted_pct=100 * float(h.mean()), wrong_halts=int((h & ~o).sum()), halted=int(h.sum()),
                         wrong_halt_pct=100 * float((h & ~o).sum() / max(h.sum(), 1)), solved_pct=100 * float(o.mean()),
                         mean_stop_segment=float(stop[sl].mean() + 1))
    return out


def unseen_official_rows():
    """Official test rows of full-all[10000:], the paper's 412,786 test puzzles, in full-all order."""
    idx = json.loads((CODE / 'data/full-test/index.json').read_text())
    names = [n for n in idx['full_all_order'] if n != 'full-dev10k']
    assert names[0] == 'full-frmtest' and idx['full_all_order']['full-frmtest'] == [10000, 1000]
    rows = np.concatenate([np.array(idx['shards'][n]['official_rows']) for n in names])
    assert len(rows) == 412786 == len(set(rows.tolist()))
    return rows


def main():
    """Compute the statistics for the three seeds and write results/ebr-halting.json."""
    rec = dict(script=str(Path(__file__).relative_to(CODE)), taus=TAUS, main_tau=MAIN_TAU, sudoku={}, maze={})
    rows = report.sweep_official_rows()
    unseen = unseen_official_rows()
    for s in range(3):
        exact, _, gtan, metas, _ = report.load_sudoku(RUNS / f'sudoku-s{s}/score')
        assert exact.shape[0] >= 64
        r = {}
        for sub, idx in (('frm1k', rows[:1000]), ('sweep10k', rows), ('unseen', unseen)):
            ex, gt = exact[:, idx], gtan[:, idx]
            r[sub] = {str(t): halting(ex, gt, 64, t) for t in TAUS}
        rec['sudoku'][f's{s}'] = dict(checkpoint=metas[0]['checkpoint'], breadth=metas[0]['breadth'], cap=64, **r)
        del exact, gtan
        a = np.load(RUNS / f'mazeunique-s{s}/score/maze.npz')
        meta = json.loads((RUNS / f'mazeunique-s{s}/score/maze.json').read_text())['meta']
        D, n = a['exact'].shape[0], meta['n']
        M = a['exact'].shape[1] // n
        ex = a['exact'].reshape(D, M, n).astype(bool)
        gt = a['gtan'].reshape(D, M, n).astype(np.float64)
        rec['maze'][f's{s}'] = dict(checkpoint=meta['checkpoint'], breadth=n, cap=D, all={str(t): halting(ex, gt, D, t) for t in TAUS})
    out = CODE / 'results/ebr-halting.json'
    out.write_text(json.dumps(rec, indent=1) + '\n')
    for task, sub in (('sudoku', 'frm1k'), ('sudoku', 'sweep10k'), ('sudoku', 'unseen'), ('maze', 'all')):
        for t in TAUS:
            for mode in ('single', 'pooled'):
                v = [rec[task][f's{s}'][sub][str(t)][mode] for s in range(3)]
                h, w = sum(x['halted'] for x in v), sum(x['wrong_halts'] for x in v)
                print(f"{task:6s} {sub:8s} tau={t:<4} {mode:6s} halted {np.mean([x['halted_pct'] for x in v]):6.2f}%  "
                      f"wrong halts {w}/{h} = {100 * w / max(h, 1):5.2f}%  solved {np.mean([x['solved_pct'] for x in v]):6.2f}%")


if __name__ == '__main__':
    main()
