"""N-Queens Cov@20 (coverage of a puzzle's solutions by 20 samples) with the any-valid rule; read-only (no training
or search). multi_solution_export.py uses these functions for every method, and `--mode ebr` writes
results/nqueens-coverage-ebr.json from the samples of the baseline EBR (see below).

Rule (EBR's evaluator): a sample counts when it is a valid board that holds every given queen; the
distinct counted samples among a row's 20, capped at min(S, 20), over min(S, 20), S the puzzle's number of solutions.
GRAM's own evaluator instead counts only samples equal to one of the puzzle's first 20 completions in enumeration order
(rows, then columns ascending), which undercounts the 19 test puzzles of 10 x 10 with more than 20 solutions.
Two pooled forms (both: completions found over completions available):
- per puzzle: every distinct test puzzle counts once; its coverage is the mean over its rows (EBR's form; EBR samples
  once per puzzle);
- per row: every test row counts (a puzzle with S solutions occurs S times in the test file; GRAM's evaluator loops over
  rows).
The EBR mode needs EBR's Python environment and EBR's source on the path, because it reads EBR's samples
(baselines/runs/ebr/nqueens<n>-s<seed>/score/nq.npz) at EBR's stopping rule (|grad_M E| < 0.2 at a segment end, else
the trained depth; evaluators.gram.gtan_stop_index in EBR's code). From experiments/, with EBR's code cloned to
third_party/ebr:
  PYTHONPATH=$PWD/..:$PWD/../third_party/ebr <python of EBR's environment> nqueens_coverage.py --mode ebr
Dimension names: r test rows, k samples, c cells, u distinct puzzles.
"""
import json
import sys
from pathlib import Path

import numpy as np

from cram import cram_nqueens as Q

HERE = Path(__file__).resolve().parent
EBR_RUNS = HERE.parent / 'baselines' / 'runs' / 'ebr'
OUT_EBR = HERE.parent / 'results' / 'nqueens-coverage-ebr.json'


def puzzles(inputs):
    """Distinct puzzles of the test rows inputs (r, c) bool: (uniq (u, c) the puzzles, inv (r,) the puzzle index of
    every row, S (u,) the number of solutions of every puzzle, by enumeration)."""
    uniq, inv = np.unique(inputs, axis=0, return_inverse=True)
    n = int(round(inputs.shape[1] ** 0.5))
    return uniq, inv.ravel(), np.array([len(Q.completions(x, n)) for x in uniq], float)


def row_coverage(inputs, boards, n, S_row):
    """Per row: distinct samples that are valid and hold every given queen, capped at min(S, 20), over min(S, 20).
    inputs (r, c) bool, boards (r, k, c) bool, S_row (r,) the number of solutions of each row's puzzle.
    Returns (coverage (r,) in [0, 1], the caps min(S, 20) (r,))."""
    ok = Q.valid(boards, n) & (boards >= inputs[:, None, :]).all(-1)
    k = np.minimum(S_row, 20)
    found = np.array([len({b.tobytes() for b in boards[i][ok[i]]}) for i in range(len(boards))])
    return np.minimum(found, k) / k, k


def pooled(inv, cov, k):
    """(per row, per puzzle) pooled coverage in %, from per-row coverage cov and caps k."""
    U = inv.max() + 1
    cov_u = np.bincount(inv, weights=cov, minlength=U) / np.bincount(inv, minlength=U)
    k_u = np.zeros(U)
    k_u[inv] = k
    return float(100 * (cov * k).sum() / k.sum()), float(100 * (cov_u * k_u).sum() / k_u.sum())


def ceiling(inv, S):
    """Expected coverage, in %, of 20 independent uniform draws over each puzzle's solutions, every draw valid: the
    value that a perfect sampler reaches. inv (r,), S (u,) as returned by puzzles. Returns (per row, per puzzle)."""
    rows = np.bincount(inv, minlength=len(S))
    k = np.minimum(S, 20)
    c = np.minimum(S * (1 - (1 - 1 / S) ** 20), 20) / k
    return float(100 * (rows * c * k).sum() / (rows * k).sum()), float(100 * (c * k).sum() / k.sum())


def ebr():
    """Write results/nqueens-coverage-ebr.json: EBR's Cov@20 per board size and seed, per row and per puzzle, with
    mean and sample SD over seeds 0-2."""
    from evaluators.gram import gtan_stop_index  # EBR's source (wwillchen/ebr 8c5271c)
    out = {}
    for n in (8, 10):
        seeds = []
        for s in (0, 1, 2):
            f = EBR_RUNS / f'nqueens{n}-s{s}' / 'score' / 'nq.npz'
            a = np.load(f)
            stop = gtan_stop_index(a['gtan'], 0.2, int(a['train_halt']))
            K, U = stop.shape
            ki, ui = np.meshgrid(np.arange(K), np.arange(U), indexing='ij')
            boards = (a['decode'][stop, ki, ui] == 2).transpose(1, 0, 2)  # (u, k, c): one sample set per puzzle
            given = a['inputs'] == 2
            rows = np.diff(a['puzzle_indices']).astype(int)
            S = np.array([len(Q.completions(x, n)) for x in given], float)
            cov, k = row_coverage(given, boards, n, S)
            seeds.append(dict(seed=s, per_row=float(100 * (rows * cov * k).sum() / (rows * k).sum()),
                              per_puzzle=float(100 * (cov * k).sum() / k.sum()), npz=str(f)))
        out[str(n)] = dict(per_seed=seeds, **{key: dict(mean=float(np.mean([x[key] for x in seeds])),
                                                        sd=float(np.std([x[key] for x in seeds], ddof=1)))
                                              for key in ('per_row', 'per_puzzle')})
    OUT_EBR.write_text(json.dumps(dict(rule=__doc__.split('Two pooled')[0].strip(), stop='|grad_M E| < 0.2, else the trained depth',
                                       sizes=out), indent=1) + '\n')
    print(json.dumps({n: {k: v for k, v in d.items() if k != 'per_seed'} for n, d in out.items()}, indent=1))


if __name__ == '__main__':
    if '--mode' in sys.argv and sys.argv[sys.argv.index('--mode') + 1] == 'ebr':
        ebr()
