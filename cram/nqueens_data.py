"""The N-Queens data of GRAM (a baseline model of the paper), the rule checker and GRAM's metrics, shared by CRAM
(cram_nqueens.py) and the scoring of the baselines.

Data source: GRAM's own dataset builder (https://github.com/ahn-ml/gram-release at commit f2e32b9,
`dataset/build_nqueens_dataset.py`, unmodified). `scripts/prepare_data.py multi-solution` runs it and writes
`<data>/gram/nqueens-{8,10}/{train,test}/all__{inputs,labels,num_solutions}.npy` (`<data>` is `data/` or `CRAM_DATA`;
`NQ_DATA` overrides the folder `<data>/gram`). The folder path must not contain "sudoku" when GRAM is trained on it:
GRAM chooses its checker from the data path.
A puzzle is a valid board of n queens with some queens removed (5, 6 or 7 of 8; 7, 8 or 9 of 10), so the input holds
1 to 3 given queens. An input with s completions occurs in s rows, one row per completion as its label.
Sizes (rows / distinct inputs): 8 x 8 train 7,224 / 4,376, test 1,240 / 772; 10 x 10 train 108,234 / 36,907,
test 18,466 / 6,513. GRAM divides train and test by distinct input.
Token encoding in the files: 0 pad, 1 empty, 2 queen; n * n tokens in row-major order. In this code a board is
(..., n * n) bool, True = queen.
Splits: 'train' and 'test' as GRAM built them; `carve_validation` holds out a seeded 10% of the distinct training
inputs as the validation set on which settings are chosen. No augmentation is applied.

NumPy only. The rules (rows, columns, diagonals) appear only in this file and only for scoring, never in CRAM's
fitting or search. Dimension names: r rows of a split, k samples per row, c cells (n * n).
"""
import os
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get('NQ_DATA', Path(os.environ.get('CRAM_DATA', Path(__file__).resolve().parents[1] / 'data')) / 'gram'))


def load_split(n, split):
    """GRAM's rows of `split` ('train' or 'test'): (inputs (r, c) bool given queens, labels (r, c) bool the row's
    solution, num_solutions (r,) as GRAM's builder counted them). An input with s solutions occurs in s rows."""
    d = DATA / f'nqueens-{n}' / split
    inputs, labels = np.load(d / 'all__inputs.npy'), np.load(d / 'all__labels.npy')
    if inputs.shape != labels.shape or inputs.shape[1] != n * n or not np.isin(inputs, [1, 2]).all() \
            or not np.isin(labels, [1, 2]).all():
        raise ValueError('unexpected GRAM N-Queens encoding (1 empty, 2 queen)')
    return inputs == 2, labels == 2, np.load(d / 'all__num_solutions.npy')


def load_tokens(n, split):
    """GRAM's raw token arrays (inputs, labels), (r, c) uint8 with 1 empty and 2 queen, as GRAM reads them."""
    d = DATA / f'nqueens-{n}' / split
    return np.load(d / 'all__inputs.npy'), np.load(d / 'all__labels.npy')


def carve_validation(inputs, fraction=0.1, seed=0):
    """Rows of the training split held out for choices: a seeded `fraction` of the distinct inputs (GRAM splits
    train/test by distinct input, so no validation input occurs in the fitting rows). Returns (fit_rows, val_rows)."""
    _, inverse = np.unique(inputs, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    held = np.random.default_rng(seed).permutation(inverse.max() + 1)[:int(round(fraction * (inverse.max() + 1)))]
    is_val = np.isin(inverse, held)
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def valid(boards, n):
    """(...,) bool: GRAM's verify_nqueens. Exactly one queen per row and column and at most one per diagonal and
    anti-diagonal (so exactly n queens). It does not compare the board with the input; see `completes`."""
    q = np.asarray(boards).reshape(*np.shape(boards)[:-1], n, n).astype(np.int32)
    rows = (q.sum(-1) == 1).all(-1)
    cols = (q.sum(-2) == 1).all(-1)
    flipped = q[..., ::-1]
    diag = np.stack([np.diagonal(q, k, -2, -1).sum(-1) for k in range(-n + 1, n)], -1)
    anti = np.stack([np.diagonal(flipped, k, -2, -1).sum(-1) for k in range(-n + 1, n)], -1)
    return rows & cols & (diag <= 1).all(-1) & (anti <= 1).all(-1)


def completes(boards, inputs, n):
    """(...,) bool: a valid board that holds every given queen of its input (boards (r, ..., c), inputs (r, c))."""
    held = np.asarray(boards) >= np.asarray(inputs).reshape(len(inputs), *([1] * (np.ndim(boards) - 2)), -1)
    return valid(boards, n) & held.all(-1)


def completions(given, n, limit=None):
    """Every valid board holding the given queens ((c,) bool), in GRAM's find_all_solutions order (free rows ascending,
    columns ascending), at most `limit`; a list of (c,) bool boards."""
    queens = [(r, col) for r, col in zip(*np.nonzero(np.asarray(given).reshape(n, n)))]
    cols, d1, d2 = {col for _, col in queens}, {r - col for r, col in queens}, {r + col for r, col in queens}
    todo = [r for r in range(n) if r not in {r for r, _ in queens}]
    found = []

    def back(k, placed):
        if limit is not None and len(found) >= limit:
            return
        if k == len(todo):
            board = np.zeros(n * n, bool)
            for r, col in queens + placed:
                board[r * n + col] = True
            found.append(board)
            return
        r = todo[k]
        for col in range(n):
            if col in cols or (r - col) in d1 or (r + col) in d2:
                continue
            cols.add(col), d1.add(r - col), d2.add(r + col)
            back(k + 1, placed + [(r, col)])
            cols.discard(col), d1.discard(r - col), d2.discard(r + col)

    back(0, [])
    return found


def gram_metrics(inputs, boards, n, tiebreak=None, k_values=(1, 2, 5, 10, 20)):
    """GRAM's N-Queens metrics (evaluators/nqueens.py) for k samples per test row; boards (r, k, c) bool.

    Correct = `valid` (GRAM's rule, which ignores the input's queens). Majority: the most frequent board among the k,
    ties to the higher mean `tiebreak` (r, k) (GRAM: sigmoid(q_halt); CRAM: minus the energy), then to the earliest
    sample. pass@j: any of the first j samples correct. Coverage (GRAM's Cov@20 with k = 20): distinct correct boards
    among the first min(20, all) completions in find_all_solutions order, over their number. Also, beyond GRAM:
    `*_completion` versions that require the input's queens, and coverage over all completions (no cap of 20).
    Returns a dict of per-row arrays; the paper's numbers are their means over rows."""
    inputs, boards = np.asarray(inputs, bool), np.asarray(boards, bool)
    r, k, _ = boards.shape
    tiebreak = np.zeros((r, k)) if tiebreak is None else np.asarray(tiebreak, float)
    ok = valid(boards, n)
    okc = completes(boards, inputs, n)
    out = {f'pass@{j}': ok[:, :j].any(1) for j in k_values if j <= k}
    out.update({f'pass@{j}_completion': okc[:, :j].any(1) for j in k_values if j <= k})
    majority, majority_c = np.zeros(r, bool), np.zeros(r, bool)
    cov20, cov_all, distinct_valid, n_solutions = (np.zeros(r) for _ in range(4))
    cache = {}
    for i in range(r):
        keys = [boards[i, j].tobytes() for j in range(k)]
        groups = {}
        for j, key in enumerate(keys):
            groups.setdefault(key, []).append(j)
        top = max(groups.values(), key=lambda js: (len(js), float(np.mean(tiebreak[i, js]))))
        majority[i], majority_c[i] = ok[i, top[0]], okc[i, top[0]]
        given_key = inputs[i].tobytes()
        if given_key not in cache:
            every = completions(inputs[i], n)
            cache[given_key] = ({b.tobytes() for b in every[:20]}, len(every))
        first20, total = cache[given_key]
        found = {key for j, key in enumerate(keys) if okc[i, j]}
        cov20[i] = len(found & first20) / len(first20)
        cov_all[i] = len(found) / total
        distinct_valid[i] = len(found)
        n_solutions[i] = total
    out.update(majority=majority, majority_completion=majority_c, coverage20=cov20, coverage_all=cov_all,
               distinct_valid=distinct_valid, n_solutions=n_solutions)
    return out
