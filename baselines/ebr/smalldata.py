"""Data efficiency: training sets for EBR (Energy-Based Reasoner) with the same rows as TRM's Sudoku-Extreme subsets.

EBR's loader (dataset/puzzle_dataset.py) takes every training batch from one epoch, one row per group, and drops the
epoch's last partial batch, so with fewer than 768 groups (the recipe's global batch) it has no batch at all. This script
stores each of the k groups of a TRM subset r = ceil(768 / k) times (r = 1 for k >= 768), every copy with the same rows
(the original puzzle and its augmented copies: 1,001 rows, or 100 rows in the sets named *-v99). An epoch then shuffles k r groups and draws one row from each, so a
batch holds about 768 / k draws of each original, as TRM's sampler gives by concatenating epochs. EBR is unmodified.

Sources (experiments/runs/trm/data, built by baselines/trm/trm_smalldata.py): seed s, k < 1,000: sudoku-smalldata/s<s>-n<k>;
k = 1,000: sudoku-extreme-1k-aug-1000 (every seed). Output: experiments/runs/ebr/data/sudoku-smalldata/s<s>-n<k>/{train,test},
test a symbolic link to EBR's own Sudoku test split (byte-identical to TRM's).
The environment variable DE_SET selects the set: `sudoku-smalldata` (default; 1,001 rows per puzzle) or
`sudoku-smalldata-v99` (100 rows per puzzle, every k per seed; the set of the paper's data-efficiency figure).

  python ebr/smalldata.py build [--seeds 0 1 2] [--sizes 1 3 10 30 100 300 1000]
  python ebr/smalldata.py selfcheck      # planted faults must fail the checks
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[2] / 'experiments'
TRM_DATA = Path(os.environ.get('TRM_DATA', EXPERIMENTS / 'runs' / 'trm' / 'data'))
EBR_DATA = Path(os.environ.get('EBR_DATA', EXPERIMENTS / 'runs' / 'ebr' / 'data'))
SET = os.environ.get('DE_SET', 'sudoku-smalldata')  # or sudoku-smalldata-v99 (100 rows per group)
OUT = EBR_DATA / SET
BATCH = 768
ROWS = 100 if SET.endswith('-v99') else 1001  # the original and its augmented copies (1,000; in a -v99 set: 99)
FIELDS = ('inputs', 'labels', 'puzzle_indices', 'group_indices', 'puzzle_identifiers')


def sha256(path):
    """SHA-256 of a file, as a hexadecimal string."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def source(k, seed):
    """Folder of TRM's training split with k unique puzzles for `seed`."""
    if SET != 'sudoku-smalldata':
        return TRM_DATA / SET / f's{seed}-n{k}' / 'train'  # a -v99 set: every k per seed, k = 1,000 included
    return TRM_DATA / ('sudoku-extreme-1k-aug-1000' if k == 1000 else f'sudoku-smalldata/s{seed}-n{k}') / 'train'


def replicate(x, y, k, r):
    """(k ROWS, n) rows of k groups -> (k r ROWS, n): group g's block repeated r times in a row."""
    assert len(x) == len(y) == k * ROWS
    xb, yb = x.reshape(k, 1, ROWS, -1), y.reshape(k, 1, ROWS, -1)
    rep = lambda a: np.broadcast_to(a, (k, r) + a.shape[2:]).reshape(k * r * ROWS, -1)  # noqa: E731
    return rep(xb), rep(yb)


def check(x, y, xs, ys, k, r):
    """Every stored group is one source group, byte for byte; each source group appears exactly r times."""
    assert len(x) == k * r * ROWS, 'row count'
    xg, yg = x.reshape(k * r, ROWS, -1), y.reshape(k * r, ROWS, -1)
    src = {(xs[g * ROWS:(g + 1) * ROWS].tobytes(), ys[g * ROWS:(g + 1) * ROWS].tobytes()): g for g in range(k)}
    assert len(src) == k, 'source groups not distinct'
    count = np.zeros(k, int)
    for i in range(k * r):
        g = src.get((xg[i].tobytes(), yg[i].tobytes()))
        assert g is not None, f'stored group {i} is not a source group'
        count[g] += 1
    assert (count == r).all(), f'copies per group {sorted(set(count.tolist()))}, want {r}'


def build(seeds, sizes):
    """Write one training set per seed and size k; skip a set whose subset.json exists."""
    test = EBR_DATA / 'sudoku-extreme-1k-aug-1000' / 'test'
    assert test.is_dir()
    for s in seeds:
        for k in sizes:
            out = OUT / f's{s}-n{k}'
            if (out / 'subset.json').exists():
                print(out, 'exists: skipped', flush=True)
                continue
            src = source(k, s)
            xs, ys = np.load(src / 'all__inputs.npy'), np.load(src / 'all__labels.npy')
            gs = np.load(src / 'all__group_indices.npy')
            assert len(gs) == k + 1 and (np.diff(gs) == ROWS).all(), f'{src}: not k groups of {ROWS} rows'
            r = max(1, math.ceil(BATCH / k))
            x, y = replicate(xs, ys, k, r)
            check(x, y, xs, ys, k, r)
            n, groups = len(x), k * r
            d = out / 'train'
            d.mkdir(parents=True, exist_ok=True)
            files = dict(inputs=x.astype(np.uint8), labels=y.astype(np.uint8), puzzle_indices=np.arange(n + 1, dtype=np.int32),
                         group_indices=np.arange(0, n + 1, ROWS, dtype=np.int32), puzzle_identifiers=np.zeros(n, dtype=np.int32))
            for f, v in files.items():
                tmp = d / f'all__{f}.tmp.npy'
                np.save(tmp, v)
                os.replace(tmp, d / f'all__{f}.npy')
            meta = dict(pad_id=0, ignore_label_id=0, blank_identifier_id=0, vocab_size=11, seq_len=81, num_puzzle_identifiers=1,
                        total_groups=groups, mean_puzzle_examples=1.0, sets=['all'], total_samples=n)  # as EBR's builder writes it
            (d / 'dataset.json').write_text(json.dumps(meta))
            (out / 'identifiers.json').write_text(json.dumps(['<blank>']))
            if not (out / 'test').exists():
                (out / 'test').symlink_to(test)
            rec = dict(seed=s, unique_puzzles=k, copies_per_group=r, groups=groups, rows=n, batches_per_epoch=groups // BATCH,
                       source=str(src), source_sha256={f: sha256(src / f'all__{f}.npy') for f in ('inputs', 'labels')},
                       sha256={f: sha256(d / f'all__{f}.npy') for f in files}, test=str(test),
                       script=str(Path(__file__).resolve()), script_sha256=sha256(__file__))
            (out / 'subset.json').write_text(json.dumps(rec, indent=1) + '\n')
            print(f'wrote {out}: k {k}, r {r}, {groups} groups, {n} rows, {groups // BATCH} batches per epoch', flush=True)


def selfcheck():
    """Each check must reject its planted fault."""
    src = source(3, 0)
    xs, ys = np.load(src / 'all__inputs.npy'), np.load(src / 'all__labels.npy')
    x, y = replicate(xs, ys, 3, 4)
    check(x, y, xs, ys, 3, 4)
    faults = {}
    x1 = x.copy(); x1[5 * ROWS + 7, 3] = 0; faults['one cell changed in a copy'] = (x1, y)
    x2, y2 = x.copy(), y.copy(); x2[:ROWS], y2[:ROWS] = x[ROWS * 4:ROWS * 5], y[ROWS * 4:ROWS * 5]
    faults['group 0 replaced by group 1 in one copy'] = (x2, y2)
    faults['one group missing'] = (x[:-ROWS], y[:-ROWS])
    for name, (a, b) in faults.items():
        try:
            check(a, b, xs, ys, 3, 4)
        except AssertionError as e:
            print(f'planted fault "{name}" rejected: {e}')
        else:
            raise SystemExit(f'planted fault "{name}" passed the checks')
    print('selfcheck passed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['build', 'selfcheck'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--sizes', type=int, nargs='+', default=[1, 3, 10, 30, 100, 300, 1000, 3000, 10000])
    a = p.parse_args()
    build(a.seeds, a.sizes) if a.mode == 'build' else selfcheck()
