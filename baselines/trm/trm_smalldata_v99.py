"""Sudoku-Extreme training sets of k unique puzzles with 100 boards each (the original and 99 augmented copies), in TRM's
data format, for the data-efficiency figure. TRM and EBR train on the same rows as CRAM (experiments/data_efficiency.py
with DE_VARIANTS=99).

- k <= 10,000: the subset of trm_smalldata.py (k = 1,000: TRM's 1,000-puzzle set) with each 1,001-row group cut to its
  first 100 rows (row 0 is the original in TRM's builder, rows 1-99 its first 99 stored copies).
- k = 30,000 and 100,000: the 10,000-puzzle set cut as above, plus puzzles that the same generator draws next:
  default_rng(s) replays trm_smalldata.py's draws (permutation(1000), choice(pool, 2000), choice(pool without the 2,000,
  7,000)) and then draws choice(pool without the 9,000, 90,000). Each new puzzle gets 99 copies from TRM's shuffle_sudoku,
  in order, in one np.random stream seeded 20,000 + s. k = 30,000 takes the first 20,000 new puzzles, k = 100,000 all 90,000,
  so 10,000 ⊂ 30,000 ⊂ 100,000 row for row (checked).
Output: experiments/runs/trm/data/sudoku-smalldata-v99/s<s>-n<k>/{train,test}, test a link to the official test split.

Run with TRM's Python (it imports shuffle_sudoku from the TRM clone), on a compute node, after trm_smalldata.py:
  python baselines/trm/trm_smalldata_v99.py build [--seeds 0 1 2] [--sizes 1 3 10 30 100 300 1000 3000 10000 30000 100000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trm_smalldata as T  # noqa: E402

KEEP = 99  # stored copies kept per original
SRC_ROWS = 1 + T.NUM_AUG  # 1,001 rows per group in the source sets
OUT = T.DATA / 'sudoku-smalldata-v99'
AUG_SEED_OFFSET = 20_000
BIG = (30000, 100000)


def source(k, s):
    return T.REF if k == 1000 else T.OUT / f's{s}-n{k}'


def cut(k, s):
    d = source(k, s) / 'train'
    x, y = np.load(d / 'all__inputs.npy'), np.load(d / 'all__labels.npy')
    g = np.load(d / 'all__group_indices.npy')
    assert len(g) == k + 1 and (np.diff(g) == SRC_ROWS).all() and g[-1] == len(x)
    keep = np.arange(len(x)) % SRC_ROWS <= KEEP
    return x[keep], y[keep]


def new_puzzles(s, qs, ans, pool):
    """The 90,000 puzzles drawn after trm_smalldata.py's 9,000, with KEEP shuffle_sudoku copies each (TRM tokens)."""
    sys.path.insert(0, str(T.TRM_SRC / 'dataset'))
    from build_sudoku_dataset import shuffle_sudoku  # TRM's augmentation, unmodified
    rng = np.random.default_rng(s)
    rng.permutation(1000)
    extra = rng.choice(pool, size=2000, replace=False)
    more = rng.choice(np.setdiff1d(pool, extra), size=7000, replace=False)
    rec10k = json.loads((T.OUT / f's{s}-n10000' / 'subset.json').read_text())
    assert rec10k['train_csv_rows'][1000:] == [int(i) for i in extra] + [int(i) for i in more], 'replayed draws differ from n10000'
    further = rng.choice(np.setdiff1d(pool, np.concatenate([extra, more])), size=90000, replace=False)
    np.random.seed(AUG_SEED_OFFSET + s)
    xs, ys = [], []
    for i in further:
        q = np.frombuffer(qs[i].replace('.', '0').encode(), np.uint8).reshape(9, 9) - ord('0')
        a = np.frombuffer(ans[i].encode(), np.uint8).reshape(9, 9) - ord('0')
        gx, gy = [q], [a]
        for _ in range(KEEP):
            b, c = shuffle_sudoku(q, a)
            gx.append(b)
            gy.append(c)
        xs.append(np.stack(gx).reshape(-1, 81) + 1)
        ys.append(np.stack(gy).reshape(-1, 81) + 1)
    return np.concatenate(xs), np.concatenate(ys), [int(i) for i in further], rec10k['train_csv_rows']


def pool_rows():
    """train.csv and the pool trm_smalldata.py draws from (every row whose question is not one of the 1,000 originals)."""
    X = np.load(T.REF / 'train' / 'all__inputs.npy', mmap_mode='r')
    G = np.load(T.REF / 'train' / 'all__group_indices.npy')
    qs, ans = T.read_pool()
    ref_q = {T.to_question(X[G[g]]) for g in range(1000)}
    return qs, ans, np.array([i for i, q in enumerate(qs) if q not in ref_q])


def build(seeds, sizes):
    T.NUM_AUG = KEEP  # T.save and T.check_rows: groups of 1 + KEEP rows
    T.OUT_V99 = OUT
    pool = None
    for s in seeds:
        big = None
        for k in sizes:
            out = OUT / f's{s}-n{k}'
            if (out / 'subset.json').exists():
                print(out, 'exists: skipped', flush=True)
                continue
            extra = dict(seed=s, size=k, kept_copies=KEEP, builder=str(Path(__file__).resolve()), builder_sha256=T.sha256(__file__))
            if k in BIG:
                if pool is None:
                    qs, ans, pool = pool_rows()
                if big is None:
                    big = new_puzzles(s, qs, ans, pool)
                bx, by, further, rows10k = big
                x10, y10 = cut(10000, s)
                m = (k - 10000) * (1 + KEEP)
                x, y = np.concatenate([x10, bx[:m]]), np.concatenate([y10, by[:m]])
                extra.update(source=str(source(10000, s)), train_csv_rows=rows10k + further[:k - 10000],
                             rule=f'n10000 cut to {1 + KEEP} rows per group, then default_rng({s}) draws after trm_smalldata.py: '
                                  f'choice(pool without the 9,000, 90,000)[:{k - 10000}], copies by shuffle_sudoku with np.random.seed({AUG_SEED_OFFSET + s})')
            else:
                x, y = cut(k, s)
                rec = json.loads((source(k, s) / 'subset.json').read_text()) if k != 1000 else None
                extra.update(source=str(source(k, s)), train_csv_rows=rec['train_csv_rows'] if rec else None,
                             rule=f'source cut to its first {1 + KEEP} rows per group')
            T.save(out, x, y, extra)
        recs = {k: json.loads((OUT / f's{s}-n{k}' / 'subset.json').read_text()) for k in sizes if k != 1000}
        ks = sorted(recs)
        for a, b in zip(ks, ks[1:]):
            assert recs[b]['train_csv_rows'][:a] == recs[a]['train_csv_rows'], f'seed {s}: {a} not a prefix of {b}'
        for a, b in ((10000, 30000), (30000, 100000)):
            if a in sizes and b in sizes:
                for f in ('inputs', 'labels'):
                    small = np.load(OUT / f's{s}-n{a}' / 'train' / f'all__{f}.npy', mmap_mode='r')
                    large = np.load(OUT / f's{s}-n{b}' / 'train' / f'all__{f}.npy', mmap_mode='r')
                    assert np.array_equal(large[:len(small)], small), f'seed {s}: n{a} {f} not the prefix of n{b}'
        print(f'seed {s}: nested {ks}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['build'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--sizes', type=int, nargs='+', default=[1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000])
    a = p.parse_args()
    build(a.seeds, a.sizes)
