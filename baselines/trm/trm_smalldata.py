"""Sudoku-Extreme training sets of 1 to 10,000 unique puzzles in TRM's data format, nested per seed, for the data-efficiency
figure. CRAM, TRM and EBR train on these sets (experiments/data_efficiency.py and baselines/ebr/smalldata.py read them).

The reference point is TRM's 1,000-puzzle set (experiments/runs/trm/data/sudoku-extreme-1k-aug-1000, built by TRM's
build_sudoku_dataset.py --subsample-size 1000 --num-aug 1000; every seed of the 1,000-puzzle runs trains on it).
That builder draws its subset and its augmentations with unseeded np.random, so this script does not call it again:

- seed k orders the 1,000 groups of that set with np.random.default_rng(k).permutation(1000);
  the sets of fewer than 1,000 puzzles (1, 3, 10, 30, 100, 300) are the first groups in that order, copied row for row (each original and its 1,000 augmented copies).
- the 3,000-puzzle set is all 1,000 groups plus 2,000 puzzles that the same generator draws from train.csv
  (sapientinc/sudoku-extreme, the pool TRM's builder reads) outside the 1,000 originals;
  each new puzzle gets 1,000 augmented copies from TRM's own shuffle_sudoku, with np.random seeded 10,000 + k.
- the 10,000-puzzle set is the 3,000-puzzle set plus 7,000 more puzzles that the same generator draws next,
  from the pool without the 2,000; the 2,000 and then the 7,000 are augmented in one np.random stream seeded 10,000 + k,
  so its first 3,000 groups are the 3,000-puzzle set row for row.
- so 100 is a subset of 300, 300 of 1,000, 1,000 of 3,000 and 3,000 of 10,000 for every seed, and a group keeps its augmented copies at every size.
- the test split is a symbolic link to the 1,000-puzzle set's test split (the official 422,786 test puzzles).

Run with TRM's Python (it imports shuffle_sudoku from the TRM clone, $TRM_SRC/dataset/build_sudoku_dataset.py), on a compute node:
  python baselines/trm/trm_smalldata.py build [--seeds 0 1 2] [--sizes 1 3 10 30 100 300 3000 10000]
  python baselines/trm/trm_smalldata.py selfcheck      # planted faults must fail the checks
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2] / 'experiments'  # run outputs and built datasets live under experiments/runs/
DATA = Path(os.environ.get('TRM_DATA', HERE / 'runs' / 'trm' / 'data'))
REF = DATA / 'sudoku-extreme-1k-aug-1000'
OUT = DATA / 'sudoku-smalldata'
CSV = Path(os.environ.get('SUDOKU_EXTREME_CSV', Path(__file__).resolve().parents[2] / 'data' / 'sudoku-extreme/train.csv'))
TRM_SRC = Path(os.environ.get('TRM_SRC', Path(os.environ.get('THIRD_PARTY', HERE.parent / 'third_party')) / 'trm'))
NUM_AUG = 1000
AUG_SEED_OFFSET = 10_000


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def to_question(tokens):
    """TRM tokens (blank 1, digit d as d + 1) -> train.csv's question string ('.' for blank)."""
    return ''.join('.' if t == 1 else str(int(t) - 1) for t in tokens)


def read_pool():
    qs, ans = [], []
    with open(CSV, newline='') as f:
        r = csv.reader(f)
        next(r)
        for _source, q, a, _rating in r:
            qs.append(q)
            ans.append(a)
    return qs, ans


def check_rows(inputs, labels, groups):
    """Every row: a valid solved grid as label, inputs = label at clues and blank elsewhere; group sizes 1 + NUM_AUG."""
    sol = labels.astype(np.int64).reshape(-1, 9, 9) - 1
    want = np.arange(1, 10)
    rows_ok = (np.sort(sol, axis=2) == want).all((1, 2))
    cols_ok = (np.sort(sol, axis=1) == want[:, None]).all((1, 2))
    boxes = sol.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    box_ok = (np.sort(boxes, axis=2) == want).all((1, 2))
    assert (rows_ok & cols_ok & box_ok).all(), 'a label is not a valid Sudoku solution'
    x = inputs.astype(np.int64)
    assert ((x == 1) | (x == labels.astype(np.int64))).all(), 'a clue disagrees with its label'
    assert (np.diff(groups) == 1 + NUM_AUG).all() and groups[0] == 0 and groups[-1] == len(inputs), 'group sizes'


def save(out, inputs, labels, meta_extra):
    d = out / 'train'
    d.mkdir(parents=True, exist_ok=True)
    n, g = len(inputs), len(inputs) // (1 + NUM_AUG)
    groups = np.arange(0, n + 1, 1 + NUM_AUG, dtype=np.int32)
    check_rows(inputs, labels, groups)
    files = dict(inputs=inputs.astype(np.uint8), labels=labels.astype(np.uint8),
                 puzzle_indices=np.arange(n + 1, dtype=np.int32), group_indices=groups,
                 puzzle_identifiers=np.zeros(n, dtype=np.int32))
    for k, v in files.items():
        tmp = d / f'all__{k}.tmp.npy'
        np.save(tmp, v)
        os.replace(tmp, d / f'all__{k}.npy')
    meta = dict(pad_id=0, ignore_label_id=0, blank_identifier_id=0, vocab_size=11, seq_len=81, num_puzzle_identifiers=1,
                total_groups=g, mean_puzzle_examples=1.0, total_puzzles=g, sets=['all'])  # as TRM's builder writes it
    (d / 'dataset.json').write_text(json.dumps(meta))
    (out / 'identifiers.json').write_text(json.dumps(['<blank>']))
    test = out / 'test'
    if not test.exists():
        test.symlink_to(REF / 'test')
    rec = dict(meta_extra, unique_puzzles=g, augmented_examples=n, num_aug=NUM_AUG,
               sha256={k: sha256(d / f'all__{k}.npy') for k in files}, test=str(REF / 'test'),
               script=str(Path(__file__).resolve()), script_sha256=sha256(__file__))
    (out / 'subset.json').write_text(json.dumps(rec, indent=1) + '\n')
    print('wrote', out, g, 'puzzles', n, 'examples', flush=True)


def build(seeds, sizes):
    sys.path.insert(0, str(TRM_SRC / 'dataset'))
    from build_sudoku_dataset import shuffle_sudoku  # TRM's augmentation, unmodified

    X = np.load(REF / 'train' / 'all__inputs.npy', mmap_mode='r')
    Y = np.load(REF / 'train' / 'all__labels.npy', mmap_mode='r')
    G = np.load(REF / 'train' / 'all__group_indices.npy')
    assert len(G) == 1001 and (np.diff(G) == 1 + NUM_AUG).all()
    qs, ans = read_pool()
    where = {}
    for i, q in enumerate(qs):
        where.setdefault(q, []).append(i)
    # The originals (aug index 0) of the 1,000-puzzle set, located in train.csv.
    ref_rows = []
    for g in range(1000):
        q = to_question(X[G[g]])
        hits = [i for i in where.get(q, []) if ans[i] == ''.join(str(int(t) - 1) for t in Y[G[g]])]
        assert hits, f'group {g}: original not found in train.csv'
        ref_rows.append(hits[0])
    ref_q = {qs[i] for i in ref_rows}
    pool = np.array([i for i, q in enumerate(qs) if q not in ref_q])  # excludes every duplicate of an original
    print(f'train.csv {len(qs)} rows; the 1,000 originals located; pool for new puzzles {len(pool)}', flush=True)

    for k in seeds:
        rng = np.random.default_rng(k)
        order = rng.permutation(1000)
        extra = rng.choice(pool, size=2000, replace=False)
        more = rng.choice(np.setdiff1d(pool, extra), size=7000, replace=False)  # drawn after extra, so the 3,000-puzzle sets do not depend on this draw
        for n in sizes:
            out = OUT / f's{k}-n{n}'
            if (out / 'subset.json').exists():
                print(out, 'exists: skipped', flush=True)
                continue
            take = order[:min(n, 1000)]
            xs = [np.asarray(X[G[g]:G[g + 1]]) for g in take]
            ys = [np.asarray(Y[G[g]:G[g + 1]]) for g in take]
            csv_rows = [int(ref_rows[g]) for g in take]
            if n > 1000:
                assert n in (3000, 10000)
                np.random.seed(AUG_SEED_OFFSET + k)
                new = list(extra) + (list(more) if n == 10000 else [])
                for i in new:
                    q = np.frombuffer(qs[i].replace('.', '0').encode(), np.uint8).reshape(9, 9) - ord('0')
                    a = np.frombuffer(ans[i].encode(), np.uint8).reshape(9, 9) - ord('0')
                    gx, gy = [q], [a]
                    for _ in range(NUM_AUG):
                        b, s = shuffle_sudoku(q, a)
                        gx.append(b)
                        gy.append(s)
                    xs.append(np.stack(gx).reshape(-1, 81) + 1)
                    ys.append(np.stack(gy).reshape(-1, 81) + 1)
                csv_rows += [int(i) for i in new]
            save(out, np.concatenate(xs), np.concatenate(ys),
                 dict(seed=k, size=n, ref_groups=[int(g) for g in take], train_csv_rows=csv_rows,
                      ref_dataset=str(REF), train_csv=str(CSV), train_csv_sha256=sha256(CSV),
                      rule=f'default_rng({k}).permutation(1000)[:min(n,1000)] of the 1,000-puzzle set; n = 3000 adds '
                           f'default_rng({k}).choice(pool, 2000) after that permutation; n = 10000 adds next '
                           f'choice(pool without the 2000, 7000); new puzzles augmented in order by TRM shuffle_sudoku '
                           f'with np.random.seed({AUG_SEED_OFFSET + k})'))
        check_nested(k, sizes)
        if 3000 in sizes and 10000 in sizes:
            for f in ('inputs', 'labels'):
                small = np.load(OUT / f's{k}-n3000' / 'train' / f'all__{f}.npy', mmap_mode='r')
                big = np.load(OUT / f's{k}-n10000' / 'train' / f'all__{f}.npy', mmap_mode='r')
                assert np.array_equal(big[:len(small)], small), f'seed {k}: n3000 {f} is not the prefix of n10000'
            print(f'seed {k}: n3000 rows are the first rows of n10000', flush=True)


def check_nested(k, sizes):
    recs = {n: json.loads((OUT / f's{k}-n{n}' / 'subset.json').read_text()) for n in sizes}
    ns = sorted(recs)
    for a, b in zip(ns, ns[1:]):
        assert recs[b]['train_csv_rows'][:a] == recs[a]['train_csv_rows'], f'seed {k}: {a} not a prefix of {b}'
        assert len(set(recs[b]['train_csv_rows'])) == b
    print(f'seed {k}: nested {ns}', flush=True)


def selfcheck():
    """Each check must reject its planted fault."""
    X = np.load(REF / 'train' / 'all__inputs.npy', mmap_mode='r')[:2 * (1 + NUM_AUG)].copy()
    Y = np.load(REF / 'train' / 'all__labels.npy', mmap_mode='r')[:2 * (1 + NUM_AUG)].copy()
    groups = np.arange(0, len(X) + 1, 1 + NUM_AUG)
    check_rows(X, Y, groups)
    faults = {}
    y = Y.copy(); y[5, [0, 1]] = y[5, [1, 0]]; faults['label with two cells swapped'] = (X, y, groups)
    x = X.copy(); c = int(np.argmax(x[7] > 1)); x[7, c] = 2 + (int(x[7, c]) - 1) % 9  # digit d -> d % 9 + 1
    faults['clue changed'] = (x, Y, groups)
    faults['group of 1,000 rows'] = (X[:-1], Y[:-1], np.array([0, 1 + NUM_AUG, 2 * (1 + NUM_AUG) - 1]))
    for name, (a, b, g) in faults.items():
        try:
            check_rows(a, b, g)
        except AssertionError as e:
            print(f'planted fault "{name}" rejected: {e}')
        else:
            raise SystemExit(f'planted fault "{name}" passed the checks')
    print('selfcheck passed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['build', 'selfcheck'])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--sizes', type=int, nargs='+', default=[1, 3, 10, 30, 100, 300, 3000, 10000])
    a = p.parse_args()
    build(a.seeds, a.sizes) if a.mode == 'build' else selfcheck()
