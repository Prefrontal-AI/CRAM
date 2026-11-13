"""Data efficiency of FRM (Flow Reasoning Model): build one run root for k unique training puzzles and seed s.

The root is a campaign folder as frm/setup.sh builds for Table 2 (the pinned upstream source, Flow-Reasoning-Models at
6149a01, its source hashes, the Table 2 development split), with the data-efficiency adapter (frm/de2_adapter.py,
copied as frm_de2.py, its SHA-256 in plan.json) and these data:
- cache/sudoku_extreme/train.npz: every stored row of experiments/runs/trm/data/sudoku-smalldata-v99/s<s>-n<k>/train (k puzzles,
  each the original and its 99 stored shuffle_sudoku copies), TRM tokens converted to FRM's format: solution = label - 1
  (digits 1-9), clue_mask = (input != 1) (TRM input 1 is a blank). FRM's loader drops the last partial batch of 128, so
  rows < 128 (k = 1: 100 rows) are stored r = ceil(128 / rows) times, as EBR's loader workaround (ebr/smalldata.py).
- data/train-s<s>.npz: a link to that file (the adapter checks the loader's rows against it).
- data/train-monitor-s<s>.npz: the first min(k, 1,000) originals, for the adapter's train monitor only.
- data/dev.npz: Table 2's 256 dev puzzles (runs/frm-compiled/sudoku/data/dev.npz), unchanged.
- data/test10k.npz: full-all rows 10,000-19,999 (data/full-test/full-all.npz).
Checks: TRM inputs equal labels at clue cells; the stored groups are k groups of 100 rows; no dev or test10k puzzle (clues
and solution) equals a training row, and dev and test10k are disjoint. Writes <root>/de2.json last; skips a finished root.

  python frm/de2_setup.py <root> <k> <seed>      # FRM's python or the repository's; numpy only
"""
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
TABLE1 = REPO / 'baselines/runs/frm-compiled/sudoku'
TRM_SETS = REPO / 'experiments/runs/trm/data/sudoku-smalldata-v99'
FULL_ALL = REPO / 'data/full-test/full-all.npz'
ROWS_PER_GROUP, BATCH, MONITOR = 100, 128, 1000


def digest(path):
    """SHA-256 of a file, as a hexadecimal string."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def keys(solutions, clues):
    """One bytes key per puzzle: the clue digits (0 elsewhere) followed by the solution."""
    a = np.concatenate([(solutions * clues).astype(np.uint8), solutions.astype(np.uint8)], 1)
    return [r.tobytes() for r in a]


def main(root, k, seed):
    """Build the run root `root` for k unique puzzles and training seed `seed`; write <root>/de2.json last."""
    root = Path(root).resolve()
    if (root / 'de2.json').exists():
        print(root, 'built: skipped', flush=True)
        return
    src = TRM_SETS / f's{seed}-n{k}' / 'train'
    x, y = np.load(src / 'all__inputs.npy'), np.load(src / 'all__labels.npy')
    g = np.load(src / 'all__group_indices.npy')
    assert len(g) == k + 1 and (np.diff(g) == ROWS_PER_GROUP).all() and g[-1] == len(x) == len(y) == k * ROWS_PER_GROUP, src
    assert y.min() >= 2 and y.max() <= 10 and x.min() >= 1 and x.max() <= 10, 'not TRM Sudoku tokens'
    clue = x != 1
    assert np.array_equal(x[clue], y[clue]), 'TRM input differs from its label at a clue cell'
    sol = (y - 1).astype(np.uint8)
    clue = clue.astype(np.uint8)
    r = math.ceil(BATCH / len(sol)) if len(sol) < BATCH else 1
    train_sol, train_clue = np.tile(sol, (r, 1)), np.tile(clue, (r, 1))

    root.mkdir(parents=True, exist_ok=True)
    if not (root / 'upstream').exists():
        shutil.copytree(TABLE1 / 'upstream', root / 'upstream.tmp', symlinks=True)
        (root / 'upstream.tmp').rename(root / 'upstream')
    for f in ('source-files.json', 'environment.txt'):
        shutil.copy2(TABLE1 / f, root / f)
    shutil.copy2(HERE / 'de2_adapter.py', root / 'frm_de2.py')
    plan = json.loads((TABLE1 / 'plan.json').read_text())
    plan.update(table1_harness_sha256=plan['harness_sha256'], harness_sha256=digest(root / 'frm_de2.py'), seeds=[seed],
                de2=dict(k=k, seed=seed, rows=len(sol), replication=r, note='data-efficiency run; adapter frm_de2.py = baselines/frm/de2_adapter.py'))
    (root / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')

    cache = root / 'cache/sudoku_extreme'
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(cache / 'train.tmp.npz', solutions=train_sol, clue_mask=train_clue, ratings=np.zeros(len(train_sol), np.int32), sources=np.full(len(train_sol), 'de2'))
    (cache / 'train.tmp.npz').replace(cache / 'train.npz')
    data = root / 'data'
    data.mkdir(exist_ok=True)
    link = data / f'train-s{seed}.npz'
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to('../cache/sudoku_extreme/train.npz')
    orig = np.arange(0, len(sol), ROWS_PER_GROUP)[:MONITOR]
    np.savez_compressed(data / f'train-monitor-s{seed}.npz', solutions=sol[orig].astype(np.int64), clue_mask=clue[orig].astype(np.int64), ratings=np.zeros(len(orig), np.int64))
    shutil.copy2(TABLE1 / 'data/dev.npz', data / 'dev.npz')
    with np.load(FULL_ALL) as f:
        t_sol, t_clue, t_rat = f['solutions'][10000:20000], f['clue_mask'][10000:20000], f['ratings'][10000:20000]
    assert t_sol.min() >= 1 and t_sol.max() <= 9 and len(t_sol) == 10000
    np.savez_compressed(data / 'test10k.npz', solutions=t_sol.astype(np.int64), clue_mask=t_clue.astype(np.int64), ratings=t_rat.astype(np.int64))

    with np.load(data / 'dev.npz') as d:
        dev = set(keys(d['solutions'], d['clue_mask']))
    test = set(keys(t_sol, t_clue))
    assert len(dev) == 256 and len(test) == 10000
    assert not dev & test, 'dev and test10k share a puzzle'
    hit = sum(1 for kk in keys(sol, clue) if kk in dev or kk in test)
    assert hit == 0, f'{hit} training rows equal a dev or test10k puzzle'

    rec = dict(k=k, seed=seed, source=str(src), source_sha256={f: digest(src / f'all__{f}.npy') for f in ('inputs', 'labels')}, rows=len(sol), replication=r, loader_rows=len(train_sol),
               train_npz_sha256=digest(cache / 'train.npz'), monitor_puzzles=len(orig), dev_sha256=digest(data / 'dev.npz'), test10k_sha256=digest(data / 'test10k.npz'), test10k_rows='full-all 10000-19999',
               checks='TRM inputs equal labels at clues; k groups of 100 rows; no training row equals a dev or test10k puzzle; dev and test10k disjoint',
               adapter_sha256=plan['harness_sha256'], script=str(Path(__file__).resolve()), script_sha256=digest(__file__))
    (root / 'de2.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(f'built {root}: k {k}, seed {seed}, {len(sol)} rows x {r} = {len(train_sol)} loader rows, monitor {len(orig)}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
