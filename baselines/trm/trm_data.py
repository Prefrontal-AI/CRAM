"""Maze-Unique in TRM's data format (SamsungSAILMontreal/TinyRecursiveModels, commit c011037, dataset/common.py).

TRM reads, per split, all__inputs.npy and all__labels.npy (int tokens, one row per example), all__puzzle_indices.npy,
all__group_indices.npy, all__puzzle_identifiers.npy and dataset.json. PAD = 0 and label 0 is ignored.

- sudoku-extreme: TRM's own builder (dataset/build_sudoku_dataset.py --subsample-size 1000 --num-aug 1000), run from
  the TRM clone; this script does not build it.
- maze-unique: EqR's split (locuslab/EqR-data, maze-30x30-unique-1k, revision bcb7d45c) is already in this format, with
  TRM's maze tokens (1 wall, 2 open, 3 S, 4 G, 5 path). Each training maze gets TRM's 8 dihedral copies in one group,
  as TRM's build_maze_dataset.py --aug does for Maze-Hard; the test mazes stay as they are.

Usage: python baselines/trm/trm_data.py maze-unique  (writes experiments/runs/trm/data/maze-unique-aug8/)
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[2] / 'experiments'  # run outputs and built datasets live under experiments/runs/
DATA = HERE / 'runs' / 'trm' / 'data'
EQR = Path(__file__).resolve().parents[2] / 'data' / 'eqr/maze-30x30-unique-1k'


def dihedral(arr, k):
    """TRM's dataset/common.py dihedral_transform, copied: the 8 symmetries of a square grid."""
    if k == 0: return arr
    if k == 1: return np.rot90(arr, k=1)
    if k == 2: return np.rot90(arr, k=2)
    if k == 3: return np.rot90(arr, k=3)
    if k == 4: return np.fliplr(arr)
    if k == 5: return np.flipud(arr)
    if k == 6: return arr.T
    if k == 7: return np.fliplr(np.rot90(arr, k=1))
    raise ValueError(k)


def save_split(out, split, inputs, labels, groups_of, vocab_size, seq_len):
    """Write one split. `groups_of[g]` = number of examples (each its own puzzle) in group g."""
    d = out / split
    d.mkdir(parents=True, exist_ok=True)
    n = len(inputs)
    assert sum(groups_of) == n
    np.save(d / 'all__inputs.npy', inputs)
    np.save(d / 'all__labels.npy', labels)
    np.save(d / 'all__puzzle_indices.npy', np.arange(n + 1, dtype=np.int32))
    np.save(d / 'all__group_indices.npy', np.concatenate([[0], np.cumsum(groups_of)]).astype(np.int32))
    np.save(d / 'all__puzzle_identifiers.npy', np.zeros(n, dtype=np.int32))
    meta = dict(seq_len=int(seq_len), vocab_size=int(vocab_size), pad_id=0, ignore_label_id=0, blank_identifier_id=0,
                num_puzzle_identifiers=1, total_groups=len(groups_of), mean_puzzle_examples=1,
                total_puzzles=len(groups_of),  # as TRM's builders write it (groups, not examples)
                sets=['all'])
    (d / 'dataset.json').write_text(json.dumps(meta))
    (out / 'identifiers.json').write_text(json.dumps(['<blank>']))


def build_maze_unique():
    """Write maze-unique-aug8: EqR's 1,000 training mazes with 8 dihedral copies each (one group per maze), and EqR's
    1,000 test mazes unchanged. Rows are 900 tokens (30 x 30, row-major), uint8."""
    out = DATA / 'maze-unique-aug8'
    for split in ('train', 'test'):
        X = np.load(EQR / split / 'all__inputs.npy')
        Y = np.load(EQR / split / 'all__labels.npy')
        assert X.shape == (1000, 900) and set(np.unique(Y)) <= {1, 2, 3, 4, 5}
        if split == 'train':
            xs = [dihedral(x.reshape(30, 30), k).reshape(-1) for x in X for k in range(8)]
            ys = [dihedral(y.reshape(30, 30), k).reshape(-1) for y in Y for k in range(8)]
            groups = [8] * len(X)
        else:
            xs, ys, groups = list(X), list(Y), [1] * len(X)
        save_split(out, split, np.stack(xs).astype(np.uint8), np.stack(ys).astype(np.uint8), groups, 6, 900)
    print('wrote', out)


if __name__ == '__main__':
    {'maze-unique': build_maze_unique}[sys.argv[1]]()
