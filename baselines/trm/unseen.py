"""TRM Sudoku on the 412,786 unseen puzzles from its full-test per-puzzle flags (CPU only).

score.right.npy (trm_bench.py score) holds one solved flag per row of TRM's 422,786-row test split in its loader order
(sequential, test_set_mode). This script checks that TRM's test split (inputs and labels) equals the official test.npz
row by row, and that dev10k's own puzzles sit at the official rows of dev10k/indices.json; then it drops those rows.
dev10k is the set of 10,000 official test rows that the baselines use as development data (data/frm/dev10k).

  python baselines/trm/unseen.py baselines/runs/trm/sudoku-s0 [more run folders]   # -> <run>/score-unseen.json
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
V1_DATA = Path(os.environ.get('V1_DATA', REPO / 'data'))
TRM_TEST = Path(os.environ.get('TRM_DATA', REPO / 'experiments/runs/trm/data')) / 'sudoku-extreme-1k-aug-1000/test'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unseen_mask():
    off = np.load(V1_DATA / 'frm/official/test.npz')
    sol, clue = off['solutions'].astype(np.int64), off['clue_mask'].astype(bool)
    inputs, labels = np.load(TRM_TEST / 'all__inputs.npy'), np.load(TRM_TEST / 'all__labels.npy')
    # TRM's builder: token = digit + 1, blank = 0 in inputs (the official solutions are digits 1..9)
    assert inputs.shape == labels.shape == sol.shape == (422786, 81), (inputs.shape, labels.shape, sol.shape)
    assert np.array_equal(labels.astype(np.int64) - 1, sol), 'TRM labels != official solutions (row order)'
    assert np.array_equal(inputs.astype(np.int64), np.where(clue, sol + 1, 1)), 'TRM inputs != official clues (row order)'
    idx = np.array(json.loads((V1_DATA / 'frm/dev10k/indices.json').read_text()))
    dev = np.load(V1_DATA / 'frm/dev10k/dev.npz')
    assert len(idx) == len(set(idx.tolist())) == 10000
    assert np.array_equal(dev['solutions'].astype(np.int64), sol[idx]) and np.array_equal(dev['clue_mask'].astype(bool), clue[idx])
    mask = np.ones(len(sol), bool)
    mask[idx] = False
    return mask, idx


def main(runs):
    mask, idx = unseen_mask()
    for run in map(Path, runs):
        score = json.loads((run / 'score.json').read_text())
        right = np.load(run / 'score.right.npy')
        assert right.dtype == bool and right.size == score['n'] == mask.size and int(right.sum()) == score['solved']
        res = dict(method='TRM', solved=int(right[mask].sum()), n=int(mask.sum()), exact=float(right[mask].mean()),
                   split='official Sudoku-Extreme test rows outside dev10k (412,786 unseen)',
                   all=dict(solved=score['solved'], n=score['n']), dev10k=dict(solved=int(right[idx].sum()), n=int(idx.size)),
                   checkpoint=score['checkpoint'], source=str(run / 'score.right.npy'), source_sha256=digest(run / 'score.right.npy'),
                   row_check='TRM test inputs and labels equal data/frm/official/test.npz row by row; dev10k puzzles at indices.json',
                   script=str(Path(__file__).resolve()), script_sha256=digest(__file__))
        (run / 'score-unseen.json').write_text(json.dumps(res, indent=1) + '\n')
        print(run, res['solved'], res['n'], round(res['exact'], 6), 'dev10k', res['dev10k'], flush=True)


if __name__ == '__main__':
    main(sys.argv[1:])
