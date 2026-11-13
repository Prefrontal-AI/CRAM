"""The paper's Sudoku-Extreme and Maze-Unique splits in EqR's data format (locuslab/EqR at commit 134c754).

EqR reads HRM's layout (dataset/puzzle_dataset.py): per split, all__inputs.npy and all__labels.npy (one row per example),
all__puzzle_indices.npy, all__group_indices.npy, all__puzzle_identifiers.npy and dataset.json. PAD = 0, label 0 is
ignored. Training draws one example per group per epoch; steps per epoch = floor(groups / global batch).

- Sudoku-Extreme, training (`sudoku-s<k>/train`): FRM's training split train-s<k> (1,000 puzzles; data/frm/split), laid
  out as EqR's builder (dataset/build_sudoku_dataset.py, imported from the EqR clone) lays out its 1k set: tokens
  digit + 1 (blank = 1, digits 2-10, vocab 11), every cell labelled (clues too); one group per puzzle holding the
  original and 1,000 augmentations drawn by the builder's own shuffle_sudoku after np.random.seed(42) (the builder's
  default seed). EqR's own 1k set is a different draw from the official training split: HRM's
  np.random.seed(42); np.random.choice(3,831,994, 1000), which shares 1, 0 and 0 puzzles with train-s0/s1/s2.
- Sudoku-Extreme, evaluation (`sudoku-eval/<split>`, one example per group): dev2k = the first 2,048 puzzles of dev10k
  (the in-training validation), dev10k, frmtest (FRM's 1,000), unseen
  (the 412,786 official test puzzles outside dev10k, in full-test's order: frmtest, then shards full-00 ... full-15) and
  first10k (unseen[:10,000]: frmtest, then the first 9,000 of full-00; the subset used for the costly
  settings).
  `sudoku-s<k>/dev` is a link to `sudoku-eval/dev2k`.
- Maze-Unique (`maze-unique/`): EqR's own maze-30x30-unique-1k (revision bcb7d45c), copied unchanged as train and test
  (EqR trains on it without augmentation); dev = dev100 (np.random.default_rng(0).permutation(1000)[:100] of the
  training mazes), a monitor of training fit only, since the final runs train on all 1,000.

Usage (EqR's Python, from anywhere): python baselines/eqr/eqr_data.py --what sudoku | maze | all | first10k
Output: experiments/runs/eqr/data (or $EQR_DATA).
"""

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1] / 'experiments'
REPO = EXPERIMENTS.parent
THIRD_PARTY = os.environ.get('THIRD_PARTY', str(Path(__file__).resolve().parents[2] / 'third_party'))  # clones of the authors' code (baselines/README.md)
EQR_SRC = Path(os.environ.get('EQR_SRC', THIRD_PARTY + '/eqr'))
OUT = Path(os.environ.get('EQR_DATA', EXPERIMENTS / 'runs' / 'eqr' / 'data'))
V1_DATA = Path(os.environ.get('V1_DATA', REPO / 'data'))  # datasets from scripts/prepare_data.py
FRM_SPLIT = V1_DATA / 'frm' / 'split'
DEV10K = V1_DATA / 'frm' / 'dev10k' / 'dev.npz'
FULL_TEST = V1_DATA / 'full-test'
EQR_DATA = Path(os.environ.get('EQR_HF_DATA', Path(__file__).resolve().parents[2] / 'data' / 'eqr'))
N_AUG = 1000
AUG_SEED = 42


def save_split(d, inputs, labels, groups_of, vocab_size, seq_len):
    """One split in EqR's layout; `groups_of[g]` = examples in group g, each example its own puzzle."""
    d.mkdir(parents=True, exist_ok=True)
    n = len(inputs)
    assert sum(groups_of) == n and inputs.shape == labels.shape == (n, seq_len)
    np.save(d / 'all__inputs.npy', inputs)
    np.save(d / 'all__labels.npy', labels)
    np.save(d / 'all__puzzle_indices.npy', np.arange(n + 1, dtype=np.int32))
    np.save(d / 'all__group_indices.npy', np.concatenate([[0], np.cumsum(groups_of)]).astype(np.int32))
    np.save(d / 'all__puzzle_identifiers.npy', np.zeros(n, dtype=np.int32))
    meta = dict(pad_id=0, ignore_label_id=0, blank_identifier_id=0, vocab_size=vocab_size, seq_len=seq_len,
                num_puzzle_identifiers=1, total_groups=len(groups_of), mean_puzzle_examples=1.0, sets=['all'],
                total_samples=n)
    (d / 'dataset.json').write_text(json.dumps(meta))
    (d.parent / 'identifiers.json').write_text(json.dumps(['<blank>']))


def sudoku_tokens(solutions, clue_mask):
    """(inputs, labels) as EqR's builder writes them: digit + 1, blank = 1."""
    sol = np.asarray(solutions).astype(np.uint8)
    inp = np.where(np.asarray(clue_mask).astype(bool), sol, 0).astype(np.uint8)
    assert sol.min() >= 1 and sol.max() <= 9
    return inp + 1, sol + 1


def build_sudoku_train(k):
    sys.path.insert(0, str(EQR_SRC))
    from dataset.build_sudoku_dataset import shuffle_sudoku  # EqR's (HRM's) augmentation, unmodified
    f = np.load(FRM_SPLIT / f'train-s{k}.npz')
    sol = f['solutions'].astype(np.uint8)
    board = np.where(f['clue_mask'].astype(bool), sol, 0).astype(np.uint8)
    np.random.seed(AUG_SEED)
    xs, ys = [], []
    for b, s in zip(board, sol):
        b, s = b.reshape(9, 9), s.reshape(9, 9)
        for a in range(1 + N_AUG):
            inp, out = (b, s) if a == 0 else shuffle_sudoku(b, s)
            xs.append(inp.reshape(-1))
            ys.append(out.reshape(-1))
    X = np.stack(xs).astype(np.uint8) + 1
    Y = np.stack(ys).astype(np.uint8) + 1
    d = OUT / f'sudoku-s{k}'
    save_split(d / 'train', X, Y, [1 + N_AUG] * len(sol), 11, 81)
    dev = d / 'dev'
    if dev.is_symlink() or dev.exists():
        dev.unlink() if dev.is_symlink() else shutil.rmtree(dev)
    dev.symlink_to('../sudoku-eval/dev2k')
    print('wrote', d / 'train', X.shape, 'from', FRM_SPLIT / f'train-s{k}.npz')


def build_sudoku_eval():
    idx = json.loads((FULL_TEST / 'index.json').read_text())
    d10 = np.load(DEV10K)
    splits = {'dev10k': (d10['solutions'], d10['clue_mask']), 'dev2k': (d10['solutions'][:2048], d10['clue_mask'][:2048])}
    order = sorted((s for s in idx['full_all_order'] if s != 'full-dev10k'), key=lambda s: idx['full_all_order'][s][0])
    parts = [np.load(FULL_TEST / f'{s}.npz') for s in order]
    for s, p in zip(order, parts):
        assert len(p['solutions']) == idx['shards'][s]['n'], s
    sol = np.concatenate([p['solutions'] for p in parts])
    clue = np.concatenate([p['clue_mask'] for p in parts])
    assert len(sol) == 412786, len(sol)
    splits['unseen'] = (sol, clue)
    splits['frmtest'] = (parts[0]['solutions'], parts[0]['clue_mask'])
    assert order[0] == 'full-frmtest'
    splits['first10k'] = (sol[:10000], clue[:10000])
    out = OUT / 'sudoku-eval'
    for name, (s, c) in splits.items():
        X, Y = sudoku_tokens(s, c)
        save_split(out / name, X, Y, [1] * len(X), 11, 81)
        print('wrote', out / name, X.shape)
    (out / 'order.json').write_text(json.dumps(dict(unseen=order, first10k='unseen[:10000] (frmtest, then the first 9,000 of full-00)')))


def build_maze():
    src = EQR_DATA / 'maze-30x30-unique-1k'
    out = OUT / 'maze-unique'
    for split in ('train', 'test'):
        X = np.load(src / split / 'all__inputs.npy')
        Y = np.load(src / split / 'all__labels.npy')
        assert X.shape == Y.shape == (1000, 900), X.shape
        save_split(out / split, X.astype(np.uint8), Y.astype(np.uint8), [1] * 1000, 6, 900)
    X = np.load(out / 'train' / 'all__inputs.npy')
    Y = np.load(out / 'train' / 'all__labels.npy')
    dev = np.random.default_rng(0).permutation(1000)[:100]
    save_split(out / 'dev', X[dev], Y[dev], [1] * 100, 6, 900)
    sha = {f'{s}/{n}': hashlib.sha256((src / s / n).read_bytes()).hexdigest() for s in ('train', 'test')
           for n in ('all__inputs.npy', 'all__labels.npy')}
    (out / 'source.json').write_text(json.dumps(dict(source=str(src), sha256=sha, dev100=dev.tolist()), indent=1))
    print('wrote', out)


@dataclass
class Args:
    what: Literal['sudoku', 'maze', 'all', 'first10k'] = 'all'  # which datasets to write under runs/eqr/data (or $EQR_DATA)


def build_first10k():
    src = OUT / 'sudoku-eval' / 'unseen'
    X = np.load(src / 'all__inputs.npy')[:10000]
    Y = np.load(src / 'all__labels.npy')[:10000]
    save_split(OUT / 'sudoku-eval' / 'first10k', X, Y, [1] * 10000, 11, 81)
    print('wrote', OUT / 'sudoku-eval' / 'first10k', X.shape)


if __name__ == '__main__':
    a = tyro.cli(Args)
    if a.what == 'first10k':
        build_first10k()
    if a.what in ('sudoku', 'all'):
        build_sudoku_eval()
        for k in range(3):
            build_sudoku_train(k)
    if a.what in ('maze', 'all'):
        build_maze()
