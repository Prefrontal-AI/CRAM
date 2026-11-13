"""Maze-Hard: the data loader and the grid-window helpers that the Maze-Hard energies share.

Data source: the Hugging Face dataset `sapientinc/maze-30x30-hard-1k` at revision 549754de (the Maze-Hard set of the
Hierarchical Reasoning Model, HRM). `scripts/prepare_data.py maze-hard` downloads `train.csv` and `test.csv` to
`<data>/maze-hard/`, where `<data>` is `data/` in the repository or the folder named by `CRAM_DATA`.
Each file holds 1,000 mazes of 30 x 30 cells. A row has the columns (source, question, answer, rating); `question` and
`answer` are strings of 900 characters in row-major order.

Token encoding of the input (`question`): '#' -> 0 wall, ' ' -> 1 open, 'S' -> 2 start, 'G' -> 3 goal.
Label (`answer`): True at the cells of the shortest path from S to G ('o'), and at S and G.

Splits: `load('train')` and `load('test')` return the two files in file order. `dev_split` divides the 1,000 training
mazes into 900 fitting mazes and 100 development mazes with a seeded permutation; every choice of a setting used the
development mazes, and the test file is scored once per final run. No augmentation is applied.

Window helpers: `offsets` lists the offsets of a square window, `read` shifts a grid by one offset, `gather` stacks the
shifted grids of several offsets, and `inside` marks the offsets that stay on the grid.

Dimension names: n mazes of a file, b mazes of a batch, k offsets of a window, h and w grid rows and columns (30, 30).
"""

import csv
import os
from pathlib import Path

import numpy as np
import torch

DATA = Path(os.environ.get('CRAM_DATA', Path(__file__).resolve().parents[1] / 'data')) / 'maze-hard'  # sapientinc/maze-30x30-hard-1k @ 549754de
NTOK = 4  # 0 wall, 1 open, 2 S, 3 G
CAP = 1e4  # the value of psi = -log(z) / q that stands for z = 0 in maze_h3.H3Energy


# --- data -------------------------------------------------------------------------------------------------------------

def load(split):
    """split: 'train' or 'test'. Returns (tokens (n, 30, 30) int64: 0 wall, 1 open, 2 S, 3 G; path (n, 30, 30) bool:
    the label, S and G included)."""
    rows = list(csv.reader(open(DATA / f'{split}.csv')))[1:]
    q = np.array([list(r[1]) for r in rows]).reshape(-1, 30, 30)
    a = np.array([list(r[2]) for r in rows]).reshape(-1, 30, 30)
    tok = np.select([q == '#', q == ' ', q == 'S', q == 'G'], [0, 1, 2, 3]).astype(np.int64)
    return tok, (a == 'o') | (a == 'S') | (a == 'G')


def dev_split():
    """(fit indices (900,), development indices (100,)) into the training file, each sorted: the development mazes are
    the first 100 entries of numpy's default_rng(0).permutation(1000), and the fitting mazes are the other 900."""
    order = np.random.default_rng(0).permutation(1000)
    return np.sort(order[100:]), np.sort(order[:100])


# --- the window -------------------------------------------------------------------------------------------------------

def offsets(w):
    """The (2w+1)^2 offsets (di, dj) of the window of radius w: self (0, 0) first, then the others in row-major order."""
    return [(0, 0)] + [(di, dj) for di in range(-w, w + 1) for dj in range(-w, w + 1) if (di, dj) != (0, 0)]


def read(x, di, dj, fill):
    """x (..., h, w) shifted by one offset: y[..., i, j] = x[..., i + di, j + dj], and `fill` beyond the grid."""
    H, W = x.shape[-2:]
    y = torch.full_like(x, fill)
    y[..., max(0, -di):H - max(0, di), max(0, -dj):W - max(0, dj)] = \
        x[..., max(0, di):H - max(0, -di), max(0, dj):W - max(0, -dj)]
    return y


def gather(x, offs, fill):
    """x (b, ..., h, w) read at each of the k offsets `offs`: (b, k, ..., h, w), with `fill` beyond the grid."""
    return torch.stack([read(x, di, dj, fill) for di, dj in offs], 1)


def inside(offs, H, W, device):
    """(k, h, w) bool: True where cell (i, j) plus the offset lands inside the H x W grid."""
    one = torch.ones(H, W, device=device)
    return gather(one[None], offs, 0.0)[0] > 0
