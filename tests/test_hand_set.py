"""Hand-set energies (experiments/hand_set.py). Ways they could be wrong, each a test below:
1. the Sudoku conflict graph misses the boxes (or rows or columns), or includes a cell paired with itself: then it does
   not have 810 pairs of degree 20, and a box-only pair is not in it;
2. the Sudoku (A, B) does not give E = h x (number of same-digit conflict pairs) in cram_sudoku_search's form
   P = 1/2 y.S y (the factor 1/2, A versus S), or a valid solution has nonzero energy;
3. the maze values do not give side coupling -1, diagonal -0.1 and no coupling to the off state;
4. the maze energy does not perform dead-end filling: T = 0 focused Gibbs from all-open on a small tree maze does not
   end at the path (an on-cost at or below 1 keeps dead ends, at or above 2 removes the path).
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import hand_set as HE  # noqa: E402
from cram import maze_unique as M  # noqa: E402

SOLUTION = [int(c) - 1 for c in
            '534678912672195348198342567859761423426853791713924856961537284287419635345286179']


def test_conflict_graph():
    Cg = HE.conflict_graph()
    assert int(Cg.triu(1).sum()) == 810 and bool((Cg.sum(1) == 20).all()) and not bool(Cg.diagonal().any())
    assert bool(Cg[0, 10]) and bool(Cg[0, 8]) and bool(Cg[0, 72]) and not bool(Cg[0, 12])  # box, row, column; none


def test_sudoku_energy_counts_conflicts():
    A, B = HE.sudoku_hand(h=1.0)
    S = A + A.T
    energy = lambda y: float(0.5 * (F.one_hot(y, 9).float() * (S @ F.one_hot(y, 9).float())).sum())
    sol = torch.tensor(SOLUTION)
    assert energy(sol) == 0 and float(B.abs().max()) == 0
    Cg = HE.conflict_graph()
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        y = torch.randint(0, 9, (81,), generator=g)
        count = int(((y[:, None] == y[None]) & Cg).triu(1).sum())
        assert energy(y) == count


def test_maze_kernel_values():
    W = HE.maze_hand().kernel().detach()
    assert torch.allclose(W[1, 1, [0, 1, 1, 2], [1, 0, 2, 1]], torch.full((4,), -1.0))
    assert torch.allclose(W[1, 1, [0, 0, 2, 2], [0, 2, 0, 2]], torch.full((4,), -0.1))
    assert float(W[1, 1, 1, 1]) == 0 and bool((W[0] == 0).all()) and bool((W[:, 0] == 0).all())


MAZE = ['#######',
        '#S..#.#',
        '###.#.#',
        '#...G.#',
        '#.#####',
        '#.....#',
        '#######']
PATH = {(1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 4)}


def _maze():
    """The small maze MAZE as (tokens (1, 7, 7), path label (1, 7, 7), free cells (1, 7, 7))."""
    tok = {'#': M.WALL, '.': M.OPEN, 'S': M.S_TOK, 'G': M.G_TOK}
    x = torch.tensor([[tok[c] for c in row] for row in MAZE])[None]
    y = torch.zeros_like(x)
    for r, c in PATH:
        y[0, r, c] = 1
    return x, y, x == M.OPEN


def test_maze_dead_end_filling():
    x, y, free = _maze()
    r = M.gibbs(HE.maze_hand(), x, y, free, T=0.0, gen=torch.Generator().manual_seed(0))
    assert bool(r['stopped'].all()) and bool(r['right'].all())
    assert int(r['moves'][0]) == int(free.sum()) - (len(PATH) - 2)  # one move per dead-end cell
