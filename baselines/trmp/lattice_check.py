"""Checks of the TRMP lattice reimplementation (trmp/lattice.py), CPU, float64 where numerical.

Adapted from the authors' test/neural/test_sudoku_lattice.py (precursor repository, commit 1695740) where it pins a
behaviour, plus a parameter count and a hand-written Eq. 12.

  ../.venv/bin/python trmp/lattice_check.py        # from baselines/
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lattice as L  # noqa: E402

torch.set_num_threads(4)


def solved_grid():
    row, col = np.divmod(np.arange(81), 9)
    return (3 * (row % 3) + row // 3 + col) % 9  # digits 0-8


def random_puzzle(rng, n_blanks):
    digits = rng.permutation(9)
    sol = digits[solved_grid()]
    clue = np.ones(81, bool)
    clue[rng.choice(81, size=n_blanks, replace=False)] = False
    return sol, clue


def lattice_of(sol, clue):
    s = torch.as_tensor(sol)
    c = torch.as_tensor(clue)
    return torch.where(c[..., None], F.one_hot(s, 9).double(), torch.ones(*s.shape, 9, dtype=torch.float64))


PEERS = [[j for j in range(81) if j != i and (i // 9 == j // 9 or i % 9 == j % 9 or (i // 27, i % 9 // 3) == (j // 27, j % 9 // 3))]
         for i in range(81)]


def check_parameters():
    net = L.LatticeNet()
    count = sum(p.numel() for p in net.parameters())
    assert count == 16543 == L.parameter_count((9, 24, 48, 12)), count
    assert L.parameter_count((13, 34, 68, 17)) == 32798  # Table 3, factor graph 32k
    units, ports = L.sudoku_units(), L.cell_ports(L.sudoku_units())
    assert sorted(units.flatten().tolist()) == sorted(list(range(81)) * 3)
    for cell in range(81):
        assert sorted(units.flatten()[ports[cell]].tolist()) == [cell] * 3
    print('parameters', count, 'ok')


def check_projection():
    rng = np.random.default_rng(1)
    pairs = [random_puzzle(rng, 40) for _ in range(16)]
    x = torch.stack([lattice_of(s, c) for s, c in pairs])
    given = x.sum(-1) == 1
    torch.manual_seed(1)
    keep, sym, conf = torch.randn(16, 81, 9).double(), torch.randn(16, 81, 9).double(), torch.randn(16, 81).double() - 8
    g = torch.Generator().manual_seed(0)
    nx, conflict, solved, kill, decided = L.project(x, given, keep, sym, conf, L.THETA_CLS_TRAIN, g)
    assert bool((nx <= x).all()) and torch.equal(nx[given], x[given])
    deduced = x.masked_fill(kill, 0.0)
    touched = (nx != deduced).any(-1)
    assert bool(decided.any())
    for r in range(16):
        if decided[r]:
            assert int(touched[r].sum()) == 1
            cell = int(touched[r].nonzero()[0, 0])
            assert float(nx[r, cell].sum()) == 1.0 and float(deduced[r, cell].sum()) > 1.5
            assert float(deduced[r, cell, int(nx[r, cell].argmax())]) == 1.0
        else:
            assert (solved[r] or conflict[r]) and not bool(touched[r].any())
    # givens are never killed, even with keep logits that say so
    nx2, *_ = L.project(x, given, torch.full_like(keep, -10.0), sym, conf, 0.5, g)
    assert torch.equal(nx2[given], x[given])
    print('projection ok')


def check_status():
    rng = np.random.default_rng(2)
    sol, clue = random_puzzle(rng, 30)
    quiet = (torch.full((1, 81, 9), 10.0).double(), torch.zeros(1, 81, 9).double(), torch.full((1, 81), -10.0).double())
    full = lattice_of(sol, np.ones(81, bool))[None]
    empty = full.clone()
    empty[0, 3] = 0.0
    _, conflict, solved, _, _ = L.project(empty, empty.sum(-1) == 1, *quiet, 0.5)
    assert bool(conflict[0]) and not bool(solved[0])
    _, conflict, solved, _, _ = L.project(full, full.sum(-1) == 1, *quiet, 0.5)
    assert bool(solved[0]) and not bool(conflict[0])
    loud = (quiet[0], quiet[1], -quiet[2])
    _, conflict, solved, _, _ = L.project(full, full.sum(-1) == 1, *loud, 0.5)
    assert bool(conflict[0]) and not bool(solved[0])
    x, y = lattice_of(sol, clue)[None], lattice_of(sol, np.ones(81, bool))[None]
    cell, board = L.conflicts(x, y)
    assert not bool(board[0]) and torch.equal(cell.any(-1), board)
    print('status ok')


def check_loss():
    rng = np.random.default_rng(8)
    sol, clue = random_puzzle(rng, 30)
    x = torch.stack([lattice_of(sol, clue)] * 2)
    y = torch.stack([lattice_of(sol, np.ones(81, bool))] * 2)
    x[1, 2] = 1.0 - y[1, 2]  # board 1 lost its solution
    given = x.sum(-1) == 1
    torch.manual_seed(8)
    c = 3
    heads = L.Heads(torch.randn(c, 2, 81, 9).double(), torch.randn(c, 2, 81, 9).double(), torch.randn(c, 2, 81).double())
    total, _ = L.lattice_loss(heads, x, y, given)
    expected = 0.0
    for i in range(c):
        t = x * y
        bce = -(4.0 * t * F.logsigmoid(heads.keep[i]) + 0.5 * (1 - t) * F.logsigmoid(-heads.keep[i])).mean()
        board = torch.tensor([False, True])
        mask = ~given & ~board[:, None]
        assert not bool(mask[1].any())
        ce = F.cross_entropy(heads.sym[i][mask], y.argmax(-1)[mask])
        conf = F.binary_cross_entropy_with_logits(torch.logsumexp(heads.conf[i], -1), board.double())
        expected = expected + (bce + 0.2 * ce + 0.1 * conf) / c
    assert torch.allclose(total, expected), (total, expected)
    print('loss ok')


def check_equivariance():
    torch.manual_seed(0)
    net = L.LatticeNet().double()
    rng = np.random.default_rng(5)
    x = torch.stack([lattice_of(*random_puzzle(rng, 45)) for _ in range(2)])
    grid = np.arange(81).reshape(9, 9)
    band = grid[[3, 4, 5, 0, 1, 2, 6, 7, 8]]
    perms = {'transpose': grid.T.reshape(-1), 'bands+rows': band[[1, 0, 2, 3, 5, 4, 6, 7, 8]].reshape(-1),
             'stacks': grid[:, [6, 7, 8, 3, 4, 5, 0, 1, 2]].reshape(-1)}
    with torch.no_grad():
        base = net(x)
        for name, perm in perms.items():
            idx = torch.as_tensor(perm.copy())
            moved = net(x[:, idx])
            err = max(float((base.keep[:, :, idx] - moved.keep).abs().max()),
                      float((base.sym[:, :, idx] - moved.sym).abs().max()),
                      float((base.conf[:, :, idx] - moved.conf).abs().max()))
            assert err < 1e-12, (name, err)
    print('equivariance ok')


class Oracle(torch.nn.Module):
    """Naked-single elimination as logits: kill the digits of singleton peers; uniform symbols; no conflict."""
    def __init__(self):
        super().__init__()
        m = torch.zeros(81, 81)
        for cell, others in enumerate(PEERS):
            m[cell, others] = 1.0
        self.register_buffer('peers', m)

    def forward(self, x, readouts='last'):
        single = (x.sum(-1) == 1).unsqueeze(-1)
        committed = torch.where(single, x, torch.zeros_like(x))
        present = torch.einsum('pq,bqd->bpd', self.peers.to(x.dtype), committed) > 0.5
        keep = torch.where(present, -10.0, 10.0).to(x.dtype)
        return L.Heads(keep[None], torch.zeros_like(x)[None], torch.full_like(x[..., 0], -10.0)[None])


def naked_singles_solve(sol, clue):
    board = np.where(clue, sol, -1)
    while True:
        progress = False
        for cell in np.where(board < 0)[0]:
            taken = {board[p] for p in PEERS[cell]} - {-1}
            opts = [d for d in range(9) if d not in taken]
            if len(opts) == 1:
                board[cell], progress = opts[0], True
        if not progress:
            return board


def check_chain_search():
    rng = np.random.default_rng(6)
    easy = []
    while len(easy) < 19:
        sol, clue = random_puzzle(rng, int(rng.integers(1, 6)))
        if (naked_singles_solve(sol, clue) == sol).all():
            easy.append((sol, clue))
    x0 = torch.stack([lattice_of(s, c) for s, c in easy]).float()
    digits, steps, resets = L.chain_search(Oracle(), x0, chains=8, budget=50, check_every=1, slots=7, amp=False,
                                           generator=torch.Generator().manual_seed(0))
    sols = torch.as_tensor(np.stack([s for s, _ in easy]))
    assert torch.equal(digits, sols), 'oracle chain search should solve every easy puzzle'
    assert int(resets.sum()) == 0
    # a wrong commit must be caught: a puzzle with many blanks forces guesses; every answer is still checked by grading
    hard = [random_puzzle(rng, 50) for _ in range(4)]
    x0 = torch.stack([lattice_of(s, c) for s, c in hard]).float()
    digits, steps, resets = L.chain_search(Oracle(), x0, chains=16, budget=400, check_every=4, slots=4, amp=False,
                                           generator=torch.Generator().manual_seed(1))
    for i, (s, c) in enumerate(hard):
        d = digits[i].numpy()
        if (d >= 0).all():  # a returned board is a valid completion of the givens (the oracle kills peers' digits)
            assert (d[c] == s[c]).all()
            for cell in range(81):
                assert all(d[cell] != d[p] for p in PEERS[cell])
    print('chain search ok: easy 19/19, hard returned', int((digits >= 0).all(-1).sum()), 'of 4, steps', steps.tolist(),
          'resets', resets.tolist())


def check_starts():
    rng = np.random.default_rng(7)
    pairs = [random_puzzle(rng, 55) for _ in range(64)]
    sols = torch.as_tensor(np.stack([s for s, _ in pairs]))
    clues = torch.as_tensor(np.stack([c for _, c in pairs]))
    g = torch.Generator().manual_seed(0)
    x0, y = L.draw_starts(sols, clues, 6000, generator=g)
    given = x0.sum(-1) == 1
    _, board = L.conflicts(x0, y)
    ngiven = given.sum(-1)
    raw = ngiven == 26
    # raw starts never conflict; every conflicting start has a wrong given
    wrong = (given & ~((x0 > .5) & (y > .5)).any(-1)).any(-1)
    assert torch.equal(board, wrong)
    frac_conf = float(board.float().mean())
    frac_raw = float(raw.float().mean())
    assert abs(frac_conf - 0.25) < 0.03, frac_conf
    assert 0.18 < frac_raw < 0.30, frac_raw  # raw kind (0.20) plus correct fills that drew zero cells
    assert bool((y.sum(-1) == 1).all())
    # the digit relabelling keeps a valid board
    yd = y.argmax(-1)
    for i in range(20):
        for cell in range(81):
            assert all(yd[i, cell] != yd[i, p] for p in PEERS[cell])
    print(f'starts ok: conflicting {frac_conf:.3f}, raw {frac_raw:.3f}')


def check_training_moves():
    """A tiny model on easy puzzles: the loss falls over 60 updates."""
    torch.manual_seed(0)
    net = L.LatticeNet(widths=(6, 12, 16, 6), cycles=2, rounds=2)
    rng = np.random.default_rng(9)
    pairs = [random_puzzle(rng, int(rng.integers(1, 4))) for _ in range(60)]
    sols = torch.as_tensor(np.stack([s for s, _ in pairs]))
    clues = torch.as_tensor(np.stack([c for _, c in pairs]))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        x0, y = L.draw_starts(sols, clues, 32)
        loss, _ = L.lattice_loss(net(x0), x0, y, x0.sum(-1) == 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert np.mean(losses[-5:]) < np.mean(losses[:5]), losses
    print(f'training moves ok: {np.mean(losses[:5]):.3f} -> {np.mean(losses[-5:]):.3f}')


if __name__ == '__main__':
    check_parameters()
    check_projection()
    check_status()
    check_loss()
    check_equivariance()
    check_chain_search()
    check_starts()
    check_training_moves()
    print('ALL OK')
