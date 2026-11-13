"""CRAM on Maze-Unique: the local pair energies and two searches that start from a supplied state.

`CRAMMazeFreeEnergy` at window 1 is the model of the paper's Maze-Unique row (`cram-sgd-fg`, 18 parameters). Its
energy, for the cell states s_i (a vector over (off, on); one-hot at a wall (off), at S and at G (on)), is

    E(s) = sum_i sum_(d in half offsets) s_i^T T_d s_(i+d)  -  sum_(i free) b . s_i  +  (1/beta) sum_(i free) s_i . log s_i,

with T_d a free 2 x 2 table for each offset d of `half_offsets(window)` (one offset of each pair {d, -d} of the
(2 window + 1)^2 window, so each unordered pair of cells inside the window is counted once), b (2,) the bias of an
open cell, and beta the inverse temperature of the soft states (1). The tables are shared over the grid and are zero
outside the window; nothing else ties them. On a hard board the entropy term is 0.
The data, the pseudolikelihood loss and the search of the paper's row (`gibbs`) are in `cram.maze_unique`.

`CRAMMazeEnergy` is the separable form of the same energy (the energy of `maze_unique.MazeEnergy`, whose docstring
gives the equation). The hand-set maze energy (`experiments/hand_set.py`) and the separable option of the
training-method study use it.

Dimension names: b mazes (or maze x chain), a and c answer states (0 off, 1 on), h and w grid rows and columns,
r relations, e factor width, d window offsets ((2 window + 1)^2 of them).

`maze_unique.MazeEnergy` stores 34 floats at R 1, de 2, window 1. Twelve of them never enter the
energy (the cell term Wc, 4, and the start logits U0, 8), and six more are the bias rows of walls, S and G, which act
only at clamped cells: pseudolikelihood takes no loss there, so they keep their zero start. `CRAMMazeEnergy` stores
the 16 that receive a training gradient and enter the energy:

    Fp (r, 2 window + 1), Gp (r, 2 window + 1)   separable offset factors, A_ij = F[dr] G[dc]
    P (r, 2, e), Q (r, 2, e)                     the state kernel K_r = Q_r P_r^T
    b (2,)                                       the bias of an open cell, B_i = b at free cells, 0 at clamped cells

It draws Fp, Gp, P, Q from the generator in the same order and scale as `MazeEnergy`, so a run with the same seed
trains the same weights; `from_legacy` converts a `MazeEnergy` state dict. `kernel`, `current`, `energy`, `w` and `beta` have
the same meaning as in `maze_unique`, so its searches (`descend`, `gibbs`) and flop counts take either model.

`CRAMMazeFreeEnergy` replaces the separable factors by a free 2 x 2 table per window offset, so that the
only supplied structure is locality and translation invariance: 4 tables + b = 18 parameters at window 1.

The searches of this file start from a supplied board or supplied logits: `gibbs_from` (single-cell moves at
temperature T, focused or random order, the first fixed point or the lowest-energy board seen) and `descend_from`
(Euler steps of the damped update, optionally with noise).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from cram import maze_unique as M


class CRAMMazeEnergy(nn.Module):
    """The separable Maze-Unique energy with R relations, factor width de and window radius `window` (module
    docstring). `init` scales the initial offset factors; beta is the inverse temperature of the soft states."""

    def __init__(self, R=1, de=2, window=1, beta=1.0, init=0.3):
        super().__init__()
        self.R, self.de, self.w, self.beta = R, de, window, beta
        self.Fp = nn.Parameter(init * torch.randn(R, 2 * window + 1))
        self.Gp = nn.Parameter(init * torch.randn(R, 2 * window + 1))
        self.P = nn.Parameter(torch.randn(R, 2, de) / 2 ** 0.5)
        self.Q = nn.Parameter(0.1 * torch.randn(R, 2, de) / 2 ** 0.5)
        self.b = nn.Parameter(torch.zeros(2))

    def n_params(self):
        """Number of learned values (16 at R 1, de 2, window 1)."""
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_legacy(cls, state, R, de, window, beta=1.0):
        """A `MazeEnergy` state dict -> this model. Refuses a state whose wall, S or G bias rows are not zero (they
        would act at clamped cells only, but then the two models' currents would differ there)."""
        B = state['B']
        assert torch.all(B[[M.WALL, M.S_TOK, M.G_TOK]] == 0), 'clamped-token bias rows are not zero'
        m = cls(R=R, de=de, window=window, beta=beta)
        m.load_state_dict({'Fp': state['Fp'], 'Gp': state['Gp'], 'P': state['P'], 'Q': state['Q'], 'b': B[M.OPEN]})
        return m

    def bias_table(self):
        """(4, 2): the token bias table of `MazeEnergy` (zero rows for walls, S and G), built so that indexing it by the
        tokens gives the same autograd graph as `MazeEnergy.B[x]`."""
        z = torch.zeros(1, 2, device=self.b.device, dtype=self.b.dtype)
        rows = [z, z, z, z]
        rows[M.OPEN] = self.b[None]
        return torch.cat(rows)

    def kernel(self):
        """W (2, 2, 2w+1, 2w+1): the current's cross-correlation kernel, W[a, c, dr + w, dc + w] (as `MazeEnergy`)."""
        K = torch.einsum('rae,rce->rac', self.Q, self.P)
        Fl, Gl = self.Fp.flip(1), self.Gp.flip(1)
        fwd = torch.einsum('ry,rx,rac->acyx', Fl, Gl, K)
        bwd = torch.einsum('ry,rx,rca->acyx', self.Fp, self.Gp, K)
        center = torch.ones(2 * self.w + 1, 2 * self.w + 1, device=fwd.device)
        center[self.w, self.w] = 0
        return (fwd + bwd) * center

    def current(self, s, x, W=None):
        """dE/ds (b, 2, h, w) of the pair and bias terms for states s (b, 2, h, w) and tokens x (b, h, w).
        W: a precomputed `kernel()`."""
        W = self.kernel() if W is None else W
        return F.conv2d(s, W, padding=self.w) - rearrange(self.bias_table()[x], 'b h w a -> b a h w')

    def energy(self, s, x, free):
        """Total energy (b,): pair + bias + entropy of the free cells (as `MazeEnergy.energy`), for states
        s (b, 2, h, w), tokens x (b, h, w) and free (b, h, w) bool."""
        W = self.kernel()
        pair = 0.5 * (F.conv2d(s, W, padding=self.w) * s).sum((1, 2, 3))
        bias = (s * rearrange(self.bias_table()[x], 'b h w a -> b a h w')).sum((1, 2, 3))
        ent = (free[:, None] * s * torch.log(s.clamp_min(1e-12))).sum((1, 2, 3)) / self.beta
        return pair - bias + ent


def half_offsets(window):
    """The offsets (dr, dc) with dr > 0, or dr = 0 and dc > 0, inside +-window: one per unordered cell pair direction
    ((2 window + 1)^2 - 1) / 2 of them, 4 at window 1)."""
    return [(dr, dc) for dr in range(0, window + 1) for dc in range(-window, window + 1) if dr > 0 or dc > 0]


class CRAMMazeFreeEnergy(nn.Module):
    """The Maze-Unique energy with a free pair cost in the window: no separable factorization.

    For each offset d of `half_offsets(window)`, a free 2 x 2 table T_d: the pair (i, j = i + d) costs s_i^T T_d s_j.
    The tables are shared over the grid (translation invariance) and zero outside the window (locality); nothing else
    ties them. The pair (i, i - d) is the same unordered pair, so the kernel holds T_d at d and T_d^T at -d. The open-cell
    bias b (2,) is that of CRAMMazeEnergy. At window 1: 4 x 4 + 2 = 18 parameters.

    `init` is the standard deviation of the initial tables (0: zero start, the default); b starts at zero.
    `kernel`, `current`, `energy`, `w`, `beta` and `bias_table` have the meaning of CRAMMazeEnergy's, so the searches of
    this module and of maze_unique, and their flop counts, take this model unchanged."""

    def __init__(self, window=1, beta=1.0, init=0.0):
        super().__init__()
        self.w, self.beta = window, beta
        self.offsets = half_offsets(window)
        self.T = nn.Parameter(init * torch.randn(len(self.offsets), 2, 2))
        self.b = nn.Parameter(torch.zeros(2))

    def n_params(self):
        """Number of learned values (18 at window 1)."""
        return sum(p.numel() for p in self.parameters())

    bias_table = CRAMMazeEnergy.bias_table

    def kernel(self):
        """W (2, 2, 2w+1, 2w+1): W[a, c, dr + w, dc + w] is the cost table between cell i in state a and cell i + d in
        state c, so that F.conv2d(s, W) at i is sum_d W[:, :, d] s_{i+d} (cross-correlation)."""
        w = self.w
        W = self.T.new_zeros(2, 2, 2 * w + 1, 2 * w + 1)
        for t, (dr, dc) in enumerate(self.offsets):
            W[:, :, w + dr, w + dc] = self.T[t]
            W[:, :, w - dr, w - dc] = self.T[t].T
        return W

    current = CRAMMazeEnergy.current
    energy = CRAMMazeEnergy.energy


def board_energy(model, board, x, free, W=None):
    """Energy (b,) of hard boards (b, h, w) with the givens clamped (no entropy: it is 0 at one-hot states)."""
    s = M.onehot(board) * free[:, None] + M.given(x, free)
    W = model.kernel() if W is None else W
    pair = 0.5 * (F.conv2d(s, W, padding=model.w) * s).sum((1, 2, 3))
    g0 = model.current(torch.zeros_like(s), x, W)  # minus the bias
    return pair + (s * g0).sum((1, 2, 3))


@torch.no_grad()
def gibbs_from(model, x, y, free, start, T=0.0, rho=0.0, max_moves=20000, stop='fixed', gen=None, W=None):
    """Single-cell moves from the hard boards `start` (b, h, w) (its clamped cells are ignored).

    Each move picks one free cell: with probability rho a uniformly random free cell, otherwise a uniformly random
    free cell that is not at its lowest-energy answer (the focused Gibbs of `maze_unique.gibbs`; if there is none, a
    uniformly random free cell). The cell is set to its lowest-energy answer (T = 0) or redrawn from the heat bath
    p(a) ~ exp(-dE/ds_i[a] / T). Fields are kept exact by updating the moved cell's (2w+1)^2 neighbourhood, and the
    energy by the moved cell's field difference.

    stop = 'fixed': a board stops at its first fixed point (no free cell has a lower-energy answer), the stopping test
    of `maze_unique.gibbs`; boards that never reach one return the lowest-energy board seen. stop = 'none': every board
    runs `max_moves` moves and returns the lowest-energy board seen.

    Returns per board: moves, stopped (at a fixed point), pred (the returned board), energy (of pred), right."""
    W = model.kernel() if W is None else W
    nb, H, Wd = x.shape
    w = model.w
    k = 2 * w + 1
    n = H * Wd
    dev = x.device
    board = torch.where(free, start, y * (~free))
    giv = M.given(x, free)
    fr = free[:, None]
    g = model.current(M.onehot(board) * fr + giv, x, W)
    Hp, Wp = H + 2 * w, Wd + 2 * w
    gp = torch.zeros(nb, 2, Hp, Wp, device=dev)
    gp[:, :, w:w + H, w:w + Wd] = g
    gp = rearrange(gp, 'b a hp wp -> b a (hp wp)')
    dr, dc = torch.meshgrid(torch.arange(-w, w + 1, device=dev), torch.arange(-w, w + 1, device=dev), indexing='ij')
    Wf = rearrange(W, 'a c y x -> a c (y x)')
    tgt_off = rearrange(-dr * Wp - dc, 'y x -> (y x)')
    flat = rearrange(board, 'b h w -> b (h w)').clone()
    freef = rearrange(free, 'b h w -> b (h w)')
    e = board_energy(model, board, x, free, W)
    best_e, best = e.clone(), flat.clone()
    moves = torch.zeros(nb, dtype=torch.long, device=dev)
    stopped = torch.zeros(nb, dtype=torch.bool, device=dev)
    ar = torch.arange(nb, device=dev)
    cells = torch.arange(n, device=dev)
    interior = lambda: rearrange(rearrange(gp, 'b a (hp wp) -> b a hp wp', hp=Hp)[:, :, w:w + H, w:w + Wd], 'b a h w -> b a (h w)')
    for _ in range(max_moves + 1):
        gin = interior()
        gy = gin.gather(1, flat[:, None]).squeeze(1)
        bad = ((gy - gin.min(1).values) > 0) & freef
        at_fixed = ~bad.any(1)
        if stop == 'fixed':
            newly = at_fixed & ~stopped
            best = torch.where(newly[:, None], flat, best)
            best_e = torch.where(newly, e, best_e)
            stopped |= at_fixed
        active = ~stopped if stop == 'fixed' else torch.ones_like(stopped)
        if not active.any() or int(moves.max()) >= max_moves:
            break
        u = torch.rand(nb, n, device=dev, generator=gen)
        pick_rand = torch.rand(nb, device=dev, generator=gen) < rho
        pool = torch.where((pick_rand | at_fixed)[:, None], freef, bad)
        cell = (u * pool).argmax(1)
        gc = gin[ar, :, cell]
        if T > 0:
            new = torch.multinomial(torch.softmax(-gc / T, 1), 1, generator=gen).squeeze(1)
        else:
            new = gc.argmin(1)
        old = flat[ar, cell]
        change = active & (new != old)
        moves += active
        e = e + torch.where(change, gc[ar, new] - gc[ar, old], torch.zeros_like(e))
        ds = (F.one_hot(new, 2) - F.one_hot(old, 2)).float() * change[:, None]
        upd = torch.einsum('acd,bc->bad', Wf, ds)
        q = (cell // Wd + w) * Wp + (cell % Wd + w)
        idx = (q[:, None] + tgt_off[None, :]).clamp(0, Hp * Wp - 1)
        gp.scatter_add_(2, repeat(idx, 'b d -> b a d', a=2), upd)
        flat = torch.where(change[:, None] & (cells[None] == cell[:, None]), new[:, None], flat)
        lower = active & (e < best_e) & ~stopped
        best = torch.where(lower[:, None], flat, best)
        best_e = torch.where(lower, e, best_e)
    pred = rearrange(best, 'b (h w) -> b h w', h=H)
    right = ((pred == y) | ~free).all((1, 2))
    return {'moves': moves, 'stopped': stopped, 'pred': pred, 'energy': best_e, 'right': right}


def gibbs_from_flop(model, moves, n=M.SIDE * M.SIDE):
    """As `maze_unique.gibbs_flop`: one full current to start, then per move the field change at (2w+1)^2 - 1 cells and
    the focus (2n). The energy bookkeeping (2 flop per move) is elementwise and not counted."""
    return M.gibbs_flop(model, moves, n)


@torch.no_grad()
def descend_from(model, x, y, free, u0, eta=0.5, sigma=0.0, max_steps=4000, tol=1e-3, check_every=16, gen=None, W=None):
    """Euler steps u <- u + eta (-dE/ds - u) + sigma sqrt(eta) xi of all free cells from logits u0 (b, 2, h, w).

    sigma = 0: a board stops at its first step with fixed-point residual < tol (the test of `maze_unique.descend`).
    Every `check_every` steps the hard board is also scored by its energy, and the lowest-energy hard board seen is kept;
    a board that never stops returns it. Currents counted: one per step plus one per check.
    Returns per board: steps, checks, stopped, pred, energy (of pred), right."""
    W = model.kernel() if W is None else W
    nb = x.shape[0]
    dev = x.device
    giv = M.given(x, free)
    fr = free[:, None]
    u = u0.clone()
    active = torch.ones(nb, dtype=torch.bool, device=dev)
    steps = torch.zeros(nb, dtype=torch.long, device=dev)
    checks = torch.zeros(nb, dtype=torch.long, device=dev)
    pred = torch.zeros_like(y)
    best_e = torch.full((nb,), float('inf'), device=dev)
    stopped = torch.zeros(nb, dtype=torch.bool, device=dev)
    for t in range(max_steps):
        v = torch.where(fr, torch.softmax(model.beta * u, 1), giv)
        g = model.current(v, x, W)
        r = (-g - u) * fr
        steps += active
        if sigma == 0:
            done = active & (r.pow(2).sum((1, 2, 3)).sqrt() < tol)
            hard = torch.where(free, v.argmax(1), y * (~free))
            pred = torch.where(done[:, None, None], hard, pred)
            best_e = torch.where(done, board_energy(model, hard, x, free, W), best_e)
            stopped |= done
            active &= ~done
        if (t + 1) % check_every == 0:
            hard = torch.where(free, v.argmax(1), y * (~free))
            eh = board_energy(model, hard, x, free, W)
            checks += active
            lower = active & (eh < best_e)
            pred = torch.where(lower[:, None, None], hard, pred)
            best_e = torch.where(lower, eh, best_e)
        if not active.any():
            break
        noise = sigma * eta ** 0.5 * torch.randn(u.shape, device=dev, generator=gen) if sigma > 0 else 0.0
        u = torch.where(active[:, None, None, None], u + eta * r + noise * fr, u)
    right = ((pred == y) | ~free).all((1, 2))
    return {'steps': steps, 'checks': checks, 'stopped': stopped, 'pred': pred, 'energy': best_e, 'right': right}
