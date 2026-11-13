"""Maze-Unique: the data loader, the pair energy with a separable window kernel, the pseudolikelihood loss and the
searches. The model of the paper's Maze-Unique row (`cram-sgd-fg`) is `cram_maze.CRAMMazeFreeEnergy`, a free kernel
with 18 parameters; it is trained with `pl_loss` and searched with `gibbs` of this file.

Data. Source: the Hugging Face dataset `locuslab/EqR-data` at revision bcb7d45c, folder `maze-30x30-unique-1k` (the
mazes of the baseline EqR, each with a unique path from start to goal). `scripts/prepare_data.py maze-unique`
downloads it to `<data>/eqr/maze-30x30-unique-1k/` (`<data>` is `data/` or `CRAM_DATA`; `EQR_DIR` overrides the
folder). The files are `{train,test}/all__inputs.npy` and `all__labels.npy`, each (1000, 900) uint8: 1,000 mazes of
30 x 30 cells in row-major order. `_read` checks the SHA-256 of each file against `SHA256`.
Encoding in the files: inputs 1 wall, 2 open, 3 start (S), 4 goal (G); labels 5 on the path cells.
Encoding returned by `load`: tokens 0 wall, 1 open, 2 S, 3 G; target 1 on the path and at S and G, else 0;
free = True at open cells. Walls (off), S and G (on) are given, so only the open cells are searched and scored.
Splits: 'train' (1,000 mazes), 'test' (1,000 mazes), and 'train900' / 'dev100', a seeded division of the training
file (`dev_split`: the first 100 entries of a permutation with seed 0 are the development mazes). Settings are chosen
on dev100. No augmentation is applied.

Dimension names: b mazes, a and c answer states (0 off, 1 on), h and w grid rows and columns (30, 30), r relations,
e factor width.

Energy of `MazeEnergy`, for the cell states s_i (a vector over (off, on)):

    E(s) = sum_r sum_(i != j) A^r_ij s_i^T K_r s_j  -  sum_i s_i . B[x_i]  +  (1/beta) sum_(i free) v_i . log v_i
    A^r_ij = F_r[row_i - row_j] G_r[col_i - col_j]  (a value per offset within +-window, zero beyond),  K_r = Q_r P_r^T

Symbols: F_r and G_r the row and column factors of relation r (one value per offset), K_r the 2 x 2 state table of
relation r, B (4, 2) the bias per input token, beta the inverse temperature, u_i the logits of cell i.
s_i = v_i = softmax(beta u_i) over (off, on) at a free (open) cell, the one-hot answer at a wall, S or G. Because the
pair term is bilinear, the current dE/ds_i is a 2D cross-correlation of s with one (2 x 2 x (2w+1) x (2w+1)) kernel W:

    dE_pair/ds_i[a] = sum_delta sum_c W[a, c, delta] s_(i + delta)[c],
    W[a, c, delta] = sum_r F_r[-dr] G_r[-dc] K_r[a, c] + F_r[dr] G_r[dc] K_r[c, a],   W[., ., 0] = 0,

which is how every search here computes it (`kernel` and `current`). The searches and the loss take any model with
`kernel`, `current`, `w` (the window radius) and `beta`: `MazeEnergy`, `cram_maze.CRAMMazeEnergy` and
`cram_maze.CRAMMazeFreeEnergy`.
In-code operation estimate (flop; a multiply-add is 2 flop): the cross-correlation over the whole grid,
2 * 2 * 2 * (2w+1)^2 per cell (583,200 per step on 30 x 30 at w = 4); elementwise work is not counted here. The
paper's operation counts follow the convention of `baselines/flops/fullcount.py`.

Searches:
- `descend`: the damped update u <- u + eta (-dE/ds - u) from every open cell on (logit `margin`), each maze
  stopped at its first step with the fixed-point residual |(-dE/ds - u) on free cells|_2 < tol, or at the board test
  (`certify_every`): the hard board is a fixed point of the energy, every free cell already at its lowest-energy
  answer given the others.
- `gibbs`: focused Gibbs on the hard board from every open cell on: each move redraws one free cell whose answer is
  not its lowest-energy one (by more than theta), at temperature T (T = 0: the lowest); a maze stops when no cell
  disagrees, which is the same board test. The paper's row uses T = 0.
Training: `pl_loss`, the pseudolikelihood at the labeled solution. It needs no search.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# locuslab/EqR-data at revision bcb7d45c (scripts/prepare_data.py maze-unique); EQR_DIR overrides the folder
EQR = Path(os.environ.get('EQR_DIR', Path(os.environ.get('CRAM_DATA', Path(__file__).resolve().parents[1] / 'data')) / 'eqr' / 'maze-30x30-unique-1k'))
SHA256 = {  # the four data files at the pinned revision
    'train/all__inputs.npy': '2922455815232dcb636ee45f335e7ce834aa45d4fab743b96ac97c17284ca7e2',
    'train/all__labels.npy': '6b1a99761606b78f85fe43085b158457f22a3d6274b1e09f12f7b16814fdb5e4',
    'test/all__inputs.npy': 'c501b3f75ec018e27fd6d00534b6738ebcc555da50536909e530787d16b1ce0f',
    'test/all__labels.npy': '20b21168459847fb260d986394098addf7cca94becc7e682dcb358967d5993dc',
}
SIDE = 30  # grid rows and columns
WALL, OPEN, S_TOK, G_TOK = 0, 1, 2, 3  # input tokens


def _read(rel):
    """The array of the data file `rel` (a key of SHA256), after its hash check."""
    path = EQR / rel
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    assert h == SHA256[rel], f'{path}: sha256 {h} is not the pinned file'
    return np.load(path)


def dev_split(n=1000, size=100, seed=0):
    """(train indices, dev indices) into the n training mazes, each sorted: dev = the first `size` entries of numpy's
    default_rng(seed).permutation(n), train = the others."""
    order = np.random.default_rng(seed).permutation(n)
    return np.sort(order[size:]), np.sort(order[:size])


def load(split):
    """split: train | test | train900 | dev100. Returns a dict of arrays, each (N, 900) in row-major order:
    'inputs' int64 tokens (0 wall, 1 open, 2 S, 3 G), 'targets' int64 (1 on the path, S and G included), 'free' bool
    (True at open cells)."""
    base = 'test' if split == 'test' else 'train'
    X = _read(f'{base}/all__inputs.npy').astype(np.int64)
    Y = _read(f'{base}/all__labels.npy').astype(np.int64)
    if split in ('train900', 'dev100'):
        tr, dv = dev_split(len(X))
        idx = tr if split == 'train900' else dv
        X, Y = X[idx], Y[idx]
    tok = X - 1
    target = ((Y == 5) | (X == 3) | (X == 4)).astype(np.int64)
    return {'inputs': tok, 'targets': target, 'free': tok == OPEN}


def to_device(data, idx, dev):
    """The mazes `idx` of a `load` dict as tensors on `dev`: (x tokens, y targets, free), each (b, 30, 30)."""
    x = torch.as_tensor(data['inputs'][idx], device=dev).view(-1, SIDE, SIDE)
    y = torch.as_tensor(data['targets'][idx], device=dev).view(-1, SIDE, SIDE)
    free = torch.as_tensor(data['free'][idx], device=dev).view(-1, SIDE, SIDE)
    return x, y, free


class MazeEnergy(nn.Module):
    """The energy of the module docstring with R relations, factor width de and window radius `window`.
    Parameters: Fp, Gp (r, 2 window + 1) the offset factors F_r, G_r; P, Q (r, 2, e) with K_r = Q_r P_r^T; B (vin, 2)
    the token biases. Wc and U0 are stored for compatibility with older checkpoints and never enter the energy.
    `cram_maze.CRAMMazeEnergy` is the same energy without the unused parameters."""

    def __init__(self, R=6, de=16, window=4, beta=1.0, init=0.3, vin=4):
        super().__init__()
        self.R, self.de, self.w, self.beta = R, de, window, beta
        self.Fp = nn.Parameter(init * torch.randn(R, 2 * window + 1))
        self.Gp = nn.Parameter(init * torch.randn(R, 2 * window + 1))
        self.P = nn.Parameter(torch.randn(R, 2, de) / 2 ** 0.5)
        self.Q = nn.Parameter(0.1 * torch.randn(R, 2, de) / 2 ** 0.5)
        self.Wc = nn.Parameter(0.1 * torch.randn(2, 2) / 2 ** 0.5)  # unused; kept so that the random draws and the state dict keep their layout
        self.B = nn.Parameter(torch.zeros(vin, 2))
        self.U0 = nn.Parameter(torch.zeros(vin, 2))  # unused (the search sets its start)

    def n_params(self):
        """Number of stored values (unused ones included)."""
        return sum(p.numel() for p in self.parameters())

    def kernel(self):
        """W (2, 2, 2w+1, 2w+1): the current's cross-correlation kernel, W[a, b, dr + w, dc + w]."""
        K = torch.einsum('rae,rbe->rab', self.Q, self.P)  # K_r[a, b] = (Q_r P_r^T)[a, b]
        Fl, Gl = self.Fp.flip(1), self.Gp.flip(1)  # value at -d
        fwd = torch.einsum('ry,rx,rab->abyx', Fl, Gl, K)  # F_r[-dr] G_r[-dc] K_r[a, b]
        bwd = torch.einsum('ry,rx,rba->abyx', self.Fp, self.Gp, K)  # F_r[dr] G_r[dc] K_r[b, a]
        center = torch.ones(2 * self.w + 1, 2 * self.w + 1, device=fwd.device)
        center[self.w, self.w] = 0  # no self-coupling (i != j)
        return (fwd + bwd) * center

    def current(self, s, x, W=None):
        """dE/ds (b, 2, h, w) of the pair and bias terms for states s (b, 2, h, w) and tokens x (b, h, w): the pair
        term's cross-correlation minus the bias. W: a precomputed `kernel()`."""
        W = self.kernel() if W is None else W
        return F.conv2d(s, W, padding=self.w) - self.B[x].permute(0, 3, 1, 2)

    def energy(self, s, x, free):
        """Total energy (b,): pair + bias + entropy of the free cells, for states s (b, 2, h, w), tokens x (b, h, w)
        and free (b, h, w) bool (s at clamped cells must be their one-hot)."""
        W = self.kernel()
        pair = 0.5 * (F.conv2d(s, W, padding=self.w) * s).sum((1, 2, 3))  # the bilinear form, each ordered pair once
        bias = (s * self.B[x].permute(0, 3, 1, 2)).sum((1, 2, 3))
        ent = (free[:, None] * s * torch.log(s.clamp_min(1e-12))).sum((1, 2, 3)) / self.beta
        return pair - bias + ent


def flop_per_step(model, H=SIDE, Wd=SIDE):
    """flop of one current over an H x Wd grid: the cross-correlation, 2 flop per multiply-add."""
    k = 2 * model.w + 1
    return 2 * 2 * 2 * k * k * H * Wd


def onehot(y):
    """Boards y (b, h, w) of 0/1 as states (b, 2, h, w) float."""
    return F.one_hot(y, 2).permute(0, 3, 1, 2).float()


def start_logits(x, free, margin=3.0, beta=1.0):
    """The start of `descend`, logits (b, 2, h, w): every free cell on with logit margin / beta, zeros elsewhere."""
    u = torch.zeros(x.shape[0], 2, *x.shape[1:], device=x.device)
    u[:, 1] = torch.where(free, torch.full_like(x, margin, dtype=torch.float) / beta, torch.zeros((), device=x.device))
    return u


def given(x, free):
    """The given one-hot answers (b, 2, h, w): walls off, S and G on (0 at free cells)."""
    on = (x == S_TOK) | (x == G_TOK)
    return onehot(on.long()) * (~free)[:, None]


def pl_loss(model, x, y, free):
    """The pseudolikelihood loss (scalar): the mean over free cells of -log p(y_i | the other cells at the label),
    p = softmax(-beta dE/ds_i) at the solution. x tokens, y targets, free: each (b, h, w)."""
    g = model.current(onehot(y), x)
    logp = torch.log_softmax(-model.beta * g, 1).gather(1, y[:, None]).squeeze(1)
    return -(logp * free).sum() / free.sum()


def disagreement(g, y, free):
    """(b, h, w): how much higher the board's answer is than the cell's lowest-energy answer (0 if it is the lowest),
    for currents g (b, 2, h, w) and a board y (b, h, w); 0 at clamped cells."""
    gy = g.gather(1, y[:, None]).squeeze(1)
    return (gy - g.min(1).values) * free


@torch.no_grad()
def descend(model, x, y, free, eta=0.5, max_steps=4000, tol=1e-3, margin=3.0, certify_every=0, W=None):
    """Descent from every open cell on, each maze stopped at its fixed point (residual < tol) or, with certify_every,
    at the first check where its hard board is a fixed point of the energy. x, y, free: (b, h, w); y gives the clamped
    answers and scores the result. Returns a dict of per-maze tensors: steps (descent steps made), checks (board tests
    made, each one more current), right (every free cell right at the stop), stop_kind (0 never stopped, 1 residual,
    2 board test), pred (b, h, w) the returned board."""
    W = model.kernel() if W is None else W
    b = x.shape[0]
    u = start_logits(x, free, margin, model.beta)
    giv = given(x, free)
    fr = free[:, None]
    active = torch.ones(b, dtype=torch.bool, device=x.device)
    steps = torch.zeros(b, dtype=torch.long, device=x.device)
    checks = torch.zeros(b, dtype=torch.long, device=x.device)
    pred = torch.zeros_like(y)
    stop_kind = torch.zeros(b, dtype=torch.long, device=x.device)  # 0 never, 1 residual, 2 certificate
    for t in range(max_steps):
        v = torch.where(fr, torch.softmax(model.beta * u, 1), giv)
        g = model.current(v, x, W)
        r = (-g - u) * fr
        res = r.pow(2).sum((1, 2, 3)).sqrt()
        steps += active
        done = active & (res < tol)
        if certify_every and (t + 1) % certify_every == 0:
            hard = torch.where(free, v.argmax(1), y * (~free))
            gh = model.current(onehot(hard) * fr + giv, x, W)
            checks += active
            ok = active & ~done & (disagreement(gh, hard, free) <= 0).all((1, 2))
            pred = torch.where(ok[:, None, None], hard, pred)
            stop_kind = torch.where(ok, torch.full_like(stop_kind, 2), stop_kind)
            active = active & ~ok
        if done.any():
            hard = torch.where(free, v.argmax(1), y * (~free))
            pred = torch.where(done[:, None, None], hard, pred)
            stop_kind = torch.where(done, torch.ones_like(stop_kind), stop_kind)
            active = active & ~done
        if not active.any():
            break
        u = torch.where(active[:, None, None, None], u + eta * r, u)
    if active.any():  # never stopped: the board at the cap
        v = torch.where(fr, torch.softmax(model.beta * u, 1), giv)
        hard = torch.where(free, v.argmax(1), y * (~free))
        pred = torch.where(active[:, None, None], hard, pred)
    right = ((pred == y) | ~free).all((1, 2))
    return {'steps': steps, 'checks': checks, 'right': right, 'stop_kind': stop_kind, 'pred': pred}


@torch.no_grad()
def gibbs(model, x, y, free, T=0.0, theta=0.0, max_moves=20000, focus='random', gen=None, W=None):
    """Focused Gibbs on the hard board from every open cell on. Each move: one free cell whose answer is above its
    lowest-energy answer by more than theta (chosen uniformly, or the most disagreeing with focus='max') is redrawn
    from the heat bath p(a) ~ exp(-dE/ds_i[a] / T) (T = 0: the lowest). A maze stops when no cell disagrees by more
    than theta. Fields are kept exact by adding the moved cell's change to its (2w+1)^2 neighbourhood.
    x, y, free: (b, h, w); gen: the generator of the random draws; max_moves: the move limit per maze.
    Returns a dict of per-maze tensors: moves, right (every free cell right), stopped (no cell disagrees), pred
    (b, h, w) the final board, fields (b, 2, h, w) the final currents."""
    W = model.kernel() if W is None else W
    b, H, Wd = x.shape
    w = model.w
    k = 2 * w + 1
    dev = x.device
    board = torch.where(free, torch.ones_like(y), y * (~free))
    fr = free[:, None]
    giv = given(x, free)
    g = model.current(onehot(board) * fr + giv, x, W)  # (b, 2, H, W)
    Hp, Wp = H + 2 * w, Wd + 2 * w
    gp = torch.zeros(b, 2, Hp, Wp, device=dev)
    gp[:, :, w:w + H, w:w + Wd] = g
    gp = gp.reshape(b, 2, Hp * Wp)
    dr, dc = torch.meshgrid(torch.arange(-w, w + 1, device=dev), torch.arange(-w, w + 1, device=dev), indexing='ij')
    Wf = W.reshape(2, 2, k * k)  # W[a, c, delta]
    # a change at cell q changes the field at q - delta by W[:, :, delta] (Delta s_q)
    tgt_off = (-dr * Wp - dc).reshape(-1)  # padded flat offset of q - delta
    moves = torch.zeros(b, dtype=torch.long, device=dev)
    flat_board = board.reshape(b, H * Wd)
    freef = free.reshape(b, H * Wd)
    ar = torch.arange(b, device=dev)
    for _ in range(max_moves):
        gin = gp.reshape(b, 2, Hp, Wp)[:, :, w:w + H, w:w + Wd].reshape(b, 2, H * Wd)
        gy = gin.gather(1, flat_board[:, None]).squeeze(1)
        dis = (gy - gin.min(1).values) * freef
        bad = (dis > theta) & freef
        active = bad.any(1)
        if not active.any():
            break
        if focus == 'max':
            cell = dis.argmax(1)
        else:
            cell = (torch.rand(b, H * Wd, device=dev, generator=gen) * bad).argmax(1)
        gc = gin[ar, :, cell]  # (b, 2)
        if T > 0:
            p = torch.softmax(-gc / T, 1)
            new = torch.multinomial(p, 1, generator=gen).squeeze(1)
        else:
            new = gc.argmin(1)
        old = flat_board[ar, cell]
        change = active & (new != old)
        moves += active
        ds = (F.one_hot(new, 2) - F.one_hot(old, 2)).float() * change[:, None]  # (b, 2)
        upd = torch.einsum('acd,bc->bad', Wf, ds)  # (b, 2, k*k)
        q = (cell // Wd + w) * Wp + (cell % Wd + w)
        idx = (q[:, None] + tgt_off[None, :]).clamp(0, Hp * Wp - 1)
        gp.scatter_add_(2, idx[:, None, :].expand(b, 2, k * k), upd)
        flat_board = torch.where(change[:, None] & (torch.arange(H * Wd, device=dev)[None] == cell[:, None]),
                                 new[:, None], flat_board)
    gin = gp.reshape(b, 2, Hp, Wp)[:, :, w:w + H, w:w + Wd].reshape(b, 2, H * Wd)
    gy = gin.gather(1, flat_board[:, None]).squeeze(1)
    stopped = ~((((gy - gin.min(1).values) * freef) > theta) & freef).any(1)
    pred = flat_board.view(b, H, Wd)
    right = ((pred == y) | ~free).all((1, 2))
    return {'moves': moves, 'right': right, 'stopped': stopped, 'pred': pred, 'fields': gin.view(b, 2, H, Wd)}


def gibbs_flop(model, moves, n=SIDE * SIDE):
    """Focused Gibbs: one full current to start, then per move the field change at (2w+1)^2 - 1 cells (4 flop each:
    two answers, a difference of two kernel entries added) and the focus (2n: a margin and a comparison per cell).
    moves: the move count per maze; returns the flop per maze."""
    k = 2 * model.w + 1
    return flop_per_step(model) + moves * (4 * (k * k - 1) + 2 * n)

