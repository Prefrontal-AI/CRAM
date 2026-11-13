"""Maze-Hard: the answer energy with attention-gated links, its pseudolikelihood loss, the answer search and the
diagnostics.

The Maze-Hard model of the paper is `cram_maze_hard_graph.GraphLatentEnergyMU`. It takes from this file the gated-link
terms of the answer energy (`H3Energy.support`), the answer search (`descend`), the diagnostics (`bfs`, `rank_corr`) and
the operation counts (`flop_*`). `H3Energy` is the complete model that these parts were written for;
`experiments/maze_hard_support.py` subclasses it.

Each cell i has the input token x_i (0 wall, 1 open, 2 S, 3 G; given) and two variables:

- z_i >= 0, the hidden (latent) potential, stored as psi_i = -log(z_i) / q. Its energy over a (2 wz + 1)^2 window is
      E_z(z) = sum_i z_i^2 / (2 g[x_i]) - 1/2 sum_i sum_(d != 0) gamma_d z_i z_(i+d) - sum_i c[x_i] z_i,
  with gamma_d = exp(-q w_d), g[x] = exp(-q rho_x) and c[x] = exp(-q beta_x). The Jacobi update of E_z in the log
  coordinate is
      psi_i <- rho_(x_i) - (1/q) log( sum_(d != 0) exp(-q (psi_(i+d) + w_d)) + exp(-q beta_(x_i)) ).
  The update is a contraction for every parameter value (rho >= 0 and w_d >= log(number of offsets) / q), so E_z has
  one fixed point. psi = CAP stands for z = 0.
- nu_i in [0, 1], the answer (1: the cell is on the path).
- The attention of cell i over a (2 wa + 1)^2 window (self included) reads the tokens and psi, never the answers:
      a_id = softmax_d(s_id),   s_id = G_d[x_i, x_(i+d)] - alpha_d sat(psi_(i+d) - psi_i),   sat(t) = DSAT tanh(t / DSAT).

Answer energy, for fixed attention a:

    E(nu; a) = sum_i [nu_i log nu_i + (1 - nu_i) log(1 - nu_i) - b[x_i] nu_i] - sum_i sum_d kappa_d a_id nu_i nu_(i+d)

Symbols: w_d the cost of offset d, rho_x and beta_x the leak and source level of token x, q the inverse temperature of
the latent, G_d a (4, 4) score table per offset, alpha_d the weight of the latent difference, kappa_d the answer
coupling of offset d (d = 0: the self term), b_x the answer bias of token x. Every one of them is learned.

Inference: psi settles to the fixed point of E_z (`H3Energy.settle`); then nu descends E with a held fixed (`descend`).
A rounded state that no single flip lowers is a fixed point of the answer energy; `descend` stops a maze there.
Training: `pl_loss`, the pseudolikelihood (PL) of the labeled answers. With `settle(grad=True)` the loss gradient
reaches the latent's parameters through the adjoint at the fixed point (the relaxation lam <- g + J^T lam, J the
Jacobian of one update at psi*). No settling step is differentiated.

Dimension names: b mazes, k offsets of a window (self first for the attention), h and w grid rows and columns,
t tokens (4).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cram.maze_learned import CAP, NTOK, dev_split, gather, inside, load, offsets, read  # noqa: F401 (re-exported)

DSAT = 8.0  # the saturation of the latent's difference in the scores


class H3Energy(nn.Module):
    """The energy of the module docstring. wz and wa: radius of the latent's and the attention's window; init: standard
    deviation of the initial parameters (drawn from a generator seeded with `seed`); q0, alpha0: initial q and mean
    alpha; param: 'lin' (w_d and rho through softplus, beta as stored) or 'log' (all three through exp);
    source: see `log_source`."""

    def __init__(self, wz=1, wa=1, init=0.1, q0=4.0, alpha0=0.0, seed=0, param='lin', source='token'):
        super().__init__()
        self.wz, self.wa, self.param = wz, wa, param
        self.source = source  # 'token': each token's source level beta_x; 'softmax': one unit of source per maze,
        # shared out over its cells by a softmax of beta_x (an attention over the whole grid: where the source is)
        self.zoffs = offsets(wz)[1:]  # the latent's window, self excluded
        zpairs = sorted({min(d, (-d[0], -d[1])) for d in self.zoffs})
        self.zpair_of = [zpairs.index(min(d, (-d[0], -d[1]))) for d in self.zoffs]
        self.aoffs = offsets(wa)  # the attention's window, self first
        self.K = len(self.aoffs)
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: init * torch.randn(*s, generator=g)
        self.raw_w = nn.Parameter(r(len(zpairs)))
        self.raw_rho = nn.Parameter(r(NTOK))
        self.beta = nn.Parameter(r(NTOK))
        self.log_q = nn.Parameter(torch.tensor(math.log(q0)))
        self.G = nn.Parameter(r(self.K, NTOK, NTOK))
        self.alpha = nn.Parameter(r(self.K) + alpha0)
        self.kappa = nn.Parameter(r(self.K))
        self.b = nn.Parameter(r(NTOK))

    def n_params(self):
        """Number of learned values."""
        return sum(p.numel() for p in self.parameters()) - 1  # alpha's self entry multiplies 0

    # --- the latent ---------------------------------------------------------------------------------------------------

    def zparams(self):
        """(q scalar, w_d (k,) per offset of the latent's window, rho (t,), beta (t,)) from the stored parameters.
        w_d >= log(k) / q by construction, which makes `psi_step` a contraction."""
        q = self.log_q.exp()
        floor = math.log(len(self.zoffs)) / q
        if self.param == 'log':
            wd = floor + self.raw_w.exp()[self.zpair_of]
            return q, wd, self.raw_rho.exp(), self.beta.exp()
        wd = floor + F.softplus(self.raw_w)[self.zpair_of]
        return q, wd, F.softplus(self.raw_rho), self.beta

    def log_source(self, tok, q, beta):
        """(b, h, w): log of each cell's source c_i ('token': c = exp(-q beta_x); 'softmax': c_i = softmax over the
        cells of the maze of beta_(x_i))."""
        if self.source == 'softmax':
            lb = beta[tok]
            return lb - torch.logsumexp(lb.flatten(1), 1)[:, None, None]
        return -q * beta[tok]

    def psi_step(self, psi, tok, zp):
        """One Jacobi update of E_z in the log coordinate. psi (b, h, w), tok (b, h, w), zp = `zparams()`. Returns the
        new psi (b, h, w), at most CAP."""
        q, wd, rho, beta = zp
        nb = gather(psi, self.zoffs, CAP)
        a = -q * (nb + wd[None, :, None, None])
        src = self.log_source(tok, q, beta)
        lse = torch.logsumexp(torch.cat([a, src[:, None]], 1), 1)
        return torch.clamp(rho[tok] - lse / q, max=CAP)

    def psi_weights(self, psi, tok, zp):
        """(b, k, h, w): d psi_i / d psi_(i+d) of one update at psi (the softmax weights of the update over the
        window); 0 for a cell whose new value is CAP."""
        q, wd, rho, beta = zp
        nb = gather(psi, self.zoffs, CAP)
        full = torch.cat([-q * (nb + wd[None, :, None, None]), self.log_source(tok, q, beta)[:, None]], 1)
        p = torch.softmax(full, 1)[:, :-1]
        new = rho[tok] - torch.logsumexp(full, 1) / q
        return torch.where((new < CAP)[:, None], p, torch.zeros_like(p))

    def settle(self, tok, grad=False, tol=1e-5, max_steps=4000, check=8, adjoint_tol=1e-6, psi0=None, adjoint_max=None):
        """psi (b, h, w) at E_z's fixed point, from z = 0 or from psi0 (a start state: the fixed point is unique, so
        only the number of steps changes). The loop checks convergence (largest change <= tol) every `check` steps.
        With grad, the returned psi carries the gradient at the fixed point: a backward hook solves the adjoint
        lam = g + J^T lam by relaxation (to adjoint_tol, or at most adjoint_max iterations when given).
        Returns (psi, steps per maze (b,))."""
        zp = self.zparams()
        B = tok.shape[0]
        steps = torch.zeros(B, dtype=torch.long, device=tok.device)
        with torch.no_grad():
            zd = tuple(t.detach() for t in zp)
            psi = torch.full(tok.shape, CAP, device=tok.device, dtype=self.G.dtype) if psi0 is None else psi0.clone()
            for t in range(1, max_steps + 1):
                new = self.psi_step(psi, tok, zd)
                if t % check == 0 or t == max_steps:
                    moved = (new - psi).abs().flatten(1).amax(1) > tol
                    steps = torch.where(moved, torch.full_like(steps, t), steps)
                    if not moved.any():
                        psi = new
                        break
                psi = new
        steps = torch.clamp(steps + check, max=max_steps)
        if not grad or not torch.is_grad_enabled():
            return psi, steps
        out = self.psi_step(psi, tok, zp)
        if not out.requires_grad:  # the latent's parameters are frozen
            return out, steps
        P = self.psi_weights(psi, tok, tuple(t.detach() for t in zp)).detach()
        offs = self.zoffs

        def adjoint(gr):
            lam = gr
            for _ in range(adjoint_max or max_steps):
                jt = sum(read(P[:, k] * lam, -di, -dj, 0.0) for k, (di, dj) in enumerate(offs))
                new = gr + jt
                if (new - lam).abs().max() < adjoint_tol:
                    return new
                lam = new
            return lam

        out.register_hook(adjoint)
        return out, steps

    # --- the attention and its read ---------------------------------------------------------------------------------

    def attention(self, tok, psi):
        """a (b, k, h, w): each cell's attention over the window; it reads tokens and the latent, never the answers."""
        return torch.softmax(self.att_scores(tok, psi), 1)

    def att_scores(self, tok, psi):
        """s (b, k, h, w): the attention's scores (-inf beyond the grid)."""
        B, H, W = tok.shape
        tn = gather(tok, self.aoffs, 0)  # (b, k, h, w) the neighbour's token (masked beyond the grid below)
        kidx = torch.arange(self.K, device=tok.device)[None, :, None, None]
        s = self.G[kidx, tok[:, None], tn]
        dpsi = DSAT * torch.tanh((gather(psi, self.aoffs, CAP) - psi[:, None]) / DSAT)
        al = torch.cat([torch.zeros_like(self.alpha[:1]), self.alpha[1:]])
        s = s - al[None, :, None, None] * dpsi
        return s.masked_fill(~inside(self.aoffs, H, W, tok.device)[None], -math.inf)

    def support(self, a, nu):
        """The gated-link terms that cell i takes part in, for couplings a (b, k, h, w) and answers nu (b, h, w).
        Returns (out, in, self), each (b, h, w): out = sum_(d != 0) kappa_d a_id nu_(i+d) (links that start at i);
        in = sum_(d != 0) kappa_d a_(i-d, d) nu_(i-d) (links that end at i); self = kappa_0 a_i0."""
        kn = self.kappa[None, :, None, None] * a
        nb = gather(nu, self.aoffs, 0.0)
        out = (kn[:, 1:] * nb[:, 1:]).sum(1)
        inn = sum(read(kn[:, k] * nu, -di, -dj, 0.0) for k, (di, dj) in enumerate(self.aoffs) if k)
        return out, inn, kn[:, 0]

    def field(self, tok, nu, a):
        """(b, h, w): -dE/dnu without the entropy term, the value that `descend` moves each logit toward."""
        out, inn, sf = self.support(a, nu)
        return self.b[tok] + out + inn + 2 * sf * nu

    def flip_delta(self, tok, nu, a):
        """(b, h, w): E(nu with cell i flipped) - E(nu) for a discrete nu (0 or 1 at every cell)."""
        out, inn, sf = self.support(a, nu)
        dl = 1 - 2 * nu
        return -dl * (self.b[tok] + out + inn + sf)

    def energy(self, tok, nu, a):
        """(b,): the answer energy E(nu; a) of the module docstring."""
        eps = 1e-12
        ent = nu * (nu + eps).log() + (1 - nu) * (1 - nu + eps).log()
        out, _, sf = self.support(a, nu)
        return (ent - self.b[tok] * nu - nu * out - sf * nu * nu).flatten(1).sum(1)


# --- loss and search --------------------------------------------------------------------------------------------------

def pl_loss(model, tok, y, a):
    """Pseudolikelihood of the labeled answers y (b, h, w) (0 or 1) for couplings a (b, k, h, w): the mean over every
    cell of -log p(nu_i = y_i | the other cells at the label) = softplus(-(E(flipped) - E(label)))."""
    return F.softplus(-model.flip_delta(tok, y, a)).mean()


def descend(model, tok, a, steps=2048, eta=0.5, margin=3.0, certify_every=16):
    """The answer search: the damped update u <- u + eta (field - u) of the answer logits u (nu = sigmoid(u)) with the
    couplings a (b, k, h, w) held fixed, from every cell on (logit `margin`). Every `certify_every` steps a maze stops
    if its rounded state is a fixed point (no single flip lowers the energy by more than 1e-6).
    Returns (rounded nu (b, h, w), steps used per maze (b,), stopped (b,) bool)."""
    B, H, W = tok.shape
    u = torch.full((B, H, W), margin, device=tok.device, dtype=a.dtype)
    stopped = torch.zeros(B, dtype=torch.bool, device=tok.device)
    used = torch.full((B,), steps, dtype=torch.long, device=tok.device)
    with torch.no_grad():
        for t in range(1, steps + 1):
            new = u + eta * (model.field(tok, torch.sigmoid(u), a) - u)
            u = torch.where(stopped[:, None, None], u, new)
            if certify_every and t % certify_every == 0:
                h = (u > 0).to(a.dtype)
                ok = (model.flip_delta(tok, h, a) >= -1e-6).flatten(1).all(1)
                used = torch.where(ok & ~stopped, torch.full_like(used, t), used)
                stopped |= ok
                if stopped.all():
                    break
    return (u > 0).to(a.dtype), used, stopped


# --- diagnostics (never targets) ---------------------------------------------------------------------------------------

def bfs(tok, source):
    """(b, h, w) float: breadth-first-search (BFS) distance over the four side neighbours through open cells (S and G
    count as open) from the cells where tok == source; inf where unreachable or a wall. For diagnosis only."""
    open_ = tok > 0
    d = torch.where(tok == source, 0.0, math.inf).to(torch.float32)
    while True:
        nb = torch.stack([read(d, di, dj, math.inf) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))]).amin(0) + 1
        new = torch.where(open_, torch.minimum(d, nb), torch.full_like(d, math.inf))
        if torch.equal(new, d):
            return d
        d = new


def rank_corr(psi, dist):
    """Mean over mazes of the Spearman rank correlation of psi (b, h, w) with dist (b, h, w) over the cells where both
    are finite. A float; nan when no maze has three such cells."""
    out = []
    for p, q in zip(psi, dist):
        m = torch.isfinite(q) & (p < CAP / 2)
        if m.sum() < 3:
            continue
        rp = p[m].argsort().argsort().double()
        rq = q[m].argsort().argsort().double()
        out.append(float(torch.corrcoef(torch.stack([rp, rq]))[0, 1]))
    return float(np.mean(out)) if out else float('nan')


# --- flop counts (a multiply-add is 2 flop, each exp, log and elementwise operation 1) ---------------------------------

def flop_psi_step(model, cells=900):
    """One update of the latent over the grid: per cell, k neighbour terms (2 each), k + 1 exponentials, their sum,
    and 4 for the logarithm, the scaling, the diagonal term and the clamp."""
    k = len(model.zoffs)
    return cells * (2 * k + 2 * (k + 1) + 4)


def flop_attention(model, cells=900):
    """The couplings of every cell, computed once per maze after the latent has settled."""
    return cells * model.K * (1 + 3 + 3)  # table read, the latent term, softmax


def flop_field(model, cells=900):
    """One evaluation of the answer field (`field`) over the grid, which is one step of `descend`."""
    return cells * model.K * (3 + 3) + cells * 4  # kappa a nu out and in, the sums, bias and self
