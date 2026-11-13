"""The hidden (latent) layer of the Maze-Hard model: a linear layer whose fixed point is found by relaxation, and
the training gradient at that fixed point (the adjoint), computed without differentiating through the relaxation.

The latent is a linear HAMUX layer (HAMUX: the hierarchical associative memory framework; a linear layer has a
quadratic Lagrangian). Its energy is a screened graph Laplacian:
    E_z(z; x) = 1/2 z^T A z - c^T z,   A = diag(a) - Gamma,   a_x = sum_(d in D) gamma_d + m_x,
with gamma_d = e^(k_d) a symmetric kernel over the window D (k_d = k_(-d)), m_x = e^(mu_x) a mass per token and
c_x = e^(s_x) a source per token. A is symmetric and strictly diagonally dominant for every parameter value, so E_z
has one minimum z* = A^-1 c. z* spans e^(-2000) within one maze, so the state is stored as psi = -log z (nats), and the
Jacobi update of A z = c is a log-sum-exp (LSE) over the window:
    psi_i <- log a_(x_i) - LSE( LSE_d(k_d - psi_(i+d)), s_(x_i) ).
The adjoint of a loss l with r = dl/dpsi at psi* is the relaxation v <- r + P^T v, with the forward shares
P_id = exp(k_d - psi_(i+d) - log a_(x_i) + psi_i), and the parameter gradient is the local term
    dl/dtheta = sum_i v_i [ d log a_(x_i) - sum_d P_id d k_d - S_i d s_(x_i) ],   S_i = exp(s_(x_i) - log a_(x_i) + psi_i),
which equals -lambda^T d_theta(grad_z E_z)(z*) with A lambda = grad_z l and v = -a z lambda. No relaxation is
differentiated: the forward relaxation runs without a graph, and `HamuxSimple` / `HamuxGated` return the gradient above
in their backward pass.

Classes. The Maze-Hard model of the paper is `cram_maze_hard_graph.GraphLatentEnergyMU`, a subclass of
`GatedLatentEnergy`: it uses the relaxation (`settle`), the shares (`gshares`), the adjoint (`adjoint`) and the local
term (`local_term`) defined here and replaces the layer's coefficients (`parts`) and the readout. `SimpleLatentEnergy`
(the equations above), `GatedLatentEnergy` and `CostLatentEnergy` are three parameterizations of the coefficients a,
Gamma and c with the same relaxation and adjoint. `tests/test_cram_maze_hard_simple.py` checks the fixed point and the
adjoint gradient of each of them against a dense float64 solve (`dense_A`, `dense_A_gated`, `dense_A_cost`).

The readout of these three classes is the attention a_id = softmax_d(B_d[x_i, x_(i+d)] + alpha (psi_i - psi_(i+d)))
over {0} and the pointer offsets. The answer energy (`support`, `field`, `flip_delta`, `energy`) is the one of
`maze_h3.H3Energy`, so `maze_h3.descend` runs on these models unchanged.
Options of the readout: readout='tanh8' saturates the latent difference as 8 tanh(./8); alpha_mode='per_offset' learns
one alpha per offset instead of one scalar.

Dimension names: b mazes, k offsets of a window, h w grid rows and columns, t tokens (4: wall, open, S, G),
n cells of one maze (h w) in the dense reference.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cram import maze_h3 as m3
from cram.maze_learned import NTOK, gather, inside, read

INF = math.inf


def symmetric(pairs):
    """Both signs of each offset pair, sorted, and the pair index of each offset."""
    pairs = sorted({min(tuple(d), (-d[0], -d[1])) for d in pairs})
    offs = sorted([d for p in pairs for d in (p, (-p[0], -p[1]))])
    return pairs, offs, [pairs.index(min(d, (-d[0], -d[1]))) for d in offs]


class SimpleLatentEnergy(nn.Module):
    """The latent and readout of the module docstring. latent_pairs: one offset (di, dj) of each pair {d, -d} of the
    latent's window D; pointer_pairs: the same for the attention's window; init: standard deviation of the initial
    k, mu, s, kappa, b (0: zero start), drawn from a generator seeded with `seed`; alpha0: initial alpha."""

    def __init__(self, latent_pairs, pointer_pairs, readout='linear', alpha_mode='scalar', alpha0=1.0, init=0.0,
                 seed=0):
        super().__init__()
        self.zpairs, self.zoffs, self.zpair_of = symmetric(latent_pairs)
        _, poffs, _ = symmetric(pointer_pairs)
        self.aoffs = [(0, 0)] + poffs  # the attention window, self first
        self.K = len(self.aoffs)
        self.readout, self.alpha_mode = readout, alpha_mode
        self.adjoint_tol = 1e-6  # relative stopping tolerance of the adjoint relaxation
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: init * torch.randn(*s, generator=g)
        self.k = nn.Parameter(r(len(self.zpairs)))  # log gamma per offset pair
        self.mu = nn.Parameter(r(NTOK))  # log mass per token
        self.s = nn.Parameter(r(NTOK))  # log source per token
        self.B = nn.Parameter(torch.zeros(self.K, NTOK, NTOK))  # score table per offset (query token, key token)
        self.alpha = nn.Parameter(torch.full((self.K - 1,) if alpha_mode == 'per_offset' else (1,), float(alpha0)))
        self.kappa = nn.Parameter(r(self.K))  # answer coupling per offset (self first)
        self.b = nn.Parameter(r(NTOK))  # answer bias per token

    # --- the latent ---------------------------------------------------------------------------------------------------

    def log_a(self, k=None, mu=None):
        """(t,): log of the diagonal of A per token, LSE of every kernel entry and the token's log mass."""
        k = self.k if k is None else k
        mu = self.mu if mu is None else mu
        kd = k[self.zpair_of]
        return torch.logsumexp(torch.cat([kd[None].expand(NTOK, -1), mu[:, None]], 1), 1)

    def psi_step(self, psi, tok, k, la, s):
        """One Jacobi update. psi (b, h, w), tok (b, h, w), k (pairs,), la = `log_a` (t,), s (t,). Returns the new psi."""
        kd = k[self.zpair_of]
        nb = gather(psi, self.zoffs, INF)  # (b, k, h, w); z = 0 beyond the grid
        terms = torch.cat([kd[None, :, None, None] - nb, s[tok][:, None]], 1)
        return la[tok] - torch.logsumexp(terms, 1)

    def settle(self, tok, grad=False, tol=1e-6, max_steps=4000, check=8):
        """psi* from z = 0 (psi = +inf) by the Jacobi update; with grad, psi* carries the adjoint gradient (HamuxSimple).
        Returns (psi (b, h, w), steps per maze)."""
        B = tok.shape[0]
        steps = torch.full((B,), max_steps, dtype=torch.long, device=tok.device)
        with torch.no_grad():
            k, s = self.k.detach(), self.s.detach()
            la = self.log_a(k, self.mu.detach())
            psi = torch.full(tok.shape, INF, device=tok.device, dtype=self.k.dtype)
            done = torch.zeros(B, dtype=torch.bool, device=tok.device)
            for t in range(1, max_steps + 1):
                new = self.psi_step(psi, tok, k, la, s)
                if t % check == 0:
                    ok = ((new - psi).abs() <= tol * (1 + new.abs())).flatten(1).all(1)
                    steps = torch.where(ok & ~done, torch.full_like(steps, t), steps)
                    done |= ok
                    if done.all():
                        psi = new
                        break
                psi = new
        if grad and torch.is_grad_enabled():
            psi = HamuxSimple.apply(psi, self, tok, self.k, self.mu, self.s)
        return psi, steps

    def shares(self, psi, tok, k, la, s):
        """(P (b, k, h, w), S (b, h, w)): the forward update's shares at psi; P is 0 beyond the grid."""
        H, W = tok.shape[1:]
        kd = k[self.zpair_of]
        nb = gather(psi, self.zoffs, 0.0)
        base = psi - la[tok]
        P = torch.exp(kd[None, :, None, None] - nb + base[:, None])
        P = torch.where(inside(self.zoffs, H, W, tok.device)[None], P, torch.zeros_like(P))
        return P, torch.exp(s[tok] + base)

    def adjoint(self, psi, tok, r, tol=None, max_steps=4000, check=8):
        """v (b, h, w) solving v = r + P^T v by relaxation (the transpose of the forward linearization), for the loss
        gradient r = dl/dpsi (b, h, w) at psi. The loop stops at the first check (every `check` steps) with largest change
        <= tol (1 + largest |v|). Returns (v, steps)."""
        tol = self.adjoint_tol if tol is None else tol
        with torch.no_grad():
            P, _ = self.shares(psi, tok, self.k, self.log_a(), self.s)
            v = r
            for t in range(1, max_steps + 1):
                new = r + sum(read(P[:, j] * v, -di, -dj, 0.0) for j, (di, dj) in enumerate(self.zoffs))
                if t % check == 0 and (new - v).abs().max() <= tol * (1 + new.abs().max()):
                    return new, t
                v = new
        return v, max_steps

    def local_term(self, psi, tok, v, k, mu, s):
        """Phi(theta) whose gradient is the parameter gradient at the fixed point (module docstring); psi, v and the
        shares are held fixed."""
        with torch.no_grad():
            P, S = self.shares(psi, tok, self.k, self.log_a(), self.s)
        la = self.log_a(k, mu)
        kd = k[self.zpair_of]
        inner = la[tok] - (P * kd[None, :, None, None]).sum(1) - S * s[tok]
        return (v * inner).sum()

    # --- the readout ----------------------------------------------------------------------------------------------------

    def att_scores(self, tok, psi):
        """s (b, k, h, w): B_d[x_i, x_(i+d)] + alpha (psi_i - psi_(i+d)); -inf beyond the grid."""
        H, W = tok.shape[1:]
        tn = gather(tok, self.aoffs, 0)
        kidx = torch.arange(self.K, device=tok.device)[:, None, None]
        sc = self.B[kidx, tok[:, None], tn]
        ok = inside(self.aoffs, H, W, tok.device)[None]
        nb = gather(psi, self.aoffs, 0.0)
        diff = torch.where(ok, psi[:, None] - nb, torch.zeros_like(nb))
        if self.readout == 'tanh8':
            diff = 8.0 * torch.tanh(diff / 8.0)
        al = self.alpha if self.alpha_mode == 'per_offset' else self.alpha.expand(self.K - 1)
        al = torch.cat([torch.zeros_like(al[:1]), al])
        sc = sc + al[None, :, None, None] * diff
        return sc.masked_fill(~ok, -INF)

    def attention(self, tok, psi):
        """a (b, k, h, w): softmax of `att_scores` over the offsets."""
        return torch.softmax(self.att_scores(tok, psi), 1)

    # the answer energy of maze_h3.H3Energy (its methods read kappa, b and aoffs of this model)
    support = m3.H3Energy.support
    field = m3.H3Energy.field
    flip_delta = m3.H3Energy.flip_delta
    energy = m3.H3Energy.energy


class HamuxSimple(torch.autograd.Function):
    """psi* in the forward pass; in the backward pass the adjoint relaxation and the local gradient term."""

    @staticmethod
    def forward(ctx, psi, model, tok, k, mu, s):
        ctx.model, ctx.tok = model, tok
        ctx.save_for_backward(psi)
        return psi.clone()

    @staticmethod
    def backward(ctx, r):
        psi, = ctx.saved_tensors
        model, tok = ctx.model, ctx.tok
        v, steps = model.adjoint(psi, tok, r)
        model.adjoint_steps = steps
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(True) for t in (model.k, model.mu, model.s)]
            phi = model.local_term(psi, tok, v, *leaves)
            grads = torch.autograd.grad(phi, leaves)
        return (None, None, None) + tuple(grads)


# --- losses -------------------------------------------------------------------------------------------------------------

def pointer_loss(model, tok, y, succ, psi, hinge=0.0, hard_fraction=1.0, none_prior=0.0, margin=0.0, cells='path'):
    """Cross-entropy of each labeled path cell's attention at its successor. succ (b, h, w): the index, in the
    attention's offsets, of the next path cell (0, the self offset, for G and off-path cells). Returns (pointer loss,
    scores (b, k, h, w)). Options, all off by default: margin > 0 subtracts a margin from the target score
    (additive-margin softmax); hinge > 0 adds relu(hinge - (target score - best other score)); hard_fraction < 1
    averages only that fraction of the cells with the largest loss; none_prior > 0 adds the cross-entropy of off-path
    cells against offset 0 with that weight; cells='open' scores every non-wall cell, off-path cells against offset 0."""
    s = model.att_scores(tok, psi)
    sm = s - margin * F.one_hot(succ, s.shape[1]).permute(0, 3, 1, 2).to(s.dtype) if margin > 0 else s
    lp = torch.log_softmax(sm, 1)
    m = (tok > 0) if cells == 'open' else (y > 0.5) & (tok > 0)
    per = -lp.gather(1, succ[:, None]).squeeze(1)
    if hinge > 0:
        st = s.gather(1, succ[:, None]).squeeze(1)
        other = s.scatter(1, succ[:, None], -INF).amax(1)
        per = per + F.relu(hinge - (st - other))
    sel = per[m]
    lu = sel.topk(max(1, math.ceil(sel.numel() * hard_fraction))).values.mean() if hard_fraction < 1 else sel.mean()
    if none_prior > 0:
        moff = (y < 0.5) & (tok > 0)
        lu = lu + none_prior * (-lp[:, 0] * moff).sum() / moff.sum()
    return lu, s


def pl_loss(model, tok, y, s, detach=True):
    """Pseudolikelihood (`maze_h3.pl_loss`) of the labeled answers y (b, h, w) for attention scores s (b, k, h, w),
    with the attention held fixed; detach=False lets its gradient reach the attention and, through the adjoint, the
    latent."""
    a = torch.softmax(s, 1)
    return m3.pl_loss(model, tok, y, a.detach() if detach else a)


# --- the dense z-space reference (tests only; z* must be representable in float64) --------------------------------------

def dense_A(model, tok_one):
    """(A (n, n), c (n,)) of one maze in float64, from the model's parameters (differentiable)."""
    H, W = tok_one.shape
    N = H * W
    k, mu, s = model.k.double(), model.mu.double(), model.s.double()
    t = tok_one.flatten()
    idx = torch.arange(N, device=tok_one.device).view(H, W)
    ok = inside(model.zoffs, H, W, tok_one.device)
    A = torch.diag(model.log_a(k, mu)[t].exp())
    for j, (di, dj) in enumerate(model.zoffs):
        m = ok[j]
        rows, cols = idx[m], read(idx, di, dj, -1)[m]
        A = A.index_put((rows, cols), -k[model.zpair_of[j]].exp().expand(len(rows)), accumulate=True)
    return A, s[t].exp()


# --- the gated latent: exact zeros from ReLU gates and sources ----------------------------------------------------------

class GatedLatentEnergy(SimpleLatentEnergy):
    """The Laplacian latent with token-gated edges and a ReLU source:
        E_z = 1/2 sum_(pairs i~j) Gamma_ij (z_i - z_j)^2 + 1/2 sum_i m_(x_i) z_i^2 - sum_i c_(x_i) z_i,
        Gamma_ij = gamma_(j-i) e_(x_i) e_(x_j),  gamma_d = e^(k_d),  e_x = relu(th_x),  c_x = relu(sg_x),  m_x = e^(mu_x).
    A = diag(a) - Gamma with a_i = sum_j Gamma_ij + m_(x_i) (no edge beyond the grid): symmetric and strictly diagonally
    dominant for every parameter value. A token whose gate is 0 is cut from the graph exactly; a token whose source is
    0 adds none. Cells with z* = 0 (psi = +inf) are masked keys of the readout, whose scores are
    B_d[x_i, x_(i+d)] - alpha psi_(i+d) over {0} and the pointer offsets (equal to the linear readout by the softmax's
    shift invariance); a cell whose whole window has z = 0 attends by B alone."""

    def __init__(self, latent_pairs, pointer_pairs, mu0=0.0, gate0=1.0, source0=1.0, mass='token', **kw):
        super().__init__(latent_pairs, pointer_pairs, **kw)
        del self.s
        self.th = nn.Parameter(torch.full((NTOK,), float(gate0)))  # gate pre-activation per token
        self.sg = nn.Parameter(torch.full((NTOK,), float(source0)))  # source pre-activation per token
        self.mu = nn.Parameter(torch.full((NTOK,) if mass == 'token' else (1,), float(mu0)))  # log mass
        self.mass = mass

    def latent_params(self):
        """The parameters of the latent, in the order that `parts` takes them."""
        return (self.k, self.th, self.sg, self.mu)

    def parts(self, tok, k, th, sg, mu, for_grad=False):
        """(log Gamma (b, k, h, w), log a (b, h, w), log c (b, h, w)); -inf where a term is exactly 0. With for_grad,
        log 0 is replaced by a finite floor so that autograd sees no inf (those terms carry zero shares)."""
        H, W = tok.shape[1:]
        e, c = F.relu(th), F.relu(sg)
        le = e.clamp_min(1e-30).log() if for_grad else e.log()
        lc = c.clamp_min(1e-30).log() if for_grad else c.log()
        lei = le[tok]
        lg = k[self.zpair_of][None, :, None, None] + lei[:, None] + gather(lei, self.zoffs, -INF)
        lg = lg.masked_fill(~inside(self.zoffs, H, W, tok.device)[None], -INF)
        lm = mu[tok] if self.mass == 'token' else mu.expand_as(tok.float())
        la = torch.logsumexp(torch.cat([lg, lm[:, None]], 1), 1)
        return lg, la, lc[tok]

    def settle(self, tok, grad=False, tol=1e-6, max_steps=4000, check=8, psi0=None):
        """psi* (b, h, w) by the Jacobi update, checked every `check` steps (|new - psi| <= tol (1 + |new|) at every
        cell), at most `max_steps`. psi0: a start state (b, h, w), for example each maze's settled state from the
        previous update; None starts from psi = +inf (z = 0). The update is the same contraction from any start.
        With grad, psi* carries the adjoint gradient (HamuxGated). Returns (psi, steps per maze (b,))."""
        B = tok.shape[0]
        steps = torch.full((B,), max_steps, dtype=torch.long, device=tok.device)
        if (getattr(self, 'triton_settle', False) and psi0 is None and tok.is_cuda and self.k.dtype == torch.float32
                and not getattr(self, 'zero_temperature', False)):  # the fused kernel (latent_settle_triton.py)
            from cram import latent_settle_triton as LT
            with torch.no_grad():
                lg, la, lc = self.parts(tok, *(p.detach() for p in self.latent_params()))
                psi, steps = LT.settle(lg, la, lc, self.zoffs, tol=tol, max_steps=max_steps, check=check)
            if grad and torch.is_grad_enabled():
                psi = HamuxGated.apply(psi, self, tok, *self.latent_params())
            return psi, steps
        with torch.no_grad():
            lg, la, lc = self.parts(tok, *(p.detach() for p in self.latent_params()))
            psi = (torch.full(tok.shape, INF, device=tok.device, dtype=self.k.dtype) if psi0 is None
                   else psi0.to(self.k.dtype).clone())
            done = torch.zeros(B, dtype=torch.bool, device=tok.device)
            terms_max = getattr(self, 'zero_temperature', False)  # the latent's zero-temperature limit: min, not soft-min
            for t in range(1, max_steps + 1):
                nb = gather(psi, self.zoffs, INF)
                terms = torch.cat([lg - nb, lc[:, None]], 1)
                new = la - (terms.amax(1) if terms_max else torch.logsumexp(terms, 1))
                if t % check == 0:
                    same = (new == psi) | ((new - psi).abs() <= tol * (1 + new.abs()))
                    ok = same.flatten(1).all(1)
                    steps = torch.where(ok & ~done, torch.full_like(steps, t), steps)
                    done |= ok
                    if done.all():
                        psi = new
                        break
                psi = new
        if grad and torch.is_grad_enabled():
            assert not getattr(self, 'zero_temperature', False), 'training runs at temperature 1'
            psi = HamuxGated.apply(psi, self, tok, *self.latent_params())
        return psi, steps

    def gshares(self, psi, tok):
        """(P (b, k, h, w), S (b, h, w)) at psi from the current parameters; 0 where a state or a term is 0."""
        with torch.no_grad():
            lg, la, lc = self.parts(tok, *(p.detach() for p in self.latent_params()))
            nb = gather(psi, self.zoffs, INF)
            fi = torch.isfinite(psi)
            lp = lg - nb - la[:, None] + psi[:, None]
            P = torch.where(fi[:, None] & torch.isfinite(nb) & torch.isfinite(lg), lp, torch.full_like(lp, -INF)).exp()
            ls = lc - la + psi
            S = torch.where(fi & torch.isfinite(lc), ls, torch.full_like(ls, -INF)).exp()
        return P, S

    def adjoint(self, psi, tok, r, tol=None, max_steps=4000, check=8):
        """As SimpleLatentEnergy.adjoint, with the shares of `gshares`; r is set to 0 where psi = +inf (z = 0)."""
        tol = self.adjoint_tol if tol is None else tol
        with torch.no_grad():
            P, _ = self.gshares(psi, tok)
            r = torch.where(torch.isfinite(psi), r, torch.zeros_like(r))
            v = r
            for t in range(1, max_steps + 1):
                new = r + sum(read(P[:, j] * v, -di, -dj, 0.0) for j, (di, dj) in enumerate(self.zoffs))
                if t % check == 0 and (new - v).abs().max() <= tol * (1 + new.abs().max()):
                    return new, t
                v = new
        return v, max_steps

    def local_term(self, psi, tok, v, *theta):
        """Sum_i v_i [log a_i - sum_d P_id log Gamma_id - S_i log c_i] with v and the shares held fixed."""
        P, S = self.gshares(psi, tok)
        lg, la, lc = self.parts(tok, *theta, for_grad=True)
        lg = torch.where(P > 0, lg, torch.zeros_like(lg))
        lc = torch.where(S > 0, lc, torch.zeros_like(lc))
        fi = torch.isfinite(psi)
        inner = torch.where(fi, la - (P * lg).sum(1) - S * lc, torch.zeros_like(la))
        return (v * inner).sum()

    def att_scores(self, tok, psi):
        """s (b, k, h, w): B_d[x_i, x_(i+d)] - alpha psi_(i+d), keys with z = 0 masked (class docstring); -inf beyond
        the grid."""
        H, W = tok.shape[1:]
        tn = gather(tok, self.aoffs, 0)
        kidx = torch.arange(self.K, device=tok.device)[:, None, None]
        sc = self.B[kidx, tok[:, None], tn]
        ok = inside(self.aoffs, H, W, tok.device)[None]
        nb = gather(psi, self.aoffs, INF)
        live = ok & torch.isfinite(nb)
        anyl = live.any(1, keepdim=True)
        lat = torch.where(live, -self.alpha * torch.where(live, nb, torch.zeros_like(nb)), torch.full_like(nb, -INF))
        lat = torch.where(anyl, lat, torch.zeros_like(lat))  # a window with z = 0 everywhere: B alone
        return (sc + lat).masked_fill(~ok, -INF)


class HamuxGated(torch.autograd.Function):
    """As HamuxSimple for GatedLatentEnergy and its subclasses: psi* in the forward pass; in the backward pass the
    adjoint relaxation and the local gradient term, by the fused Triton kernel when the model provides one."""

    @staticmethod
    def forward(ctx, psi, model, tok, *theta):
        ctx.model, ctx.tok = model, tok
        ctx.save_for_backward(psi)
        return psi.clone()

    @staticmethod
    def backward(ctx, r):
        psi, = ctx.saved_tensors
        model, tok = ctx.model, ctx.tok
        fast = getattr(model, 'triton_adjoint_grad', None)
        if fast is not None and getattr(model, 'triton_settle', False) and psi.is_cuda:  # fused kernel
            out = fast(psi, tok, r)
            if out is not None:
                grads, model.adjoint_steps = out
                return (None, None, None) + tuple(grads)
        v, steps = model.adjoint(psi, tok, r)
        model.adjoint_steps = steps
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(True) for t in model.latent_params()]
            grads = torch.autograd.grad(model.local_term(psi, tok, v, *leaves), leaves)
        return (None, None, None) + tuple(grads)


def dense_A_gated(model, tok_one):
    """(A (n, n), c (n,)) of one maze in float64 for the gated latent (differentiable; tests only)."""
    H, W = tok_one.shape
    N = H * W
    k, th, sg, mu = (p.double() for p in model.latent_params())
    e, c = F.relu(th), F.relu(sg)
    t = tok_one.flatten()
    idx = torch.arange(N, device=tok_one.device).view(H, W)
    ok = inside(model.zoffs, H, W, tok_one.device)
    m = mu[t] if model.mass == 'token' else mu.expand(N)
    A = torch.diag(m.exp())
    for j, (di, dj) in enumerate(model.zoffs):
        sel = ok[j]
        rows, cols = idx[sel], read(idx, di, dj, -1)[sel]
        w = k[model.zpair_of[j]].exp() * e[t[rows]] * e[t[cols]]
        A = A.index_put((rows, cols), -w, accumulate=True).index_put((rows, rows), w, accumulate=True)
    return A, c[t]


# --- the cost latent: diagonal = row sum times e^(delta) ------------------------------------------------------------------

class CostLatentEnergy(GatedLatentEnergy):
    """E_z = 1/2 z^T A z - c^T z with A = diag(a) - Gamma, Gamma_ij = gamma_(j-i) = e^(k_d) (a symmetric kernel, no edge
    beyond the grid), a_x = (sum_(d in D) gamma_d) e^(delta_x) and a per-token excess cost delta_x > 0 (nats per step),
    delta_x = e^(r_x) (delta='exp') or softplus(r_x) (delta='softplus'). A is strictly diagonally dominant for every
    parameter value (e^delta > 1) and does not change when every gamma is scaled, so the coupling scale cannot drift.
    One step costs log(sum_d gamma_d / gamma_d) + delta_x nats. Source c_x = relu(sg_x) (source='relu') or e^(sg_x)
    (source='exp'). The update, adjoint, local term and readout are those of GatedLatentEnergy."""

    def __init__(self, latent_pairs, pointer_pairs, delta0=4.0, delta='exp', source='relu', source0=1.0, **kw):
        super().__init__(latent_pairs, pointer_pairs, source0=source0, **kw)
        del self.th, self.mu
        self.delta_mode, self.source_mode = delta, source
        r0 = math.log(delta0) if delta == 'exp' else math.log(math.expm1(delta0))
        self.r = nn.Parameter(torch.full((NTOK,), r0))  # excess cost pre-activation per token

    def latent_params(self):
        """The parameters of the latent, in the order that `parts` takes them."""
        return (self.k, self.r, self.sg)

    def delta(self, r=None):
        """(t,): the excess cost per token, e^r or softplus(r)."""
        r = self.r if r is None else r
        return r.exp() if self.delta_mode == 'exp' else F.softplus(r)

    def parts(self, tok, k, r, sg, for_grad=False):
        """(log Gamma (b, k, h, w), log a (b, h, w), log c (b, h, w)), as GatedLatentEnergy.parts."""
        H, W = tok.shape[1:]
        kd = k[self.zpair_of]
        lg = kd[None, :, None, None].expand(tok.shape[0], -1, H, W)
        lg = lg.masked_fill(~inside(self.zoffs, H, W, tok.device)[None], -INF)
        la = (torch.logsumexp(kd, 0) + self.delta(r))[tok]
        if self.source_mode == 'exp':
            lc = sg[tok]
        else:
            c = F.relu(sg)
            lc = (c.clamp_min(1e-30).log() if for_grad else c.log())[tok]
        return lg, la, lc


def dense_A_cost(model, tok_one):
    """(A (n, n), c (n,)) of one maze in float64 for the cost latent (differentiable; tests only)."""
    H, W = tok_one.shape
    N = H * W
    k, r, sg = (p.double() for p in model.latent_params())
    t = tok_one.flatten()
    idx = torch.arange(N, device=tok_one.device).view(H, W)
    ok = inside(model.zoffs, H, W, tok_one.device)
    kd = k[model.zpair_of]
    A = torch.diag((torch.logsumexp(kd, 0) + model.delta(r))[t].exp())
    for j, (di, dj) in enumerate(model.zoffs):
        sel = ok[j]
        rows, cols = idx[sel], read(idx, di, dj, -1)[sel]
        A = A.index_put((rows, cols), -kd[j].exp().expand(len(rows)), accumulate=True)
    c = sg[t].exp() if model.source_mode == 'exp' else F.relu(sg)[t]
    return A, c
