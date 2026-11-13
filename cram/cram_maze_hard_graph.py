"""The Maze-Hard model of the paper: a hidden (latent) layer on the graph of open cells, and an answer energy that
reads it. `GraphLatentEnergyMU` is the model of the row `cram-latent-zmu`; `experiments/maze_hard_latent.py` trains and
scores it with the recipe `experiments/recipes/maze-hard-latent.json`.

The latent lives on the graph of open cells (tokens open, S and G); walls are not nodes. Edges join side neighbours
that are both open (the standard 4-neighbour adjacency, no estimated support). The latent energy is the screened
graph-Laplacian (Gaussian random field) energy

    E_z(z; x) = 1/2 sum_(i~j) (z_i - z_j)^2 + 1/2 sum_i m(x_i) z_i^2 - sum_i c(x_i) z_i,     A = L + diag(m),

with L the Laplacian of the open-cell graph, a mass m(x) > 0 per token (or one shared mass) and a source c(x) >= 0 per
token (or c = 1 at G and 0 elsewhere). A is symmetric and strictly diagonally dominant for every parameter value, so
E_z has one minimum z* = A^-1 c. The model is this linear system. z* spans e^(-8000) within one maze, so the solve runs
in the log coordinate psi = -log z, where the Jacobi update of A z = c is a soft-min Bellman backup (soft value
iteration with step cost log(deg_i + m_i)):

    psi_i <- log(deg_i + m(x_i)) - log( sum_(j~i) e^(-psi_j) + c(x_i) ).

Walls and open cells that no source reaches have z* = 0 (psi = +inf). The update, the adjoint relaxation and the local
gradient term are those of `cram_maze_hard_simple.GatedLatentEnergy` (its `parts` is replaced here), so the training
gradient is -lambda^T d_theta(grad_z E_z)(z*) with A lambda = grad_z l; no relaxation is differentiated.
The recipe of the paper uses one shared mass (mass='shared', m = e^mu), the source at G (source='goal') and the
diagonal 4 + m at every cell in place of deg_i + m (`degree_in_diag = False`), so the latent has one learned value, mu.

Couplings (the readout): a_id = softmax_(d in {0} u D)(beta_d - psi_(i+d)) = e^(beta_d) z*_(i+d) / sum_e e^(beta_e) z*_(i+e),
over the self offset 0 and the four side offsets D; keys with z* = 0 are masked, and a window with z* = 0 everywhere
attends by beta alone. Option coupling='gate' replaces the attention by edge gates g_id = sigmoid(beta_d + psi_i -
psi_(i+d)) on the maze graph's edges (0 on any other pair, no self term); the recipe of the paper uses the attention.

Answer energy of `GraphLatentEnergy`: the one of `maze_h3.H3Energy` (`support`, `field`, `flip_delta`, `energy`, which
read kappa, b and aoffs). Answer energy of `GraphLatentEnergyMU`, with p_i in [0, 1] the answer of cell i,
s_i = (1 - p_i, p_i), walls clamped to p = 0 and S, G clamped to p = 1:

    E(p; a) = sum_i sum_(d in half offsets) s_i^T T_d s_(i+d) - sum_(i free) b^T s_i
              - sum_i sum_d kappa_d a_id p_i p_(i+d) + sum_(i free) [p_i log p_i + (1 - p_i) log(1 - p_i)].

The first line is the Maze-Unique energy (`cram_maze.CRAMMazeFreeEnergy` at window 1: a free 2 x 2 table T_d for each
of the 4 half offsets of the 3 x 3 window and the open-cell bias b (2,)). The second line holds the links gated by the
latent (kappa_d: one coupling per offset, self first) and the entropy of the free cells. `maze_h3.descend` searches
this energy with a held fixed.

Dimension names: b mazes, k offsets of a window, h w grid rows and columns, t tokens (4: wall, open, S, G),
a c answer states (2: off, on).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cram import cram_maze_hard_simple as S
from cram.maze_learned import NTOK, gather, inside

INF = math.inf
SIDES = [(-1, 0), (0, -1)]  # one offset of each side pair: the 4-neighbour adjacency


class GraphLatentEnergy(S.GatedLatentEnergy):
    """mass: 'token' (m per token: open, S, G) or 'shared' (one m); mass_fn: 'exp' (m = e^mu) or 'softplus';
    source: 'token' (c = relu(sg) per token) or 'goal' (c = 1 at G, 0 elsewhere; not a parameter: the scale of c
    shifts psi by a constant, which the readout's softmax cancels)."""

    def __init__(self, mass='token', mass_fn='exp', source='token', mass0=4 * math.e ** 4, source0=1.0,
                 coupling='attention'):
        nn.Module.__init__(self)
        self.zpairs, self.zoffs, self.zpair_of = S.symmetric(SIDES)
        self.aoffs = [(0, 0)] + self.zoffs  # the attention window, self first
        self.K = len(self.aoffs)
        self.mass, self.mass_fn, self.source, self.coupling = mass, mass_fn, source, coupling
        self.adjoint_tol = 1e-6
        self.register_buffer('k', torch.zeros(len(self.zpairs)))  # log edge weight: fixed 0 (unit couplings); sets the dtype
        r0 = math.log(mass0) if mass_fn == 'exp' else mass0 + math.log(-math.expm1(-mass0))  # softplus^-1
        self.mu = nn.Parameter(torch.full((NTOK,) if mass == 'token' else (1,), r0))  # mass pre-activation (wall entry unused)
        if source == 'token':
            self.sg = nn.Parameter(torch.full((NTOK,), float(source0)))  # source pre-activation (wall entry unused)
        self.beta = nn.Parameter(torch.zeros(self.K))  # relative-position bias per offset (self first)
        self.kappa = nn.Parameter(torch.zeros(self.K))  # answer coupling per offset (self first)
        self.b = nn.Parameter(torch.zeros(NTOK))  # answer bias per token

    def latent_params(self):
        """The learned parameters of the latent, in the order that `parts` takes them."""
        return (self.mu, self.sg) if self.source == 'token' else (self.mu,)

    def masses(self, mu=None):
        """The mass m per token (t,), or the shared mass (1,): e^mu or softplus(mu)."""
        mu = self.mu if mu is None else mu
        return mu.exp() if self.mass_fn == 'exp' else F.softplus(mu)

    def parts(self, tok, mu, sg=None, for_grad=False):
        """(log Gamma (b, k, h, w), log a (b, h, w), log c (b, h, w)): -inf where a term is 0 (edges that touch a wall,
        sources off G or relu(sg) = 0); log a = 0 on walls (unused: every term of a wall is -inf)."""
        H, W = tok.shape[1:]
        op = tok > 0
        edge = op[:, None] & gather(op, self.zoffs, False) & inside(self.zoffs, H, W, tok.device)[None]
        lg = torch.zeros(edge.shape, dtype=mu.dtype, device=tok.device).masked_fill(~edge, -INF)
        deg = edge.sum(1).to(mu.dtype)
        if self.mass_fn == 'exp':
            lm = mu[tok] if self.mass == 'token' else mu.expand(tok.shape)
            if getattr(self, 'degree_in_diag', True):
                la = torch.logaddexp(deg.log(), lm)
            else:  # diagonal 4 + m for every cell (no degree), so A = (4 + m) I - Adj stays positive definite
                la = torch.logaddexp(torch.full_like(lm, math.log(4.0)), lm)
        else:
            m = F.softplus(mu)
            la = (deg + (m[tok] if self.mass == 'token' else m.expand(tok.shape))).log()
        la = torch.where(op, la, torch.zeros_like(la))
        if self.source == 'token':
            c = F.relu(sg)
            lc = (c.clamp_min(1e-30).log() if for_grad else c.log())[tok]
        else:
            lc = torch.where(tok == 3, 0.0, -INF).to(mu.dtype)
        lc = lc.masked_fill(~op, -INF)
        return lg, la, lc

    def att_scores(self, tok, psi):
        """s (b, k, h, w): beta_d - psi_(i+d), keys with z* = 0 masked; beta alone where the whole window has z* = 0;
        -inf beyond the grid."""
        H, W = tok.shape[1:]
        ok = inside(self.aoffs, H, W, tok.device)[None]
        nb = gather(psi, self.aoffs, INF)
        live = ok & torch.isfinite(nb)
        anyl = live.any(1, keepdim=True)
        lat = torch.where(live, -torch.where(live, nb, torch.zeros_like(nb)), torch.full_like(nb, -INF))
        lat = torch.where(anyl, lat, torch.zeros_like(lat))
        return (self.beta[None, :, None, None] + lat).masked_fill(~ok, -INF)

    def gates(self, tok, psi):
        """The edge gates g (b, k, h, w): sigmoid(beta_d + psi_i - psi_(i+d)) on the maze graph's edges, 0 elsewhere
        and at the self offset."""
        H, W = tok.shape[1:]
        nb = gather(psi, self.aoffs, INF)
        live = inside(self.aoffs, H, W, tok.device)[None] & torch.isfinite(nb) & torch.isfinite(psi)[:, None]
        live[:, 0] = False
        x = self.beta[None, :, None, None] + torch.where(live, psi[:, None] - nb, torch.zeros_like(nb))
        return torch.where(live, torch.sigmoid(x), torch.zeros_like(x))

    def triton_adjoint_grad(self, psi, tok, r):
        """(grads of latent_params, adjoint steps) by the fused kernel (latent_settle_triton.adjoint_mu_grad), for the
        shared mass with the source at G (the only case it covers; None otherwise, and the PyTorch path runs)."""
        if not (self.mass == 'shared' and self.mass_fn == 'exp' and self.source == 'goal' and psi.dtype == torch.float32):
            return None
        from cram import latent_settle_triton as LT
        with torch.no_grad():
            lg, la, _ = self.parts(tok, self.mu.detach())
            g, steps = LT.adjoint_mu_grad(psi, r, lg, la, float(self.mu.detach()[0]), self.zoffs, tol=self.adjoint_tol)
        return (g.reshape(1).to(self.mu.dtype),), int(steps.max())

    def couplings(self, tok, psi):
        """The coefficients a_id (b, k, h, w) that the answer energy reads: the attention, or the edge gates."""
        return torch.softmax(self.att_scores(tok, psi), 1) if self.coupling == 'attention' else self.gates(tok, psi)


# --- the Maze-Unique answer energy plus the latent-gated links ------------------------------------------------------------

BIG = 1e4  # field and flip cost of a clamped cell (walls off, S and G on): its answer never moves


class GraphLatentEnergyMU(GraphLatentEnergy):
    """The hidden layer of GraphLatentEnergy and an answer energy that contains the Maze-Unique energy
    (cram_maze.CRAMMazeFreeEnergy, window 1: a free 2 x 2 table T_d per half offset of the 3 x 3 window and the open-cell
    bias b (2,), walls clamped off and S, G clamped on, 18 parameters) plus the links gated by the latent,
    - sum_i sum_d kappa_d a_id p_i p_(i+d) (module docstring). With kappa = 0 it is the Maze-Unique energy exactly.
    Each cell's answer p_i in [0, 1] is the two-state answer s_i = (1 - p_i, p_i). The methods keep the H3Energy
    interface (field = -dE/dp without the entropy; flip_delta; energy; pl_loss) that the descent, scoring and counters use.
    In the method arguments, nu (b, h, w) is the answer p and a (b, k, h, w) the couplings.
    Dimension names: b mazes, a c answer states (2), k offsets, h w grid rows and columns."""

    def __init__(self, **kw):
        super().__init__(**kw)
        del self.b
        self.b = nn.Parameter(torch.zeros(2))  # the open-cell bias table of the Maze-Unique energy
        from cram.cram_maze import half_offsets
        self.mu_offsets = half_offsets(1)
        self.T = nn.Parameter(torch.zeros(len(self.mu_offsets), 2, 2))  # free pair tables, as CRAMMazeFreeEnergy

    def mu_kernel(self):
        """W (2, 2, 3, 3): the cross-correlation kernel of the Maze-Unique pair term, T_d at offset d and T_d^T at -d
        (as CRAMMazeFreeEnergy.kernel)."""
        W = self.T.new_zeros(2, 2, 3, 3)
        for t, (dr, dc) in enumerate(self.mu_offsets):
            W[:, :, 1 + dr, 1 + dc] = self.T[t]
            W[:, :, 1 - dr, 1 - dc] = self.T[t].T
        return W

    @staticmethod
    def free(tok):
        """(b, h, w) bool: the cells whose answer is not given."""
        return tok == 1  # open cells other than S and G

    def clamped(self, tok, nu):
        """nu with the givens: walls 0, S and G 1."""
        return torch.where(self.free(tok), nu, (tok >= 2).to(nu.dtype))

    def mu_current(self, tok, nu):
        """dE_MU/ds (b, 2, h, w) at s = (1 - nu, nu) (givens clamped), as CRAMMazeFreeEnergy.current."""
        nu = self.clamped(tok, nu)
        s = torch.stack([1 - nu, nu], 1)
        bias = torch.where(self.free(tok)[:, None], self.b[None, :, None, None], torch.zeros_like(s))
        return F.conv2d(s, self.mu_kernel(), padding=1) - bias

    def field(self, tok, nu, a):
        """(b, h, w): -dE/dp without the entropy at free cells; +BIG at S and G and -BIG at walls, so that the descent
        keeps the givens."""
        nc = self.clamped(tok, nu)
        out, inn, sf = self.support(a, nc)
        g = self.mu_current(tok, nu)
        f = out + inn + 2 * sf * nc - (g[:, 1] - g[:, 0])
        return torch.where(self.free(tok), f, torch.where(tok >= 2, torch.full_like(f, BIG), torch.full_like(f, -BIG)))

    def flip_delta(self, tok, nu, a):
        """(b, h, w): E(p with cell i flipped) - E(p) for a discrete p at free cells; BIG at clamped cells."""
        nc = self.clamped(tok, nu)
        out, inn, sf = self.support(a, nc)
        g = self.mu_current(tok, nu)
        dl = 1 - 2 * nc
        d = -dl * (out + inn + sf) + dl * (g[:, 1] - g[:, 0])
        return torch.where(self.free(tok), d, torch.full_like(d, BIG))

    def energy(self, tok, nu, a):
        """(b,): the answer energy E(p; a) of the module docstring."""
        eps = 1e-12
        nc = self.clamped(tok, nu)
        fr = self.free(tok).to(nu.dtype)
        ent = fr * (nc * (nc + eps).log() + (1 - nc) * (1 - nc + eps).log())
        out, _, sf = self.support(a, nc)
        s = torch.stack([1 - nc, nc], 1)
        pair = 0.5 * (F.conv2d(s, self.mu_kernel(), padding=1) * s).sum(1)
        bias = fr * (s * self.b[None, :, None, None]).sum(1)
        return (ent - nc * out - sf * nc * nc + pair - bias).flatten(1).sum(1)

    def pl_loss(self, tok, y, a):
        """Pseudolikelihood of the free cells' labels (the givens are clamped), as maze_unique.pl_loss."""
        fr = self.free(tok)
        return F.softplus(-self.flip_delta(tok, y, a))[fr].mean()
