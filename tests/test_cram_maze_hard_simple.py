"""The simple Maze-Hard latent (cram/cram_maze_hard_simple.py): fixed point and adjoint gradient against dense
float64 z-space solves.

Ways it can fail, each checked below: the log-space update settles to a point other than z* = A^-1 c (a wrong diagonal
or boundary term); the adjoint uses P in place of P^T; the local term drops the source share S or the kernel share P;
the gradient has the wrong sign or scale.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from cram import cram_maze_hard_simple as S  # noqa: E402


def mazes(n=3, h=7, w=9, seed=0):
    g = np.random.default_rng(seed)
    tok = (g.random((n, h, w)) < 0.7).astype(np.int64)  # open 1, wall 0
    for b in range(n):
        tok[b, 0, 0], tok[b, h - 1, w - 1] = 2, 3
    return torch.as_tensor(tok)


def model(seed=0, pairs=((0, -1), (-1, 0))):
    torch.manual_seed(seed)
    m = S.SimpleLatentEnergy(list(pairs), [(0, -1), (-1, 0)]).double()
    with torch.no_grad():
        m.k.copy_(torch.randn(m.k.shape) * 0.5)
        m.mu.copy_(torch.tensor([3.0, 0.5, 0.8, 0.2]) + 0.2 * torch.randn(4))
        m.s.copy_(torch.tensor([-4.0, -3.0, -2.0, 2.0]) + 0.2 * torch.randn(4))
    return m


def dense_psi(m, tok):
    out = []
    for b in range(tok.shape[0]):
        A, c = S.dense_A(m, tok[b])
        out.append(-torch.linalg.solve(A, c).log().view(tok.shape[1:]))
    return torch.stack(out)


def test_fixed_point_matches_dense_solve():
    for pairs in (((0, -1), (-1, 0)), ((0, -1), (-1, 0), (-1, -1), (-1, 1))):
        m, tok = model(pairs=pairs), mazes()
        psi, _ = m.settle(tok, tol=1e-12)
        assert torch.allclose(psi, dense_psi(m, tok).detach(), atol=1e-9, rtol=0)


def adjoint_grads(m, tok, weights):
    m.zero_grad()
    m.adjoint_tol = 1e-13
    psi, _ = m.settle(tok, grad=True, tol=1e-13)
    (weights * torch.sin(psi)).sum().backward()
    return [p.grad.clone() for p in (m.k, m.mu, m.s)]


def dense_grads(m, tok, weights):
    m.zero_grad()
    (weights * torch.sin(dense_psi(m, tok))).sum().backward()
    return [p.grad.clone() for p in (m.k, m.mu, m.s)]


def test_adjoint_gradient_matches_dense():
    for pairs in (((0, -1), (-1, 0)), ((0, -1), (-1, 0), (-1, -1), (-1, 1))):
        m, tok = model(1, pairs), mazes(seed=1)
        wts = torch.randn(tok.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        ga, gd = adjoint_grads(m, tok, wts), dense_grads(m, tok, wts)
        for a, d in zip(ga, gd):
            assert torch.allclose(a, d, rtol=1e-6, atol=1e-9), (a, d)


def test_planted_bugs_fail(monkeypatch):
    """Each planted bug makes the adjoint gradient differ from the dense one."""
    m, tok = model(2), mazes(seed=2)
    wts = torch.randn(tok.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    gd = dense_grads(m, tok, wts)
    from cram.maze_learned import read
    orig_shares = S.SimpleLatentEnergy.shares

    def adjoint_P_not_PT(self, psi, tok, r, tol=None, max_steps=4000, check=8):
        P, _ = orig_shares(self, psi, tok, self.k, self.log_a(), self.s)
        v = r
        for _ in range(2000):
            v = r + sum(P[:, j] * read(v, di, dj, 0.0) for j, (di, dj) in enumerate(self.zoffs))
        return v, 2000

    def no_source_share(self, psi, tok, k, la, s):
        P, Sh = orig_shares(self, psi, tok, k, la, s)
        return P, torch.zeros_like(Sh)

    for name, bug in (('adjoint', adjoint_P_not_PT), ('shares', no_source_share)):
        with monkeypatch.context() as mp:
            mp.setattr(S.SimpleLatentEnergy, name, bug)
            ga = adjoint_grads(m, tok, wts)
        assert not all(torch.allclose(a, d, rtol=1e-4, atol=1e-7) for a, d in zip(ga, gd)), name


# --- the gated latent: exact zeros (dead gates and sources), z* = 0 cells ------------------------------------------------

def gated(seed=0, mass='token'):
    torch.manual_seed(seed)
    m = S.GatedLatentEnergy([(0, -1), (-1, 0)], [(0, -1), (-1, 0)], mass=mass).double()
    with torch.no_grad():
        m.k.copy_(torch.randn(2) * 0.5)
        m.th.copy_(torch.tensor([-0.3, 1.2, 0.7, 0.9]))  # walls cut from the graph
        m.sg.copy_(torch.tensor([-1.0, -0.2, 0.4, 2.0]))  # sources only at S and G
        m.mu.copy_((torch.tensor([0.5, 1.0, 1.5, 0.3]) if mass == 'token' else torch.tensor([0.8])) + 0.1 * torch.randn(m.mu.shape))
    return m


def dense_gated(m, tok, mask):
    out = []
    for b in range(tok.shape[0]):
        A, c = S.dense_A_gated(m, tok[b])
        z = torch.linalg.solve(A, c)
        out.append(-z[mask[b].flatten()].log())
    return out


def test_gated_fixed_point_and_zero_cells():
    for mass in ('token', 'scalar'):
        m, tok = gated(mass=mass), mazes(seed=5)
        psi, _ = m.settle(tok, tol=1e-12)
        fin = torch.isfinite(psi)
        assert (~fin).any() and fin.any()
        for b in range(tok.shape[0]):
            A, c = S.dense_A_gated(m, tok[b])
            z = torch.linalg.solve(A, c).detach().view(tok.shape[1:])
            assert torch.equal(z > 1e-300, fin[b])
            assert torch.allclose(psi[b][fin[b]], -z[fin[b]].log(), atol=1e-9, rtol=0)


def test_gated_adjoint_matches_dense():
    for mass in ('token', 'scalar'):
        m, tok = gated(1, mass), mazes(seed=6)
        wts = torch.randn(tok.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
        m.zero_grad()
        m.adjoint_tol = 1e-13
        psi, _ = m.settle(tok, grad=True, tol=1e-13)
        fin = torch.isfinite(psi.detach())
        (wts[fin] * torch.sin(psi[fin])).sum().backward()
        ga = [p.grad.clone() for p in m.latent_params()]
        m.zero_grad()
        sum((w[f] * torch.sin(p)).sum() for w, f, p in zip(wts, fin, dense_gated(m, tok, fin))).backward()
        gd = [p.grad.clone() for p in m.latent_params()]
        for a, d in zip(ga, gd):
            assert torch.isfinite(a).all()
            assert torch.allclose(a, d, rtol=1e-6, atol=1e-9), (a, d)


def test_gated_readout_masks_zero_cells():
    m, tok = gated(2), mazes(seed=8)
    psi, _ = m.settle(tok)
    a = m.attention(tok, psi)
    assert torch.isfinite(a).all()
    assert torch.allclose(a.sum(1), torch.ones_like(a[:, 0]))


# --- the cost latent: scale-free diagonal, ReLU or exp source --------------------------------------------------------------

def cost_model(seed=0, delta='exp', source='relu'):
    torch.manual_seed(seed)
    m = S.CostLatentEnergy([(0, -1), (-1, 0)], [(0, -1), (-1, 0)], delta=delta, source=source).double()
    with torch.no_grad():
        m.k.copy_(torch.randn(2) * 0.5)
        m.r.copy_(torch.tensor([1.5, 0.2, 0.5, -0.5]) + 0.1 * torch.randn(4))
        m.sg.copy_(torch.tensor([-1.0, -0.2, 0.4, 2.0]) if source == 'relu' else torch.tensor([-6.0, -5.0, -1.0, 2.0]))
    return m


def test_cost_fixed_point_and_adjoint():
    for delta, source in (('exp', 'relu'), ('softplus', 'relu'), ('exp', 'exp')):
        m, tok = cost_model(delta=delta, source=source), mazes(seed=9)
        m.adjoint_tol = 1e-13
        psi, _ = m.settle(tok, tol=1e-13)
        fin = torch.isfinite(psi)
        z = []
        for b in range(tok.shape[0]):
            A, c = S.dense_A_cost(m, tok[b])
            z.append(torch.linalg.solve(A, c).detach().view(tok.shape[1:]))
        z = torch.stack(z)
        assert torch.equal(z > 1e-300, fin)
        assert torch.allclose(psi[fin], -z[fin].log(), atol=1e-9, rtol=0)
        wts = torch.randn(tok.shape, dtype=torch.float64, generator=torch.Generator().manual_seed(10))
        m.zero_grad()
        psi, _ = m.settle(tok, grad=True, tol=1e-13)
        (wts[fin] * torch.sin(psi[fin])).sum().backward()
        ga = [p.grad.clone() for p in m.latent_params()]
        m.zero_grad()
        loss = 0
        for b in range(tok.shape[0]):
            A, c = S.dense_A_cost(m, tok[b])
            zb = torch.linalg.solve(A, c)
            loss = loss + (wts[b][fin[b]] * torch.sin(-zb[fin[b].flatten()].log())).sum()
        loss.backward()
        gd = [p.grad.clone() for p in m.latent_params()]
        for a, d in zip(ga, gd):
            assert torch.isfinite(a).all() and torch.allclose(a, d, rtol=1e-6, atol=1e-9), (delta, source, a, d)


def test_cost_diagonal_is_scale_free():
    m, tok = cost_model(3), mazes(seed=11)
    psi, _ = m.settle(tok, tol=1e-13)
    with torch.no_grad():
        m.k.add_(5.0)
    psi2, _ = m.settle(tok, tol=1e-13)
    fin = torch.isfinite(psi)
    assert torch.allclose(psi[fin] + 5.0, psi2[fin], atol=1e-8)  # A scales by e^5: z* by e^-5, psi by +5


def test_zero_temperature_is_the_limit_of_scaled_costs():
    """min in place of log-sum-exp equals tau times the soft fixed point of the system whose log-terms (log Gamma,
    log a, log c) are divided by tau, as tau -> 0 (here tau = 1e-3), on the finite cells."""
    m, tok = cost_model(4, source='exp'), mazes(seed=12)  # exp source: log c = sg scales exactly
    m.zero_temperature = True
    hard, _ = m.settle(tok, tol=1e-13)
    m.zero_temperature = False
    tau = 1e-3
    with torch.no_grad():  # every log-term / tau: log Gamma = k, log a = LSE(k) + delta, log c
        la = torch.logsumexp(m.k[m.zpair_of], 0) + m.delta()
        m.k.div_(tau)
        m.r.copy_((la / tau - torch.logsumexp(m.k[m.zpair_of], 0)).log())
        m.sg.div_(tau)
    soft, _ = m.settle(tok, tol=1e-13)
    fin = torch.isfinite(hard)
    assert torch.equal(fin, torch.isfinite(soft))
    assert torch.allclose(tau * soft[fin], hard[fin], atol=0.05)
