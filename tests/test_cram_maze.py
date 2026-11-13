"""CRAM Maze-Unique (cram/cram_maze.py). Ways it could be wrong, each a test below:
1. `from_legacy` maps a tensor wrongly (the bias row, P and Q swapped), so the 16-parameter model's current or energy
   differs from the 34-float model it was converted from;
2. `from_legacy` accepts a checkpoint whose wall, S or G bias rows are nonzero (they would change clamped currents);
3. the stored count is not 16 at R 1, de 2, window 1, or the initial draws differ from MazeEnergy's at the same seed
   (then a seeded run would not reproduce the old weights);
4. `board_energy` disagrees with `energy` at one-hot states (bias sign, the 1/2 of the bilinear form);
5. `gibbs_from` keeps a wrong running energy (the lowest-energy board is chosen by it), or a stopped board is not a
   fixed point;
6. `descend_from` from the all-open logits differs from maze_unique.descend at its fixed points.
CRAMMazeFreeEnergy (the free 3 x 3 kernel of the paper's Maze-Unique row):
7. its kernel puts T_d and T_d^T at the wrong offsets (sign of d, or no transpose), so the current is not the gradient of
   the energy;
8. its pair energy differs from the explicit sum over unordered pairs i, j = i + d of s_i^T T_d s_j;
9. it cannot express the separable model (an offset left out of `half_offsets`), or the count is not 18 at window 1, or
   an entry gets no gradient, or the center (self-interaction) is nonzero.
"""
import pytest
from einops import rearrange
import torch

from cram import cram_maze as C
from cram import maze_unique as M


def _maze(b=4, H=9, seed=0):
    """Random grids (b, H, H) of walls and open cells with S and G at two corners, a random 0/1 board as the label,
    and the mask of free (open) cells. The boards need not be valid mazes for these checks."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 2, (b, H, H), generator=g)  # walls and open cells
    x[:, 0, 0] = M.S_TOK
    x[:, -1, -1] = M.G_TOK
    free = x == M.OPEN
    y = torch.randint(0, 2, (b, H, H), generator=g) * free + (~free) * ((x == M.S_TOK) | (x == M.G_TOK))
    return x, y.long(), free


def _legacy(seed=1, R=1, de=2, window=1):
    """A maze_unique.MazeEnergy at a seeded random start with a random open-cell bias."""
    torch.manual_seed(seed)
    m = M.MazeEnergy(R=R, de=de, window=window)
    m.B.data[M.OPEN] = torch.randn(2)
    return m


@pytest.mark.parametrize('R,de,window', [(1, 2, 1), (2, 3, 2)])
def test_from_legacy_same_current_and_energy(R, de, window):
    old = _legacy(R=R, de=de, window=window)
    new = C.CRAMMazeEnergy.from_legacy(old.state_dict(), R, de, window)
    x, y, free = _maze()
    s = torch.softmax(torch.randn(x.shape[0], 2, *x.shape[1:]), 1)
    assert torch.equal(new.current(s, x), old.current(s, x))
    assert torch.allclose(new.energy(s, x, free), old.energy(s, x, free), atol=1e-5)


def test_from_legacy_refuses_clamped_bias():
    old = _legacy()
    old.B.data[M.S_TOK, 1] = 0.5
    with pytest.raises(AssertionError):
        C.CRAMMazeEnergy.from_legacy(old.state_dict(), 1, 2, 1)


def test_count_and_seeded_init():
    torch.manual_seed(7)
    new = C.CRAMMazeEnergy(R=1, de=2, window=1)
    torch.manual_seed(7)
    old = M.MazeEnergy(R=1, de=2, window=1)
    assert new.n_params() == 16 and old.n_params() == 34
    for k in ('Fp', 'Gp', 'P', 'Q'):
        assert torch.equal(getattr(new, k), getattr(old, k))


def test_board_energy_matches_energy():
    m = C.CRAMMazeEnergy.from_legacy(_legacy(seed=3).state_dict(), 1, 2, 1)
    x, y, free = _maze(seed=2)
    board = torch.randint(0, 2, x.shape)
    s = M.onehot(torch.where(free, board, y)) * free[:, None] + M.given(x, free)
    with torch.no_grad():
        assert torch.allclose(C.board_energy(m, board, x, free), m.energy(s, x, free), atol=1e-4)


@pytest.mark.parametrize('T,stop', [(1.0, 'none'), (0.0, 'fixed'), (0.5, 'fixed')])
def test_gibbs_from_energy_and_fixed_points(T, stop):
    m = C.CRAMMazeEnergy.from_legacy(_legacy(seed=4).state_dict(), 1, 2, 1)
    x, y, free = _maze(b=6, seed=5)
    gen = torch.Generator().manual_seed(0)
    start = (torch.rand(x.shape, generator=gen) < 0.5).long()
    r = C.gibbs_from(m, x, y, free, start, T=T, rho=0.3, max_moves=60, stop=stop, gen=gen)
    with torch.no_grad():
        e = C.board_energy(m, r['pred'], x, free)
        assert torch.allclose(r['energy'], e, atol=1e-4)
        assert torch.all(r['energy'] <= C.board_energy(m, start, x, free) + 1e-4) or stop == 'fixed'
        if stop == 'fixed':
            g = m.current(M.onehot(r['pred']) * free[:, None] + M.given(x, free), x)
            gy = g.gather(1, r['pred'][:, None]).squeeze(1)
            dis = ((gy - g.min(1).values) * free).amax((1, 2))
            assert torch.all(dis[r['stopped']] <= 1e-5)


def test_descend_from_matches_descend():
    m = C.CRAMMazeEnergy.from_legacy(_legacy(seed=6).state_dict(), 1, 2, 1)
    x, y, free = _maze(b=6, seed=8)
    ref = M.descend(m, x, y, free, max_steps=3000)
    u0 = M.start_logits(x, free, margin=3.0)
    r = C.descend_from(m, x, y, free, u0, max_steps=3000)
    st = ref['stop_kind'] > 0
    assert st.any()
    assert torch.equal(r['stopped'], st)
    assert torch.equal(r['steps'][st], ref['steps'][st])
    assert torch.equal(r['pred'][st], ref['pred'][st])


# ---- CRAMMazeFreeEnergy ----------------------------------------------------------------------------------------------

def _free(seed=3, window=1):
    """A CRAMMazeFreeEnergy with random tables and a random bias."""
    torch.manual_seed(seed)
    m = C.CRAMMazeFreeEnergy(window=window, init=1.0)
    m.b.data = torch.randn(2)
    return m


@pytest.mark.parametrize('window', [1, 2])
def test_free_current_is_energy_gradient(window):
    m = _free(window=window).double()
    x, y, free = _maze(H=7)
    s = torch.softmax(torch.randn(x.shape[0], 2, *x.shape[1:], dtype=torch.float64), 1).requires_grad_()
    pair_bias = lambda s: m.energy(s, x, torch.zeros_like(free))  # entropy off: the current is d(pair + bias)/ds
    g, = torch.autograd.grad(pair_bias(s).sum(), s)
    assert torch.allclose(m.current(s.detach(), x), g, atol=1e-10)


def test_free_pair_energy_is_explicit_sum():
    m = _free()
    x, y, free = _maze(b=2, H=6)
    s = torch.softmax(torch.randn(2, 2, 6, 6), 1)
    want = torch.zeros(2)
    for t, (dr, dc) in enumerate(m.offsets):
        for r in range(6):
            for c in range(6):
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < 6 and 0 <= c2 < 6:
                    want += torch.einsum('ba,ac,bc->b', s[:, :, r, c], m.T[t].detach(), s[:, :, r2, c2])
    bias = (s * rearrange(m.bias_table()[x], 'b h w a -> b a h w')).sum((1, 2, 3))
    got = m.energy(s, x, torch.zeros_like(free)) + bias
    assert torch.allclose(got, want, atol=1e-5)


def test_free_count_gradient_center_and_contains_separable():
    m = C.CRAMMazeFreeEnergy(window=1)
    assert m.n_params() == 18 and len(m.offsets) == 4
    assert torch.all(m.T == 0) and torch.all(m.b == 0)  # zero start by default
    m = _free()
    W = m.kernel()
    assert torch.all(W[:, :, 1, 1] == 0)
    x, y, free = _maze()
    M.pl_loss(m, x, y, free).backward()
    assert (m.T.grad != 0).all() and (m.b.grad != 0).all()
    sep = C.CRAMMazeEnergy.from_legacy(_legacy().state_dict(), 1, 2, 1)
    Ws = sep.kernel().detach()
    for t, (dr, dc) in enumerate(m.offsets):
        m.T.data[t] = Ws[:, :, 1 + dr, 1 + dc]
    m.b.data = sep.b.data.clone()
    s = torch.softmax(torch.randn(x.shape[0], 2, *x.shape[1:]), 1)
    assert torch.allclose(m.current(s, x), sep.current(s, x), atol=1e-6)
