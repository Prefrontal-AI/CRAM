"""Tests of cram.cram_sudoku_general (the search of the paper's Sudoku model on the general pairwise energy).
The tests need a GPU. The "equality kernel" is the sparse kernel of the same-digit energy (cram_sudoku_sparse).

Ways the general kernel can fail, and the test that catches each:
- a neighbour block is read transposed, or with old and new swapped (wrong field update)  -> test_general_state_exact
- the input cost of a cell is dropped from the heat bath, the energy or the focus flags    -> test_general_state_exact
- padded neighbour slots change a field, or the flags of padded cells go stale             -> test_general_state_exact (ragged degrees)
- the moves differ from the equality kernel's on an equality energy (cell choice, keys)    -> test_general_matches_sparse_on_equality
- a certified board does not pass the zero test                                            -> test_general_certified_pass
"""
import pytest
import torch

from cram import cram_sudoku as CS

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
N, V = 81, 9


def boards(b=6, seed=0):
    """Random boards (b, 81) of digits 0-8 and a random mask of free cells (b, 81)."""
    g = torch.Generator().manual_seed(seed)
    sol = torch.randint(0, 9, (b, 81), generator=g)
    free = torch.rand(b, 81, generator=g) < 0.7
    return sol, free


def general_model(seed=0):
    """Random asymmetric blocks on the conflict graph plus a few extra pairs (ragged degrees), random input costs."""
    g = torch.Generator().manual_seed(seed)
    nz = CS.conflict_graph()
    extra = torch.rand(N, N, generator=g) < 0.02
    nz = (nz | extra | extra.T) & ~torch.eye(N, dtype=torch.bool)
    iu = torch.triu_indices(N, N, 1)
    W = torch.randn(iu.shape[1], V, V, generator=g) * nz[iu[0], iu[1]][:, None, None]
    S = torch.zeros(N, N, V, V).index_put((iu[0], iu[1]), W).index_put((iu[1], iu[0]), W.transpose(1, 2))
    Bc = 0.3 * torch.randn(N, V, generator=g)
    return S.cuda(), Bc.cuda()


@cuda
def test_general_state_exact():
    from cram.cram_sudoku_general import GeneralSearch, neighbour_blocks
    S, Bc = general_model(3)
    _, _, D = neighbour_blocks(S)
    deg = ((S != 0).flatten(2).any(-1)).sum(1)
    assert int(deg.min()) < D  # ragged: padded slots are exercised
    sol, free = boards(32, 5)
    s = GeneralSearch(S, Bc, sol.numpy(), free.numpy(), temp=0.7, theta=0.2, tau=-1e9, cap=1e12, seed=1)
    s._moves(3000)
    G, E, flag = s.G.clone(), s.P.clone(), s.flag.clone()
    s._refresh()
    assert torch.allclose(G, s.G, atol=1e-3) and torch.allclose(E, s.P, atol=1e-2)
    assert torch.equal(flag, s.flag) and bool(flag.any())


@cuda
def test_general_matches_sparse_on_equality():
    """S = 0.5 * 2 * I on the conflict graph (h = 1, exact in float), no input costs: the general kernel makes the moves
    of the equality kernel."""
    from cram.cram_sudoku_general import GeneralSearch
    from cram.cram_sudoku_sparse import SparseSearch
    A = 0.5 * CS.conflict_graph('cuda').float()
    S = (A + A.T)[:, :, None, None] * torch.eye(V, device='cuda')
    sol, free = boards(16, 4)
    kw = dict(temp=0.4, theta=0.4, tau=-1e9, cap=1e12, seed=2, check=4096)
    a = SparseSearch(A, torch.zeros(10, V), sol.numpy(), free.numpy(), test='pair', **kw)
    b = GeneralSearch(S, torch.zeros(N, V), sol.numpy(), free.numpy(), **kw)
    for _ in range(3):
        a.step_check(); b.step_check()
    assert torch.equal(a.y, b.y) and torch.equal(a.G * a.hscale, b.G)


@cuda
def test_general_certified_pass():
    from cram.cram_sudoku_general import GeneralSearch, board_energy
    S, Bc = general_model(4)
    sol, free = boards(64, 6)
    s = GeneralSearch(S, Bc, sol.numpy(), free.numpy(), temp=0.5, theta=0.2, tau=0.0, tol=5.0, cap=1e12, seed=3)
    s._moves(2000)
    got = s.found >= 0
    assert bool(got.any())
    E = board_energy(S, Bc, s.ycert[got])
    assert bool((E <= 5.0 + 1e-2).all())
