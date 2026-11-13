"""Tests of cram.cram_sudoku, cram.cram_sudoku_search and cram.cram_sudoku_sparse.

The tests that fit a model or search real puzzles read the Sudoku-Extreme split files (`scripts/prepare_data.py sudoku`);
the search tests need a GPU. PL is the pseudolikelihood loss.

Ways these can fail, and the test that catches each:
- the proximal step shrinks the wrong way or past zero                                 -> test_prox_l1
- a conflict coupling turns negative under PL (+L1) from zero (per-term test unsound)  -> test_conflict_couplings_nonnegative
- the kernel's incremental fields, P or Q drift from the exact values                  -> test_incremental_state_exact
- a passing board fails its test (wrong certificate)                                   -> test_certified_boards_pass
- a resumed search differs from an uninterrupted one                                   -> test_resume_bitwise
- the per-term count of a board is wrong                                               -> test_term_violations
"""
import pytest
import torch

from cram import cram_sudoku as CS

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')


def boards(b=6, seed=0):
    """Random boards (b, 81) of digits 0-8 and a random mask of free cells (b, 81)."""
    g = torch.Generator().manual_seed(seed)
    sol = torch.randint(0, 9, (b, 81), generator=g)
    free = torch.rand(b, 81, generator=g) < 0.7
    return sol, free


def train_split(device):
    """The 1,000 training puzzles of split 0 as tensors: (solutions (1000, 81), free (1000, 81))."""
    from cram.cram_sudoku_search import load_puzzles
    sol, free, _ = load_puzzles('train-s0')
    return torch.as_tensor(sol, device=device), torch.as_tensor(free, device=device)


def test_prox_l1():
    A = torch.tensor([[-2.0, -0.5, 0.0], [0.3, 1.0, 5.0], [0.0, -1.0, 0.7]])
    CS.prox_l1(A, torch.tensor(0.6))
    assert torch.allclose(A, torch.tensor([[-1.4, 0.0, 0.0], [0.0, 0.4, 4.4], [0.0, -0.4, 0.1]]))


def test_term_violations():
    A = -1.0 * (1 - torch.eye(81))  # S = -2 on every pair: no pair blocks
    A[0, 1] = A[1, 0] = 0.0  # except (0, 1): S = 0 blocks (the test is strict)
    sol = torch.arange(81).remainder(9)[None].clone()  # cells 0 and 9 share digit 0, cells 0 and 1 do not
    assert CS.term_violations(A, sol).item() == 0
    sol[0, 1] = 0  # now (0, 1) is a same-digit pair with S = 0: blocks
    assert CS.term_violations(A, sol).item() == 1


@pytest.mark.parametrize('opt', ['sgd', 'adam'])
def test_conflict_couplings_nonnegative(opt):
    sol, free = train_split('cpu')
    peer = CS.conflict_graph()
    for l1 in (0.0, 1e-3):
        cfg = CS.FitConfig(aug='digit', updates=8, opt=opt, lr=30.0 if opt == 'sgd' else 0.1, l1=l1, device='cpu')
        A, _, _ = CS.train(cfg, sol[:200], free[:200], graph=False)
        S = A + A.T
        assert (S[peer] >= 0).all()
        assert (S[peer] > 0).any()


def random_model(seed=0, device='cuda'):
    """Random symmetric couplings A (81, 81) with a zero diagonal and small random input biases B (10, 9)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(81, 81, generator=g) * 0.3
    A = (A + A.T) / 2 * (1 - torch.eye(81))
    return A.to(device), (0.05 * torch.randn(10, 9, generator=g)).to(device)


@cuda
@pytest.mark.parametrize('test', ['pair', 'term'])
def test_incremental_state_exact(test):
    from cram import cram_sudoku_search as SS
    A, B = random_model(2)
    A = A - 0.1 * (1 - torch.eye(81, device='cuda'))
    sol, free = boards(16, 2)
    ss = SS.Search(A, B, sol.numpy(), free.numpy(), temp=0.5, theta=0.2, test=test, tau=-1e9, cap=1e12, seed=1)
    ss._moves(2000)
    G, P, Q = ss.G.clone(), ss.P.clone(), ss.Q.clone()
    ss._refresh()
    assert torch.allclose(G, ss.G, atol=1e-3) and torch.allclose(P, ss.P, atol=1e-2)
    if test == 'term':
        assert torch.equal(Q, ss.Q) and (Q > 0).any()


@cuda
@pytest.mark.parametrize('test', ['pair', 'term'])
def test_certified_boards_pass(test):
    from cram import cram_sudoku_search as SS
    sol, free = train_split('cuda')
    A, B, _ = CS.train(CS.FitConfig(aug='digit', updates=8), sol, free)
    A, _ = CS.postfit(A, 'k2')  # every non-conflict coupling negative: the per-term test can pass
    tau, ebar = SS.solution_stats(A, sol)
    from cram.cram_sudoku_search import load_puzzles
    ds, df, _ = load_puzzles('dev10k', 0, 64)
    ss = SS.Search(A, B, ds, df, temp=0.5, theta=1.0, test=test, tau=tau, ebar=ebar, cap=2e9, seed=0)
    ss.run()
    hit = ss.found >= 0
    assert bool(hit.any())
    yc = ss.ycert[hit].long()
    if test == 'term':
        assert (CS.term_violations(A, yc) == 0).all()
    else:
        assert (CS.pair_energy(A, yc) <= tau + 1e-3 + 1e-3).all()


@cuda
def test_resume_bitwise():
    from cram import cram_sudoku_search as SS
    sol, free = train_split('cuda')
    A, B, _ = CS.train(CS.FitConfig(aug='sudoku', updates=8), sol, free)
    tau, ebar = SS.solution_stats(A, sol)
    from cram.cram_sudoku_search import load_puzzles
    ds, df, _ = load_puzzles('dev10k', 0, 128)
    kw = dict(temp=0.5, theta=1.0, test='pair', tau=tau, ebar=ebar, cap=3e9, seed=0, restart=8192, growth=1.5, C=2)
    ref = SS.Search(A, B, ds, df, **kw).run()
    a = SS.Search(A, B, ds, df, **kw)
    while a.t < 3 * 4096:
        a.step_check()
    sd = a.state_dict()
    b = SS.Search(A, B, ds, df, **kw)
    b.load_state_dict(sd)
    res = b.run()
    for k in ('flops', 'solved', 'stopped'):
        assert res[k] == ref[k]
    other = SS.Search(A, B, ds, df, **{**kw, 'seed': 1}).run()
    assert other['flops'] != ref['flops']


# ---------------------------------------------------------------- sparse kernel
# Ways it can fail: the neighbour fields, flags or pair energy drift from their exact values (test_sparse_state_exact);
# its moves differ from the dense kernel's on the same energy (test_sparse_matches_dense); the one-level count mode
# changes the moves (test_sparse_one_level_matches_dense); resume is not bitwise (test_sparse_resume_bitwise).
def support_model(weights='random', seed=0, device='cuda'):
    """A on the Sudoku peer graph (a 20-regular support; a test fixture only), with random positive or equal weights."""
    g = torch.Generator().manual_seed(seed)
    peer = CS.conflict_graph()
    W = 0.3 + torch.rand(81, 81, generator=g)
    W = (W + W.T) / 2 if weights == 'random' else torch.full((81, 81), 0.5)
    A = torch.where(peer, W, torch.zeros(()))
    return A.to(device), torch.zeros(10, 9, device=device)


@cuda
def test_sparse_matches_dense():
    from cram import cram_sudoku_search as SS
    from cram.cram_sudoku_sparse import SparseSearch
    A, B = support_model('random')
    B = B + 0.05 * torch.randn(10, 9, generator=torch.Generator().manual_seed(1)).cuda()
    sol, free = boards(16, 3)
    kw = dict(temp=0.6, theta=0.4, test='pair', tau=-1e9, cap=1e12, seed=5, check=1024)
    a = SS.Search(A, B, sol.numpy(), free.numpy(), **kw)
    b = SparseSearch(A, B, sol.numpy(), free.numpy(), **kw)
    assert not b.one_level and b.D == 20
    for _ in range(3):  # 3,072 moves: before the 4,096-move refresh
        a._moves(1024); b._moves(1024); a.t += 1024; b.t += 1024
    assert torch.equal(a.y, b.y)
    assert torch.allclose(a.G, b.G, atol=1e-5) and torch.allclose(a.P, b.P, atol=1e-4)


@cuda
def test_sparse_one_level_matches_dense():
    """With one level h = 0.5 (a power of 2), dense sums and integer counts are exact: identical moves."""
    from cram import cram_sudoku_search as SS
    from cram.cram_sudoku_sparse import SparseSearch
    A, B = support_model('equal')
    sol, free = boards(16, 4)
    kw = dict(temp=0.4, theta=0.4, test='pair', tau=-1e9, cap=1e12, seed=2, check=4096)
    a = SS.Search(A, B, sol.numpy(), free.numpy(), **kw)
    b = SparseSearch(A, B, sol.numpy(), free.numpy(), **kw)
    assert b.one_level and b.hscale == 1.0  # S = A + A^T = 2 * 0.5
    for _ in range(3):
        a.step_check(); b.step_check()
    assert torch.equal(a.y, b.y) and torch.equal(a.G, b.G * b.hscale)


@cuda
@pytest.mark.parametrize('weights', ['random', 'equal'])
def test_sparse_state_exact(weights):
    from cram.cram_sudoku_sparse import SparseSearch
    A, B = support_model(weights, seed=7)
    sol, free = boards(32, 5)
    s = SparseSearch(A, B, sol.numpy(), free.numpy(), temp=0.5, theta=0.3, test='pair', tau=-1e9, cap=1e12, seed=1)
    s._moves(3000)
    G, P, flag = s.G.clone(), s.P.clone(), s.flag.clone()
    s._refresh()
    assert torch.allclose(G, s.G, atol=1e-4) and torch.allclose(P, s.P, atol=1e-3)
    assert torch.equal(flag, s.flag) and bool(flag.any())
    if weights == 'equal':
        assert torch.equal(G, s.G) and torch.equal(P, s.P)


@cuda
def test_sparse_resume_bitwise():
    from cram.cram_sudoku_sparse import SparseSearch
    sol, free = train_split('cuda')
    cfg = CS.FitConfig(aug='digit', updates=32, l1=3e-3)
    A, B, _ = CS.train(cfg, sol, free)
    A, _ = CS.postfit(A, 'k2')
    from cram.cram_sudoku_search import load_puzzles
    ds, df, _ = load_puzzles('dev10k', 0, 128)
    kw = dict(temp=0.3, theta=0.2, test='pair', tau=0.0, ebar=0.0, cap=3e9, seed=0, C=2, check=4096,
              anneal=(0.4, 0.25, 8 * 4096))
    ref = SparseSearch(A, B, ds, df, **kw).run()
    assert sum(ref['solved']) > 0
    a = SparseSearch(A, B, ds, df, **kw)
    while a.t < 3 * 4096:
        a.step_check()
    b = SparseSearch(A, B, ds, df, **kw)
    b.load_state_dict(a.state_dict())
    res = b.run()
    for k in ('flops', 'solved', 'stopped'):
        assert res[k] == ref[k]
    assert all(st <= sv for st, sv in zip(res['stopped'], res['solved']) if st)  # every stop is a solution
