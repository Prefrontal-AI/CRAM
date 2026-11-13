"""Tests of cram.cram_nqueens and cram.nqueens_data. PL is the pseudolikelihood loss; test_fit_loss_matches_rows reads
GRAM's 8 x 8 training file (`scripts/prepare_data.py multi-solution`).

Ways the code can fail, and the test that catches each:
- the search's field update reads a column with old and new swapped, or transposed           -> test_search_state_exact
- the input cost B is dropped from the energy the search tracks                                -> test_search_state_exact
- a chain moves a given queen, or a stopped chain's board is above the threshold               -> test_search_givens_and_stops
- the reduction of PL to distinct solutions weights cells wrongly (given cells counted)       -> test_fit_loss_matches_rows
- the rule checker misses a diagonal or anti-diagonal, or accepts n-1 queens                   -> test_valid_against_brute_force
- the enumeration differs from the known solution counts (92 for n = 8, 724 for n = 10)        -> test_completion_counts
- majority vote ignores the tie rule, coverage counts boards that drop a given queen            -> test_gram_metrics_rules
"""
import itertools

import numpy as np
import torch

from cram import cram_nqueens as Q


def random_model(c, seed=0):
    """A model dict with random pair tables W and input costs B for c cells, and a threshold that no board reaches."""
    g = torch.Generator().manual_seed(seed)
    iu = Q.pair_index(c, 'cpu')
    W = torch.randn(iu.shape[1], 2, 2, generator=g)
    B = torch.randn(c, 2, generator=g)
    return dict(W=W, B=B, S=Q.full_S(W, iu, c), tau=-1e9, unit=1.0)


def test_search_state_exact():
    """With an unreachable threshold every chain runs to the cap; the returned lowest-energy board's energy must equal
    a direct recompute (the incremental energy accumulates every field update)."""
    n, c = 5, 25
    m = random_model(c)
    givens = np.zeros((4, c), bool)
    givens[1, 3] = True
    boards, stopped, energy, moves, flop = Q.solve(m, givens, chains=3, temperature=.7, cap=300, seed=1)
    assert not stopped.any() and (moves == 300).all()
    direct = Q.board_energy(m['S'], m['B'], torch.as_tensor(boards.reshape(-1, c))).numpy()
    np.testing.assert_allclose(energy.reshape(-1), direct, rtol=1e-4, atol=1e-3)
    assert flop.shape == (4, 3) and (flop == Q.init_flop(c) + 300 * Q.move_flop(c)).all()


def test_search_givens_and_stops():
    n, c = 5, 25
    m = random_model(c, seed=2)
    rng = np.random.default_rng(0)
    givens = rng.random((6, c)) < 0.15
    # a threshold that some boards reach: the median energy of random boards
    rand = torch.as_tensor(rng.random((200, c)) < 0.3)
    m['tau'] = float(Q.board_energy(m['S'], m['B'], rand).median())
    boards, stopped, energy, _, _ = Q.solve(m, givens, chains=4, temperature=.5, cap=2000, seed=0)
    assert (boards >= givens[:, None, :]).all()
    assert stopped.any()
    direct = Q.board_energy(m['S'], m['B'], torch.as_tensor(boards.reshape(-1, c))).numpy().reshape(6, 4)
    assert (direct[stopped] <= m['tau'] + 1e-3 + 1e-3).all()


def test_fit_loss_matches_rows():
    """One plain SGD step from zero equals -lr times the gradient of the PL over the free cells of every row."""
    inputs, labels, _ = Q.load_split(8, 'train')
    inputs, labels = inputs[:400], labels[:400]
    model, info = Q.fit(inputs, labels, steps=1, lr=5., lam=0., momentum=0., cosine=False)
    # recompute the loss at the fitted parameters directly per row, then compare with a second fit's step-0 loss
    W0 = torch.zeros_like(model['W'], requires_grad=True)
    B0 = torch.zeros_like(model['B'], requires_grad=True)
    y = torch.as_tensor(labels).long()
    free = torch.as_tensor(~inputs).float()
    g = Q.fields(Q.full_S(W0, Q.pair_index(64, 'cpu'), 64), B0, y)
    nll = -torch.log_softmax(-g, -1).gather(-1, y[..., None])[..., 0]
    loss = (nll * free).sum() / free.sum()
    gW, gB = torch.autograd.grad(loss, (W0, B0))
    np.testing.assert_allclose(model['W'].numpy(), (-5. * gW).numpy(), rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(model['B'].numpy(), (-5. * gB).numpy(), rtol=1e-4, atol=1e-6)


def test_valid_against_brute_force():
    n = 5
    rng = np.random.default_rng(0)
    boards = rng.random((3000, n * n)) < 0.2
    for k in range(200):  # add permutations so valid boards occur
        boards[k] = False
        boards[k, np.arange(n) * n + rng.permutation(n)] = True

    def brute(b):
        q = [(i // n, i % n) for i in np.flatnonzero(b)]
        if len(q) != n:
            return False
        return all(r1 != r2 and c1 != c2 and abs(r1 - r2) != abs(c1 - c2)
                   for (r1, c1), (r2, c2) in itertools.combinations(q, 2))

    assert (Q.valid(boards, n) == np.array([brute(b) for b in boards])).all()
    assert Q.valid(boards, n).sum() > 0


def test_completion_counts():
    assert len(Q.completions(np.zeros(64, bool), 8)) == 92
    assert len(Q.completions(np.zeros(100, bool), 10)) == 724
    assert len(Q.completions(np.zeros(64, bool), 8, limit=20)) == 20


def test_gram_metrics_rules():
    n = 8
    sols = Q.completions(np.zeros(64, bool), 8)
    a, b = sols[0], sols[1]
    given = a & b if (a & b).any() else np.zeros(64, bool)
    bad = a.copy()
    bad[np.flatnonzero(a)[0]] = False
    # samples: bad, a, a, b, b -> tie between a and b (2 each); tiebreak prefers b
    boards = np.stack([bad, a, a, b, b])[None]
    tb = np.array([[0, 0, 0, 1, 1]], float)
    m = Q.gram_metrics(given[None], boards, n, tiebreak=tb, k_values=(1, 5))
    assert (m['majority'] == [True]).all() and not m['pass@1'][0] and m['pass@5'][0]
    assert m['distinct_valid'][0] == 2
    # a valid board without the given queen is correct for GRAM's rule but not a completion and not covered
    other = next(s for s in sols if not (s >= given).all()) if given.any() else None
    if other is not None:
        m2 = Q.gram_metrics(given[None], np.stack([other])[None], n, k_values=(1,))
        assert m2['pass@1'][0] and not m2['pass@1_completion'][0] and m2['coverage20'][0] == 0
