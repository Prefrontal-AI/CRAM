"""Tests of cram.cram_graph_coloring and cram.graph_coloring_data.

Failure modes: a token routed to the wrong vertex pair, a transposed or doubled pair table, fields that disagree with
the energy, a cached field update with the wrong sign, a stop accepted above the threshold, conflicts counted on
non-edges, a non-canonical coloring counted as covered, and a majority vote that ignores frequency.
The environment variables GC_TEST_MODULE and GCD_TEST_MODULE name a file to test in place of each module, so that the
tests can be run against a deliberately broken copy to check that they fail.
"""
import importlib.util
import os

import numpy as np
import torch

from cram import cram_graph_coloring as G
from cram import graph_coloring_data as D


def _mutant(var, default):
    """The module loaded from the file named by the environment variable `var`, or `default` when it is not set."""
    if not os.environ.get(var):
        return default
    spec = importlib.util.spec_from_file_location('mutant', os.environ[var])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _mutant('GC_TEST_MODULE', G)
D = _mutant('GCD_TEST_MODULE', D)


def gram_encode(adj):
    """GRAM's encode_adjacency (dataset/build_graph_coloring_dataset.py), copied."""
    n = adj.shape[0]
    seq = []
    for i in range(n):
        for j in range(i, n):
            seq.append(0 if i == j else (2 if adj[i, j] else 1))
    return np.array(seq, dtype=np.uint8)


def random_graphs(rng, b, n, p=.4):
    """Random symmetric adjacency matrices (b, n, n) bool with edge probability p."""
    A = np.triu(rng.random((b, n, n)) < p, 1)
    return A | A.transpose(0, 2, 1)


def random_model(rng, arm, n):
    """A model dict of one arm with random parameters and fixed search statistics."""
    prm = {k: rng.normal(size=v.shape) for k, v in G.init_params(arm, n).items()}
    return prm | dict(arm=arm, n=n, theta=.45, tau=0., ebar=0., hbar=1.)


def test_convert_routes_each_token_to_its_pair():
    rng = np.random.default_rng(0)
    n = 6
    A = random_graphs(rng, 5, n)
    x = np.stack([gram_encode(a) for a in A])
    y = np.zeros((5, n * (n + 1) // 2), np.uint8)
    y[:, :n] = rng.integers(3, 6, size=(5, n))
    a, col = D.convert(x, y, n)
    np.testing.assert_array_equal(D.adjacency(a, n), A)
    np.testing.assert_array_equal(col, y[:, :n] - 3)


def test_fields_equal_energy_differences_all_arms():
    rng = np.random.default_rng(1)
    n = 5
    A = random_graphs(rng, 4, n)
    a = A[:, *G.pairs(n)].astype(np.float64)
    y = rng.integers(3, size=(4, n))
    for arm in G.ARMS:
        m = random_model(rng, arm, n)
        prm = {k: torch.tensor(m[k]) for k in ('W0', 'D', 'B')}
        S = G.pair_tensor(G.assemble(prm, torch.tensor(a)), n)
        f = G.fields(S, G.bias(prm, n), torch.tensor(y)).numpy()
        E0 = G.energy_np(m, a, y)
        np.testing.assert_allclose(G.energy_t(S, G.bias(prm, n), torch.tensor(y)).numpy(), E0, rtol=1e-10)
        for i in range(n):
            for c in range(3):
                y2 = y.copy()
                y2[:, i] = c
                np.testing.assert_allclose(G.energy_np(m, a, y2) - E0, f[:, i, c] - f[np.arange(4), i, y[:, i]],
                                           rtol=1e-9, atol=1e-9)


def test_energy_matches_direct_sum_over_pairs():
    rng = np.random.default_rng(2)
    n = 5
    A = random_graphs(rng, 3, n)
    a = A[:, *G.pairs(n)].astype(np.float64)
    y = rng.integers(3, size=(3, n))
    m = random_model(rng, 'full', n)
    I, J = G.pairs(n)
    for b in range(3):
        e = sum(m['B'][i, y[b, i]] for i in range(n))
        for p, (i, j) in enumerate(zip(I, J)):
            K = m['W0'][p] + sum(a[b, t] * m['D'][p, t] for t in range(len(I)))
            e += K[y[b, i], y[b, j]]
        np.testing.assert_allclose(G.energy_np(m, a[b:b + 1], y[b:b + 1])[0], e, rtol=1e-10)


def test_full_arm_with_own_token_only_equals_pos_arm():
    rng = np.random.default_rng(3)
    n = 5
    P = n * (n - 1) // 2
    pos = random_model(rng, 'pos', n)
    full = dict(pos)
    full['D'] = np.zeros((P, P, 3, 3))
    full['D'][np.arange(P), np.arange(P)] = pos['D']
    A = random_graphs(rng, 4, n)
    a = A[:, *G.pairs(n)].astype(np.float64)
    y = rng.integers(3, size=(4, n))
    np.testing.assert_allclose(G.energy_np(full, a, y), G.energy_np(pos, a, y), rtol=1e-12)


def test_search_energy_is_exact_and_stops_obey_threshold():
    rng = np.random.default_rng(4)
    n = 6
    A = random_graphs(rng, 8, n, .5)
    a = A[:, *G.pairs(n)].astype(np.float64)
    m = random_model(rng, 'pos', n)
    E = G.energy_np(m, a, rng.integers(3, size=(8, n)))
    m['tau'] = float(np.median(E))
    res = G.solve(m, a, samples=3, temperature=.5, cap=200, seed=0)
    boards = res['boards'].reshape(-1, n)
    exact = G.energy_np(m, np.repeat(a, 3, 0), boards)
    np.testing.assert_allclose(res['energy'].ravel(), exact, rtol=1e-9, atol=1e-9)
    assert (res['energy'][res['stopped']] <= m['tau'] + 1e-3).all()
    assert (res['moves'] > 0).any()


def test_equality_energy_solves_and_scores_proper():
    rng = np.random.default_rng(5)
    n = 8
    A = random_graphs(rng, 20, n, .3)
    a = A[:, *G.pairs(n)].astype(np.float64)
    m = dict(G.init_params('shared', n)) | dict(arm='shared', n=n, theta=.45, tau=0., ebar=0., hbar=1.)
    m = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in m.items()}
    m['D'] = np.eye(3)
    sols = G.proper_canonical_sets(A)
    keep = [b for b in range(20) if sols[b]]
    res = G.solve(m, a[keep], samples=20, temperature=.3, cap=5000, seed=1)
    s = G.score(A[keep], res)
    assert s['proper_1'] == 1. and s['conflicts_1'] == 0 and s['wrong_stops_all'] == 0
    assert s['coverage_canonicalized_20'] >= s['coverage_20']


def test_conflicts_count_only_edges():
    A = np.zeros((1, 3, 3), bool)
    A[0, 0, 1] = A[0, 1, 0] = True
    boards = np.array([[[0, 0, 0], [0, 1, 1], [2, 2, 1]]])
    np.testing.assert_array_equal(D.conflicts(A, boards), [[1, 0, 1]])


def test_canonical_form_and_solution_sets():
    assert D.is_canonical(np.array([0, 1, 0, 2])) and not D.is_canonical(np.array([0, 2, 1, 0]))
    assert not D.is_canonical(np.array([1, 0, 2]))
    np.testing.assert_array_equal(D.canonical(np.array([2, 1, 2, 0])), [0, 1, 0, 2])
    tri = np.ones((1, 3, 3), bool) & ~np.eye(3, dtype=bool)
    assert D.proper_canonical_sets(tri) == [{(0, 1, 2)}]
    path = np.zeros((1, 3, 3), bool)
    path[0, 0, 1] = path[0, 1, 0] = path[0, 1, 2] = path[0, 2, 1] = True
    assert D.proper_canonical_sets(path) == [{(0, 1, 0), (0, 1, 2)}]


def test_score_coverage_and_majority():
    path = np.zeros((1, 3, 3), bool)
    path[0, 0, 1] = path[0, 1, 0] = path[0, 1, 2] = path[0, 2, 1] = True
    boards = np.array([[[0, 1, 0], [1, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]])
    res = dict(boards=boards, energy=np.zeros((1, 5)), stopped=np.ones((1, 5), bool), moves=np.ones((1, 5), int),
               chain_flop=np.ones((1, 5)), assembly_flop=0)
    s = G.score(path, res, ks=(1, 5))
    assert s['coverage_5'] == .5  # (0,1,0) found; (1,0,1) is proper but not canonical
    assert s['coverage_canonicalized_5'] == .5
    assert s['conflicts_majority_5'] == 1  # (0,0,1) appears three times and has one conflict
    assert s['conflicts_1'] == 0 and s['proper_1'] == 1.


def test_check_tokens_matches_gram_definitions():
    """GRAM-format predictions: conflicts, canonical form, validity, coverage and a stray token."""
    path = np.zeros((3, 3), bool)
    path[0, 1] = path[1, 0] = path[1, 2] = path[2, 1] = True
    x = gram_encode(path)[None]
    L = x.shape[1]
    preds = np.zeros((1, 4, L), np.uint8)
    for s, c in enumerate([(0, 1, 0), (1, 0, 1), (0, 0, 1), (0, 1, 2)]):
        preds[0, s, :3] = np.array(c) + 3
    m = D.check_tokens(x, preds)
    np.testing.assert_array_equal(m['conflicts'], [[0, 0, 1, 0]])
    np.testing.assert_array_equal(m['valid'], [[True, False, False, True]])
    assert m['coverage'] == 1. and m['solutions_total'] == 2
    bad = preds[:, 0].copy()
    bad[0, 1] = 1  # a non-color token
    m1 = D.check_tokens(x, bad)
    assert m1['conflicts'][0] == -1 and not m1['valid'][0]
    m2 = D.check_tokens(x, preds[:, 2])
    assert m2['conflicts'][0] == 1 and m2['coverage'] == 0.


def test_term_test_matches_its_definition_and_search_stops_obey_it():
    """The search's incremental term test equals accepts_terms on the returned boards; thresholds fitted by fit()."""
    rng = np.random.default_rng(6)
    n = 6
    A = random_graphs(rng, 40, n, .4)
    a = A[:, *G.pairs(n)].astype(np.float32)
    y = rng.integers(3, size=(40, n))
    x = np.stack([gram_encode(g) for g in A])
    lab = np.zeros_like(x)
    lab[:, :n] = y + 3
    for arm in G.ARMS:
        m, _ = G.fit(x, lab, n, arm=arm, updates=8, lr=10., lam=1e-3)
        assert G.accepts_terms(m, a.astype(np.float64), y).all()  # every training coloring passes its own thresholds
        res = G.solve(m, a[:10].astype(np.float64), samples=2, temperature=.5, cap=300, seed=0, stop='terms')
        ok = G.accepts_terms(m, np.repeat(a[:10], 2, 0).astype(np.float64), res['boards'].reshape(-1, n))
        np.testing.assert_array_equal(ok, res['stopped'].ravel())
