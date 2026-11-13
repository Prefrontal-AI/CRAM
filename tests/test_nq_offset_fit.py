"""Tests of the dense pair-matrix map of experiments/nqueens_offset.py: fit(dense=True) must fit the same objective
with the same optimizer, on the same rows, as the index route (dense=False). The dense map builds the same pair matrix
in another way and is the route of the reported fit times.

Ways the code can fail, and the test that catches each:
- dense_map puts a parameter at the wrong (a, s) entry, or drops the transposed half S[j, i] = W_p^T  -> test_dense_map_equals_index_route
- dense_map gives the ising arm entries other than queen-queen                                         -> test_dense_map_equals_index_route
- the dense route builds the unary costs B differently (table b, or ising [0, h])                      -> test_dense_map_equals_index_route
- the dense route draws from the training generator differently (mpl masks), or changes the update   -> test_dense_fit_matches_index_fit
"""
import math

import numpy as np
import pytest
import torch

import nqueens_offset as O


def random_task(n, arm, dense, seed=0):
    """An OffsetTask of one arm on an n x n board with random parameters A and b."""
    y = torch.zeros(1, n * n, dtype=torch.long)
    task = O.OffsetTask(torch.zeros(1, n * n, dtype=torch.bool), y, n, arm, dense=dense)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        task.A.copy_(torch.randn(task.A.shape, generator=g))
        task.b.copy_(torch.randn(task.b.shape, generator=g))
    return task


def random_rows(n, rows, labels, seed=0):
    """GRAM-style tokens (1 empty, 2 queen): `rows` rows drawn from `labels` random label boards, each input a random
    subset of its label's queens. Labels need not be valid boards for the fit's arithmetic."""
    rng = np.random.default_rng(seed)
    lab = rng.random((labels, n * n)) < 0.3
    lab = lab[rng.integers(0, labels, rows)]
    inp = lab & (rng.random(lab.shape) < 0.4)
    return np.where(inp, 2, 1), np.where(lab, 2, 1)


@pytest.mark.parametrize('arm', ['ising', 'table'])
@pytest.mark.parametrize('n', [4, 5])
def test_dense_map_equals_index_route(n, arm):
    index, dense = random_task(n, arm, False), random_task(n, arm, True)
    index.begin()
    dense.begin()
    assert torch.equal(index.M, dense.M)  # each entry of M is one parameter or zero: exact
    assert torch.equal(index.B, dense.B)


@pytest.mark.parametrize('method,knob', [('pl', None), ('mpl', 0.3)])
@pytest.mark.parametrize('arm', ['ising', 'table'])
def test_dense_fit_matches_index_fit(arm, method, knob, monkeypatch):
    """A short fit (500 updates, one divergence check) by both routes ends at the same parameters and the same
    inference-ready model (float32 sums in another order: close, not equal)."""
    monkeypatch.setattr(O, 'UPDATES', 500)
    n = 5
    inputs, labels = random_rows(n, 300, 9)
    cfg = dict(method=method, arm=arm, lr=3.0, knob=knob, lam=0.0, precond=True)
    ref, _ = O.fit(cfg, inputs, labels, n, seed=0, device='cpu', dense=False)
    new, info = O.fit(cfg, inputs, labels, n, seed=0, device='cpu', dense=True)
    assert not info['diverged'] and math.isfinite(new['tau'])
    scale = float(ref['A'].abs().max())
    assert scale > 0.1  # the fit moved the parameters
    np.testing.assert_allclose(new['A'].numpy(), ref['A'].numpy(), rtol=0, atol=1e-5 * scale)
    np.testing.assert_allclose(new['b'].numpy(), ref['b'].numpy(), rtol=0, atol=1e-5 * scale)
    np.testing.assert_allclose([new[k] for k in ('tau', 'unit', 'delta')], [ref[k] for k in ('tau', 'unit', 'delta')], rtol=1e-5, atol=1e-5)
