"""Tests of FullCounter. Ways it can fail: (1) an operator with arithmetic is silently counted 0 (unknown op, custom
op, fused attention); (2) a known cost formula is wrong (linear with bias, softmax, layer norm); (3) operators run by
autograd.grad are missed; (4) random draws enter the total; (5) the matmul-only total differs from FlopCounterMode.

  ../../.venv/bin/python -m pytest -q test_fullcount.py      (from baselines/flops, on a compute node)
"""
import pytest
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

from fullcount import COSTS, FullCounter


def count(fn):
    with FullCounter() as c:
        fn()
    return c


def test_linear_bias():
    x, w, b = torch.randn(3, 5), torch.randn(7, 5), torch.randn(7)
    c = count(lambda: F.linear(x, w, b)).classes()
    assert c['matmul'] == 2 * 3 * 5 * 7 and c['arith'] == 3 * 7 and c['transc'] == 0 and c['compare'] == 0


def test_softmax_five_per_element():
    x = torch.randn(4, 10)
    c = count(lambda: torch.softmax(x, -1)).classes()
    assert (c['arith'], c['transc'], c['compare']) == (3 * 40, 40, 40)  # the max pass is a comparison (not counted)


def test_layer_norm():
    x = torch.randn(6, 8)
    c = count(lambda: F.layer_norm(x, (8,), torch.ones(8), torch.zeros(8))).classes()
    assert c['arith'] == 7 * 48 and c['transc'] == 6


def test_autograd_grad_counted():
    x = torch.randn(4, 3, requires_grad=True)
    w = torch.randn(3, 3)

    def f():
        e = torch.tanh(x @ w).sum()
        torch.autograd.grad(e, x)
    c = count(f).classes()
    # forward: mm 2*4*3*3, tanh 12, sum 12; backward: tanh_backward 3*12, mm 2*4*3*3
    assert c['matmul'] == 2 * (2 * 4 * 3 * 3)
    assert c['transc'] == 12
    assert c['arith'] == 12 + 3 * 12


def test_rng_not_counted():
    c = count(lambda: torch.randn(100) * 2.0)
    assert c.classes()['rng'] == 100 and c.get_total_flops() == 100


def test_unknown_op_raises(monkeypatch):
    monkeypatch.delitem(COSTS, 'exp')
    with pytest.raises(RuntimeError, match='no rule'):
        count(lambda: torch.exp(torch.randn(3)))


def test_fused_attention_raises():
    q = torch.randn(1, 2, 5, 4)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION), pytest.raises(RuntimeError):
        count(lambda: F.scaled_dot_product_attention(q, q, q))


def test_math_attention_counted():
    q = torch.randn(1, 2, 5, 4)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        c = count(lambda: F.scaled_dot_product_attention(q, q, q)).classes()
    assert c['matmul'] == 2 * (2 * 2 * 5 * 5 * 4)  # QK^T and PV
    assert c['transc'] >= 2 * 25  # one exp per score


def test_matmul_only_matches_flopcountermode():
    m = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.SiLU(), torch.nn.Linear(32, 8))
    x = torch.randn(5, 16)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fc:
            m(x)
        c = count(lambda: m(x))
    assert c.matmul_only == fc.get_total_flops()
    assert c.get_total_flops() > fc.get_total_flops()


def test_integer_and_compare():
    a = torch.randint(0, 9, (10,))
    c = count(lambda: (a + 1) == 3)
    assert c.classes()['arith'] == 10 and c.classes()['compare'] == 10
    assert c.get_total_flops() == 10 and c.total_with_compare() == 20  # comparisons recorded, not counted


def test_inference_mode_attention_decomposed():
    # under inference_mode, scaled_dot_product_attention reaches the counter whole (and carries the seeded tag):
    # it must be decomposed and its matmuls counted, as FlopCounterMode does
    q = torch.randn(1, 2, 5, 4)
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
        with FlopCounterMode(display=False) as fc:
            F.scaled_dot_product_attention(q, q, q)
        c = count(lambda: F.scaled_dot_product_attention(q, q, q))
    assert c.classes()['matmul'] == fc.get_total_flops() == 2 * (2 * 2 * 5 * 5 * 4)
    assert c.classes()['rng'] == 0


def test_eval_dropout_free():
    x = torch.randn(10)
    with torch.inference_mode():
        c = count(lambda: F.dropout(x, 0.1, training=False))
    assert c.get_total_flops() == 0


def test_clip_counts_as_clamp():
    x = torch.randn(10)
    assert count(lambda: x.clone().clip_(-1, 1)).classes()['compare'] == 20


def test_bucketize_binary_search():
    b = torch.linspace(0, 1, 10_000)
    assert count(lambda: torch.bucketize(torch.rand(7), b)).classes()['compare'] == 7 * 14


def test_inplace_generator_is_rng():
    x = torch.empty(50)
    c = count(lambda: x.uniform_(-1, 1).erfinv_())
    assert c.classes()['rng'] == 50 and c.classes()['transc'] == 50


def test_scatter_add_counts_source_values():
    dst = torch.zeros(2, 1000)
    idx = torch.randint(0, 1000, (2, 18))
    c = count(lambda: dst.scatter_add_(1, idx, torch.ones(2, 18))).classes()
    assert c['arith'] == 36


def test_adam_step_counted():
    w = torch.nn.Parameter(torch.randn(10, 3))
    opt = torch.optim.Adam([w], lr=0.1, foreach=True)
    (w ** 2).sum().backward()
    c = count(opt.step)
    assert c.get_total_flops() > 10 * 30  # every Adam element update counted (moments, bias correction, sqrt, divide)
