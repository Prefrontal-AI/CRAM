"""One operation-counting convention for every method in the paper: count arithmetic, not only matrix products.

`FullCounter` is a drop-in replacement for torch.utils.flop_counter.FlopCounterMode (same context-manager use,
`get_total_flops()`, `get_flop_counts()`), written as a TorchDispatchMode so that it also sees the operators that
`torch.autograd.grad` runs (EBR differentiates its energy at every inference step). It classifies every aten operator
that reaches it, and raises on an operator it has no rule for, so no arithmetic is dropped without notice.

Convention (arithmetic only; the paper's appendix "Operation counts" states it; CRAM's kernels are counted under the
same rule through their transcriptions in cram_kernels.py):
- one flop per elementary arithmetic operation on one element: add, subtract, multiply, divide, negate, absolute value;
  a multiply-add counts 2 (class "arith");
- one flop per evaluation of exp, log, sqrt, rsqrt, sigmoid, tanh, erf, sin, cos (class "transc", so that they can be
  reweighted in a sensitivity check);
- matrix products: 2 flop per multiply-add (torch's flop_registry formulas), plus 1 per element for a fused bias add;
- composite operators count their elementary steps (softmax: max, subtract, exp, sum, divide = 4 arith + 1 transc per
  element; layer and RMS norms, activations and their backward formulas as listed in COSTS);
- integer arithmetic counts like floating-point arithmetic;
- comparisons, selections (where, masked_fill), min/max, argmax/argmin, clamps, ReLU-type thresholds and logical or
  bitwise mask operations are recorded (class "compare") but NOT counted;
- 0 for operators that only move, create, cast or index data (views, copies, cat, gather, embedding lookups, fills);
  a digit extraction written as a masked sum (sum(where(cols == a, x, 0)) = x[a]) is indexing (CRAM's transcriptions
  write it as indexing);
- random-number generation counts 0 per number drawn, for every method; the numbers drawn are recorded separately
  (class "rng", excluded from the total) for a sensitivity check. Arithmetic applied to a drawn number by the
  method's own code (scaling noise, a Gumbel transform) is ordinary arithmetic and counts.
Classes: matmul, arith, transc (counted); compare and rng (recorded, not counted). `summary()` also gives the total
with comparisons counted, for the sensitivity check.

Composite operators that reach the counter whole (under torch.inference_mode, CompositeImplicitAutograd operators such
as scaled_dot_product_attention are not decomposed before dispatch) are decomposed here, as FlopCounterMode does, and
their parts counted; one that cannot be decomposed and has no rule raises.

Fused attention kernels (flash, memory-efficient, cuDNN) raise: run counted code with the math SDPA backend, whose
decomposition the counter sees operator by operator.

Dimension names: none (counts only).
"""
import math
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import flop_registry

CLASSES = ('matmul', 'arith', 'transc')  # counted
COMPARE = 'compare'  # recorded, not counted
RNG = 'rng'  # numbers drawn, recorded but not counted


def _numel(x):
    return x.numel() if isinstance(x, torch.Tensor) else 1


def _first_tensor(xs):
    for x in tree_flatten(xs)[0]:
        if isinstance(x, torch.Tensor):
            return x
    return None


# Per-element costs. basis 'out': per element of the first output; 'in': per element of the first tensor input.
# A tuple (arith, transc, compare); extra per output element given as 'out_extra'.
def _c(arith=0, transc=0, compare=0, basis='out', out_extra=(0, 0, 0)):
    return dict(cost=(arith, transc, compare), basis=basis, out_extra=out_extra)


ZERO = _c()
COSTS = {}
for n in ('add', 'sub', 'rsub', 'mul', 'div', 'neg', 'abs', 'reciprocal', 'square', 'floor', 'ceil', 'round', 'trunc',
          'frac', 'fmod', 'remainder', 'floor_divide', 'copysign', 'xlogy_noop'):
    COSTS[n] = _c(arith=1)
for n in ('sign', 'sgn', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_not', 'bitwise_left_shift',
          'bitwise_right_shift', '__and__', '__or__', '__xor__', '__lshift__', '__rshift__', '__ilshift__', '__irshift__'):
    COSTS[n] = _c(compare=1)  # sign tests and mask logic
for n in ('exp', 'exp2', 'expm1', 'log', 'log2', 'log10', 'log1p', 'sqrt', 'rsqrt', 'sin', 'cos', 'tan', 'tanh',
          'sinh', 'cosh', 'asin', 'acos', 'atan', 'atan2', 'erf', 'erfc', 'erfinv', 'sigmoid', 'lgamma', 'digamma',
          'pow'):  # pow: one evaluation (x ** 2 would be 1 arith; counted as transc, the larger class, for every method)
    COSTS[n] = _c(transc=1)
for n in ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'isnan', 'isinf', 'isfinite', 'isneginf', 'isposinf', 'logical_and',
          'logical_or', 'logical_xor', 'logical_not', 'maximum', 'minimum', 'fmax', 'fmin', 'clamp_min', 'clamp_max',
          'where', 'masked_fill', 'relu', 'threshold', 'threshold_backward', 'heaviside', 'signbit'):
    COSTS[n] = _c(compare=1)
COSTS.update({
    'clamp': _c(compare=2), 'clip': _c(compare=2),  # clip = clamp (lower and upper bound)
    'hardtanh': _c(compare=2), 'hardtanh_backward': _c(compare=3),
    'leaky_relu': _c(arith=1, compare=1), 'leaky_relu_backward': _c(arith=1, compare=1),
    'lerp': _c(arith=3), 'addcmul': _c(arith=3), 'addcdiv': _c(arith=3), 'xlogy': _c(arith=1, transc=1),
    'logit': _c(arith=2, transc=1),
    # activations: elementary steps of their definitions
    'silu': _c(arith=1, transc=1),                       # x * sigmoid(x)
    'silu_backward': _c(arith=5, transc=1),              # g * s * (1 + x (1 - s)), s = sigmoid(x)
    'sigmoid_backward': _c(arith=3),                     # g * y * (1 - y)
    'tanh_backward': _c(arith=3),                        # g * (1 - y * y)
    'gelu': _c(arith=4, transc=1),                       # 0.5 x (1 + erf(x / sqrt 2)); tanh form: same order
    'gelu_backward': _c(arith=8, transc=2),              # erf and exp terms
    'softplus': _c(arith=2, transc=2, compare=1),        # log1p(exp(beta x)) / beta, threshold test
    'logaddexp': _c(arith=3, transc=2, compare=1),       # max(a, b) + log1p(exp(-|a - b|)): sub, abs, add; exp, log1p; max
    'softplus_backward': _c(arith=3, transc=1, compare=1),
    'mish': _c(arith=1, transc=3), 'elu': _c(arith=2, transc=1, compare=1),
    'hardswish': _c(arith=3, compare=2), 'hardsigmoid': _c(arith=2, compare=2),
    'log_sigmoid_forward': _c(arith=3, transc=2, compare=1),
    'glu': _c(arith=1, transc=1),
    # reductions, per input element (+ per output element)
    'sum': _c(arith=1, basis='in'), 'nansum': _c(arith=1, compare=1, basis='in'), 'prod': _c(arith=1, basis='in'),
    'mean': _c(arith=1, basis='in', out_extra=(1, 0, 0)),
    'amax': _c(compare=1, basis='in'), 'amin': _c(compare=1, basis='in'), 'max': _c(compare=1, basis='in'),
    'min': _c(compare=1, basis='in'), 'argmax': _c(compare=1, basis='in'), 'argmin': _c(compare=1, basis='in'),
    'aminmax': _c(compare=2, basis='in'), 'any': _c(compare=1, basis='in'), 'all': _c(compare=1, basis='in'),
    'count_nonzero': _c(arith=1, compare=1, basis='in'), 'nonzero': _c(compare=1, basis='in'),
    'cumsum': _c(arith=1, basis='in'), 'cumprod': _c(arith=1, basis='in'), 'cummax': _c(compare=1, basis='in'),
    'logcumsumexp': _c(arith=3, transc=2, basis='in'),
    'logsumexp': _c(arith=3, transc=1, compare=1, basis='in', out_extra=(1, 1, 0)),  # max, sub, exp, sum; log, add
    '_softmax': _c(arith=3, transc=1, compare=1, basis='in'),                      # max, sub, exp, sum, divide
    '_safe_softmax': _c(arith=3, transc=1, compare=2, basis='in'),                 # softmax + fully-masked-row test
    '_log_softmax': _c(arith=3, transc=1, compare=1, basis='in'),  # max, sub, exp, sum, sub (+ log per row)
    '_softmax_backward_data': _c(arith=4, basis='in'),                             # y (g - sum(g y))
    '_log_softmax_backward_data': _c(arith=3, transc=1, basis='in'),               # g - exp(y) sum(g)
    'var': _c(arith=4, basis='in', out_extra=(1, 0, 0)), 'var_mean': _c(arith=4, basis='in', out_extra=(1, 0, 0)),
    'std': _c(arith=4, basis='in', out_extra=(1, 1, 0)), 'std_mean': _c(arith=4, basis='in', out_extra=(1, 1, 0)),
    'linalg_vector_norm': _c(arith=2, basis='in', out_extra=(0, 1, 0)), 'norm': _c(arith=2, basis='in', out_extra=(0, 1, 0)),
    'native_layer_norm': _c(arith=7, basis='in'),          # mean, sub, square, sum, scale, affine mul + add (+ rsqrt per row)
    'native_layer_norm_backward': _c(arith=12, basis='in'),
    '_fused_rms_norm': _c(arith=4, basis='in'),             # square, sum, scale, weight (+ rsqrt per row)
    '_fused_rms_norm_backward': _c(arith=9, basis='in'),
    'native_group_norm': _c(arith=7, basis='in'), 'native_batch_norm': _c(arith=7, basis='in'),
    '_native_batch_norm_legit_no_training': _c(arith=2, basis='in'),
    'embedding_dense_backward': _c(arith=1, basis='in'),    # scatter-add of the incoming gradient
    'scatter_add': _c(arith=1, basis='src'), 'scatter_reduce': _c(arith=1, basis='src'), 'index_add': _c(arith=1, basis='src'),  # one add per source value
    'bincount': _c(arith=1, basis='in'), 'histc': _c(arith=2, compare=1, basis='in'),
    'mse_loss': _c(arith=3, basis='in'), 'nll_loss_forward': _c(arith=1, basis='out'),
    'dot': _c(arith=2, basis='in'), 'vdot': _c(arith=2, basis='in'),
    'trace': _c(arith=1, basis='out'),
})
# Data movement, creation, casts, views and indexing: 0.
for n in ('_to_copy', 'copy', 'clone', 'contiguous', 'cat', 'stack', 'index', 'index_select', 'gather', 'scatter',
          'slice_scatter', 'select_scatter', 'diagonal_scatter', 'as_strided_scatter', 'embedding', '_unsafe_index',
          'masked_select', 'repeat', 'repeat_interleave', 'roll', 'flip', 'constant_pad_nd', 'pad', 'reflection_pad1d',
          'reflection_pad2d', 'replication_pad1d', 'replication_pad2d', '_local_scalar_dense', 'lift_fresh',
          'lift_fresh_copy', 'detach', 'alias', 'set', 'resize', 'fill', 'zero', 'empty', 'empty_like', 'empty_strided',
          'new_empty', 'new_empty_strided', 'zeros', 'zeros_like', 'new_zeros', 'ones', 'ones_like', 'new_ones', 'full',
          'full_like', 'new_full', 'arange', 'linspace', 'scalar_tensor', 'eye', 'tril', 'triu', 'triu_indices', 'tril_indices', 't', 'transpose',
          'permute', 'view', '_unsafe_view', 'reshape', '_reshape_alias', 'expand', 'expand_as', 'squeeze', 'unsqueeze',
          'select', 'slice', 'narrow', 'split', 'split_with_sizes', 'unbind', 'chunk', 'unfold', 'as_strided',
          'view_as_real', 'view_as_complex', 'movedim', 'unflatten', 'flatten', 'diagonal', '_assert_async',
          '_assert_tensor_metadata', 'sym_size', 'sym_stride', 'sym_numel', 'sym_storage_offset', 'is_nonzero',
          'equal_noop', '_unsafe_view_copy', 'index_put', 'masked_scatter', 'col2im', 'im2col', 'one_hot',
          'stride', 'size', 'dim', 'is_same_size', 'record_stream', '_has_compatible_shallow_copy_type',
          'item', 'clone_', 'resize_as', 'prim_noop', 'alias_copy', 'detach_copy', 'view_copy', 'permute_copy',
          'expand_copy', 'slice_copy', 'select_copy', 'transpose_copy', 'unsqueeze_copy', 'squeeze_copy', 't_copy',
          'slice_backward', 'select_backward', 'diagonal_backward', 'unfold_backward', 'as_strided_backward',
          'index_select_backward', 'embedding_backward_noop', 'new_empty_like', 'resolve_conj', 'resolve_neg',
          'conj', 'conj_physical_noop', 'pin_memory', '_pin_memory', 'to', 'type_as'):
    COSTS.setdefault(n, ZERO)

MATMUL_OPS = {'mm', 'bmm', 'addmm', 'baddbmm', 'addbmm', 'convolution', '_convolution', 'convolution_backward', 'mv',
              'addmv', 'addr', 'outer', 'ger'}
RNG_OPS = {'rand', 'randn', 'randint', 'randperm', 'rand_like', 'randn_like', 'randint_like', 'normal', 'uniform',
           'bernoulli', 'exponential', 'geometric', 'cauchy', 'log_normal', 'random', 'multinomial', 'native_dropout',
           '_fused_dropout', 'poisson'}
FUSED_ATTENTION = {'_scaled_dot_product_flash_attention', '_scaled_dot_product_efficient_attention',
                   '_scaled_dot_product_cudnn_attention', '_flash_attention_forward', '_efficient_attention_forward',
                   '_scaled_dot_product_flash_attention_for_cpu', '_scaled_dot_product_fused_attention_overrideable'}


def _name(func):
    n = func._overloadpacket.__name__
    if n.endswith('_') and not n.endswith('__') and (n[:-1] in COSTS or n[:-1] in MATMUL_OPS or n[:-1] in RNG_OPS):
        n = n[:-1]  # in-place form costs the same (uniform_, normal_, bernoulli_: generators)
    return n


class FullCounter(TorchDispatchMode):
    """Counts every aten operator under the module docstring's convention. Interface of FlopCounterMode."""

    def __init__(self, display=False, depth=None, mods=None, custom_mapping=None):
        super().__init__()
        self.by_op = defaultdict(lambda: defaultdict(int))  # op -> class -> flop
        self.matmul_only = 0  # FlopCounterMode's own total (matmuls and attention), for comparison with old counts

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = _name(func)
        if func.namespace == 'profiler':  # record_function bookkeeping: no arithmetic
            return func(*args, **kwargs)
        if func.namespace == 'aten' and name.startswith('_foreach_'):  # list form of an elementwise op (optimizers)
            out = func(*args, **kwargs)
            self._count_foreach(name, args)
            return out
        if (func.namespace == 'aten' and not func.is_view and name not in COSTS and name not in MATMUL_OPS
                and name not in RNG_OPS and name not in FUSED_ATTENTION and name not in ('linear', 'index_put')
                and func._overloadpacket not in flop_registry):
            with self:  # a composite operator reached the counter whole: count its parts (as FlopCounterMode does)
                r = func.decompose(*args, **kwargs)
            if r is not NotImplemented:
                return r
        out = func(*args, **kwargs)
        self._count(func, args, kwargs, out)
        return out

    def _count_foreach(self, name, args):
        """_foreach_<op>(list, ...): the rule of <op> on every tensor of the first list (a reduction such as
        _foreach_norm: its per-element rule plus its per-output term once per tensor)."""
        base = name[len('_foreach_'):].rstrip('_')
        base = {'pow_scalar': 'pow'}.get(base, base)
        rule = COSTS.get(base)
        if base in ('copy', 'zero', 'clone'):
            return
        if rule is None:
            raise RuntimeError(f'FullCounter: no rule for aten::{name} (base op {base})')
        tensors = [t for t in args[0] if isinstance(t, torch.Tensor)] if isinstance(args[0], (list, tuple)) else [args[0]]
        n = sum(t.numel() for t in tensors)
        a, t, c = rule['cost']
        ea, et, ec = rule['out_extra']
        k = len(tensors) if rule['basis'] == 'in' else n  # per-output term: one output per tensor for a reduction
        self._add(name, 'arith', a * n + ea * k)
        self._add(name, 'transc', t * n + et * k)
        self._add(name, 'compare', c * n + ec * k)

    def _add(self, op, cls, n):
        if n:
            self.by_op[op][cls] += int(n)

    def _count(self, func, args, kwargs, out):
        packet = func._overloadpacket
        ns = func.namespace
        name = _name(func)
        if ns not in ('aten', 'prims'):
            raise RuntimeError(f'FullCounter: custom operator {ns}::{name} has no rule (count its arithmetic analytically '
                               f'or run the reference path)')
        if name in FUSED_ATTENTION:
            raise RuntimeError(f'FullCounter: fused attention {name}; run with the math SDPA backend')
        if name in RNG_OPS:
            if name == 'native_dropout' and not (args[2] if len(args) > 2 else kwargs.get('train')):
                return  # evaluation: identity
            self._add(name, RNG, _numel(_first_tensor(out)))
            if name == 'multinomial':
                self._add(name, 'arith', _numel(args[0]))  # cumulative sum of the probabilities
            if name == 'native_dropout':
                self._add(name, 'arith', _numel(args[0]))  # scale of the kept values
            return
        if torch.Tag.nondeterministic_seeded in func.tags:
            raise RuntimeError(f'FullCounter: seeded operator {name} is not a known generator; add a rule')
        if packet in flop_registry or name in MATMUL_OPS:
            if packet not in flop_registry:
                raise RuntimeError(f'FullCounter: matmul-class operator {name} without a flop_registry formula')
            f = int(flop_registry[packet](*args, **kwargs, out_val=out))
            self.matmul_only += f
            self._add(name, 'matmul', f)
            if name in ('addmm', 'baddbmm', 'addbmm', 'addmv', 'addr'):
                self._add(name, 'arith', _numel(_first_tensor(out)))  # bias / input add (beta = alpha = 1 assumed)
            if name == 'convolution' and len(args) > 2 and isinstance(args[2], torch.Tensor):
                self._add(name, 'arith', _numel(_first_tensor(out)))
            return
        if name in ('linear',):  # reaches the counter undecomposed under inference_mode
            x, w = args[0], args[1]
            m = x.numel() // x.shape[-1]
            f = 2 * m * w.shape[0] * w.shape[1]
            self.matmul_only += f
            self._add(name, 'matmul', f)
            if len(args) > 2 and args[2] is not None:
                self._add(name, 'arith', m * w.shape[0])
            return
        if func.is_view:
            return
        if name == 'index_put' and (kwargs.get('accumulate') or (len(args) > 3 and args[3])):
            self._add(name, 'arith', _numel(args[2]))  # accumulate: one add per value
            return
        if name in ('bucketize', 'searchsorted'):  # binary search: ceil(log2(n + 1)) comparisons per looked-up element
            bounds = args[1] if name == 'bucketize' else args[0]
            n = bounds.shape[-1] if isinstance(bounds, torch.Tensor) else 1
            self._add(name, 'compare', _numel(_first_tensor(out)) * math.ceil(math.log2(n + 1)))
            return
        if name in ('sort', 'argsort', 'topk', 'kthvalue', 'median', 'msort'):  # comparisons only (not counted)
            x = _first_tensor(args)
            dim = args[1] if len(args) > 1 and isinstance(args[1], int) and name != 'topk' else kwargs.get('dim', -1)
            size = x.shape[dim] if x.dim() else 1
            self._add(name, 'compare', x.numel() * max(1, math.ceil(math.log2(max(size, 2)))))
            return
        if name == 'native_batch_norm' and not (args[5] if len(args) > 5 else kwargs.get('training', True)):
            # evaluation with running statistics (CPU eager reaches here, not the no-training operator): x * scale + shift
            name = '_native_batch_norm_legit_no_training'
        rule = COSTS.get(name)
        if rule is None:
            raise RuntimeError(f'FullCounter: no rule for aten::{name} ({func}); add it to COSTS with its elementary cost')
        a, t, c = rule['cost']
        if not (a or t or c or any(rule['out_extra'])):
            return
        o = _first_tensor(out)
        if rule['basis'] == 'in':
            base = _numel(_first_tensor(args))
        elif rule['basis'] == 'src':  # scatter_add(self, dim, index, src), index_add(self, dim, index, source)
            base = _numel(args[3] if len(args) > 3 else kwargs.get('src', kwargs.get('source')))
        else:
            base = _numel(o)
        nout = _numel(o)
        ea, et, ec = rule['out_extra']
        if name in ('native_layer_norm', '_fused_rms_norm', 'native_group_norm', 'native_batch_norm',
                    '_log_softmax'):
            x = _first_tensor(args)
            rows = x.numel() // max(1, x.shape[-1]) if x.dim() else 1
            self._add(name, 'transc', rows)  # one rsqrt (norms) or log (log-softmax) per row
        self._add(name, 'arith', a * base + ea * nout)
        self._add(name, 'transc', t * base + et * nout)
        self._add(name, 'compare', c * base + ec * nout)

    # ---- results ----
    def classes(self):
        tot = defaultdict(int)
        for op, d in self.by_op.items():
            for cls, n in d.items():
                tot[cls] += n
        return {k: int(tot.get(k, 0)) for k in (*CLASSES, COMPARE, RNG)}

    def get_total_flops(self):
        c = self.classes()
        return sum(c[k] for k in CLASSES)

    def total_with_compare(self):
        """Sensitivity check: comparisons, selections and mask logic counted too."""
        return self.get_total_flops() + self.classes()[COMPARE]

    def get_flop_counts(self):
        return {'Global': {op: sum(n for cls, n in d.items() if cls in CLASSES) for op, d in self.by_op.items()}}

    def summary(self):
        """JSON-ready record: total, per class, per operator, FlopCounterMode's matmul-only total."""
        return dict(total=self.get_total_flops(), total_with_compare=self.total_with_compare(), classes=self.classes(), matmul_only_flopcountermode=int(self.matmul_only),
                    by_op={op: dict(d) for op, d in sorted(self.by_op.items())}, convention=CONVENTION)


CONVENTION = ('baselines/flops/fullcount.py FullCounter: arithmetic only; 1 flop per elementary arithmetic operation per element '
              '(add, sub, mul, div, neg, abs; exp/log/sqrt/sigmoid/tanh/erf 1 each), multiply-add 2, matrix products by '
              'flop_registry plus bias adds; integer arithmetic counted; comparisons, selections, min/max, argmax and '
              'mask logic recorded but not counted; data movement, casts, indexing and random-number generation 0')


def weighted(classes, transc_weight=1, compare_weight=0, rng_weight=0):
    """Total under a sensitivity reweighting of transcendental evaluations, comparisons and random numbers."""
    return (classes['matmul'] + classes['arith'] + transc_weight * classes['transc'] + compare_weight * classes[COMPARE]
            + rng_weight * classes[RNG])
