"""Inference operation count (flop: floating-point operations) of EBR (Energy-Based Reasoner), by component.

Calls EBR's own loaders and model (github.com/wwillchen/ebr at 8c5271c: evaluators.common, dataset.common) on one test
puzzle. Settings: math attention, batch 1, eval mode, float32, uncompiled. Two counters:
- with --full: all arithmetic, by the operation-count convention of baselines/flops/fullcount.py (the paper's counts);
- without: the per-operator formulas of torch's FlopCounterMode (matrix multiplications and attention only;
  multiply-add = 2 flop).

Components, each for one sample (one trajectory) of one puzzle:
- segment: one evaluation segment of EBR's model (InferenceLossHead.forward): the input embedding, `cycles` descent
  microsteps (each an energy forward and dE/dz, no noise draw counted), the decode, the convergence logit;
- measure: evaluators.common.measure at a segment end: the energy and its gradient (forward + backward). The
  gradient-norm stopping rule (stop at |grad_M E| < tau, the N-Queens and Graph Coloring protocol) needs it at every
  segment end;
- energy: one energy forward at a state, which the lowest-energy pick among N samples needs at the answer segment.
The flop of an inference setting is assembled from these in ebr/report.py: per sample, D x segment, plus D x measure
under the gradient-norm stopping rule, plus one energy when N > 1.

Run from EBR's source tree, on CPU:
  python <repository>/baselines/ebr/flops.py <out.json> <checkpoint> <data_path> [--full]
With --full, the field `ops` holds each component's FullCounter summary (per class and per operator).
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import flop_registry

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'flops'))
from fullcount import FullCounter  # noqa: E402
from evaluators.common import eval_config, load_checkpoint, load_model, measure  # noqa: E402
from dataset.common import read_metadata  # noqa: E402


class Counter(TorchDispatchMode):
    """FlopCounterMode's arithmetic (torch.utils.flop_counter.flop_registry) without its module hooks, which fail
    inside autograd.grad (EBR takes dE/dz with autograd.grad in every microstep)."""

    def __init__(self):
        super().__init__()
        self.ops = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        packet = func._overloadpacket
        if packet not in flop_registry and not func.is_view and str(packet) in ('aten.linear', 'aten.matmul', 'aten.einsum', 'aten.bmm'):
            raise RuntimeError(f'{packet} reached the counter undecomposed')
        if packet in flop_registry:
            self.ops[str(packet)] = self.ops.get(str(packet), 0) + int(flop_registry[packet](*args, **kwargs, out_val=out))
        return out


FULL = '--full' in sys.argv  # all arithmetic (baselines/flops/fullcount.py); without it, the original matmul-only count
if FULL:
    sys.argv.remove('--full')


def count(fn):
    """Run `fn()` under the selected counter. Returns (total flop, per-operator breakdown)."""
    if FULL:
        with FullCounter() as c:
            fn()
        s = c.summary()
        return s['total'], s
    with Counter() as c:
        fn()
    return sum(c.ops.values()), c.ops


def main(out, checkpoint, data_path):
    """Count the components on the first test puzzle of `data_path` with the model in `checkpoint`; write `out` (JSON)."""
    dev = torch.device('cpu')
    ckpt = load_checkpoint(checkpoint, map_location=dev)
    config = eval_config(ckpt, device=dev)  # on CPU: energy_compile off, float32
    for key, value in dict(attn_backend='math', attn_nograd_backend='math', rope_impl='broadcast').items():
        config.arch.__pydantic_extra__[key] = value
    config.dataset.data_path = data_path
    meta = read_metadata(data_path, 'test')
    model, _ = load_model(ckpt, config, meta, device=dev, build_device=dev)
    inner = model.model.inner
    test = Path(data_path) / 'test'
    row = int(np.load(test / 'all__puzzle_indices.npy')[0])
    batch = {'inputs': torch.as_tensor(np.load(test / 'all__inputs.npy', mmap_mode='r')[row:row + 1].astype(np.int32)),
             'labels': torch.as_tensor(np.load(test / 'all__labels.npy', mmap_mode='r')[row:row + 1].astype(np.int64)),
             'puzzle_identifiers': torch.zeros(1, dtype=torch.int32)}
    arch = config.arch.__pydantic_extra__
    torch.manual_seed(0)
    with torch.inference_mode(False), torch.no_grad():  # under inference_mode aten.linear reaches the counter undecomposed
        carry = model.initial_carry(batch)
        if FULL:  # the prediction only: the model's segment and the argmax decode, not the loss head's losses and metrics
            seg, seg_ops = count(lambda: torch.argmax(model.model(carry=carry, batch=batch)[1]['logits'], dim=-1))
        else:
            seg, seg_ops = count(lambda: model(carry=carry, batch=batch, return_keys=['preds']))
        carry2, *_ = model(carry=model.initial_carry(batch), batch=batch, return_keys=['preds'])
        x = inner.input_embeddings(batch['inputs'])
        cos_sin = inner.cos_sin()
        z = carry2.inner_carry.z
        mea, mea_ops = count(lambda: measure(inner.level, z, x, cos_sin))
        v = inner.level.normalize_descent_state(z)
        ene, ene_ops = count(lambda: inner.level.energy(v, x, cos_sin))
        microstep, micro_ops = count(lambda: inner.level(z, x, cos_sin))
    rec = dict(checkpoint=checkpoint, data_path=data_path, seq_len=meta.seq_len, cycles=arch['cycles'],
               halt_max_steps=arch['halt_max_steps'], hidden_size=arch.get('hidden_size'), num_layers=arch.get('num_layers'),
               params=sum(p.numel() for p in model.parameters()),
               segment_flop=seg, measure_flop=mea, energy_flop=ene, microstep_flop=microstep,
               check_segment_vs_cycles_x_microstep=seg / (arch['cycles'] * microstep),
               ops=dict(segment=seg_ops, measure=mea_ops, energy=ene_ops, microstep=micro_ops),
               convention=('all arithmetic (baselines/flops/fullcount.py), math attention, batch 1, eval mode, float32, uncompiled' if FULL else
                           'FlopCounterMode formulas (flop_registry, dispatch mode), math attention, batch 1, eval mode, float32, uncompiled; multiply-add = 2 flop; '
                           'matmuls and attention only'), torch=torch.__version__)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k != 'ops'}, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
