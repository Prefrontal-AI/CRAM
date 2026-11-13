"""Parameters and inference flop per fixed-point iteration of FPRM on Maze-Hard, with the authors' code (baselines/fprm/common.py).

FullCounter (baselines/flops/fullcount.py) on the authors' eval-mode model with their released Maze-Hard checkpoint, for
one batch-1 test maze, uncompiled, under bf16 autocast as their evaluation runs it, with SDPA's math kernel so that the
counter sees the attention arithmetic. One call model(carry, batch) is one fixed-point iteration: the input embedding,
the two-block transformer with its depthwise convolution, the damped update with the residue test, and both heads.
Every iteration runs the same operators, so the count is taken on the first 3 and checked equal (the first
iteration differs in uncounted operators only: it creates the state that later iterations select). FlopCounterMode on the
same call must equal FullCounter's matmul class. The decode is the argmax of the final logits, once per maze.
The flop per maze is setup + steps x iteration + decode, with the step counts of baselines/fprm/score.py.

Run (one GPU): cd $FPRM_SRC && .venv/bin/python <repo>/baselines/flops/fprm_count.py <out.json>
"""
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile
from torch.utils.flop_counter import FlopCounterMode

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / 'fprm'))
from common import load, sha256, test_batches  # noqa: E402
from fullcount import CONVENTION, FullCounter  # noqa: E402

STEPS = 3


def main(out):
    torch.manual_seed(0)
    model, meta, rec = load(batch_size=1)
    batch, real = next(iter(test_batches(1)))
    assert real == 1
    steps, old = [], []
    # no_grad, not inference_mode: composite operators must decompose before they reach the counter (as hrm_count.py)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            with FullCounter() as c, torch.device('cuda'):
                carry = model.initial_carry(batch)
            setup = c.summary()
            for _ in range(STEPS):
                with FullCounter() as c:
                    carry, outputs = model(carry=carry, batch=batch)
                    done = bool(carry.halted.all())
                steps.append(c.summary())
                assert not done, 'halted within the counted iterations'
            with FullCounter() as c:
                preds = outputs['logits'].argmax(dim=-1)
            decode = c.summary()
        for _ in range(STEPS):
            with FlopCounterMode(display=False) as fc:
                carry, outputs = model(carry=carry, batch=batch)
            old.append(int(fc.get_total_flops()))
    kernels = sorted({e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA})
    suspect = [k for k in kernels if any(t in k.lower() for t in ('triton', 'flash', 'fmha', 'attention', 'liger', 'apex'))]
    assert not suspect, f'kernels invisible to FullCounter: {suspect}'
    step = steps[1]
    counted = lambda s: (s['total'], {k: s['classes'][k] for k in ('matmul', 'arith', 'transc')})
    # the first iteration builds the state instead of selecting it (uncounted operators); the counted classes must agree
    assert all(counted(s) == counted(step) for s in steps), [counted(s) for s in steps]
    assert all(s['by_op'] == step['by_op'] for s in steps[1:])
    first_only = {op: [steps[0]['by_op'].get(op), step['by_op'].get(op)] for op in set(steps[0]['by_op']) | set(step['by_op']) if steps[0]['by_op'].get(op) != step['by_op'].get(op)}
    assert len(set(old)) == 1 and step['classes']['matmul'] == old[0], (step['classes'], old)
    rec.update(method='FPRM', benchmark='maze-hard', cwd=str(Path.cwd()), command=' '.join([sys.executable] + sys.argv),
               slurm_job=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
               components=dict(iteration=dict(full=step, old_matmul_only=old[0], ratio=step['total'] / old[0],
                                              note='one fixed-point iteration = one model(carry, batch) call; every iteration identical'),
                               decode=dict(full=decode, note='argmax over the vocabulary of the final logits, once per maze'),
                               setup=dict(full=setup, note='initial carry (allocation only)')),
               flop_per_iteration=step['total'], flop_per_iteration_with_compare=step['total_with_compare'], flop_per_iteration_old=old[0],
               first_iteration_operator_differences=first_only, flop_decode=decode['total'], flop_setup=setup['total'], rng_per_iteration=step['classes'].get('rng', 0),
               flop_per_puzzle_formula='setup + steps x iteration + decode; steps from baselines/fprm/score.py',
               cuda_kernels=kernels, attention_counted='SDPA math kernel during counting only',
               convention=CONVENTION + '; batch 1, eval mode, uncompiled, bf16 autocast',
               script=str(HERE), script_sha256=sha256(HERE), fullcount_sha256=sha256(HERE.parent / 'fullcount.py'),
               common_sha256=sha256(HERE.parents[1] / 'fprm' / 'common.py'),
               created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k not in ('components', 'cuda_kernels', 'params_by_tensor')}, indent=1), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
