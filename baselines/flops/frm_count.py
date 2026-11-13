"""FRM's inference operation count under the all-arithmetic convention (fullcount.py), beside the old matmul-only count.

The old counts are FlopCounterMode on one batch-1 network call, `model(z, tau, self_cond=z)` under the math SDPA kernel
(each campaign's `profile`), times the calls per puzzle: Sudoku-Extreme 943,069,184 x 4,097 and Maze-Unique
17,971,810,304 x 257 (recorded in each campaign's result files). A campaign is one training run folder of FRM made by
the adapters in baselines/frm/adapters/, with a copy of the authors' code in its `upstream/` folder. This script
counts, with FullCounter, on the paper's checkpoint (seed 0 of each campaign):
- call: one network call exactly as `profile` makes it (the old count's component);
- puzzle: one complete batch-1 `run_sampler` call with the campaign's selected sampler: every network call, the
  sampler's arithmetic around them (logit cap tanh, log-softmax, exp, the held-tau re-noise or the Euler/SDE update,
  the clue re-pin, the time lookup table) and the final argmax. Within it, the arithmetic of the network calls is
  separated from the sampler's by forward hooks (`network` and `sampler` in the record).
Per puzzle (full) = the `puzzle` count (one puzzle, nothing scaled). Per puzzle (old) = old flop per call x calls.
Matmul check: the `call` matmul class equals FlopCounterMode on the same call and the recorded per-call count; the
`puzzle` matmul class equals FlopCounterMode's count of the first call (no self-conditioning input yet, so without the
self-conditioning projection) plus (calls - 1) x the recorded per-call count. The old per-puzzle total charged the
first call the full per-call count, so it is that projection (a few Mflop) higher than the executed matmuls.
Totals follow fullcount.py: `total` counts matmul, arith and transc; `total_with_compare` adds the compare class.
Not counted: the scoring against the solution (it reads the labels) and the clue check.

Kernels: the counted path is eager PyTorch. DIT_USE_COMPILE=0 and FLM_DISABLE_FLASH=1 are set before the DiT module is
imported, as every scoring job of these campaigns set them (baselines/frm/score_unseen.sbatch, the campaigns' env.sh), and
the script asserts dit.USE_COMPILE and dit.HAS_FLASH_ATTN are False; the DiT's jit_deco is the identity, so no
TorchScript fusion runs. Attention runs under the math SDPA backend.

FRM's Python, on a GPU:
  python baselines/flops/frm_count.py --task sudoku|maze-unique --out baselines/runs/flops-full/frm-<benchmark>.json

The campaign folders are read from $FRM_REPRO (default: $THIRD_PARTY/frm-reproduction); the folder names below are
those of the campaigns that the paper's checkpoints come from.

Dimension names: b puzzles, l cells, v cell states.
"""

import os

os.environ['DIT_USE_COMPILE'] = '0'
os.environ['FLM_DISABLE_FLASH'] = '1'
os.environ['SPECFLOW_DISABLE_FLASH'] = '1'

import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from fullcount import FullCounter  # noqa: E402

THIRD_PARTY = os.environ.get('THIRD_PARTY', str(Path(__file__).resolve().parents[2] / 'third_party'))  # clones of the authors' code (baselines/README.md)
REASONING = Path(os.environ.get('FRM_REPRO', THIRD_PARTY + '/frm-reproduction'))
SUDOKU_ROOT = REASONING / 'outputs/frm-reproduction-20260918/weekend-20260918T134701Z/groups/primary'
MAZE_ROOT = REASONING / 'outputs/frm-maze-20260924/20260924T094000Z'


def sha(p):
    """SHA-256 hex digest of the file at `p`."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setup(task):
    """(root, checkpoint, data file, sampler kwargs, vocab, cells, old per call, old calls, benchmark, source)."""
    if task == 'sudoku':
        final = json.loads((REASONING / 'runs/260918--frm-reproduction/weekend/final-results.json').read_text())
        rec = [r for r in final['selected_results'] if r['group'] == 'primary' and r['seed'] == 0][0]
        it = 4096
        sampler = dict(kind='held_t', self_cond=True, steps=it, held_iters=it, noise_scale=0.0, t_hold=0.4, renoise=1.0, batch_size=1)
        return (SUDOKU_ROOT, Path(rec['checkpoint']), REPO / 'data/full-test/full-03.npz', sampler, 10, 81, 943_069_184, it + 1,
                'sudoku-extreme', "the Sudoku-Extreme campaign's profile record; calls = held_iters + 1 readout")
    root = MAZE_ROOT
    src = 'outputs/frm-maze-20260924/20260924T094000Z/summary.json of the Maze-Unique campaign (paper1000 lanes, test nfe 257)'
    sampler = dict(recipe='sde_selfcond', steps=256)  # the pre-declared point: sde_selfcond, NFE 257 = 256 steps + 1 readout
    return (root, root / 'runs/paper1000/s0/det/stageB/step_00100000.pt', root / 'data/paper1000/test.npz', sampler, 6, 900,
            17_971_810_304, 257, task, src)


def main(task, out):
    """Count one network call and one complete sampler run of FRM on `task` and write the record to `out` (JSON)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode
    root, ckpt, data, sampler_kw, vocab, cells, old_call_recorded, calls_old, bench, source = setup(task)
    sys.path.insert(0, str(root / 'upstream'))
    sys.path.insert(0, str(root / 'upstream/experiments/sudoku_extreme_train'))
    from flow_map_reasoning import SamplerConfig, build_sampler_config, run_sampler
    import flow_map_reasoning.dit as dit
    from evaluate import load_checkpoint_model
    assert not dit.USE_COMPILE and not dit.HAS_FLASH_ATTN, 'compiled or flash DiT: kernels the counter cannot see'
    model, meta = load_checkpoint_model(str(ckpt), 'cuda', use_ema=True)
    model.eval()
    if 'recipe' in sampler_kw:
        recipe = root / 'upstream/experiments/sudoku_extreme_train/samplers' / f"{sampler_kw['recipe']}.yaml"
        cfg, _ = build_sampler_config([str(recipe)], ['sampler.num_samples=1', 'sampler.batch_size=1'])
        cfg.steps = sampler_kw['steps']
    else:
        cfg = SamplerConfig(**sampler_kw)
    with np.load(data, allow_pickle=False) as z:
        if 'solutions' in z:
            sols, clues = z['solutions'][:1].astype(np.int64), z['clue_mask'][:1].astype(np.int64)
        else:
            from frm_reproduction import arrays  # noqa: F401  (Sudoku shards: the campaign's own reader)
            sols, clues = None, None
    if sols is None:
        sys.path.insert(0, str(root))
        from frm_reproduction import arrays
        sols, clues = arrays(data)
        sols, clues = sols[:1], clues[:1]
    assert sols.shape == (1, cells)
    params = sum(p.numel() for p in model.parameters())

    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        z0 = torch.zeros((1, cells, vocab), device='cuda')
        tau = torch.full((1,), 0.5, device='cuda')
        with torch.random.fork_rng(devices=[0]), FlopCounterMode(display=False) as fcm:
            model(z0, tau, self_cond=z0)
        old_call = int(fcm.get_total_flops())
        with torch.random.fork_rng(devices=[0]), FlopCounterMode(display=False) as fcm:
            model(z0, tau, self_cond=None)  # the sampler's first call: no self-conditioning buffer yet
        old_call_first = int(fcm.get_total_flops())
        with torch.random.fork_rng(devices=[0]), FullCounter() as fc:
            model(z0, tau, self_cond=z0)
        call = fc.summary()

        fc = FullCounter()
        net = dict(calls=0, total=0, matmul=0, start=None, start_mm=None)

        def pre(_m, _a, _k=None):
            net['calls'] += 1
            net['start'], net['start_mm'] = fc.get_total_flops(), fc.classes()['matmul']

        def post(_m, _a, _o):
            net['total'] += fc.get_total_flops() - net['start']
            net['matmul'] += fc.classes()['matmul'] - net['start_mm']
        h1, h2 = model.register_forward_pre_hook(pre), model.register_forward_hook(post)
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            with fc:
                pred = run_sampler(model, sols, clues, cfg, lut_a2g=meta['lut_a2g'], vocab_size=vocab, device='cuda')
        h1.remove()
        h2.remove()
        puzzle = fc.summary()
    assert net['calls'] == calls_old, (net['calls'], calls_old)
    assert np.array_equal(pred[clues.astype(bool)], sols[clues.astype(bool)])
    checks = dict(call_matmul_equals_flopcountermode=call['classes']['matmul'] == call['matmul_only_flopcountermode'] == old_call,
                  call_flopcountermode_equals_recorded=old_call == old_call_recorded,
                  puzzle_matmul_equals_calls_x_old_call=puzzle['classes']['matmul'] == calls_old * old_call,
                  puzzle_matmul_equals_first_plus_rest=puzzle['classes']['matmul'] == old_call_first + (calls_old - 1) * old_call,
                  network_matmul_equals_puzzle_matmul=net['matmul'] == puzzle['classes']['matmul'])
    per_puzzle_old = old_call_recorded * calls_old
    rec = dict(method='FRM', benchmark=bench, sampler=asdict(cfg), params=params,
               components=dict(call=call, puzzle=puzzle),
               puzzle_split=dict(network_calls=net['calls'], network_flop=net['total'], sampler_flop=puzzle['total'] - net['total'],
                                 network_flop_per_call_mean=net['total'] / net['calls']),
               old=dict(flop_per_call_recorded=old_call_recorded, flop_per_call_flopcountermode=old_call, calls=calls_old,
                        flop_first_call_flopcountermode=old_call_first,
                        per_puzzle=per_puzzle_old, source=source),
               ratio_call=call['total'] / old_call, per_puzzle_full=puzzle['total'], per_puzzle_ratio=puzzle['total'] / per_puzzle_old,
               per_puzzle_full_with_compare=puzzle['total_with_compare'], per_puzzle_ratio_with_compare=puzzle['total_with_compare'] / per_puzzle_old,
               checks=checks,
               formula='full = FullCounter total of one batch-1 run_sampler call (all calls, sampler arithmetic, argmax); '
                       'old = flop per call x calls (calls = sampler NFE, counted here by a forward pre-hook)',
               kernels=dict(DIT_USE_COMPILE=dit.USE_COMPILE, HAS_FLASH_ATTN=dit.HAS_FLASH_ATTN, sdpa='math', jit_deco='identity'),
               checkpoint=str(ckpt), checkpoint_sha256=sha(ckpt), data=str(data), data_row=0,
               script=str(Path(__file__).resolve()), script_sha256=sha(__file__), command=' '.join([sys.executable, *sys.argv]),
               slurm_job=os.environ.get('SLURM_JOB_ID'), host=os.uname().nodename, gpu=torch.cuda.get_device_name(),
               torch=torch.__version__, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k != 'components'}, indent=1))
    for k, s in rec['components'].items():
        print(k, s['total'], s['classes'])
    assert checks['call_matmul_equals_flopcountermode'] and checks['puzzle_matmul_equals_first_plus_rest'], checks


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=('sudoku', 'maze-unique'), required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    main(a.task, a.out)
