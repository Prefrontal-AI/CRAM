"""Score a timed TRMP lattice fit on the Sudoku-Extreme test split (outside the fit clock).

Inference is the paper's lattice evaluation (B.2.1, B.4): 64 independent stochastic chains per puzzle, a budget of
1,000 deep-supervision steps per puzzle, chains checked for acceptance or restart at every step, board conflict
threshold 0.8. The first chain on a solved board gives the answer, graded afterwards. Test: the 412,786 official test
puzzles outside dev10k (as our TRMP ACT row), against the paper's 1,000. bf16 autocast, as in training.
Predictions and per-puzzle step counts are written per shard (resumable) before any label is compared; the generator
of a shard is seeded with 20261112 + its first index.

  python trmp/lattice_score.py --run runs/trmp-lattice/sudoku-extreme-s0        # from baselines/
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from clock import write_json  # noqa: E402
import lattice as L  # noqa: E402
import lattice_fit as LF  # noqa: E402

RNG_BASE = 20261112


@dataclasses.dataclass
class Args:
    run: str = 'runs/trmp-lattice/sudoku-extreme-s0'  # fit folder (fit.json, model.pt); outputs go to <run>/score
    slots: int = 1024  # puzzles searched at once (each with 64 chains)
    shard: int = 16384  # puzzles per durable prediction shard
    test_limit: Optional[int] = None  # score only the first N test puzzles (smoke tests; the full run leaves it unset)
    budget: int = L.RECIPE['test_budget']  # deep-supervision steps per puzzle
    chains: int = L.RECIPE['test_chains']  # chains per puzzle
    check_every: int = L.RECIPE['test_check_every']  # steps between acceptance/restart checks
    compile: bool = True  # torch.compile the network forward


def load_test():
    f = np.load(LF.V1_DATA / 'full-test/full-all.npz')
    sol, clue = f['solutions'], f['clue_mask']
    dev = np.load(LF.V1_DATA / 'frm/dev10k/dev.npz')
    assert (dev['solutions'] == sol[:10000]).all(), 'dev10k is not full-all[:10000]'
    assert len(sol) == 422786
    x0, d = LF.lattice_of(sol[10000:], clue[10000:])
    return x0, d, 'full-all[10000:422786] (412,786 official test puzzles outside dev10k)'


def main(args: Args):
    run = Path(args.run)
    rec = json.loads((run / 'fit.json').read_text())
    out = run / 'score'
    out.mkdir(exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    net = L.LatticeNet().cuda().eval()
    net.load_state_dict(torch.load(run / 'model.pt', map_location='cuda', weights_only=True))
    model_sha = hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest()
    assert model_sha == rec['model_sha256'], 'model.pt differs from the fit record'
    fn = torch.compile(net) if args.compile else net
    name = 'test' if args.test_limit is None else f'test-first{args.test_limit}'
    if (out / f'{name}.json').exists():
        print(f'{out}/{name}.json exists', flush=True)
        return
    x0, d, desc = load_test()
    if args.test_limit is not None:
        x0, d, desc = x0[:args.test_limit], d[:args.test_limit], f'{desc}, first {args.test_limit} only'
    folder = out / name
    folder.mkdir(exist_ok=True)
    preds, steps, resets, seconds = [], [], [], 0.0
    for lo in range(0, len(x0), args.shard):
        path = folder / f'{lo:07d}.npz'
        if not path.exists():
            hi = min(lo + args.shard, len(x0))
            torch.cuda.synchronize()
            t = time.perf_counter()
            gen = torch.Generator(device='cuda').manual_seed(RNG_BASE + lo)
            p, s, r = L.chain_search(fn, x0[lo:hi].cuda(), chains=args.chains, budget=args.budget,
                                     check_every=args.check_every, slots=args.slots, amp=True, generator=gen)
            torch.cuda.synchronize()
            np.savez(path.with_suffix('.tmp.npz'), prediction=p.cpu().numpy().astype(np.int8),
                     steps=s.cpu().numpy().astype(np.int16), resets=r.cpu().numpy().astype(np.int32),
                     seconds=time.perf_counter() - t)
            os.replace(path.with_suffix('.tmp.npz'), path)
            print(f'  shard {lo}: {time.perf_counter() - t:.1f} s', flush=True)
        f = np.load(path)
        preds.append(f['prediction']); steps.append(f['steps']); resets.append(f['resets']); seconds += float(f['seconds'])
    pred, steps, resets = np.concatenate(preds), np.concatenate(steps).astype(np.int64), np.concatenate(resets)
    solved = (pred == d.numpy()).all(-1)
    timeout = (pred < 0).any(-1)
    np.save(out / f'{name}.solved.npy', solved)
    np.save(out / f'{name}.steps.npy', steps.astype(np.int16))
    res = dict(method='TRMP lattice (factor graph, 16k)', task='sudoku-extreme', split=desc, count=len(d),
               solved=int(solved.sum()), solve_rate=float(solved.mean()), timeouts=int(timeout.sum()),
               wrong=int((~timeout & ~solved).sum()), chains=args.chains, budget=args.budget,
               check_every=args.check_every, theta_cls=L.THETA_CLS_EVAL, temperature=L.TEMP, numerics='bf16 autocast',
               mean_steps=float(steps.mean()), max_steps=int(steps.max()), chain_steps_per_puzzle=float(steps.mean() * args.chains),
               mean_resets=float(resets.mean()), rng_base=RNG_BASE, slots=args.slots,
               inference_seconds_total=seconds, model_sha256=model_sha, fit_seconds=rec['fit_seconds'],
               training_seed=rec['config']['seed'], params=rec['params'], gpu=torch.cuda.get_device_name(),
               slurm_job=os.environ.get('SLURM_JOB_ID'), unsolved_indices=np.flatnonzero(~solved).tolist()[:1000],
               steps_file=f'{name}.steps.npy (steps run on each puzzle; chain steps = steps x chains)')
    write_json(out / f'{name}.json', res)
    print(f'TEST {res["solved"]}/{res["count"]} = {100 * res["solve_rate"]:.3f}% (timeouts {res["timeouts"]}, wrong '
          f'{res["wrong"]}, mean steps {res["mean_steps"]:.2f})', flush=True)


if __name__ == '__main__':
    main(tyro.cli(Args))
