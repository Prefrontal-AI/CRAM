"""Compute sweep of the TRMP lattice model ("TRMP-fast", baselines/trmp/lattice.py) at several inference settings.

The model, its fit and its test search are the reimplementation in baselines/trmp/ (lattice.py, lattice_fit.py,
lattice_score.py); TRMP has no public code.

Each setting (chains K, budget B) is one real run of the Table-2 test search, lattice.chain_search, with K chains per
puzzle and at most B steps per puzzle, checked every step, board conflict threshold 0.8, slots 1,024, bf16 autocast,
on the first 10,000 puzzles of lattice_score.py's test set (full-all rows 10,000-19,999, the puzzles of the compute
sweep of the other baselines). The generator is seeded with 20261112 + 0, as shard 0 of lattice_score.py, so the setting K = 64, B = 1,000 should
reproduce the first 10,000 puzzles of the Table-2 shard puzzle by puzzle (checked when that shard exists).

Operation count (the convention of baselines/flops/fullcount.py): the arithmetic of chain_search does not depend on the data, so the
count of one puzzle with K chains is a_K + b_K x steps. a_K and b_K are measured by FullCounter over chain_search on dev
puzzles one at a time (slots = 1, budget 100 and 1, 2, 3 for different step counts), fitted on two rows and checked
exactly on the others; written to <out>/count-k<K>.json. Every chain, every projection (removal, conflict and solved
tests, commit) and the slot bookkeeping are inside the counted region.

Outputs per setting: <out>/k<K>-b<B>.npz (prediction int8 (p, v) with -1 for a timeout, steps, resets) and
<out>/k<K>-b<B>.json, written last; a setting whose json exists is skipped, so a requeued job redoes only unfinished ones.

Run from baselines/ on one GPU (sweep/trmp_lattice.sbatch):
  ../.venv/bin/python sweep/trmp_lattice_sweep.py --run runs/trmp-lattice/sudoku-extreme-s1 --out runs/compute-sweep/trmp-lattice/sudoku-extreme-s1
Dimensions: p puzzles, v cells.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tyro

HERE = Path(__file__).resolve().parent
BASELINES = HERE.parent
sys.path.insert(0, str(BASELINES))
sys.path.insert(0, str(BASELINES / 'trmp'))
sys.path.insert(0, str(BASELINES / 'flops'))
from clock import write_json  # noqa: E402
from fullcount import CONVENTION, FullCounter  # noqa: E402
import lattice as L  # noqa: E402
import lattice_fit as LF  # noqa: E402
import lattice_score as LS  # noqa: E402

BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000)  # at 64 chains (the step caps 1, 2, 4, ... of sweep/trmp_caps.py, then the Table-2 budget)
CHAINS = (1, 2, 4, 8, 16, 32)  # at budget 1,000 (the rollout counts K = 1, 2, 4, ..., 32 of sweep/trmp_sweep.py; K = 64 is above)
GRID = tuple((64, b) for b in BUDGETS) + tuple((k, 1000) for k in CHAINS)
N = 10000


@dataclasses.dataclass
class Args:
    run: str = 'runs/trmp-lattice/sudoku-extreme-s1'  # fit folder (fit.json, model.pt)
    out: str = 'runs/compute-sweep/trmp-lattice/sudoku-extreme-s1'  # output folder of this seed
    settings: tuple[str, ...] = ()  # subset of the grid as K:B (default: the whole grid, cheapest first)
    slots: int = 1024  # puzzles searched at once, as lattice_score.py
    count_puzzles: int = 4  # dev puzzles counted one at a time per chain count
    compile: bool = True  # torch.compile the network forward, as lattice_score.py


def count(net, K, theta, puzzles):
    """Operation count of chain_search per puzzle with K chains: (a, b, rows), count = a + b x steps.

    net: the uncompiled LatticeNet; theta: the conflict threshold of the fit recipe; puzzles: number of dev puzzles counted."""
    x0, _ = LF.read_dev(puzzles)
    rows = []
    jobs = [(i, 100) for i in range(puzzles)] + [(0, 1), (0, 2), (0, 3)]
    for j, (i, budget) in enumerate(jobs):
        gen = torch.Generator(device='cuda').manual_seed(7 + j)
        with FullCounter() as fc:
            _, steps, _ = L.chain_search(net, x0[i:i + 1].cuda(), chains=K, budget=budget, check_every=1,
                                         theta_cls=L.THETA_CLS_EVAL, slots=1, amp=True, generator=gen, theta=theta)
        s = fc.summary()
        rows.append(dict(puzzle=i, budget=budget, steps=int(steps[0]), flop=int(s['total']), flop_with_compare=int(s['total_with_compare'])))
        print(f'  count K={K}', rows[-1], flush=True)
    st = np.array([r['steps'] for r in rows])
    fl = np.array([r['flop'] for r in rows])
    i, j = int(st.argmin()), int(st.argmax())
    assert st[i] != st[j], 'need two rows with different step counts'
    b = (fl[j] - fl[i]) // (st[j] - st[i])
    a = fl[i] - b * st[i]
    assert all(f == a + b * s for f, s in zip(fl, st)), 'the count is not linear in the number of steps'
    return int(a), int(b), rows


def main(args: Args):
    """Run every requested (chains, budget) setting on the first N test puzzles and write its count and its record."""
    torch._dynamo.config.cache_size_limit = 64  # one compiled graph per chain count (batch = slots x K)
    torch.backends.cuda.matmul.allow_tf32 = True
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fit = json.loads((run / 'fit.json').read_text())
    theta = fit['recipe'].get('theta', L.THETA)
    model_sha = hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest()
    assert model_sha == fit['model_sha256'], 'model.pt differs from the fit record'
    net = L.LatticeNet().cuda().eval()
    net.load_state_dict(torch.load(run / 'model.pt', map_location='cuda', weights_only=True))
    fn = torch.compile(net) if args.compile else net
    grid = [tuple(int(v) for v in s.split(':')) for s in args.settings] if args.settings else sorted(GRID, key=lambda kb: kb[0] * kb[1])
    assert all(kb in GRID for kb in grid), f'not in the grid GRID: {grid}'
    x0, d, desc = LS.load_test()
    x0, d = x0[:N], d[:N]
    desc = f'{desc}, first {N} = full-all rows 10,000-19,999'
    ref = run / 'score/test/0000000.npz'  # shard 0 of the Table-2 search (103,197 puzzles, same generator seed)
    for K, B in grid:
        name = f'k{K}-b{B}'
        if (out / f'{name}.json').exists():
            print(f'{name}: exists', flush=True)
            continue
        cpath = out / f'count-k{K}.json'
        if not cpath.exists():
            t = time.perf_counter()
            a, b, rows = count(net, K, theta, args.count_puzzles)
            write_json(cpath, dict(convention=CONVENTION, chains=K, per_puzzle_constant=a, per_step=b, rows=rows,
                                   linear_exact=True, model_sha256=model_sha, seconds=time.perf_counter() - t))
        cnt = json.loads(cpath.read_text())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        gen = torch.Generator(device='cuda').manual_seed(LS.RNG_BASE + 0)
        p, s, r = L.chain_search(fn, x0.cuda(), chains=K, budget=B, check_every=1, slots=args.slots, amp=True,
                                 generator=gen, theta=theta)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t
        pred, steps, resets = p.cpu().numpy().astype(np.int8), s.cpu().numpy().astype(np.int64), r.cpu().numpy()
        np.savez(out / f'{name}.tmp.npz', prediction=pred, steps=steps.astype(np.int16), resets=resets.astype(np.int32))
        os.replace(out / f'{name}.tmp.npz', out / f'{name}.npz')
        solved = (pred == d.numpy()).all(-1)
        timeout = (pred < 0).any(-1)
        flop = cnt['per_puzzle_constant'] + cnt['per_step'] * steps
        res = dict(method='TRMP lattice (factor graph, 16k)', task='sudoku-extreme', split=desc, count=N, chains=K, budget=B,
                   check_every=1, theta_cls=L.THETA_CLS_EVAL, theta=theta, temperature=L.TEMP, numerics='bf16 autocast',
                   solved=int(solved.sum()), solve_rate=float(solved.mean()), timeouts=int(timeout.sum()),
                   wrong=int((~timeout & ~solved).sum()), mean_steps=float(steps.mean()), max_steps=int(steps.max()),
                   mean_resets=float(resets.mean()), flop_per_puzzle_mean=float(flop.mean()),
                   flop_per_puzzle_p99=float(np.percentile(flop, 99)), count_file=cpath.name, rng_base=LS.RNG_BASE, slots=args.slots,
                   seconds=seconds, peak_gpu_bytes=torch.cuda.max_memory_allocated(), model=str(run / 'model.pt'),
                   model_sha256=model_sha, training_seed=fit['config']['seed'], gpu=torch.cuda.get_device_name(),
                   slurm_job=os.environ.get('SLURM_JOB_ID'), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if (K, B) == (64, 1000) and ref.exists():
            z = np.load(ref)
            res['check_table1_shard0'] = dict(file=str(ref), prediction_mismatch_puzzles=int((z['prediction'][:N] != pred).any(-1).sum()),
                                              steps_mismatch_puzzles=int((z['steps'][:N].astype(np.int64) != steps).sum()))
        write_json(out / f'{name}.json', res)
        print(f'{name}: {res["solved"]}/{N} = {100 * res["solve_rate"]:.2f}% (timeouts {res["timeouts"]}, wrong {res["wrong"]}, '
              f'mean steps {res["mean_steps"]:.2f}), {res["flop_per_puzzle_mean"]:.3e} flop/puzzle, {seconds:.1f} s, '
              f'peak {res["peak_gpu_bytes"] / 2**30:.1f} GiB', flush=True)


if __name__ == '__main__':
    main(tyro.cli(Args))
