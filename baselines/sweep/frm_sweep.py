"""FRM at several sampler iteration counts (compute sweep); one run per count.

Upstream code: the FRM clone in <root>/upstream (package flow_map_reasoning) and the adapter frm_reproduction.py in the
campaign root <root> (the folder of one FRM fit, baselines/frm/setup.sh).
Sudoku-Extreme: as baselines/frm/score_unseen.py (the campaign's selected checkpoint, EMA weights, its held_t sampler
with steps = held_iters = N, so N + 1 network calls per puzzle; torch/numpy/random seeded 0 before each shard; batch
1,024). Puzzles: full-frmtest (1,000) and the first 9,000 of full-00, i.e. full-all rows 10,000-19,999.

Run with FRM's Python, cwd = the campaign root, PYTHONPATH = <root>/upstream (sweep/run.sbatch does this):
  python frm_sweep.py sudoku <root> --seed K --iters 8 32 ... --out <dir> [--check4096]
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V1_DATA = Path(os.environ.get('V1_DATA', HERE.parents[1] / 'data'))
FULL = V1_DATA / 'full-test'
SUBSET = (('full-frmtest', 1000), ('full-00', 9000))


def digest(path):
    """SHA-256 of a file, as hex."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sudoku(a):
    """Score the sweep puzzles at each iteration count of a.iters; with a.check4096 also FRM's 1,000 test puzzles at 4,096."""
    import torch
    root = a.root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(HERE.parent / 'frm'))
    from frm_reproduction import arrays, load_model, profile
    from flow_map_reasoning import SamplerConfig, run_sampler
    from score_unseen import selection

    sel = selection(root, a.seed)
    ckpt = Path(sel['checkpoint'])
    sha = digest(ckpt)
    assert sha == sel['checkpoint_sha256']
    model, meta = load_model(root, ckpt)
    cost = profile(model)
    assert cost['flops_per_forward'] == sel['flops_per_forward']
    model.eval()
    calls = [0]
    model.register_forward_pre_hook(lambda _m, x: calls.__setitem__(0, calls[0] + len(x[0])))
    index = json.loads((FULL / 'index.json').read_text())
    jobs = [(n, SUBSET) for n in a.iters] + ([(4096, (('full-frmtest', 1000),))] if a.check4096 else [])
    for n, shards in jobs:
        tag = f'held{n}' + ('-check-frmtest' if len(shards) == 1 else '')
        out = a.out / f'{tag}.json'
        if out.exists():
            continue
        sampler = SamplerConfig(**{**sel['sampler'], 'batch_size': a.batch, 'steps': n, 'held_iters': n})
        solved, rows, seconds = [], [], 0.0
        calls[0] = 0
        for name, m in shards:
            sol, clue = arrays(FULL / f'{name}.npz')
            r = np.array(index['shards'][name]['official_rows'])
            sol, clue, r = sol[:m], clue[:m], r[:m]
            with torch.no_grad(), torch.random.fork_rng(devices=[0]):
                torch.manual_seed(0)
                np.random.seed(0)
                random.seed(0)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pred = run_sampler(model, sol, clue, sampler, lut_a2g=meta['lut_a2g'], vocab_size=10, device='cuda')
                torch.cuda.synchronize()
                seconds += time.perf_counter() - t0
            assert np.array_equal(pred[clue.astype(bool)], sol[clue.astype(bool)])
            solved.append(np.all(pred == sol, axis=1))
            rows.append(r)
        solved, rows = np.concatenate(solved), np.concatenate(rows)
        assert calls[0] == len(solved) * (n + 1), (calls[0], n)
        res = dict(method='FRM', task='sudoku-extreme', seed=a.seed, held_iters=n, nfe=n + 1, n=int(len(solved)), solved=int(solved.sum()),
                   flops_per_forward=cost['flops_per_forward'], flops_per_puzzle=(n + 1) * cost['flops_per_forward'],
                   puzzles='full-frmtest + full-00[:9000] (full-all rows 10,000-19,999)' if len(shards) == 2 else 'full-frmtest',
                   checkpoint=str(ckpt), checkpoint_sha256=sha, step=sel['step'], sampler=asdict(sampler), seconds=seconds,
                   gpu=torch.cuda.get_device_name(), slurm_job=os.environ.get('SLURM_JOB_ID'), script_sha256=digest(__file__))
        if len(shards) == 1:  # check: the Table-2 flags (frm/score_unseen.py) of the same puzzles at 4,096 iterations
            ref = np.load(root / 'score-unseen.solved.npz')
            pos = np.searchsorted(ref['official_rows'], rows)
            assert np.array_equal(ref['official_rows'][pos], rows)
            u = json.loads((root / f'runs/s{a.seed}/unseen/full-frmtest.json').read_text())
            res['check'] = dict(ref='score-unseen.solved.npz', ref_solved=int(ref['solved'][pos].sum()),
                                mismatches=int((ref['solved'][pos] != solved).sum()), ref_batch=u['sampler']['batch_size'])
        a.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out.with_suffix('.npz'), solved=solved, official_rows=rows)
        out.write_text(json.dumps(res, indent=1) + '\n')
        print(json.dumps({k: res[k] for k in ('seed', 'held_iters', 'n', 'solved', 'seconds')} | ({'check': res['check']} if 'check' in res else {})), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('task', choices=['sudoku'])
    p.add_argument('root', type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--iters', type=int, nargs='*', default=[])
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--check4096', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sudoku(a)
