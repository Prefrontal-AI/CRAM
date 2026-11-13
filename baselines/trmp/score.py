"""Score a timed TRMP fit (outside the fit clock): choose the rollout noise on validation, then score the test split.

Inference is the paper's TRMP row: 64 confidence-selected noisy rollouts, each halting at q > 0 or at the step cap
(64 for Maze-Unique, B.5; 1,024 for Sudoku-Extreme, B.4). The noise scale is
chosen from {0.1, 0.2, 0.3, 0.5} by validation solves (ties: the smallest).
Validation: Sudoku the first 500 dev10k puzzles (the fit's checkpoint-selection set), Maze the 100 generated mazes.
Test: Sudoku the 412,786 official test puzzles outside dev10k (full-all 10,000-422,785), Maze the 1,000 test mazes.
The model runs under bf16 autocast, as in training. Predictions are written per shard (resumable) before any label
is compared. Puzzles run in batches; the noise generator of a batch is seeded with 20261020 + its first index.

  python trmp/score.py --run runs/trmp/sudoku-extreme-s0        # from baselines/
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
sys.path.insert(0, str(HERE))
import fit  # noqa: E402
from infer import best_of, matrix_count  # noqa: E402
from model import RECIPES, Trainable  # noqa: E402

SIGMAS = (.1, .2, .3, .5)
RNG_BASE = 20261020
ROLLOUTS = 64


@dataclasses.dataclass
class Args:
    run: str = 'runs/trmp/sudoku-extreme-s0'  # fit folder (fit.json, model.pt); outputs go to <run>/score
    batch: Optional[int] = None  # puzzles per inference batch (default: 256 Sudoku, 16 Maze)
    test_limit: Optional[int] = None  # score only the first N test puzzles (smoke tests; the full run leaves it unset)
    shard: int = 8192  # puzzles per durable prediction shard
    cap: Optional[int] = None  # rollout step cap (smoke tests only; default: 1,024 Sudoku, 64 Maze)
    dev_limit: Optional[int] = None  # score only the first N validation puzzles (smoke tests only)
    test10k: bool = False  # Sudoku: score test10k (full-all rows 10,000-19,999) only; also writes <run>/test10k.json and test10k.solved.npy; unused by the reported scores


def load_split(task, split):
    """(x, target, indices, description) without opening anything but inputs and solutions of that split."""
    if task == 'sudoku-extreme':
        if split == 'dev':
            f = np.load(fit.V1_DATA / 'frm/dev10k/dev.npz')
            x, y = fit.encode_sudoku(f['solutions'][:500], f['clue_mask'][:500])
            return x, y, np.arange(500), 'dev10k[:500]'
        f = np.load(fit.V1_DATA / 'full-test/full-all.npz')
        sol, clue = f['solutions'], f['clue_mask']
        dev = np.load(fit.V1_DATA / 'frm/dev10k/dev.npz')
        assert (dev['solutions'] == sol[:10000]).all(), 'dev10k is not full-all[:10000]'
        assert len(sol) == 422786
        x, y = fit.encode_sudoku(sol[10000:], clue[10000:])
        return x, y, np.arange(10000, 422786), 'full-all[10000:422786] (412,786 official test puzzles outside dev10k)'
    if split == 'dev':
        x, y = fit.read_dev(fit.Config(task=task), 100)
        return x, y, np.arange(100), 'generated dev100'
    sys.path.insert(0, str(fit.ROOT / 'experiments'))
    sys.path.insert(0, str(fit.ROOT))  # the cram package
    from cram.maze_unique import _read
    inputs, labels = _read('test/all__inputs.npy'), _read('test/all__labels.npy')
    tokens = torch.as_tensor(inputs.astype(np.int64) - 1).reshape(-1, 900)
    target_path = (labels == 5) | (inputs == 3) | (inputs == 4)
    x, y = fit.encode_maze(tokens, torch.as_tensor(target_path).reshape(-1, 900))
    return x, y, np.arange(1000), 'EqR maze-30x30-unique-1k test (1,000)'


def predict(model, x, sigma, cap, batch, folder, shard):
    """Predictions (puzzles, cells) and rollout depths, in durable shards of `shard` puzzles."""
    folder.mkdir(parents=True, exist_ok=True)
    preds, depths, seconds = [], [], 0.0
    for lo in range(0, len(x), shard):
        path = folder / f'{lo:07d}.npz'
        if not path.exists():
            torch.cuda.synchronize()
            t = time.perf_counter()
            p, d = [], []
            for b in range(lo, min(lo + shard, len(x)), batch):
                xb = x[b:min(b + batch, lo + shard, len(x))].cuda()
                gen = torch.Generator(device='cuda').manual_seed(RNG_BASE + b)
                with fit.amp_context(True):
                    answer, work = best_of(model, xb, rollouts=ROLLOUTS, steps=cap, sigma=sigma, generator=gen)
                p.append(answer.cpu().numpy().astype(np.int8))
                d.append(work['rollout_steps'].cpu().numpy().astype(np.int16))
            torch.cuda.synchronize()
            np.savez(path.with_suffix('.tmp.npz'), prediction=np.concatenate(p), steps=np.concatenate(d),
                     seconds=time.perf_counter() - t)
            os.replace(path.with_suffix('.tmp.npz'), path)
            print(f'  shard {lo}: {time.perf_counter() - t:.1f} s', flush=True)
        f = np.load(path)
        preds.append(f['prediction']); depths.append(f['steps']); seconds += float(f['seconds'])
    return np.concatenate(preds), np.concatenate(depths), seconds


def summary(task, pred, steps, y, seconds, sigma, cap, desc, batch):
    solved = (pred == y.numpy()).all(-1)
    r = RECIPES[task]
    return dict(split=desc, count=len(y), solved=int(solved.sum()), solve_rate=float(solved.mean()), sigma=sigma,
                rollouts=ROLLOUTS, step_cap=cap, adaptive=True, batch_puzzles=batch, rng_base=RNG_BASE,
                numerics='bf16 autocast', mean_rollout_steps=float(steps.mean()),
                max_rollout_steps=int(steps.max()), capped_rollouts=int((steps == cap).sum()),
                recursion_matrix_flop_per_puzzle_partial=float(steps.sum(1).mean()) * matrix_count(r),
                inference_seconds_total=seconds, unsolved_indices=np.flatnonzero(~solved).tolist()[:1000])


def main(args: Args):
    run = Path(args.run)
    rec = json.loads((run / 'fit.json').read_text())
    task = rec['task']
    cap = args.cap or RECIPES[task]['inference_steps']
    batch = args.batch or (256 if task == 'sudoku-extreme' else 16)
    out = run / 'score'
    out.mkdir(exist_ok=True)
    model = Trainable(task).cuda().eval()
    model.load_state_dict(torch.load(run / 'model.pt', map_location='cuda', weights_only=True))
    model_sha = hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest()
    assert model_sha == rec['model_sha256'], 'model.pt differs from the fit record'

    sel_path = out / 'selection.json'
    if not sel_path.exists():
        x, y, _, desc = load_split(task, 'dev')
        if args.dev_limit is not None:
            x, y, desc = x[:args.dev_limit], y[:args.dev_limit], f'{desc}, first {args.dev_limit} only'
        dev = {}
        for sigma in SIGMAS:
            path = out / f'dev-sigma{sigma}.json'
            if not path.exists():
                pred, steps, sec = predict(model, x, sigma, cap, batch, out / f'dev-sigma{sigma}', args.shard)
                fit.write_json(path, summary(task, pred, steps, y, sec, sigma, cap, desc, batch))
            dev[sigma] = json.loads(path.read_text())
            print(f'dev sigma {sigma}: {dev[sigma]["solved"]}/{dev[sigma]["count"]}, mean steps {dev[sigma]["mean_rollout_steps"]:.2f}', flush=True)
        sigma = min(SIGMAS, key=lambda s: (-dev[s]['solved'], s))
        fit.write_json(sel_path, dict(sigma=sigma, solved={str(s): dev[s]['solved'] for s in SIGMAS}, model_sha256=model_sha,
                                      rule='most validation solves, ties to the smallest sigma'))
    sigma = json.loads(sel_path.read_text())['sigma']

    if args.test10k:
        assert task == 'sudoku-extreme' and args.test_limit in (None, 10000)
        args = dataclasses.replace(args, test_limit=10000)  # full-all[10000:][:10000] = full-all rows 10,000-19,999
    name = 'test' if args.test_limit is None else f'test-first{args.test_limit}'
    if (out / f'{name}.json').exists():
        print(f'{out}/{name}.json exists', flush=True)
        return
    x, y, idx, desc = load_split(task, 'test')
    if args.test_limit is not None:
        x, y, desc = x[:args.test_limit], y[:args.test_limit], f'{desc}, first {args.test_limit} only'
    pred, steps, sec = predict(model, x, sigma, cap, batch, out / name, args.shard)
    res = summary(task, pred, steps, y, sec, sigma, cap, desc, batch)
    res.update(method='TRMP', task=task, model_sha256=model_sha, fit_seconds=rec['fit_seconds'], training_seed=rec['config']['seed'],
               params=rec['params'], gpu=torch.cuda.get_device_name(), slurm_job=os.environ.get('SLURM_JOB_ID'))
    fit.write_json(out / f'{name}.json', res)
    if args.test10k:
        solved = (pred == y.numpy()).all(-1)
        assert len(solved) == 10000 and idx[0] == 10000 and idx[9999] == 19999
        np.save(run / 'test10k.solved.npy', solved)
        fit.write_json(run / 'test10k.json', dict(n=10000, solved=int(solved.sum()), rate=float(solved.mean()), rows='full-all 10000-19999',
                                                  checkpoint=str(run / 'model.pt'), model_sha256=model_sha, steps=rec['steps'], selected_step=rec['selected_step'],
                                                  sigma=sigma, rollouts=ROLLOUTS, step_cap=cap, mean_rollout_steps=res['mean_rollout_steps'],
                                                  inference_seconds=sec, fit_seconds=rec['fit_seconds'], training_data=rec['training_data'],
                                                  flags='test10k.solved.npy: one bool per full-all row 10,000-19,999, in order'))
    print(f'TEST {res["solved"]}/{res["count"]} = {100 * res["solve_rate"]:.3f}% (sigma {sigma}, mean steps {res["mean_rollout_steps"]:.2f})', flush=True)


if __name__ == '__main__':
    main(tyro.cli(Args))
