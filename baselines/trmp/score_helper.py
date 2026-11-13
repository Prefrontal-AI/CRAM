"""Fill test prediction shards of a running trmp/score.py job from extra GPUs. The results don't change.

A shard's predictions depend only on the model, the selected sigma, the step cap, the batch size and the puzzle index
of each batch (its noise generator is seeded with 20261020 + its first index). So a shard computed here is identical
to one score.py would compute; score.py loads finished shards instead of recomputing them and writes test.json as usual.
The helper walks the shards from the end (score.py walks from the start) and claims each one with an exclusive
<shard>.claim file, so helpers never duplicate each other; a shard that score.py reaches first is computed twice at most.
--verify recomputes one finished shard into a scratch folder and checks that it matches exactly.

  python trmp/score_helper.py --run runs/trmp/sudoku-extreme-s2        # from baselines/
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fit  # noqa: E402
from infer import best_of  # noqa: E402
from model import RECIPES, Trainable  # noqa: E402
from score import RNG_BASE, ROLLOUTS, load_split  # noqa: E402


@dataclasses.dataclass
class Args:
    run: str = 'runs/trmp/sudoku-extreme-s0'  # fit folder whose score.py job is running
    shard: int = 8192  # puzzles per shard (must equal score.py's)
    verify: Optional[int] = None  # recompute this finished shard start into a scratch folder, compare, and exit


def compute(model, x, lo, shard, batch, sigma, cap):
    p, d = [], []
    t = time.perf_counter()
    for b in range(lo, min(lo + shard, len(x)), batch):  # the same batch boundaries and seeds as score.predict
        xb = x[b:min(b + batch, lo + shard, len(x))].cuda()
        gen = torch.Generator(device='cuda').manual_seed(RNG_BASE + b)
        with fit.amp_context(True):
            answer, work = best_of(model, xb, rollouts=ROLLOUTS, steps=cap, sigma=sigma, generator=gen)
        p.append(answer.cpu().numpy().astype(np.int8))
        d.append(work['rollout_steps'].cpu().numpy().astype(np.int16))
    torch.cuda.synchronize()
    return np.concatenate(p), np.concatenate(d), time.perf_counter() - t


def main(args: Args):
    run = Path(args.run)
    rec = json.loads((run / 'fit.json').read_text())
    task = rec['task']
    cap, batch = RECIPES[task]['inference_steps'], (256 if task == 'sudoku-extreme' else 16)  # score.py defaults
    sigma = json.loads((run / 'score' / 'selection.json').read_text())['sigma']
    folder = run / 'score' / 'test'
    model = Trainable(task).cuda().eval()
    model.load_state_dict(torch.load(run / 'model.pt', map_location='cuda', weights_only=True))
    x, _, _, _ = load_split(task, 'test')  # inputs only are used; labels are never compared here
    if args.verify is not None:
        ref = np.load(folder / f'{args.verify:07d}.npz')
        p, d, sec = compute(model, x, args.verify, args.shard, batch, sigma, cap)
        same = bool((p == ref['prediction']).all() and (d == ref['steps']).all())
        print(f'verify shard {args.verify}: identical={same} ({sec:.1f} s vs {float(ref["seconds"]):.1f} s)', flush=True)
        sys.exit(0 if same else 1)
    host = f'{os.environ.get("SLURM_JOB_ID", "local")}'
    for lo in reversed(range(0, len(x), args.shard)):
        path = folder / f'{lo:07d}.npz'
        if path.exists():
            continue
        try:
            fd = os.open(folder / f'{lo:07d}.claim', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        os.write(fd, host.encode()); os.close(fd)
        p, d, sec = compute(model, x, lo, args.shard, batch, sigma, cap)
        tmp = folder / f'{lo:07d}.helper{host}.tmp.npz'
        np.savez(tmp, prediction=p, steps=d, seconds=sec)
        if not path.exists():
            os.replace(tmp, path)
        else:
            tmp.unlink()
        print(f'  shard {lo}: {sec:.1f} s', flush=True)
    print('no unclaimed shards left', flush=True)


if __name__ == '__main__':
    main(tyro.cli(Args))
