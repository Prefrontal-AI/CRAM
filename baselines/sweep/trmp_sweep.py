"""Instrumented TRMP pass (compute sweep): every rollout's answer, confidence and depth.

TRMP has no public code; its model is the reimplementation in baselines/trmp/ (model.py, fit.py, score.py). Same model, inputs, batches, noise generators (seeded with RNG_BASE + the batch's first test index), sigma (the run's
validation choice, score/selection.json), step cap and bf16 autocast as baselines/trmp/score.py. `rollouts_all` repeats
baselines/trmp/infer.best_of line by line and also returns the per-rollout predictions; its selected answer is checked
against the Table-2 prediction shards (score/test/*.npz). TRMP's answer for K rollouts is the rollout with the largest
terminal confidence among K (ties to the lowest index); the build script takes the first K of the 64 rollouts, so one
run gives every K <= 64. Rollouts are independent given their own noise draws.

Test puzzles [start, stop) of score.py's test order (Sudoku: full-all[10000:]; Maze: the 1,000 test mazes), with start a
multiple of the batch so that each batch and its generator equal score.py's.

Run from baselines/ on one GPU:
  ../.venv/bin/python sweep/trmp_sweep.py --run runs/trmp/sudoku-extreme-s0 --start 0 --stop 2560 --out <prefix>
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
sys.path.insert(0, str(HERE.parent / 'trmp'))
import fit  # noqa: E402
import score  # noqa: E402
from model import RECIPES, Trainable  # noqa: E402


@dataclasses.dataclass
class Args:
    run: str  # fit folder (fit.json, model.pt, score/)
    out: str  # output prefix (<out>.npz, <out>.json)
    start: int = 0  # first test puzzle (a multiple of the batch)
    stop: int = 0  # end test puzzle (0: all)
    batch: int = 0  # puzzles per batch (0: score.py's default, 256 Sudoku, 16 Maze)


@torch.no_grad()
def rollouts_all(model, x, *, rollouts, steps, sigma, generator):
    """infer.best_of (adaptive) with the per-rollout predictions returned as well.

    x: (puzzles, cells, symbols), the input encoding that trmp/score.py load_split returns; returns the selected answer (puzzles, cells), every rollout's answer
    (puzzles, rollouts, cells), its terminal confidence (puzzles, rollouts) and its number of steps (puzzles, rollouts)."""
    puzzles, cells, _ = x.shape
    expanded = x.repeat_interleave(rollouts, dim=0)
    state = model.initial(expanded)
    clues = model.embedding(expanded)
    active = torch.arange(len(expanded), device=x.device)
    prediction = torch.empty((len(expanded), cells), dtype=torch.long, device=x.device)
    confidence = torch.empty(len(expanded), device=x.device)
    depth = torch.zeros(len(expanded), dtype=torch.long, device=x.device)
    for index in range(1, steps+1):
        state = model.step(state, clues)
        depth[active] += 1
        q = -torch.logsumexp(-model.halt_logits(state[1]).float(), dim=-1)
        if not torch.isfinite(q).all():
            raise ValueError('Nonfinite halting confidence')
        done = (q > 0) if index < steps else torch.ones_like(q, dtype=torch.bool)
        if done.any():
            finished = active[done]
            logits = model.symbol(state[1][done]).float()
            prediction[finished] = logits.masked_fill(expanded[finished] == 0, -torch.inf).argmax(-1)
            confidence[finished] = q[done]
        remaining = ~done
        if not remaining.any():
            break
        active = active[remaining]
        clues = clues[remaining]
        state = tuple(value[remaining] for value in state)
        if sigma:
            h, a, messages, units = state
            noise = torch.randn(a.shape, dtype=a.dtype, device=a.device, generator=generator)
            state = h, a+sigma*noise, messages, units
    confidence = confidence.reshape(puzzles, rollouts)
    depth = depth.reshape(puzzles, rollouts)
    winner = confidence.argmax(-1)
    preds = prediction.reshape(puzzles, rollouts, cells)
    return preds[torch.arange(puzzles, device=x.device), winner], preds, confidence, depth


def main(a: Args):
    """Run test puzzles [a.start, a.stop) with 64 rollouts, check the answers against the Table-2 shards, write the records."""
    run = Path(a.run)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.with_suffix('.json').exists():
        print(f'{out}.json exists', flush=True)
        return
    rec = json.loads((run / 'fit.json').read_text())
    task = rec['task']
    cap = RECIPES[task]['inference_steps']
    rollouts = score.ROLLOUTS
    batch = a.batch or (256 if task == 'sudoku-extreme' else 16)
    assert a.start % batch == 0
    sigma = json.loads((run / 'score/selection.json').read_text())['sigma']
    model = Trainable(task).cuda().eval()
    model.load_state_dict(torch.load(run / 'model.pt', map_location='cuda', weights_only=True))
    sha = hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest()
    assert sha == rec['model_sha256']
    x, y, idx, desc = score.load_split(task, 'test')
    stop = a.stop or len(x)
    answers, exact, conf, steps = [], [], [], []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for b in range(a.start, stop, batch):
        xb = x[b:min(b + batch, len(x))].cuda()  # full score.py batches, so each generator draws as in score.py
        gen = torch.Generator(device='cuda').manual_seed(score.RNG_BASE + b)
        with fit.amp_context(True):
            ans, preds, c, d = rollouts_all(model, xb, rollouts=rollouts, steps=cap, sigma=sigma, generator=gen)
        yb = y[b:b + len(xb)].cuda()
        keep = min(len(xb), stop - b)
        answers.append(ans[:keep].cpu().numpy().astype(np.int8))
        exact.append((preds == yb[:, None]).all(-1)[:keep].cpu().numpy())
        conf.append(c[:keep].float().cpu().numpy())
        steps.append(d[:keep].cpu().numpy().astype(np.int16))
        print(f'{b + keep}/{stop} {time.perf_counter() - t0:.0f} s', flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    answers, exact, conf, steps = map(np.concatenate, (answers, exact, conf, steps))
    # check against the Table-2 shards (score/test/<lo>.npz hold predictions and steps of puzzles lo...)
    shard = 8192
    ref_p, ref_s = [], []
    for lo in range(a.start - a.start % shard, stop, shard):
        f = np.load(run / f'score/test/{lo:07d}.npz')
        ref_p.append(f['prediction']); ref_s.append(f['steps'])
    off = a.start % shard
    ref_p, ref_s = np.concatenate(ref_p)[off:off + len(answers)], np.concatenate(ref_s)[off:off + len(answers)]
    sel_exact = (answers == y[a.start:stop].numpy()).all(-1)
    check = dict(answer_mismatch_puzzles=int((ref_p != answers).any(1).sum()), steps_mismatch_puzzles=int((ref_s != steps).any(1).sum()),
                 ref_solved=int((ref_p == y[a.start:stop].numpy()).all(-1).sum()), sweep_solved=int(sel_exact.sum()))
    np.savez_compressed(out.with_suffix('.npz'), exact=exact, confidence=conf, steps=steps, answer=answers, test_index=np.arange(a.start, stop),
                        dataset_index=idx[a.start:stop])
    res = dict(method='TRMP', task=task, run=str(run), model_sha256=sha, sigma=sigma, rollouts=rollouts, step_cap=cap, batch=batch,
               rng_base=score.RNG_BASE, split=desc, start=a.start, stop=stop, n=len(answers), solved_best_of_64=int(sel_exact.sum()),
               check=check, seconds=seconds, gpu=torch.cuda.get_device_name(), slurm_job=os.environ.get('SLURM_JOB_ID'),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               score_py_sha256=hashlib.sha256((HERE.parent / 'trmp/score.py').read_bytes()).hexdigest())
    out.with_suffix('.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main(tyro.cli(Args))
