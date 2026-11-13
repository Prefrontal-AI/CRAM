"""Instrumented TRMP pass for the per-rollout step cap (compute sweep).

TRMP has no public code; its model is the reimplementation in baselines/trmp/ (model.py, fit.py, score.py). Same model, inputs, batches, noise generators, sigma, bf16 autocast and 64 rollouts as baselines/trmp/score.py (through
sweep/trmp_sweep.py's loading). For every cap c on the grid this records each rollout's answer and confidence as
infer.best_of with steps = c would return them:
- a rollout that halted (q > 0) at a step < c keeps the answer and q of that step;
- a rollout still running after step c - 1 is forced to stop at step c: its answer is the masked argmax of the symbol
  logits at step c and its confidence is q at step c (whether or not q > 0), exactly as best_of's last step.
Up to step c, best_of with cap c and with the Table-2 cap run the same steps on the same active rollouts and draw the same
noise (noise is drawn after the halting test, only for continuing rollouts), so the record at c is the capped run, not an
approximation. TRMP's answer at cap c = the most confident of the 64 rollouts (ties to the lowest index); rollout steps at
cap c = min(realized steps, c). At the Table-2 cap the selected answers and steps are checked against score/test/*.npz.

Run from baselines/ on one GPU:
  ../.venv/bin/python sweep/trmp_caps.py --run runs/trmp/sudoku-extreme-s0 --start 0 --stop 2560 --out <prefix>
Writes <prefix>.npz: caps (c,), exact_at (p, c, r) bool, confidence_at (p, c, r), steps (p, r) at the Table-2 cap,
answer (p, cells) at the Table-2 cap, test_index, dataset_index; and <prefix>.json. Dimensions: p puzzles, c caps,
r rollouts.
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


@torch.no_grad()
def rollouts_caps(model, x, *, rollouts, steps, sigma, generator, caps):
    """infer.best_of (adaptive) that also records every rollout's answer and confidence at each cap in `caps`.

    x: (puzzles, cells, symbols), the input encoding that trmp/score.py load_split returns; returns the selected answer (puzzles, cells), the steps of every rollout
    (puzzles, rollouts), and per cap the answers (caps, puzzles, rollouts, cells) and confidences (caps, puzzles, rollouts)."""
    puzzles, cells, _ = x.shape
    expanded = x.repeat_interleave(rollouts, dim=0)
    state = model.initial(expanded)
    clues = model.embedding(expanded)
    active = torch.arange(len(expanded), device=x.device)
    prediction = torch.empty((len(expanded), cells), dtype=torch.long, device=x.device)
    confidence = torch.empty(len(expanded), device=x.device)
    depth = torch.zeros(len(expanded), dtype=torch.long, device=x.device)
    cap_pred = torch.empty((len(caps), len(expanded), cells), dtype=torch.int8, device=x.device)
    cap_conf = torch.empty((len(caps), len(expanded)), device=x.device)
    recorded = set()
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
        if index in caps:  # a run capped here would force every remaining rollout to answer now
            ci = caps.index(index)
            cap_pred[ci] = prediction.to(torch.int8)  # rows halted at or before this step (others overwritten below)
            cap_conf[ci] = confidence
            if remaining.any():
                rows = active[remaining]
                logits = model.symbol(state[1][remaining]).float()
                cap_pred[ci, rows] = logits.masked_fill(expanded[rows] == 0, -torch.inf).argmax(-1).to(torch.int8)
                cap_conf[ci, rows] = q[remaining]
            recorded.add(index)
        if not remaining.any():
            break
        active = active[remaining]
        clues = clues[remaining]
        state = tuple(value[remaining] for value in state)
        if sigma:
            h, a, messages, units = state
            noise = torch.randn(a.shape, dtype=a.dtype, device=a.device, generator=generator)
            state = h, a+sigma*noise, messages, units
    for ci, c in enumerate(caps):  # caps past the step where every rollout had halted: the halted answers
        if c not in recorded:
            cap_pred[ci] = prediction.to(torch.int8)
            cap_conf[ci] = confidence
    confidence = confidence.reshape(puzzles, rollouts)
    depth = depth.reshape(puzzles, rollouts)
    winner = confidence.argmax(-1)
    preds = prediction.reshape(puzzles, rollouts, cells)
    return (preds[torch.arange(puzzles, device=x.device), winner], depth,
            cap_pred.view(len(caps), puzzles, rollouts, cells), cap_conf.view(len(caps), puzzles, rollouts))


def main(a: Args):
    """Run test puzzles [a.start, a.stop) with 64 rollouts and record every rollout's answer and confidence at each cap."""
    run, out = Path(a.run), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.with_suffix('.json').exists():
        print(f'{out}.json exists', flush=True)
        return
    rec = json.loads((run / 'fit.json').read_text())
    task = rec['task']
    cap = RECIPES[task]['inference_steps']
    caps = [2 ** i for i in range(20) if 2 ** i <= cap]
    assert caps[-1] == cap
    rollouts = score.ROLLOUTS
    batch = 256 if task == 'sudoku-extreme' else 16  # score.py's defaults (its Table-2 runs used them)
    assert a.start % batch == 0
    sigma = json.loads((run / 'score/selection.json').read_text())['sigma']
    model = Trainable(task).cuda().eval()
    model.load_state_dict(torch.load(run / 'model.pt', map_location='cuda', weights_only=True))
    sha = hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest()
    assert sha == rec['model_sha256']
    x, y, idx, desc = score.load_split(task, 'test')
    stop = a.stop or len(x)
    answers, steps, exact_at, conf_at = [], [], [], []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for b in range(a.start, stop, batch):
        xb = x[b:min(b + batch, len(x))].cuda()  # full score.py batches, so each generator draws as in score.py
        gen = torch.Generator(device='cuda').manual_seed(score.RNG_BASE + b)
        with fit.amp_context(True):
            ans, d, cp, cc = rollouts_caps(model, xb, rollouts=rollouts, steps=cap, sigma=sigma, generator=gen, caps=caps)
        yb = y[b:b + len(xb)].cuda()
        keep = min(len(xb), stop - b)
        answers.append(ans[:keep].cpu().numpy().astype(np.int8))
        steps.append(d[:keep].cpu().numpy().astype(np.int16))
        exact_at.append((cp.long() == yb[None, :, None]).all(-1).permute(1, 0, 2)[:keep].cpu().numpy())  # (p, c, r)
        conf_at.append(cc.permute(1, 0, 2)[:keep].float().cpu().numpy())
        print(f'{b + keep}/{stop} {time.perf_counter() - t0:.0f} s', flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    answers, steps, exact_at, conf_at = map(np.concatenate, (answers, steps, exact_at, conf_at))
    shard = 8192
    ref_p, ref_s = [], []
    for lo in range(a.start - a.start % shard, stop, shard):
        f = np.load(run / f'score/test/{lo:07d}.npz')
        ref_p.append(f['prediction']); ref_s.append(f['steps'])
    off = a.start % shard
    ref_p, ref_s = np.concatenate(ref_p)[off:off + len(answers)], np.concatenate(ref_s)[off:off + len(answers)]
    yt = y[a.start:stop].numpy()
    n = len(answers)
    win = conf_at.argmax(2)  # (p, c): the most confident rollout at each cap, ties to the lowest index
    solved_at = exact_at[np.arange(n)[:, None], np.arange(len(caps))[None], win].sum(0)
    check = dict(answer_mismatch_puzzles=int((ref_p != answers).any(1).sum()), steps_mismatch_puzzles=int((ref_s != steps).any(1).sum()),
                 ref_solved=int((ref_p == yt).all(-1).sum()), sweep_solved=int((answers == yt).all(-1).sum()),
                 table1_cap_selection_equals_answer_solved=int(solved_at[-1]))
    np.savez_compressed(out.with_suffix('.npz'), caps=np.array(caps), exact_at=exact_at, confidence_at=conf_at, steps=steps, answer=answers,
                        test_index=np.arange(a.start, stop), dataset_index=idx[a.start:stop])
    res = dict(method='TRMP', task=task, run=str(run), model_sha256=sha, sigma=sigma, rollouts=rollouts, table1_step_cap=cap, caps=caps,
               batch=batch, rng_base=score.RNG_BASE, split=desc, start=a.start, stop=stop, n=n,
               solved_best_of_64_at_cap=dict(zip(map(str, caps), solved_at.tolist())), check=check, seconds=seconds,
               gpu=torch.cuda.get_device_name(), slurm_job=os.environ.get('SLURM_JOB_ID'),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               score_py_sha256=hashlib.sha256((HERE.parent / 'trmp/score.py').read_bytes()).hexdigest())
    out.with_suffix('.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main(tyro.cli(Args))
