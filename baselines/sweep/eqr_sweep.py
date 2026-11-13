"""Instrumented EqR pass (compute sweep): per-trajectory answers at several depths, one run.

Calls the authors' code through baselines/eqr/eqr_score.py (which loads the model from the EqR clone). Same model, loader, compilation, seeding and inference as baselines/eqr/eqr_score.py (imported): B trajectories
per puzzle, each for D network evaluations with path noise, no halting at evaluation. EqR draws the same noise
tensors at every step whatever D is, so the first d steps of a D-step trajectory are a d-step trajectory. At each depth
d in DEPTHS (d <= D) this script records, for every trajectory, whether its argmax answer is exact and its convergence
score (the mean over the last `window` steps of the per-cell L2 norm of the logit change, eqr_score.py's top1_conv
score). EqR's answer for breadth K is the trajectory with the lowest score among K; the build script takes the first K
of the B trajectories (independent restarts), so one B = 128 run gives every K <= 128.

Puzzles [start, stop) only (multiples of --batch), for sharding. As eqr_score.py's resume, a shard that starts at
puzzle s > 0 reseeds the evaluation noise with seed + s; the shard at 0 repeats eqr_score.py's draws exactly.

Run with EqR's Python from baselines/eqr (sweep/run.sbatch does this):
  python ../sweep/eqr_sweep.py --ckpt <model.pt> --data <dir> --split first10k --depth 64 --breadth 128 \
      --noise 0.5 --batch 128 --start 0 --stop 2560 --out <prefix>
Writes <prefix>.npz (exact, score: (puzzles, len(depths), B); depths; puzzle indices) and <prefix>.json.
"""

import hashlib
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path.cwd()))
import eqr_common as C  # noqa: E402
import eqr_score  # noqa: E402

DEPTHS = (4, 8, 16, 32, 64, 128)


@dataclass
class Args:
    ckpt: Path  # checkpoint in EqR's format (model, ema, config)
    data: Path  # EqR-format dataset directory
    out: Path  # output prefix
    split: str = 'test'  # split folder inside --data
    depth: int = 64  # D: network evaluations per trajectory
    breadth: int = 1  # B: trajectories per puzzle
    noise: float = C.EVAL_NOISE  # path noise scale
    init_std: float = C.EVAL_INIT_STD  # random initial latent scale (used when B > 1, as eqr_score.py)
    window: int = 3  # convergence window
    batch: int = 128  # puzzles per batch
    start: int = 0  # first puzzle (a multiple of batch)
    stop: int = 0  # end puzzle (0: the whole split)
    seed: int = 0  # evaluation seed
    compile: bool = True  # torch.compile, as eqr_score.py


def main(a: Args):
    """Run puzzles [a.start, a.stop) with a.breadth trajectories each and record exactness and convergence score at each depth."""
    out_json = a.out.with_suffix('.json')
    if out_json.exists():
        print(f'{out_json} exists', flush=True)
        return
    assert a.start % a.batch == 0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    C.install_stand_ins()
    from evaluate import _seed
    from models.losses import IGNORE_LABEL_ID
    sa = eqr_score.Args(ckpt=a.ckpt, data=a.data, out=a.out, split=a.split, depth=a.depth, breadth=a.breadth, noise=a.noise,
                        init_std=a.init_std, window=a.window, batch=a.batch, seed=a.seed, compile=a.compile)
    model, loader, meta, cfg, ck = eqr_score.load_model(sa, a.split)
    stop = a.stop or meta.total_groups
    if a.start:
        _seed(a.seed + a.start, 0)
    depths = [d for d in DEPTHS if a.window < d <= a.depth]
    B = a.breadth
    exact_all, score_all = [], []
    done, t0 = 0, time.perf_counter()
    with torch.inference_mode():
        for _, batch, _ in loader:
            bs = batch['inputs'].shape[0]
            if done < a.start:
                done += bs
                continue
            if done >= stop:
                break
            batch = {k: v.cuda() for k, v in batch.items()}
            labels = batch['labels']
            nb = min(int((labels != IGNORE_LABEL_ID).any(-1).sum()), stop - done)
            rows = batch if B == 1 else {k: v.repeat_interleave(B, dim=0) for k, v in batch.items()}
            carry = model.initial_carry(rows)
            last = deque(maxlen=a.window + 1)
            mask = labels != IGNORE_LABEL_ID
            ex, sc = [], []
            steps = 0
            while True:
                carry, _, _, preds, finished = model(carry=carry, batch=rows, return_keys={'preds', 'logits'})
                last.append(preds['logits'].float())
                steps += 1
                if steps in depths:
                    pred = preds['preds'].view(bs, B, -1)
                    ex.append((pred.eq(labels[:, None]) | ~mask[:, None]).all(-1)[:nb])
                    lg = torch.stack(list(last))
                    sc.append((lg[1:] - lg[:-1]).norm(dim=-1).mean(dim=(0, -1)).view(bs, B)[:nb])
                if bool(finished):
                    break
            assert steps == a.depth, steps
            exact_all.append(torch.stack(ex, 1).cpu().numpy())
            score_all.append(torch.stack(sc, 1).cpu().numpy())
            done += bs
            el = time.perf_counter() - t0
            print(f'{min(done, stop)}/{stop}  top1_conv@D{depths[-1]} {np.mean([e[np.arange(len(e)), -1, s[:, -1].argmin(1)].mean() for e, s in zip(exact_all, score_all)]):.4f}  {el:.0f} s', flush=True)
    exact, score = np.concatenate(exact_all), np.concatenate(score_all)
    n = len(exact)
    assert n == stop - a.start, (n, stop, a.start)
    np.savez_compressed(a.out.with_suffix('.npz'), exact=exact, score=score, depths=np.array(depths), puzzles=np.arange(a.start, stop))
    top = exact[np.arange(n)[:, None], np.arange(len(depths))[None], score.argmin(2)]  # (n, depths)
    summary = dict(ckpt=str(a.ckpt), ckpt_sha256=hashlib.sha256(a.ckpt.read_bytes()).hexdigest(), ckpt_step=ck.get('step'), data=str(a.data),
                   split=a.split, start=a.start, stop=stop, n=n, depth=a.depth, depths=depths, breadth=B, noise=a.noise,
                   init_std=a.init_std, random_init=B > 1, window=a.window, batch=a.batch, seed=a.seed,
                   noise_seed=a.seed + a.start, top1_conv_solved=dict(zip(map(str, depths), top.sum(0).tolist())),
                   first_solved=dict(zip(map(str, depths), exact[:, :, 0].sum(0).tolist())),
                   seconds=time.perf_counter() - t0, gpu=torch.cuda.get_device_name(), slurm_job=os.environ.get('SLURM_JOB_ID'),
                   eqr_commit=C.EQR_COMMIT, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   eqr_score_sha256=hashlib.sha256(Path(eqr_score.__file__).read_bytes()).hexdigest())
    tmp = out_json.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(summary, indent=1))
    os.replace(tmp, out_json)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main(tyro.cli(Args))
