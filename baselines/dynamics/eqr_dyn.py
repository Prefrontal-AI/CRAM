"""Latent dynamics and q-head halting of EqR run past the trained depth.

Calls the authors' code through baselines/eqr/eqr_score.py (which loads the model from the EqR clone). Same model, loader, EMA weights, compilation, seeding and inference as baselines/eqr/eqr_score.py (imported) at
breadth 1: one trajectory per puzzle from EqR's random initial latents (init_std 1.0), --depth network evaluations,
path noise --noise at every inner update (0 gives the deterministic map; 0.01 is the training and Table-2 value).
After every evaluation t this records, per puzzle: the squared norms of z_H and z_L and of their change since t - 1,
whether the argmax answer changed since t - 1, whether it is exact, and q_halt.
EqR's own ACT evaluation halts a trajectory at the first step with q_halt > 0 (evaluators/eval_fn.py, threshold 0).

Run with EqR's Python from baselines/eqr (dynamics/run.sbatch does this):
  python eqr_dyn.py --ckpt <model.pt> --data <dir> --split first10k --depth 128 --noise 0 --batch 250 --stop 1000 --out <prefix>
Writes <prefix>.npz and, last, <prefix>.json; exits at once when <prefix>.json exists.
Arrays in <prefix>.npz are (n, t), n puzzles and t network evaluations (1-based): zh2, zl2, dzh2, dzl2, changed, exact, q_halt.
"""

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

sys.path.insert(0, str(Path.cwd()))
import eqr_common as C  # noqa: E402
import eqr_score  # noqa: E402


@dataclass
class Args:
    ckpt: Path  # checkpoint in EqR's format (model, ema, config)
    data: Path  # EqR-format dataset directory
    out: Path  # output prefix
    split: str = 'test'  # split folder inside --data
    depth: int = 128  # network evaluations per trajectory
    noise: float = 0.0  # path noise scale (0: deterministic map)
    init_std: float = C.EVAL_INIT_STD  # random initial latent scale, as eqr_score.py
    batch: int = 250  # puzzles per batch
    stop: int = 1000  # puzzles scored: the first `stop` of the split
    seed: int = 0  # evaluation seed
    trained_depth: int = 16  # halt_max_steps of the training recipe
    compile: bool = True  # torch.compile, as eqr_score.py
    fp32: bool = False  # run the model in float32 instead of its bf16 forward_dtype (separates convergence from bf16 rounding)


def main(a: Args):
    """Run the first a.stop puzzles of the split for a.depth network evaluations and write the per-step records."""
    out_json = a.out.with_suffix('.json')
    if out_json.exists():
        print(f'{out_json} exists', flush=True)
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    C.install_stand_ins()
    from models.losses import IGNORE_LABEL_ID
    if a.fp32:  # the checkpoint's config sets arch.forward_dtype; override it before eqr_score builds the model
        load = eqr_score.safe_load.load

        def load32(*args, **kw):
            ck = load(*args, **kw)
            ck['config']['arch']['forward_dtype'] = 'float32'
            return ck
        eqr_score.safe_load.load = load32
    sa = eqr_score.Args(ckpt=a.ckpt, data=a.data, out=a.out, split=a.split, depth=a.depth, breadth=1, noise=a.noise,
                        init_std=a.init_std, batch=a.batch, seed=a.seed, compile=a.compile)
    model, loader, meta, cfg, ck = eqr_score.load_model(sa, a.split)
    acc = {k: [] for k in ('zh2', 'zl2', 'dzh2', 'dzl2', 'changed', 'exact', 'q_halt')}
    done, t0 = 0, time.perf_counter()
    with torch.inference_mode():
        for _, batch, _ in loader:
            if done >= a.stop:
                break
            bs = batch['inputs'].shape[0]
            batch = {k: v.cuda() for k, v in batch.items()}
            labels = batch['labels']
            mask = labels != IGNORE_LABEL_ID
            nb = min(int(mask.any(-1).sum()), a.stop - done)
            carry = model.initial_carry(batch)
            per = {k: [] for k in acc}
            prev = None
            steps = 0
            while True:
                carry, _, _, preds, finished = model(carry=carry, batch=batch, return_keys={'preds', 'q_halt_logits'})
                steps += 1
                p = preds['preds']
                zh, zl = carry.inner_carry.z_H.float(), carry.inner_carry.z_L.float()
                per['zh2'].append(zh.square().sum((1, 2))[:nb])
                per['zl2'].append(zl.square().sum((1, 2))[:nb])
                if prev is None:
                    nan = torch.full((nb,), float('nan'), device=zh.device)
                    per['dzh2'].append(nan)
                    per['dzl2'].append(nan)
                    per['changed'].append(torch.zeros(nb, dtype=torch.bool, device=zh.device))
                else:
                    per['dzh2'].append((zh - prev[0]).square().sum((1, 2))[:nb])
                    per['dzl2'].append((zl - prev[1]).square().sum((1, 2))[:nb])
                    per['changed'].append(((p != prev[2]) & mask).any(-1)[:nb])
                per['exact'].append((p.eq(labels) | ~mask).all(-1)[:nb])
                per['q_halt'].append(preds['q_halt_logits'].float().view(-1)[:nb])
                prev = (zh, zl, p)
                if bool(finished):
                    break
            assert steps == a.depth, steps
            for k in acc:
                acc[k].append(torch.stack(per[k], 1).cpu().numpy())
            done += bs
            ex = np.concatenate(acc['exact'])
            print(f'{min(done, a.stop)}/{a.stop}  exact@16 {ex[:, 15].mean():.4f} @64 {ex[:, min(63, a.depth - 1)].mean():.4f}  {time.perf_counter() - t0:.0f} s', flush=True)
    torch.cuda.synchronize()
    rec = {k: np.concatenate(v) for k, v in acc.items()}
    n = len(rec['exact'])
    assert n == a.stop, (n, a.stop)
    summary = dict(method='EqR', ckpt=str(a.ckpt), ckpt_sha256=hashlib.sha256(a.ckpt.read_bytes()).hexdigest(), ckpt_step=ck.get('step'),
                   data=str(a.data), split=a.split, n=n, steps=a.depth, trained_depth=a.trained_depth,
                   depth_unit='network evaluation (one ACT step: H_cycles x L_cycles inner updates)', halt_rule='q_halt > 0',
                   noise=a.noise, init_std=a.init_std, batch=a.batch, seed=a.seed,
                   forward_dtype='float32' if a.fp32 else 'checkpoint config (bfloat16)',
                   state='concatenation of z_H and z_L (all positions, fp32 copy of the carry)',
                   solved_per_step=rec['exact'].sum(0).tolist(), seconds=time.perf_counter() - t0, gpu=torch.cuda.get_device_name(),
                   slurm_job=os.environ.get('SLURM_JOB_ID'), eqr_commit=C.EQR_COMMIT,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   eqr_score_sha256=hashlib.sha256(Path(eqr_score.__file__).read_bytes()).hexdigest())
    np.savez_compressed(a.out.with_suffix('.npz'), t=np.arange(1, a.depth + 1), **rec)
    tmp = out_json.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(summary, indent=1))
    os.replace(tmp, out_json)
    print(json.dumps({k: v for k, v in summary.items() if k != 'solved_per_step'}, indent=1), flush=True)


if __name__ == '__main__':
    main(tyro.cli(Args))
