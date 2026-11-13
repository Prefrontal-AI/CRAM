"""Score an EqR checkpoint per puzzle at a depth D and breadth B, as EqR's evaluate.py scores it.

EqR's inference (evaluators/eval_fn.py, non-streaming path): each puzzle runs B independent trajectories (the batch
repeated B times; each trajectory's latents start from a fresh Gaussian draw, scale --init-std), each for D outer
iterations (D network evaluations; ACT halting is not used at evaluation, so every trajectory runs all D), with path
noise --noise added at every inner update. The answer of a trajectory is the argmax of its last logits. Three
selections over the B trajectories, none of which reads the label:
- top1_conv (the paper's headline, "Top-1 convergence"): the trajectory whose logits changed least over the last 3
  iterations (mean over cells of the L2 norm of the logit change, averaged over the window).
- majority: the most frequent answer grid (ties to the earliest trajectory, as EqR's code).
- first: trajectory 0 alone (B = 1 is the plain run).
Also recorded: any (some trajectory is right; an oracle, never a result) and mean (mean exact over trajectories).

EqR's own totals count the padding rows of the last batch as solved (their labels are all ignored); this file scores
only real puzzles, and checks its totals against EqR's own accumulators (_update_di, _update_conv) run on the real rows.

The checkpoint may be EqR's released file (loaded with safe_load's allowlist unpickler) or a model.pt/best.pt of
eqr_train.py. A stage whose <out>.json exists exits at once; a killed run resumes from <out>.partial.npz (saved every --save-minutes
at a batch boundary), reseeding the evaluation noise with seed + puzzles done. Seconds per puzzle are measured past the
first batch of each start (which includes compilation), synchronized, on the GPU named in the output.
Run with EqR's Python:
  python eqr_score.py --ckpt <file> --data <dataset dir> --split test --depth 64 --breadth 128 --out <prefix>
"""

import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro

import eqr_common as C
import safe_load


@dataclass
class Args:
    ckpt: Path  # checkpoint in EqR's format (model, ema, config)
    data: Path  # EqR-format dataset directory
    out: Path  # output prefix: <out>.json and <out>.npz (per-puzzle results)
    split: str = 'test'  # split folder inside --data
    depth: int = 16  # D: outer iterations (network evaluations) per trajectory
    breadth: int = 1  # B: independent trajectories (restarts) per puzzle
    noise: float = C.EVAL_NOISE  # path noise scale at evaluation (EqR's eval configs: 0.5)
    init_std: float = C.EVAL_INIT_STD  # scale of the random latent initialisation (EqR's eval configs: 1.0)
    window: int = 3  # convergence window L for top1_conv (the paper's L = 3)
    top_k: int = 4  # k of EqR's convergence_top_k metric (recorded for comparison with its logs)
    batch: int = 128  # puzzles per batch (each repeated B times)
    limit: Optional[int] = None  # score only the first `limit` puzzles of the split
    budget_minutes: Optional[float] = None  # stop after the batch that passes this wall-clock budget (partial result)
    seed: int = 0  # evaluation seed (EqR's evaluate.py: torch, numpy and random seeded with it)
    compile: bool = True  # torch.compile the model, as EqR's evaluate.py does
    save_minutes: float = 10.0  # minutes between partial-result saves (<out>.partial.npz); a restart resumes from the last


def load_model(a: Args, meta_split):
    import pretrain
    from config.schema import PretrainConfig
    from evaluate import _apply_runtime_arch_overrides, _seed, _setup_random_reset_carry
    ck = safe_load.load(a.ckpt, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    cfg = PretrainConfig(**ck['config'])
    cfg.dataset.data_path = str(a.data)
    cfg.global_batch_size = a.batch * a.breadth
    cfg.checkpoint_path = None
    loader, meta = pretrain.create_dataloader(cfg, meta_split, rank=0, world_size=1, test_set_mode=True, epochs_per_iter=1,
                                              global_batch_size=a.batch, enable_shared_sampler_state=False)
    model, _, _ = pretrain.create_model(cfg, meta, rank=0, world_size=1)
    sd = {k.replace('_orig_mod.', ''): v for k, v in ck['model'].items()}
    res = model.load_state_dict(sd, strict=False)
    assert all(k.endswith(('local_weights', 'local_ids')) for k in res.missing_keys) and not res.unexpected_keys, res
    shadow = {k.replace('_orig_mod.', ''): v for k, v in ck['ema']['shadow'].items()}
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(shadow[name])  # EqR evaluates the EMA weights (utils/checkpoint.load_ema)
    model.eval()
    model.model.config.halt_max_steps = a.depth
    combo = {'arch.halt_max_steps': a.depth, 'arch.noise_scale': a.noise, 'arch.H_init_std': a.init_std, 'arch.L_init_std': a.init_std}
    _apply_runtime_arch_overrides(model, combo, 0)
    if a.breadth > 1:
        _setup_random_reset_carry(model, a.init_std, 0)
    _seed(a.seed, 0)
    if a.compile and 'DISABLE_COMPILE' not in os.environ:
        model = torch.compile(model, dynamic=False)
    return model, loader, meta, cfg, ck


def main(a: Args):
    out_json, part = a.out.with_suffix('.json'), a.out.with_suffix('.partial.npz')
    if out_json.exists():
        print(f'{out_json} exists: nothing to do', flush=True)
        return
    C.install_stand_ins()
    from evaluate import _seed
    from evaluators.eval_fn import _conv_state, _di_state, _update_conv, _update_di
    from models.losses import IGNORE_LABEL_ID
    model, loader, meta, cfg, ck = load_model(a, a.split)
    n_total = meta.total_groups if a.limit is None else min(a.limit, meta.total_groups)
    B, dev = a.breadth, torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    keys = ('first', 'top1_conv', 'majority', 'any', 'mean', 'top1_score')
    out = {k: [] for k in keys}
    di, conv = _di_state(dev), _conv_state(dev, min(a.top_k, B))
    done, el0, warm = 0, 0.0, [0.0, 0]  # puzzles scored before this start, their seconds, (seconds, puzzles) past the first batch
    if part.exists():  # resume at a batch boundary; the evaluation noise is reseeded with seed + puzzles done
        z = np.load(part)
        out = {k: [z[k]] for k in keys}
        done, el0, warm = int(z['done']), float(z['seconds']), [float(z['warm_s']), int(z['warm_n'])]
        for st, name in ((di, 'di'), (conv, 'conv')):
            for k in st:
                st[k] += float(z[f'{name}_{k}'])
        _seed(a.seed + done, 0)
        print(f'RESUME scoring at puzzle {done} ({el0:.0f} s so far)', flush=True)
    t0 = last_save = time.perf_counter()
    skipped, resume_at = 0, done
    first_batch = True

    def save_partial():
        el = el0 + time.perf_counter() - t0
        tmp = part.with_suffix('.tmp.npz')
        np.savez(tmp, **{k: np.concatenate(v) for k, v in out.items()}, done=done, seconds=el, warm_s=warm[0], warm_n=warm[1],
                 **{f'di_{k}': float(v) for k, v in di.items()}, **{f'conv_{k}': float(v) for k, v in conv.items()})
        os.replace(tmp, part)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for _, batch, _ in loader:
            if skipped < resume_at:  # scored before the restart
                skipped += batch['inputs'].shape[0]
                continue
            tb = time.perf_counter()
            batch = {k: v.to(dev) for k, v in batch.items()}
            labels = batch['labels']
            real = (labels != IGNORE_LABEL_ID).any(-1)
            nb = min(int(real.sum()), n_total - done)  # real rows come first; padding last
            if nb <= 0:
                break
            rows = batch if B == 1 else {k: v.repeat_interleave(B, dim=0) for k, v in batch.items()}
            carry = model.initial_carry(rows)
            last = deque(maxlen=a.window + 1)
            steps = 0
            while True:
                carry, _, _, preds, finished = model(carry=carry, batch=rows, return_keys={'preds', 'logits'})
                last.append(preds['logits'].float())
                steps += 1
                if bool(finished):
                    break
            assert steps == a.depth, steps
            bs = labels.shape[0]
            pred = preds['preds'].view(bs, B, -1)
            mask = labels != IGNORE_LABEL_ID
            exact = (pred.eq(labels[:, None]) | ~mask[:, None]).all(-1)  # (bs, B)
            lg = torch.stack(list(last))
            score = (lg[1:] - lg[:-1]).norm(dim=-1).mean(dim=(0, -1)).view(bs, B)  # EqR's convergence score
            sel = score.argmin(1)
            ar = torch.arange(bs, device=dev)
            # majority vote, EqR's rule: the most frequent grid, ties to the earliest trajectory
            maj = torch.zeros(bs, dtype=torch.bool, device=dev)
            if B > 1:
                for i in range(nb):
                    votes = {}
                    for r, row in enumerate(pred[i].cpu().numpy()):
                        votes.setdefault(row.tobytes(), [0, r])[0] += 1
                    win = max(votes.values(), key=lambda v: v[0])[1]  # max() keeps the first of equal counts
                    maj[i] = exact[i, win]
            else:
                maj = exact[:, 0]
            for k, v in (('first', exact[:, 0]), ('top1_conv', exact[ar, sel]), ('majority', maj), ('any', exact.any(1)),
                         ('mean', exact.float().mean(1)), ('top1_score', score[ar, sel])):
                out[k].append(v[:nb].cpu().numpy())
            if B > 1:  # EqR's own accumulators on the real rows, as a check of the selections above
                _update_di(di, preds['preds'].view(bs, B, -1)[:nb].reshape(nb * B, -1), labels[:nb], B)
                _update_conv(conv, preds['preds'].view(bs, B, -1)[:nb].reshape(nb * B, -1), labels[:nb], score[:nb], min(a.top_k, B))
            done += nb
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            if not first_batch:  # the first batch after a start includes compilation
                warm[0] += time.perf_counter() - tb
                warm[1] += nb
            first_batch = False
            el = el0 + time.perf_counter() - t0
            print(f'{done}/{n_total} puzzles  top1_conv {np.concatenate(out["top1_conv"]).mean():.4f}  '
                  f'first {np.concatenate(out["first"]).mean():.4f}  {el:.0f} s', flush=True)
            if time.perf_counter() - last_save > 60 * a.save_minutes:
                save_partial()
                last_save = time.perf_counter()
            if done >= n_total or (a.budget_minutes is not None and time.perf_counter() - t0 > 60 * a.budget_minutes):
                break
    el = el0 + time.perf_counter() - t0
    res = {k: np.concatenate(v)[:n_total] for k, v in out.items()}
    n = len(res['first'])
    summary = dict(ckpt=str(a.ckpt), ckpt_step=ck.get('step'), data=str(a.data), split=a.split, n=n,
                   n_split=meta.total_groups, depth=a.depth, breadth=B, nfe_per_puzzle=a.depth * B, noise=a.noise,
                   init_std=a.init_std, window=a.window, seed=a.seed,
                   exact={k: float(res[k].mean()) for k in ('first', 'top1_conv', 'majority', 'any', 'mean')},
                   solved={k: int(res[k].sum()) for k in ('first', 'top1_conv', 'majority', 'any')},
                   seconds=el, puzzles_per_second=n / el, seconds_per_puzzle_warm=warm[0] / warm[1] if warm[1] else None,
                   warm_puzzles_timed=warm[1], gpu=torch.cuda.get_device_name() if dev.type == 'cuda' else 'cpu',
                   max_memory_gb=torch.cuda.max_memory_allocated() / 2**30 if dev.type == 'cuda' else None, batch=a.batch, eqr_commit=C.EQR_COMMIT)
    if B > 1:
        tot = {k: float(v) for k, v in di.items()}
        ct = {k: float(v) for k, v in conv.items()}
        summary['eqr_metrics'] = dict(any_correct=tot['any'] / tot['samples'], majority_vote=tot['majority_exact'] / tot['samples'],
                                      avg_pass_rate=tot['pass_sum'] / tot['samples'],
                                      cumulative_exact_acc_top1=ct['prefix_0'] / ct['samples'], samples=tot['samples'])
        assert tot['samples'] == ct['samples'] == n or n < tot['samples'], (tot['samples'], n)
        if tot['samples'] == n:
            assert abs(summary['eqr_metrics']['any_correct'] - summary['exact']['any']) < 1e-9
            assert abs(summary['eqr_metrics']['majority_vote'] - summary['exact']['majority']) < 1e-9
            assert abs(summary['eqr_metrics']['cumulative_exact_acc_top1'] - summary['exact']['top1_conv']) < 1e-9
    np.savez(a.out.with_suffix('.npz'), **res)
    tmp = out_json.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(summary, indent=1))
    os.replace(tmp, out_json)
    if part.exists():
        part.unlink()
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main(tyro.cli(Args))
