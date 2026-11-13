"""Latent dynamics and q-head halting of HRM or TRM run past the trained depth.

HRM and TRM wrap their network in adaptive computation time (ACT): one ACT step is one forward call of the wrapper,
and a halting head (the q-head) decides in training when to stop. This script calls the authors' code: the model,
loader, compilation, prediction rule and exact-solve test are those of baselines/sweep/act_steps.py (load_hrm,
load_trm), which builds them from the clone's pretrain.py.
Evaluation mode never halts early, so one pass with arch.halt_max_steps = K runs every puzzle for K ACT steps.
After every ACT step t this records, per puzzle:
- the squared norms of the carried latents z_H and z_L and of their change since step t - 1 (fp32, from the bf16 carry);
- whether the decoded answer differs from the answer at step t - 1, and whether it is exactly right;
- the halting logits q_halt and q_continue.
The q-head halting of training (without exploration) is applied afterwards from these logits:
a puzzle halts at the first step where q_halt > q_continue (HRM) or q_halt > 0 (TRM, no_ACT_continue).
Each batch row is computed independently, so a puzzle that would halt at step k has the same state at step k here.

Puzzles: the rows in --rows (a .npy of test-split row indices), built into batches of the config's global batch size
with the official loader's collation and padding.

Run from the method's source tree (cwd), with its Python (dynamics/run.sbatch does this):
  python act_dyn.py <hrm|trm> <out prefix> <checkpoint> --rows <rows.npy> [--ref <act32.npz>] <hydra overrides...>
Writes <out>.npz and, last, <out>.json; exits at once when <out>.json exists.
Dimension names: n puzzles, t ACT steps (1-based in the files).
"""

import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path.cwd()
sys.path.insert(0, str(SRC))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'sweep'))
import act_steps  # noqa: E402  (load_hrm, load_trm: the loaders of the compute-sweep scorer)


def collate(data, rows, bs, ignore_label_id, pad_id, blank_id, IGNORE):
    """The official test loader's _collate_batch for the given rows, padded to bs rows."""
    b = {'inputs': data['inputs'][rows].astype(np.int32), 'labels': data['labels'][rows].astype(np.int32),
         'puzzle_identifiers': data['puzzle_identifiers'][data['puzzle_of_row'][rows]].astype(np.int32)}
    if ignore_label_id is not None:
        b['labels'][b['labels'] == ignore_label_id] = IGNORE
    pad = bs - len(rows)
    if pad:
        values = {'inputs': pad_id, 'labels': IGNORE, 'puzzle_identifiers': blank_id}
        b = {k: np.pad(v, ((0, pad),) + ((0, 0),) * (v.ndim - 1), constant_values=values[k]) for k, v in b.items()}
    return {k: torch.from_numpy(v).cuda() for k, v in b.items()}


def main(method, out, checkpoint, rows_file, overrides, ref=None):
    """Run the puzzles of rows_file for K = arch.halt_max_steps ACT steps and write <out>.npz and <out>.json.

    method: 'hrm' or 'trm'; checkpoint: the trained weights; overrides: the hydra overrides of the fit;
    ref: optional act32.npz of sweep/act_steps.py, whose per-step exact flags must equal the ones recorded here.
    Arrays in <out>.npz are (n, t): zh2, zl2 (squared norms), dzh2, dzl2 (squared change), changed, exact, q_halt, q_continue."""
    from models.losses import IGNORE_LABEL_ID
    out = Path(out)
    if out.with_suffix('.json').exists():
        print(f'{out}.json exists', flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    model, loader, K, data_dir, key = (act_steps.load_hrm if method == 'hrm' else act_steps.load_trm)(overrides, checkpoint)
    inner = getattr(model, '_orig_mod', model).model
    cfg = inner.config
    no_continue = bool(getattr(cfg, 'no_ACT_continue', False))
    rule = 'q_halt > 0' if no_continue else 'q_halt > q_continue'
    meta = json.loads((data_dir / 'test' / 'dataset.json').read_text())
    test = data_dir / 'test'
    data = {f: np.load(test / f'all__{f}.npy', mmap_mode='r') for f in ('inputs', 'labels', 'puzzle_identifiers', 'puzzle_indices')}
    pidx = np.asarray(data['puzzle_indices'])
    data['puzzle_of_row'] = np.searchsorted(pidx, np.arange(len(data['inputs'])), side='right') - 1
    rows = np.load(rows_file)
    bs = int(next(iter(loader))[1]['inputs'].shape[0])
    t0 = time.perf_counter()
    acc = {k: [] for k in ('zh2', 'zl2', 'dzh2', 'dzl2', 'changed', 'exact', 'q_halt', 'q_continue')}
    whole = torch.device('cuda') if method == 'trm' else contextlib.nullcontext()  # as act_steps.py
    with torch.inference_mode(), whole:
        for lo in range(0, len(rows), bs):
            r = rows[lo:lo + bs]
            nr = len(r)
            batch = collate(data, r, bs, meta.get('ignore_label_id'), meta['pad_id'], meta['blank_identifier_id'], IGNORE_LABEL_ID)
            with torch.device('cuda'):
                carry = model.initial_carry(batch)
            labels = batch['labels']
            mask = labels != IGNORE_LABEL_ID
            prev = None
            per = {k: [] for k in acc}
            for t in range(K):
                carry, _, _, preds, done = model(carry=carry, batch=batch, return_keys=[key, 'q_halt_logits', 'q_continue_logits'])
                p = preds[key].argmax(-1) if key == 'logits' else preds[key]
                zh, zl = carry.inner_carry.z_H.float(), carry.inner_carry.z_L.float()
                per['zh2'].append(zh.square().sum((1, 2))[:nr])
                per['zl2'].append(zl.square().sum((1, 2))[:nr])
                if prev is None:
                    nan = torch.full((nr,), float('nan'), device=zh.device)
                    per['dzh2'].append(nan)
                    per['dzl2'].append(nan)
                    per['changed'].append(torch.zeros(nr, dtype=torch.bool, device=zh.device))
                else:
                    per['dzh2'].append((zh - prev[0]).square().sum((1, 2))[:nr])
                    per['dzl2'].append((zl - prev[1]).square().sum((1, 2))[:nr])
                    per['changed'].append(((p != prev[2]) & mask).any(-1)[:nr])
                per['exact'].append((((p == labels) & mask) | ~mask).all(-1)[:nr])
                per['q_halt'].append(preds['q_halt_logits'].float()[:nr])
                qc = preds.get('q_continue_logits')
                per['q_continue'].append(qc.float()[:nr] if qc is not None else torch.full((nr,), float('nan'), device=zh.device))
                prev = (zh, zl, p)
                if t + 1 < K:
                    assert not bool(done), ('halted early', t + 1)
            assert bool(done), 'not done after K steps'
            for k in acc:
                acc[k].append(torch.stack(per[k], 1).cpu().numpy())
            print(f'{lo + nr}/{len(rows)}  exact@16 {np.concatenate(acc["exact"])[:, 15].mean():.4f}  {time.perf_counter() - t0:.0f} s', flush=True)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    rec = {k: np.concatenate(v) for k, v in acc.items()}
    res = dict(method=method.upper(), checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
               overrides=list(overrides), data=str(data_dir), rows_file=str(rows_file), n=int(len(rows)), steps=K,
               trained_depth=16, depth_unit='ACT step (one forward call of the ACT wrapper)', halt_rule=rule,
               state='concatenation of z_H and z_L (all positions, fp32 copy of the bf16 carry)',
               solved_per_step=rec['exact'].sum(0).tolist(), seconds=seconds, gpu=torch.cuda.get_device_name(),
               slurm_job=os.environ.get('SLURM_JOB_ID'), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if ref:  # check: step 16 (and every step up to 32) equals the compute-sweep pass (sweep/act_steps.py) on the same rows
        r = np.load(ref)['right'][rows]
        k = min(r.shape[1], K)
        res['check'] = dict(ref=str(ref), steps_compared=k, mismatches_step16=int((r[:, 15] != rec['exact'][:, 15]).sum()),
                            mismatches_all_steps=int((r[:, :k] != rec['exact'][:, :k]).sum()), ref_solved_step16=int(r[:, 15].sum()))
    np.savez_compressed(out.with_suffix('.npz'), t=np.arange(1, K + 1), rows=rows, **rec)
    tmp = out.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(res, indent=1) + '\n')
    os.replace(tmp, out.with_suffix('.json'))
    print(json.dumps({k: v for k, v in res.items() if k not in ('overrides', 'solved_per_step')}, indent=1), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    method, out, ckpt = args[:3]
    rest = args[3:]
    assert rest[0] == '--rows'
    rows_file, rest = rest[1], rest[2:]
    ref = None
    if rest and rest[0] == '--ref':
        ref, rest = rest[1], rest[2:]
    main(method, out, ckpt, rows_file, rest, ref)
