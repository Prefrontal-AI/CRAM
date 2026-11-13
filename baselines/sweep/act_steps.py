"""Instrumented ACT pass for HRM or TRM (compute sweep): exact solve after every ACT step.

HRM and TRM wrap their network in adaptive computation time (ACT): one ACT step is one forward call of the wrapper.
This script calls the authors' code: pretrain.py of the HRM or TRM clone builds the config, the loader and the model.
Evaluation mode never halts early in HRM or TRM (the Q-halt test applies only in training), so the prediction after
ACT step k of a K-step run is the prediction of a k-step run. One pass with arch.halt_max_steps = K therefore gives
the solve rate at every step count k <= K. The model, loader, compilation, batch size and prediction rule are those of
the Table-2 scorers: baselines/hrm/score.py (HRM: argmax of the logits) and baselines/trm/trm_bench.py score (TRM: the
model's preds). A puzzle is solved at step k when every labelled cell is right.

Run from the method's source tree (cwd), with its Python (sweep/run.sbatch does this):
  python <repository>/baselines/sweep/act_steps.py <hrm|trm> <out prefix> <checkpoint> [--ref <score.right.npy>] <hydra overrides...>
Writes <out>.npz (right: (puzzles, K) bool, in the test split's order) and <out>.json (solved per step; with --ref,
the comparison of step halt_max_steps(ref) = 16 with the Table-2 per-puzzle flags).
"""

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
BASELINES = HERE.parent


def load_hrm(overrides, checkpoint):
    """Build HRM as hrm/score.py does. overrides: the hydra overrides of the fit; checkpoint: the state dict file.

    Returns (model in eval mode, test loader, K = arch.halt_max_steps, data folder, the key of the prediction in the model output)."""
    sys.path.insert(0, str(BASELINES / 'hrm'))
    import pretrain
    from fit import alias_adam_atan2
    from hydra import compose, initialize_config_dir
    alias_adam_atan2()
    with initialize_config_dir(config_dir=str(SRC / 'config'), version_base=None):
        hcfg = compose(config_name='cfg_pretrain', overrides=list(overrides))
    config = pretrain.load_synced_config(hcfg, rank=0, world_size=1)
    config.checkpoint_path = None
    bs = config.global_batch_size
    _, train_meta = pretrain.create_dataloader(config, 'train', test_set_mode=False, epochs_per_iter=1, global_batch_size=bs, rank=0, world_size=1)
    loader, _ = pretrain.create_dataloader(config, 'test', test_set_mode=True, epochs_per_iter=1, global_batch_size=bs, rank=0, world_size=1)
    state = pretrain.init_train_state(config, train_meta, world_size=1)
    sd = torch.load(checkpoint, map_location='cuda')
    try:
        state.model.load_state_dict(sd, assign=True)
    except Exception:
        state.model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in sd.items()}, assign=True)
    model = state.model
    model.eval()
    K = config.arch.__pydantic_extra__['halt_max_steps']
    return model, loader, K, Path(config.data_path), 'logits'


def load_trm(overrides, checkpoint):
    """Build TRM as trm/trm_bench.py score does (EMA weights, compiled). Arguments and return value as load_hrm."""
    sys.path.insert(0, str(BASELINES / 'trm'))
    import pretrain
    from trm_bench import build_eval_model, load_config
    cfg = load_config(overrides)
    loader, meta = pretrain.create_dataloader(cfg, 'test', test_set_mode=True, epochs_per_iter=1, global_batch_size=cfg.global_batch_size, rank=0, world_size=1)
    model = build_eval_model(cfg, meta, checkpoint, compile=True)
    K = getattr(model, '_orig_mod', model).model.config.halt_max_steps
    return model, loader, K, Path(cfg.data_paths[0]), 'preds'


def main(method, out, checkpoint, overrides, ref=None):
    """Score the test split after every ACT step and write <out>.npz (right: (puzzles, K) bool) and <out>.json.

    ref: optional score.right.npy of the Table-2 scorer; step 16 of this pass is compared with it."""
    from models.losses import IGNORE_LABEL_ID
    out = Path(out)
    if out.with_suffix('.json').exists():
        print(f'{out}.json exists', flush=True)
        return
    model, loader, K, data, key = (load_hrm if method == 'hrm' else load_trm)(overrides, checkpoint)
    t0 = time.perf_counter()
    rows = []
    import contextlib
    whole = torch.device('cuda') if method == 'trm' else contextlib.nullcontext()  # as trm_bench.score / hrm/score.py
    with torch.inference_mode(), whole:
        for _, batch, _ in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device('cuda'):
                carry = model.initial_carry(batch)
            labels = batch['labels']
            mask = labels != IGNORE_LABEL_ID
            real = mask.any(-1)
            per_step = []
            while True:
                carry, _, _, preds, done = model(carry=carry, batch=batch, return_keys=[key])
                p = preds[key].argmax(-1) if key == 'logits' else preds[key]
                per_step.append((((p == labels) & mask) | ~mask).all(-1)[real])
                if done:
                    break
            assert len(per_step) == K, (len(per_step), K)
            rows.append(torch.stack(per_step, 1).cpu().numpy())
    torch.cuda.synchronize()
    right = np.concatenate(rows)
    n_labels = len(np.load(data / 'test' / 'all__labels.npy', mmap_mode='r'))
    assert right.shape == (n_labels, K), (right.shape, n_labels)
    res = dict(method=method.upper(), checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
               overrides=list(overrides), data=str(data), n=int(len(right)), act_steps=K, solved_per_step=right.sum(0).tolist(),
               seconds=time.perf_counter() - t0, gpu=torch.cuda.get_device_name(), slurm_job=os.environ.get('SLURM_JOB_ID'),
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if ref:
        r = np.load(ref)
        assert r.shape == (len(right),)
        k = 16
        res['check'] = dict(ref=str(ref), step=k, ref_solved=int(r.sum()), sweep_solved=int(right[:, k - 1].sum()),
                            mismatches=int((r != right[:, k - 1]).sum()))
    np.savez_compressed(out.with_suffix('.npz'), right=right)
    out.with_suffix('.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: v for k, v in res.items() if k != 'overrides'}, indent=1), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    method, out, ckpt = args[:3]
    rest = args[3:]
    ref = None
    if rest and rest[0] == '--ref':
        ref, rest = rest[1], rest[2:]
    main(method, out, ckpt, rest, ref)
