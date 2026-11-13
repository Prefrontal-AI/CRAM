"""Score a trained HRM model on the test split with HRM's own evaluation (outside the fit clock).

As HRM's evaluate.py: the model from init_train_state (compiled as pretrain.py compiles it unless DISABLE_COMPILE),
the checkpoint loaded into it, eval mode, the test loader in test_set_mode at the training global batch size, and the
ACT loop run until all sequences halt. In eval mode HRM halts every sequence at halt_max_steps (16), the authors'
inference setting. Prediction = argmax of the logits, as ACTLossHead's exact_accuracy; a puzzle is solved when every
labelled cell is right. HRM's own exact_accuracy metric is summed on the same batches as a check.

Sudoku: the HRM/TRM test split holds the official 422,786 test puzzles in the official order (checked against
data/frm/official/test.npz). dev10k is a set of 10,000 official rows (data/frm/dev10k/indices.json),
not the first 10,000, so the unseen 412,786 are the rows outside those indices. Top-level solved/n = unseen.

Run from the HRM source tree:
  python baselines/hrm/score.py <out.json> <checkpoint> <hydra overrides...>
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path.cwd()
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[2]
V1_DATA = Path(os.environ.get('V1_DATA', ROOT / 'data'))


def dev10k_rows(labels):
    """The official row indices of dev10k, checked against dev10k's own solutions."""
    idx = np.array(json.loads((V1_DATA / 'frm/dev10k/indices.json').read_text()))
    dev = np.load(V1_DATA / 'frm/dev10k/dev.npz')
    assert len(idx) == 10000 and len(set(idx.tolist())) == 10000
    assert (dev['solutions'].reshape(-1, 81) == labels[idx].astype(np.int64) - 1).all(), 'dev10k rows do not match'
    return idx


def main(out, checkpoint, overrides):
    import pretrain
    from fit import alias_adam_atan2, attention_backend
    from hydra import compose, initialize_config_dir
    from models.losses import IGNORE_LABEL_ID
    alias_adam_atan2()
    with initialize_config_dir(config_dir=str(SRC / 'config'), version_base=None):
        hcfg = compose(config_name='cfg_pretrain', overrides=list(overrides))
    config = pretrain.load_synced_config(hcfg, rank=0, world_size=1)
    config.checkpoint_path = None
    bs = config.global_batch_size
    train_loader, train_meta = pretrain.create_dataloader(config, 'train', test_set_mode=False, epochs_per_iter=1,
                                                          global_batch_size=bs, rank=0, world_size=1)
    loader, meta = pretrain.create_dataloader(config, 'test', test_set_mode=True, epochs_per_iter=1,
                                              global_batch_size=bs, rank=0, world_size=1)
    state = pretrain.init_train_state(config, train_meta, world_size=1)
    sd = torch.load(checkpoint, map_location='cuda')
    try:
        state.model.load_state_dict(sd, assign=True)
    except Exception:
        state.model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in sd.items()}, assign=True)
    model = state.model
    model.eval()
    t0 = time.perf_counter()
    right, steps_per_batch, sums = [], [], {}
    with torch.inference_mode():
        for _, batch, _ in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device('cuda'):
                carry = model.initial_carry(batch)
            n_steps = 0
            while True:
                carry, _, metrics, preds, done = model(carry=carry, batch=batch, return_keys=['logits'])
                n_steps += 1
                if done:
                    break
            steps_per_batch.append(n_steps)
            labels = batch['labels']
            mask = labels != IGNORE_LABEL_ID
            real = mask.any(-1)  # padding rows of the last batch have no label
            hit = (preds['logits'].argmax(-1) == labels) & mask
            right.append((hit | ~mask).all(-1)[real].cpu().numpy())
            for k in ('count', 'exact_accuracy'):
                sums[k] = sums.get(k, 0.0) + float(metrics[k])
    torch.cuda.synchronize()
    right = np.concatenate(right)
    labels_all = np.load(Path(config.data_path) / 'test' / 'all__labels.npy', mmap_mode='r')
    assert right.size == len(labels_all) == sums['count'], (right.size, len(labels_all), sums['count'])
    assert int(sums['exact_accuracy']) == int(right.sum()), (sums, int(right.sum()))
    res = dict(checkpoint=str(checkpoint), overrides=list(overrides), data_path=config.data_path,
               act_steps=sorted(set(steps_per_batch)), halt_max_steps=config.arch.__pydantic_extra__['halt_max_steps'],
               global_batch_size=bs, attention=attention_backend(), compiled='DISABLE_COMPILE' not in os.environ,
               score_seconds=time.perf_counter() - t0, gpu=torch.cuda.get_device_name(),
               slurm_job=os.environ.get('SLURM_JOB_ID'),
               hrm_metric=dict(exact_accuracy_sum=sums['exact_accuracy'], count=sums['count']))
    if 'sudoku' in Path(config.data_path).name:
        dev = dev10k_rows(np.asarray(labels_all))
        unseen = np.ones(right.size, bool)
        unseen[dev] = False
        res.update(solved=int(right[unseen].sum()), n=int(unseen.sum()), exact=float(right[unseen].mean()),
                   split='official test rows outside dev10k (412,786 unseen)',
                   all=dict(solved=int(right.sum()), n=int(right.size), exact=float(right.mean())),
                   dev10k=dict(solved=int(right[dev].sum()), n=int(dev.size)))
    else:
        res.update(solved=int(right.sum()), n=int(right.size), exact=float(right.mean()), split='test')
    np.save(Path(out).with_suffix('.right.npy'), right)
    Path(out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
