"""Choose an HRM checkpoint on development data, then score only that checkpoint on test (outside the fit clock).

Candidates: every checkpoint pretrain.py's checkpoint_every_eval wrote in the run folder (step_<n>, one per eval
interval, the last one being the final model). Each is scored on development data with HRM's eval loop (eval mode,
halt_max_steps = 16 ACT steps, argmax; the same loop as score.py):
- Sudoku: dev10k, the 10,000 official test rows in data/frm/dev10k/indices.json (excluded from the unseen
  412,786 that score.py reports), written as a small HRM-format folder <run>/dev-data/test/;
- Maze-Unique: the generated dev100 (EqR's generator, 100 perfect mazes with unique paths, no overlap with the train
  or test split; experiments/runs/TRMP-audit/generated-validation/dev), the set the TRMP Maze-Unique row selects on.
  EqR's dev100 is 100 of the training mazes, on which HRM trains, so it is not used.
Rule: the highest dev exact solve count; ties go to the latest step. Test is never read for the choice.
Outputs in the run folder: selection.json (dev score of every candidate, the rule, the choice); score-final.json (the
final checkpoint on test: the score fit.sbatch wrote as score.json, renamed, or computed here); score.json (the
selected checkpoint on test via score.py, plus selected_step, dev_solved, dev_n). fit.json is not touched.

Run from the HRM source tree:
  python baselines/hrm/select.py <run_dir> <hydra overrides...>
"""

import json
import os
import shutil
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
MAZE_DEV = Path(os.environ.get('HRM_MAZE_DEV', ROOT / 'experiments/runs/TRMP-audit/generated-validation'))


def sudoku_dev(test_dir, out):
    """dev10k as an HRM test split: the official rows at dev10k's indices, checked against dev10k's solutions."""
    from score import dev10k_rows
    labels = np.load(test_dir / 'all__labels.npy')
    idx = dev10k_rows(labels)
    d = out / 'test'
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / 'all__inputs.npy', np.load(test_dir / 'all__inputs.npy')[idx])
    np.save(d / 'all__labels.npy', labels[idx])
    np.save(d / 'all__puzzle_indices.npy', np.arange(len(idx) + 1, dtype=np.int32))
    np.save(d / 'all__group_indices.npy', np.arange(len(idx) + 1, dtype=np.int32))
    np.save(d / 'all__puzzle_identifiers.npy', np.zeros(len(idx), dtype=np.int32))
    meta = json.loads((test_dir / 'dataset.json').read_text())
    meta.update(total_groups=len(idx), mean_puzzle_examples=1.0)
    (d / 'dataset.json').write_text(json.dumps(meta))
    return out, 'dev10k (10,000 official test rows at data/frm/dev10k/indices.json)'


def maze_dev():
    manifest = json.loads((MAZE_DEV / 'final.json').read_text())
    import hashlib
    for kind in ('inputs', 'labels'):
        assert hashlib.sha256((MAZE_DEV / f'dev/all__{kind}.npy').read_bytes()).hexdigest() == manifest[f'{kind}_sha256']
    assert manifest['overlap_count'] == 0
    return MAZE_DEV, 'generated dev100 (experiments/runs/TRMP-audit/generated-validation/dev)'


def dev_right(model, config, data_path, split):
    """Per-puzzle exact solve on one split, HRM's eval loop (as score.py)."""
    import pretrain
    from models.losses import IGNORE_LABEL_ID
    cfg = config.model_copy(update=dict(data_path=str(data_path)))
    loader, _ = pretrain.create_dataloader(cfg, split, test_set_mode=True, epochs_per_iter=1,
                                           global_batch_size=cfg.global_batch_size, rank=0, world_size=1)
    right, total = [], 0.0
    with torch.inference_mode():
        for _, batch, _ in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device('cuda'):
                carry = model.initial_carry(batch)
            while True:
                carry, _, metrics, preds, done = model(carry=carry, batch=batch, return_keys=['logits'])
                if done:
                    break
            labels = batch['labels']
            mask = labels != IGNORE_LABEL_ID
            real = mask.any(-1)
            hit = (preds['logits'].argmax(-1) == labels) & mask
            right.append((hit | ~mask).all(-1)[real].cpu().numpy())
            total += float(metrics['exact_accuracy'])
    right = np.concatenate(right)
    assert int(total) == int(right.sum())
    return right


def main(run, overrides):
    import pretrain
    import score
    from fit import alias_adam_atan2
    from hydra import compose, initialize_config_dir
    run = Path(run)
    fit = json.loads((run / 'fit.json').read_text())
    if (run / 'score.json').exists() and 'selected_step' in json.loads((run / 'score.json').read_text()):
        print(f'{run}/score.json is already the selected score: nothing to do', flush=True)
        return
    alias_adam_atan2()
    with initialize_config_dir(config_dir=str(SRC / 'config'), version_base=None):
        hcfg = compose(config_name='cfg_pretrain', overrides=list(overrides))
    config = pretrain.load_synced_config(hcfg, rank=0, world_size=1)
    config.checkpoint_path = None
    steps = sorted(int(p.name.removeprefix('step_')) for p in run.glob('step_*') if p.name.removeprefix('step_').isdigit())
    assert steps and steps[-1] == fit['steps'] and f'step_{steps[-1]}' == fit['final_checkpoint'], (steps, fit['steps'])
    iters = config.epochs // config.eval_interval
    assert len(steps) == iters, f'{len(steps)} checkpoints, expected one per eval interval ({iters})'

    # 1. the final checkpoint's test score (fit.sbatch wrote it as score.json)
    if not (run / 'score-final.json').exists():
        if (run / 'score.json').exists():
            os.replace(run / 'score.right.npy', run / 'score-final.right.npy')
            os.replace(run / 'score.json', run / 'score-final.json')
        else:
            score.main(str(run / 'score-final.json'), str(run / fit['final_checkpoint']), overrides)
    final = json.loads((run / 'score-final.json').read_text())
    assert final['checkpoint'].endswith(fit['final_checkpoint'])

    # 2. every candidate on development data
    if 'sudoku' in Path(config.data_path).name:
        dev_path, dev_name = sudoku_dev(Path(config.data_path) / 'test', run / 'dev-data')
        dev_split = 'test'
    else:
        dev_path, dev_name = maze_dev()
        dev_split = 'dev'
    train_loader, train_meta = pretrain.create_dataloader(config, 'train', test_set_mode=False, epochs_per_iter=1,
                                                          global_batch_size=config.global_batch_size, rank=0,
                                                          world_size=1)
    state = pretrain.init_train_state(config, train_meta, world_size=1)
    model = state.model
    model.eval()
    t0 = time.perf_counter()
    cands = []
    for s in steps:
        sd = torch.load(run / f'step_{s}', map_location='cuda')
        sd = {k.removeprefix('_orig_mod.'): v for k, v in sd.items()}
        getattr(model, '_orig_mod', model).load_state_dict(sd)  # copied into the same tensors (no recompilation)
        right = dev_right(model, config, dev_path, dev_split)
        cands.append(dict(step=s, dev_solved=int(right.sum()), dev_n=int(right.size)))
        print('DEV ' + json.dumps(cands[-1]), flush=True)
    best = max(cands, key=lambda c: (c['dev_solved'], c['step']))  # highest dev solve count; ties: latest step
    sel = dict(rule='highest dev exact solve count among the per-eval-interval checkpoints; ties: latest step',
               dev=dev_name, candidates=cands, selected_step=best['step'], dev_solved=best['dev_solved'],
               dev_n=best['dev_n'], dev_seconds=time.perf_counter() - t0, slurm_job=os.environ.get('SLURM_JOB_ID'),
               overrides=list(overrides))
    (run / 'selection.json').write_text(json.dumps(sel, indent=1) + '\n')
    del model, state
    torch.cuda.empty_cache()

    # 3. the selected checkpoint on test, once
    if best['step'] == steps[-1]:
        shutil.copy(run / 'score-final.right.npy', run / 'score.right.npy')
        res = dict(final)
    else:
        score.main(str(run / 'score.json'), str(run / f"step_{best['step']}"), overrides)
        res = json.loads((run / 'score.json').read_text())
    res.update(selected_step=best['step'], dev_solved=best['dev_solved'], dev_n=best['dev_n'], dev=dev_name,
               selection='selection.json', final_step=steps[-1], final_solved=final['solved'], final_n=final['n'])
    tmp = run / 'score.json.tmp'
    tmp.write_text(json.dumps(res, indent=1) + '\n')
    os.replace(tmp, run / 'score.json')
    print('SELECTED ' + json.dumps({k: res[k] for k in ('selected_step', 'dev_solved', 'dev_n', 'solved', 'n')}), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
