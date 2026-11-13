"""FPRM (Fixed-Point Reasoning Model, arXiv 2606.18206) on Maze-Hard: the authors' model with the authors' checkpoint.

Source: github.com/nilskiKonjIzDunava/fprm (commit b291252). Checkpoint and its configuration: the Hugging Face model
repository fixed-point-reasoners/fprm, folder maze/ (step_78120 = final EMA weights, all_config.yaml). The repository's
models/fixed_point_reasoning/fprm.py equals the checkpoint's fp_trm_singlez.py except for class names.
Evaluation setting of the published 87.0% (maze/README.md there): at most 35,000 fixed-point iterations, step-size
decay 0.996 with patience 10, restart seed 4, batches of 768.

Run from the FPRM source tree with its Python (FPRM_SRC, default $THIRD_PARTY/fprm/fprm; THIRD_PARTY defaults to <repository>/third_party).
"""
import hashlib
import os
import sys
from pathlib import Path

import torch
import yaml

THIRD_PARTY = Path(os.environ.get('THIRD_PARTY', Path(__file__).resolve().parents[2] / 'third_party'))  # clones of the authors' code (baselines/README.md)
SRC = Path(os.environ.get('FPRM_SRC', THIRD_PARTY / 'fprm' / 'fprm'))
HF = Path(os.environ.get('FPRM_HF', SRC.parent / 'hf' / 'maze'))
DATA = Path(os.environ.get('FPRM_DATA', SRC / 'data' / 'maze-30x30-hard-1k-noaug'))
CHECKPOINT = 'step_78120'
PUBLISHED = dict(max_iter_eval=35000, stepsize_decay_eval=0.996, decay_patience=10, seed=4, batch_size=768)
sys.path.insert(0, str(SRC))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(batch_size):
    """The eval-mode model with the published evaluation setting, and the record of what was loaded."""
    import json
    from models.fixed_point_reasoning.fprm import FixedPointReasoningModel_ACTV1
    arch = yaml.safe_load((HF / 'all_config.yaml').read_text())['arch']
    arch = {k: v for k, v in arch.items() if k not in ('name', 'loss')}
    meta = json.loads((DATA / 'test' / 'dataset.json').read_text())
    cfg = dict(arch, batch_size=batch_size, vocab_size=meta['vocab_size'], seq_len=meta['seq_len'], num_puzzle_identifiers=meta['num_puzzle_identifiers'])
    cfg.update(max_iter_eval=PUBLISHED['max_iter_eval'], stepsize_decay_eval=PUBLISHED['stepsize_decay_eval'], decay_patience=PUBLISHED['decay_patience'])
    with torch.device('cuda'):
        model = FixedPointReasoningModel_ACTV1(cfg)
    sd = torch.load(HF / CHECKPOINT, map_location='cuda', weights_only=False)
    sd = {k.removeprefix('_orig_mod.model.'): v for k, v in sd.items()}
    result = model.load_state_dict(sd, strict=True, assign=True)  # strict: every saved tensor is used, none is missing
    model.eval()
    model.set_num_iters()
    opt = model.inner.L_optimizer
    assert (model.max_iter, opt.stepsize_decay_eval, opt.decay_patience) == (35000, 0.996, 10)
    named = {k: v.numel() for k, v in model.named_parameters()}
    buffers = {k: v.numel() for k, v in model.named_buffers()}
    masks = {k: v for k, v in model.named_buffers() if k.endswith('.mask')}
    assert all(bool((m == 1).all()) for m in masks.values()), 'a weight-dropout mask is not all ones'
    rec = dict(source=str(SRC), checkpoint=str(HF / CHECKPOINT), checkpoint_sha256=sha256(HF / CHECKPOINT), config_sha256=sha256(HF / 'all_config.yaml'),
               load_result=str(result), params=sum(named.values()), params_by_tensor=named,
               puzzle_embedding_buffer=buffers['inner.puzzle_emb.weights'],
               params_with_puzzle_embedding=sum(named.values()) + buffers['inner.puzzle_emb.weights'],
               params_note='sum over model.parameters(), as for HRM and TRM; the puzzle embedding (one 512-vector on Maze-Hard) is a buffer; the .mask buffers are weight-dropout masks, all ones',
               seq_len_incl_puzzle_emb=meta['seq_len'] + model.inner.puzzle_emb_len, setting=dict(PUBLISHED, batch_size=batch_size))
    return model, meta, rec


def test_batches(batch_size):
    """The authors' test loader, as their evaluation script builds it: file order; the last batch is padded to
    batch_size with pad rows (labels all -100). Yields (batch on the GPU, number of real mazes in it)."""
    from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig
    ds = PuzzleDataset(PuzzleDatasetConfig(seed=0, dataset_paths=[str(DATA)], rank=0, num_replicas=1, test_set_mode=True, epochs_per_iter=1, global_batch_size=batch_size), split='test')
    for _, batch, _ in ds:
        batch = {k: v.cuda() for k, v in batch.items()}
        yield batch, int((batch['labels'] != -100).any(dim=1).sum())
