"""Shared setup for the EqR wrappers: the clone on sys.path, the recipes (presets), the config and the model.

EqR = locuslab/EqR at commit 134c754 (a clone in $EQR_SRC, unmodified), run with its own Python ($EQR_SRC/.venv).
Two stand-ins replace packages that would need a CUDA build, without editing EqR's files:
- adam_atan2: the pure-PyTorch AdamATan2 in ../trm/trm_shim (the same update as the CUDA kernel of adam-atan2 0.0.3),
  put first on sys.path.
- flash attention: EqR's Attention calls flash_attn_func whenever head_dim % 8 == 0 on CUDA (the paper's Maze model,
  head_dim 16) and raises if FlashAttention is missing. `models.layers.flash_attn_func` is set to an exact stand-in
  that calls torch's scaled_dot_product_attention (non-causal, softmax scale 1/sqrt(head_dim), as flash's default).

Recipes (each is EqR's own hydra train config plus overrides; total updates = epochs x floor(groups / 768)):
- sudoku: config/train/eqr_sudoku.yaml unchanged (the released checkpoint's config): MLP-mixer token mixer, hidden 512,
  2 layers, H 3 x L 6, 16 ACT segments, lr 1e-4, wd 1.0, batch 768, EMA 0.999, warmup 2,000, 50,000 updates.
- maze-paper: the released Maze-Unique checkpoint's config (paper Table 27): attention, hidden 128, 8 heads, 1 layer,
  H 3 x L 4, lr 1e-4, wd 1.0, noise 0.01, 150,000 updates (the checkpoint's epochs; Table 27 prints 100k).
- maze-repo: config/train/eqr_maze_unique.yaml unchanged (the repository's current default, from the Maze-Unique
  supplement): hidden 44, 2 heads, lr 1e-3, wd 0.1, 150,000 updates.
"""

import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
THIRD_PARTY = os.environ.get('THIRD_PARTY', str(Path(__file__).resolve().parents[2] / 'third_party'))  # clones of the authors' code (baselines/README.md)
EQR_SRC = Path(os.environ.get('EQR_SRC', THIRD_PARTY + '/eqr'))
EQR_COMMIT = '134c7541711125860954c78463023c583b2ef21b'
DATA = Path(os.environ.get('EQR_DATA', HERE.parents[1] / 'experiments' / 'runs' / 'eqr' / 'data'))
sys.path.insert(0, str(EQR_SRC))
sys.path.insert(0, str(HERE.parent / 'trm' / 'trm_shim'))

PRESETS = {
    'sudoku': ('train/eqr_sudoku', []),
    'maze-paper': ('train/eqr_maze_unique', ['arch.hidden_size=128', 'arch.num_heads=8', 'arch.puzzle_emb_ndim=128',
                                             'lr=1e-4', 'weight_decay=1.0', 'arch.noise_scale=0.01']),
    'maze-repo': ('train/eqr_maze_unique', []),
}
# The test-time settings of the paper (README "Reproduce Sudoku results"; Table 4): depth D (outer iterations =
# network evaluations per trajectory), breadth B (restarts), path noise and latent reset scale at evaluation.
EVAL_NOISE = 0.5
EVAL_INIT_STD = 1.0


def _sdpa_as_flash(q, k, v, causal=False, **kw):
    """flash_attn_func(q, k, v) with (batch, seq, heads, head_dim) tensors, computed by torch's SDPA."""
    assert not kw, kw
    y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
    return y.transpose(1, 2)


def install_stand_ins():
    import models.layers
    if models.layers.flash_attn_func is None:
        models.layers.flash_attn_func = _sdpa_as_flash


def load_config(preset, data_path, seed=0, overrides=()):
    """EqR's PretrainConfig for a preset, with the dataset path, the seed and extra hydra overrides."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from config.schema import PretrainConfig
    name, pre = PRESETS[preset]
    ov = [*pre, f'dataset.data_path={data_path}', f'+seed={seed}', *overrides]
    with initialize_config_dir(config_dir=str(EQR_SRC / 'config'), version_base=None):
        hcfg = compose(config_name=name, overrides=ov)
    cfg = PretrainConfig(**OmegaConf.to_container(hcfg, resolve=True))
    cfg.wandb_mode = 'disabled'
    return cfg


def total_updates(cfg, meta):
    return cfg.epochs * max(math.floor(meta.total_groups * meta.mean_puzzle_examples / cfg.global_batch_size), 1)
