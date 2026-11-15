"""Load the trained CRAM models that the paper reports. Needs only numpy and torch.

    from checkpoints.load import load, names
    weights = load('maze-unique', seed=0)      # dict: tensor name -> numpy array
    weights = load('graph-coloring-8')         # one model: its fit is deterministic

Run it as a script to check every file against its recorded SHA-256 and to print each model's tensors:

    uv run python checkpoints/load.py

README.md in this folder gives the meaning of each tensor and the script that evaluates each model.
"""
import hashlib
import io
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MANIFEST = json.loads((HERE / 'manifest.json').read_text())


def names():
    """The model names: sudoku-extreme, maze-unique, nqueens-8, nqueens-10, graph-coloring-8, graph-coloring-10."""
    return list(MANIFEST)


def path(name, seed=0):
    """Locate the weights without verifying them. A model with one fit ignores the training seed."""
    entry = MANIFEST[name]
    return ROOT / _model(name, seed)['folder'] / entry['weights']


def _model(name, seed):
    models = MANIFEST[name]['models']
    return models[seed if len(models) > 1 else 0]


def _read_verified(model, filename):
    path = ROOT / model['folder'] / filename
    expected = model['sha256'].get(filename)
    if expected is None:
        raise ValueError(f'{path}: missing SHA-256 in the manifest')
    resolved = path.resolve()
    if not resolved.is_relative_to(HERE):
        raise ValueError(f'{path}: artifact is outside the checkpoint directory')
    data = resolved.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f'{path}: SHA-256 differs from the manifest')
    return data


def read_artifact(name, filename, seed=0):
    """Read a file listed for this model and seed, raising if its recorded SHA-256 differs."""
    return _read_verified(_model(name, seed), filename)


def load_torch(name, seed=0, *, map_location='cpu'):
    """Load a verified .pt checkpoint as tensors and its original scalar metadata."""
    import torch

    filename = MANIFEST[name]['weights']
    if Path(filename).suffix != '.pt':
        raise ValueError(f'{name}: expected a .pt checkpoint')
    return torch.load(io.BytesIO(read_artifact(name, filename, seed)), map_location=map_location, weights_only=True)


def load(name, seed=0):
    """The weights of model `name` as a dict of numpy arrays (scalars and strings become 0-dimensional arrays)."""
    filename = MANIFEST[name]['weights']
    if Path(filename).suffix == '.npz':
        with np.load(io.BytesIO(read_artifact(name, filename, seed)), allow_pickle=False) as weights:
            return dict(weights)
    import torch
    return {k: (v.numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in load_torch(name, seed).items()}


def check():
    """Compare every file with its recorded SHA-256; return the number of files that differ."""
    bad = 0
    for entry in MANIFEST.values():
        for model in entry['models']:
            for filename in dict.fromkeys((entry['weights'], *model['sha256'])):
                try:
                    _read_verified(model, filename)
                except (OSError, ValueError) as error:
                    print(f'DIFFERS: {error}')
                    bad += 1
    return bad


if __name__ == '__main__':
    bad = check()
    print('all files match their SHA-256' if bad == 0 else f'{bad} checkpoint file(s) failed verification')
    if bad:
        raise SystemExit(1)
    for name, entry in MANIFEST.items():
        print(f'\n{name}  ({entry["table"]}; {len(entry["models"])} fit(s))')
        for seed in range(len(entry['models'])):
            w = load(name, seed)
            print(f'  {entry["models"][seed]["folder"]}')
            for k, v in w.items():
                if v.dtype.kind in 'US' or v.ndim == 0:
                    print(f'    {k:6s} = {v}')
                else:
                    print(f'    {k:6s} shape {str(v.shape):16s} nonzero {int(np.count_nonzero(v)):7d}')
