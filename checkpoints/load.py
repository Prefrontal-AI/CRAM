"""Load the trained CRAM models that the paper reports. Needs only numpy and torch.

    from checkpoints.load import load, names
    weights = load('maze-unique', seed=0)      # dict: tensor name -> numpy array
    weights = load('graph-coloring-8')         # one model: its fit is deterministic

Run it as a script to check every file against its recorded SHA-256 and to print each model's tensors:

    uv run python checkpoints/load.py

README.md in this folder gives the meaning of each tensor and the script that evaluates each model.
"""
import hashlib
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
    """The weights file of model `name`. `seed` is the training seed (0, 1, 2); a model with one fit ignores it."""
    entry = MANIFEST[name]
    models = entry['models']
    return ROOT / models[seed if len(models) > 1 else 0]['folder'] / entry['weights']


def load(name, seed=0):
    """The weights of model `name` as a dict of numpy arrays (scalars and strings become 0-dimensional arrays)."""
    p = path(name, seed)
    if p.suffix == '.npz':
        return dict(np.load(p))
    import torch
    return {k: (v.numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in torch.load(p, map_location='cpu', weights_only=True).items()}


def check():
    """Compare every file with its recorded SHA-256; return the number of files that differ."""
    bad = 0
    for name, entry in MANIFEST.items():
        for model in entry['models']:
            for f, digest in model['sha256'].items():
                if hashlib.sha256((ROOT / model['folder'] / f).read_bytes()).hexdigest() != digest:
                    print(f'DIFFERS: {model["folder"]}/{f}')
                    bad += 1
    return bad


if __name__ == '__main__':
    bad = check()
    print('all files match their SHA-256' if bad == 0 else f'{bad} files differ from the manifest')
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
