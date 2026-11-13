"""Maze-Unique (EqR's maze-30x30-unique-1k, locuslab/EqR-data bcb7d45c) in EBR's maze layout.

EqR's files are already HRM token ids with EBR's maze charset ("# SGo" -> 1..5, 0 = padding), one row per maze and no
augmentation, as EBR's Maze-Hard builder writes them (aug=False). This adds the index arrays EBR's loader needs.
The four source files are checked against the SHA-256 values that the CRAM Maze-Unique runs pin (cram/maze_unique.py).
Calls EBR's dataset.common.write_hrm_split (github.com/wwillchen/ebr at 8c5271c).
Source: data/eqr/maze-30x30-unique-1k (scripts/prepare_data.py), or $EQR_DIR.
Run from EBR's source tree (ebr/build_data.sh does this): python <this file> <output dir>
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

from dataset.common import write_hrm_split

EQR = Path(os.environ.get('EQR_DIR', Path(__file__).resolve().parents[2] / 'data' / 'eqr/maze-30x30-unique-1k'))
SHA256 = {
    'train/all__inputs.npy': '2922455815232dcb636ee45f335e7ce834aa45d4fab743b96ac97c17284ca7e2',
    'train/all__labels.npy': '6b1a99761606b78f85fe43085b158457f22a3d6274b1e09f12f7b16814fdb5e4',
    'test/all__inputs.npy': 'c501b3f75ec018e27fd6d00534b6738ebcc555da50536909e530787d16b1ce0f',
    'test/all__labels.npy': '20b21168459847fb260d986394098addf7cca94becc7e682dcb358967d5993dc',
}


def read(rel):
    """Load one source array (path relative to the EqR folder) after checking its SHA-256."""
    path = EQR / rel
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == SHA256[rel], f'{path}: sha256 {digest} is not the pinned file'
    return np.load(path)


def main(out):
    """Write the train and test splits, 1,000 mazes of 900 cells each, to the folder `out`."""
    for split in ('train', 'test'):
        x, y = read(f'{split}/all__inputs.npy'), read(f'{split}/all__labels.npy')
        n = len(x)
        assert x.shape == y.shape == (1000, 900) and set(np.unique(x)) <= {1, 2, 3, 4} and set(np.unique(y)) <= {1, 2, 3, 4, 5}
        arrays = dict(inputs=x.astype(np.uint8), labels=y.astype(np.uint8), group_indices=np.arange(n + 1, dtype=np.int32),
                      puzzle_indices=np.arange(n + 1, dtype=np.int32), puzzle_identifiers=np.zeros(n, dtype=np.int32))
        meta = dict(pad_id=0, ignore_label_id=0, blank_identifier_id=0, vocab_size=6, seq_len=900, num_puzzle_identifiers=1,
                    total_groups=n, mean_puzzle_examples=1.0, sets=['all'], total_samples=None)
        write_hrm_split(out, split, arrays, meta)
        print(f'[maze-unique/{split}] {n} examples')
    Path(out, 'identifiers.json').write_text(json.dumps(['<blank>']))


if __name__ == '__main__':
    main(sys.argv[1])
