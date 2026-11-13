"""Data efficiency: the 10,000 test puzzles that every point of the data-efficiency figure is scored on, as a test split
in the format of EBR (Energy-Based Reasoner). The puzzles are rows 10,000-19,999 of data/full-test/full-all.npz (the
first 10,000 Sudoku-Extreme test puzzles outside the development set dev10k; CRAM's `test10k`, the compute sweep's puzzles).

Rows: ebr/report.py sweep_official_rows() of EBR's Sudoku test split (byte-identical to the official test order), in
that order. Checks each row against full-all.npz rows 10,000-19,999 (solutions and clue masks), the file CRAM reads.
Output: experiments/runs/ebr/data/sudoku-test10k/{test, train -> the 1k set's train}; evaluate.py reads only test.

  python ebr/test10k.py      (numpy only; needs the dataset that ebr/build_data.sh wrote)
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from report import sweep_official_rows  # noqa: E402

EXPERIMENTS = HERE.parents[1] / 'experiments'
SRC = EXPERIMENTS / 'runs/ebr/data/sudoku-extreme-1k-aug-1000'  # EBR's own Sudoku build (ebr/build_data.sh)
OUT = EXPERIMENTS / 'runs/ebr/data/sudoku-test10k'
FULL_ALL = HERE.parents[1] / 'data/full-test/full-all.npz'

rows = sweep_official_rows()
assert len(rows) == 10000 and len(set(rows.tolist())) == 10000
x = np.load(SRC / 'test/all__inputs.npy')[rows]
y = np.load(SRC / 'test/all__labels.npy')[rows]
f = np.load(FULL_ALL)
sol, clue = f['solutions'][10000:20000].astype(np.int64), f['clue_mask'][10000:20000].astype(bool)
assert np.array_equal(y.astype(np.int64) - 1, sol), 'labels differ from full-all rows 10,000-19,999'
assert np.array_equal(x != 1, clue), 'clue masks differ'
assert np.array_equal(x[clue].astype(np.int64), y[clue].astype(np.int64)), 'a clue disagrees with its label'
d = OUT / 'test'
d.mkdir(parents=True, exist_ok=True)
n = len(x)
files = dict(inputs=x, labels=y, puzzle_indices=np.arange(n + 1, dtype=np.int32), group_indices=np.arange(n + 1, dtype=np.int32),
             puzzle_identifiers=np.zeros(n, dtype=np.int32))
for k, v in files.items():
    np.save(d / f'all__{k}.npy', v)
meta = json.loads((SRC / 'test/dataset.json').read_text())
meta.update(total_groups=n, total_puzzles=n, total_samples=n)
(d / 'dataset.json').write_text(json.dumps(meta))
for name in ('train', 'identifiers.json'):
    if not (OUT / name).exists() and (SRC / name).exists():
        (OUT / name).symlink_to(SRC / name)
rec = dict(official_rows=rows.tolist(), full_all_rows=[10000, 20000], source=str(SRC / 'test'),
           sha256={k: hashlib.sha256((d / f'all__{k}.npy').read_bytes()).hexdigest() for k in files},
           script=str(Path(__file__).resolve()), source_dataset_json=json.loads((SRC / 'test/dataset.json').read_text()))
(OUT / 'subset.json').write_text(json.dumps(rec) + '\n')
print('wrote', OUT, n, 'boards; rows match full-all 10,000-19,999')
