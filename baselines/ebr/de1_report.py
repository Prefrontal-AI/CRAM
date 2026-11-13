"""Data efficiency: scores and operation counts of one EBR (Energy-Based Reasoner) checkpoint on test10k.

test10k is the 10,000 Sudoku-Extreme test puzzles of the data-efficiency figure: rows 10,000-19,999 of
data/full-test/full-all.npz, stored in EBR's format by ebr/test10k.py (experiments/runs/ebr/data/sudoku-test10k).
Input: the single shard that EBR's evaluate.py wrote for that set (ebr/score.sbatch with --num_shards 1, submitted by
ebr/de2_submit.sh), and the per-component operation counts of ebr/flops.py.
Output: one JSON record with every inference setting of ebr/report.py (fixed depth, and the stop at
|grad_M E| < tau), each with its solved count and mean operations per puzzle. CPU only; no model runs.

  python ebr/de1_report.py <score10k dir> <flops.json> <out.json>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from report import digest, load_sudoku, settings, summarize  # noqa: E402


def main(folder, flops, out):
    """Read the shard in `folder` and the counts in `flops` (both paths); write the record to `out` and print three settings."""
    exact, energy, gtan, metas, shards = load_sudoku(folder)
    assert exact.shape[1] == 10000, exact.shape
    sets = settings(exact, energy, gtan, json.loads(Path(flops).read_text()))
    rec = dict(method='EBR', test='test10k: full-all rows 10,000-19,999 (runs/ebr/data/sudoku-test10k)',
               checkpoint=metas[0]['checkpoint'], breadth=metas[0]['breadth'], depth=metas[0]['depth'],
               flops=flops, flops_sha256=digest(flops), shards={str(p): digest(p) for p in shards},
               script=str(Path(__file__)), script_sha256=digest(__file__), test10k=summarize(sets, np.ones(10000, bool)))
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    for key in ('D64-N1', 'D64-N5', 'gtan0.2-cap64-N5'):
        u = rec['test10k'][key]
        print(f"{key}: {u['rate'] * 100:.3f}% ({u['solved']}/{u['n']}), {u['mean_flop']:.3g} flop")


if __name__ == '__main__':
    main(*sys.argv[1:4])
