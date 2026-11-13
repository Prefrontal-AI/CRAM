"""One results row per (benchmark, model), written as experiments/runs/table/<benchmark>__<row>.json.

The run scripts write the rows of their final results with `write_row`. A row holds the accuracy of each training
seed on the test set, the parameter count, the mean inference operation count and the fit time.
"""

import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

TABLE = Path(__file__).resolve().parents[1] / 'experiments' / 'runs' / 'table'

BENCHMARKS = ('sudoku-extreme', 'maze-unique', 'maze-hard',
              'nqueens-8', 'nqueens-10',  # GRAM's N-Queens sets (cram/nqueens_data.py)
              'graph-coloring-8', 'graph-coloring-10')  # GRAM's graph-coloring sets (cram/graph_coloring_data.py)
ROWS = (
    'hand-energy',       # the CRAM energy and search with hand-set weights, no learned parameters (hand_set.py)
)  # every other row key starts with 'cram-'


def _commit():
    """The short hash of the repository's HEAD commit, '' outside a git checkout, None without git."""
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except OSError:
        return None


def write_row(benchmark, row, *, label, per_seed, n_test, test_set, params, flop_per_puzzle,
              train_gpu_s, train_cpu_s=None, source='measured', provenance='', notes='', extra=None):
    """Write one row and return its path. `per_seed` is the exact-solve fraction of each training seed on the test set
    (0..1).

    `flop_per_puzzle` is the mean inference flop per test puzzle; `train_gpu_s` the training wall clock of one seed on
    one H200 (mean over seeds); `train_cpu_s` the same on one CPU thread, or None when not measured.
    `label`, `test_set`, `provenance` and `notes` are free text; `extra` is a dict stored under the key 'extra'.
    """
    assert benchmark in BENCHMARKS, benchmark
    assert row in ROWS or row.startswith('cram-'), row
    per_seed = [float(s) for s in per_seed]
    mean = statistics.fmean(per_seed) if per_seed else math.nan
    sd = statistics.stdev(per_seed) if len(per_seed) > 1 else None
    out = {
        'benchmark': benchmark, 'row': row, 'label': label,
        'solved_mean': mean, 'solved_sd': sd, 'solved_peak': max(per_seed) if per_seed else None,
        'per_seed': per_seed, 'n_test': n_test, 'test_set': test_set,
        'params': params, 'flop_per_puzzle': flop_per_puzzle,
        'train_gpu_s': train_gpu_s, 'train_cpu_s': train_cpu_s,
        'source': source, 'provenance': provenance, 'notes': notes,
        'written': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'commit': _commit(),
    }
    if extra:
        out['extra'] = extra
    TABLE.mkdir(parents=True, exist_ok=True)
    path = TABLE / f'{benchmark}__{row}.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    return path


def read_rows():
    """Every row written so far, as a list of dicts."""
    return [json.loads(p.read_text()) for p in sorted(TABLE.glob('*.json'))]


if __name__ == '__main__':
    for r in read_rows():
        sd = f" ± {100 * r['solved_sd']:.2f}" if r['solved_sd'] is not None else ''
        print(f"{r['benchmark']:15s} {r['row']:18s} {100 * r['solved_mean']:7.3f}{sd:9s} params {r['params']}"
              f"  flop {r['flop_per_puzzle']:.2g}  gpu {r['train_gpu_s']}s  cpu {r['train_cpu_s']}  [{r['source']}]")
