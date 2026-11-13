"""Per-puzzle operation counts under the paper's arithmetic-only convention, rebuilt from saved per-puzzle test records
without rescoring. The counted components are in results/op-counts.json (written by baselines/flops/fc2_export.py from
the counter records in baselines/runs/flops-full/).

The operation-count convention of baselines/flops/fullcount.py: matrix products (2 per multiply-add), elementwise
arithmetic and transcendental evaluations count; comparisons, selections and mask logic are recorded but not counted;
indexing, data movement and random-number generation are free. CRAM's Triton kernels are counted through line-by-line
PyTorch transcriptions (baselines/flops/cram_kernels.py).

"flop" in this file is one counted floating-point operation. A search record stores, per puzzle, the in-kernel
estimate `flops` (moves x chains x `fl_move`); the functions here recover the number of moves from it and multiply by
the counted cost of a move.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OPCOUNTS = ROOT / 'results/op-counts.json'


@lru_cache(maxsize=1)
def components():
    """The `components` dict of results/op-counts.json: the counted cost of each part of each method."""
    return json.loads(OPCOUNTS.read_text())['components']


def cram0_sudoku(old_flops, signature, with_compare=False):
    """Per-puzzle counted operations of the Sudoku search of the paper (cram_sudoku_general.GeneralSearch, no restarts).

    old_flops: the per-puzzle `flops` of a test record (moves x C x fl_move, C the chains per puzzle); signature: the
    record's search settings (fl_move, C, D the neighbour count, check the moves between two checks).
    Returns an array: moves x C x move(D) + start + ceil(moves / check) x check cost. A puzzle is charged up to the move
    at which its stopping test first passes (the kernel tests every move). with_compare also counts comparisons."""
    c = components()['cram0_sudoku']
    key = 'total_with_compare' if with_compare else 'total'
    old = np.asarray(old_flops, float)
    steps = old / (signature['fl_move'] * signature['C'])
    assert np.allclose(steps, np.round(steps), atol=1e-6), 'record has restarts; this rebuild assumes none'
    steps = np.round(steps)
    move = c['move'][str(signature['D'])][key]
    assert signature['check'] == c['check_moves'] and signature['C'] == c['C']
    return steps * signature['C'] * move + c['start'][key] + np.ceil(steps / signature['check']) * c['check'][key]


def unit(method, task, name='per_step'):
    """A baseline's counted unit (operations per ACT step, per network evaluation, per call, ...) under the convention.
    ACT (adaptive computation time) is the halting loop of the looped baselines."""
    return components()['baselines'][method][task][name]


__all__ = ['components', 'cram0_sudoku', 'unit']
