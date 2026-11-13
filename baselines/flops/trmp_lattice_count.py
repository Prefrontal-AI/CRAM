"""Inference operation count of the TRMP lattice model on Sudoku-Extreme (baselines/trmp/lattice.py, a reimplementation
from the TRMP paper; no upstream code), under the operation-count convention of baselines/flops/fullcount.py.

The counted region is the whole chain search, lattice.chain_search, for one puzzle (64 chains): every step runs the
network on 64 chains, projects them, and does the slot bookkeeping. Its arithmetic does not depend on the data, so the
count of a puzzle is a + b * steps, with steps the number of search steps run on that puzzle. a and b are measured by
FullCounter over chain_search on dev puzzles one at a time (slots = 1), fitted on two puzzles with different step counts
and checked exactly on the others. Per-puzzle test totals then come from the step counts that lattice_score.py saved.

  python flops/trmp_lattice_count.py --run runs/trmp-lattice/sudoku-extreme-s0        # from baselines/
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import tyro

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'trmp'))
from fullcount import CONVENTION, FullCounter  # noqa: E402
from clock import write_json  # noqa: E402
import lattice as L  # noqa: E402
import lattice_fit as LF  # noqa: E402


@dataclasses.dataclass
class Args:
    run: str = 'runs/trmp-lattice/sudoku-extreme-s0'  # fit folder with model.pt and score/test.steps.npy
    puzzles: int = 6  # dev puzzles counted one at a time
    out: str = ''  # output json (default: <run>/score/flops.json)
    task: str = 'sudoku-extreme'  # benchmark name written to the record (only sudoku-extreme is supported)


def main(a: Args):
    """Count chain_search on dev puzzles, fit count = a + b * steps, and write the record with the per-puzzle test totals."""
    run = Path(a.run)
    assert a.task == 'sudoku-extreme', a.task
    net = L.LatticeNet().cuda().eval()
    net.load_state_dict(torch.load(run / 'model.pt', map_location='cuda', weights_only=True))
    x0, _ = LF.read_dev(a.puzzles)
    recipe = L.RECIPE
    theta_cls = L.THETA_CLS_EVAL
    rows = []
    for i in range(a.puzzles):
        gen = torch.Generator(device='cuda').manual_seed(7 + i)
        with FullCounter() as fc:
            _, steps, _ = L.chain_search(net, x0[i:i + 1].cuda(), chains=recipe['test_chains'], budget=recipe['test_budget'],
                                         check_every=1, theta_cls=theta_cls, slots=1, amp=True, generator=gen,
                                         theta=json.loads((run / 'fit.json').read_text())['recipe'].get('theta', L.THETA))
        s = fc.summary()
        rows.append(dict(puzzle=i, steps=int(steps[0]), flop=int(s['total']), flop_with_compare=int(s['total_with_compare'])))
        print(rows[-1], flush=True)
    st = np.array([r['steps'] for r in rows])
    fl = np.array([r['flop'] for r in rows])
    i, j = int(st.argmin()), int(st.argmax())
    assert st[i] != st[j], 'need two puzzles with different step counts'
    b = (fl[j] - fl[i]) // (st[j] - st[i])
    a0 = fl[i] - b * st[i]
    exact = bool(all(f == a0 + b * s for f, s in zip(fl, st)))
    assert exact, 'the count is not linear in the number of steps'
    res = dict(convention=CONVENTION, per_puzzle_constant=int(a0), per_step=int(b), per_chain_step=b / recipe['test_chains'],
               chains=recipe['test_chains'], task=a.task, validation_rows=rows, linear_exact=exact,
               model_sha256=hashlib.sha256((run / 'model.pt').read_bytes()).hexdigest())
    steps_path = run / 'score' / 'test.steps.npy'
    if steps_path.exists():
        t = np.load(steps_path).astype(np.int64)
        per = a0 + b * t
        res.update(test_puzzles=len(t), test_mean_steps=float(t.mean()), flop_per_puzzle_mean=float(per.mean()),
                   flop_per_puzzle_median=float(np.median(per)), flop_per_puzzle_max=float(per.max()))
    write_json(Path(a.out) if a.out else run / 'score' / 'flops.json', res)
    print(json.dumps({k: v for k, v in res.items() if k != 'validation_rows'}, indent=1), flush=True)


if __name__ == '__main__':
    main(tyro.cli(Args))
