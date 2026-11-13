"""CRAM Sudoku-Extreme solve rate against a per-puzzle cap on the operation count, from the saved test records (no reruns).

  python compute_curve_export.py      (from experiments/; reads about 60 MB of records)

"flop" means one counted floating-point operation.
Each test record experiments/runs/CRAM-S/full/test/cram0-hb-fit-s{0,1,2}.json (sudoku.py, with the stopping threshold
fitted from the training solutions, E <= tau + 1e-3) holds, for each of the 412,786 test puzzles, the flop at which its
search stopped and whether the answer is correct (cap 4e11). A puzzle's search follows a fixed random sequence, so with
a lower cap b it stops at the same point if it used at most b flop; otherwise it would hit the cap and return its
lowest-energy board, which this export counts as unsolved below the records' cap (a slight underestimate). At the
records' cap the returned board is the record's, so a run that hit the cap with the right board counts as solved there,
as in the table record.
The mean flop at cap b is mean(min(flop, b)); p99 is the 99th-percentile per-puzzle flop at cap b. Two puzzle sets: all
412,786 test puzzles, and full-all rows 10,000-19,999 (the first 10,000 test puzzles), the subset of the baseline
compute sweeps (results/compute-sweep/).
Operation counts: each puzzle's count is rebuilt from its record under the operation-count convention of
baselines/flops/fullcount.py (cram/opcount.cram0_sudoku: steps x C x counted move + start + checks). The records' cap
of 4e11 flop, which is in the search's own per-move count, becomes, per seed, the count of its move cap T_cap.
Dimensions: p = puzzles, s = seeds.
Writes results/cram-anytime/sudoku-extreme.json; its `description` field is this text.
"""
import json
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[1]
RECORDS = [CODE / f'experiments/runs/CRAM-S/full/test/cram0-hb-fit-s{s}.json' for s in range(3)]
OUT = CODE / 'results/cram-anytime/sudoku-extreme.json'
GRID = 10 ** np.arange(6.0, 11.75 + 1e-9, 0.125)  # 8 caps per decade; the records' cap (per seed, under the operation-count convention) is added


def curve(solved, flop, cap, caps):
    """Solve rate and compute used at each cap.

    solved: (s, p) bool. flop: (s, p) operation count at which each search stopped. cap: (s,) the records' cap per
    seed. caps: the caps to evaluate. Returns one dict per cap: the solve rate per seed, its mean and sample SD, and
    the mean and 99th percentile of min(flop, cap)."""
    rows = []
    for b in caps:
        rate = (solved & ((flop <= b) | (b >= cap[:, None]))).mean(1)  # (s,)
        used = np.minimum(flop, b)  # (s, p)
        rows.append(dict(cap=float(b), solved_per_seed=rate.tolist(), solved_mean=float(rate.mean()), solved_sd=float(rate.std(ddof=1)),
                         mean_flop_per_seed=used.mean(1).tolist(), mean_flop=float(used.mean()),
                         p99_flop_per_seed=np.percentile(used, 99, axis=1).tolist(), p99_flop=float(np.percentile(used, 99, axis=1).mean())))
    return rows


def main():
    """Read the three test records, rebuild the operation counts and write the curves of both puzzle sets."""
    from cram import opcount
    solved, flop, puzzle, cap = [], [], [], []
    for path in RECORDS:
        rec = json.loads(path.read_text())
        sig = rec['signature']
        assert sig['test'] == 'fit', path
        pp = rec['per_puzzle']
        solved.append(np.asarray(pp['solved'], bool))
        # the records hold steps x C x fl_move; rebuild each puzzle's count under the operation-count convention
        flop.append(opcount.cram0_sudoku(pp['flops'], sig))
        t_end = -(-pp['T_cap'] // sig['check']) * sig['check']  # the search runs whole checks: it ends at the first check >= T_cap
        cap.append(float(opcount.cram0_sudoku([t_end * sig['fl_move'] * sig['C']], sig)[0]))  # the records' move cap
        puzzle.append(np.asarray(pp['puzzle']))
    assert all(np.array_equal(puzzle[0], x) for x in puzzle)
    cap = np.asarray(cap)
    assert all(f.max() <= c + 1 for f, c in zip(flop, cap))
    CAPS = np.unique(np.concatenate([GRID[GRID < cap.min()], [cap.max()]]))
    solved, flop, puzzle = np.stack(solved), np.stack(flop), puzzle[0]  # (s, p), (s, p), (p,) full-all rows
    sub = (puzzle >= 10000) & (puzzle < 20000)
    assert sub.sum() == 10000
    sets = {'unseen-412786': curve(solved, flop, cap, CAPS), 'first10k': curve(solved[:, sub], flop[:, sub], cap, CAPS)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(description=__doc__.strip(), op_counts='FC2: the arithmetic-only operation-count convention of baselines/flops/fullcount.py (cram/opcount.py, results/op-counts.json)',
                                   record_caps=cap.tolist(), records=[str(p.relative_to(CODE)) for p in RECORDS],
                                   n_puzzles={'unseen-412786': int(flop.shape[1]), 'first10k': int(sub.sum())}, sets=sets), indent=1) + '\n')
    for name, rows in sets.items():
        print(name)
        for r in rows[::4] + rows[-1:]:
            print(f"{r['cap']:9.3g}  solved {100 * r['solved_mean']:8.4f} +/- {100 * r['solved_sd']:.4f}  mean TFLOPs {r['mean_flop'] / 1e12:.3g}  p99 TFLOPs {r['p99_flop'] / 1e12:.3g}")


if __name__ == '__main__':
    main()
