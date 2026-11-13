"""Per-puzzle compute and accuracy against Sudoku-Extreme difficulty, for CRAM and the Table 2 baselines.

  python difficulty_export.py      (from experiments/; reads about 0.5 GB of records)

Subset: full-all rows 10,000-19,999 (the first 10,000 test puzzles: FRM's 1,000 test puzzles, then the first 9,000 of
shard full-00), which every saved baseline score covers. Difficulty: the dataset's `rating` column (full-all.npz
`ratings`, the test.csv value). Every model is scored from its saved records; nothing is run again.
"flop" means one counted floating-point operation; ACT is adaptive computation time, the halting loop of HRM and TRM.
  CRAM   experiments/runs/CRAM-S/full/test/cram0-hb-s{0,1,2}.json   per-puzzle flops and solved (412,786 test puzzles, full-all
         order); the records of the search with the zero-energy stopping test (sudoku.py --scale hbar --stop zero)
  HRM    baselines/runs/hrm/sudoku-e40000-s{k}/score.right.npy   official order; 16 ACT steps
  TRM    baselines/runs/trm/sudoku-s{k}/score.right.npy          official order; 16 ACT steps
  EqR    baselines/runs/eqr/sudoku-s{k}/score-*.npz              top1_conv (Table 2's metric); depth D x breadth B network evaluations
  FRM    baselines/runs/frm-compiled{,-s1,-s2}/sudoku/score-unseen.solved.npz; 4,097 calls
  TRMP   baselines/runs/trmp/sudoku-extreme-s{k}/score/test/*.npz  64 rollouts, counted per rollout step
  EBR    baselines/runs/ebr/sudoku-s{k}/report-fc2.flags.npz      per-puzzle solved and flop of gtan0.2-cap64-N8, the compute
         sweep's most accurate EBR setting (8 samples, each stopped at |grad_M E| < 0.2 or 64 segments); official order
Operation counts: the cost of each unit (ACT step, network evaluation, call, rollout step) is the count of
results/op-counts.json (cram/opcount.py) under the operation-count convention of baselines/flops/fullcount.py, and
CRAM's per-puzzle counts are rebuilt from its records (opcount.cram0_sudoku).
The paper's figure shows CRAM, EBR and FRM; the file also holds the other Table 2 baselines.
Writes results/difficulty/sudoku-difficulty.json. Dimensions: p = puzzles of the subset, s = seeds, k = rollouts.
"""
import glob
import json
from pathlib import Path

import numpy as np

from cram import opcount
from cram.paths import DATA

CODE = Path(__file__).resolve().parents[1]
T = CODE / 'baselines/runs'
LO, N = 10000, 10000  # the subset: full-all rows LO .. LO + N - 1
BINS = [0, 1, 2, 4, 8, 16, 32, 64, 10**9]  # rating edges: bin b holds lo <= rating < hi; the last bin holds every rating >= 64
BUDGETS = np.logspace(6, 12, 121)  # per-puzzle operation budgets of the solved-within-budget curves
EBR_KEY = 'gtan0.2-cap64-N8'  # EBR's most accurate point in the compute sweep


def full_all_to_official():
    """(422,786,) int64: the official test.csv row of each row of full-all.npz (from data/full-test/index.json)."""
    idx = json.loads((DATA / 'full-test/index.json').read_text())
    out = np.empty(idx['n_official'], dtype=np.int64)
    for shard, (start, n) in idx['full_all_order'].items():
        out[start:start + n] = idx['shards'][shard]['official_rows']
    return out


def main():
    """Collect per-puzzle (solved, flop) of every model on the subset, bin them by rating and write the JSON.

    For each model and rating bin: the solve rate per seed and the median, 10th and 90th percentile and mean of the
    per-puzzle operation count over the seeds; and for each budget, the share of puzzles solved within it."""
    fa = np.load(DATA / 'full-test/full-all.npz')
    rows = np.arange(LO, LO + N)
    rating = fa['ratings'][rows]
    official = full_all_to_official()[rows]
    models = {}

    # CRAM: per-puzzle flops and solved, full-all rows 10,000 onward
    cram = []
    for s in range(3):
        rec = json.loads((CODE / f'experiments/runs/CRAM-S/full/test/cram0-hb-s{s}.json').read_text())
        pp = rec['per_puzzle']
        puzzle = np.asarray(pp['puzzle'])
        pos = np.searchsorted(puzzle, rows) if np.all(np.diff(puzzle) > 0) else None
        assert pos is not None and np.array_equal(puzzle[pos], rows), 'CRAM puzzle indices are not full-all rows'
        cram.append({'solved': np.asarray(pp['solved'], bool)[pos], 'flop': opcount.cram0_sudoku(np.asarray(pp['flops'], float)[pos], rec['signature'])})
    models['CRAM'] = cram

    def fixed(name, paths, flop, key=None, by='official'):
        """A baseline that spends the same operation count `flop` on every puzzle; `paths` holds its per-puzzle flags."""
        runs = []
        for p in paths:
            if p.endswith('.npy'):
                right = np.load(p).astype(bool)[official]
            else:
                z = np.load(p)
                if 'official_rows' in z.files:  # FRM: (official_rows, solved) over the unseen puzzles
                    m = dict(zip(z['official_rows'].tolist(), z['solved'].astype(bool).tolist()))
                    right = np.array([m[o] for o in official])
                else:  # EqR: full-all order; unseen files start at row 10,000, first10k files are rows 10,000-19,999
                    right = z[key].astype(bool)[:N]
            runs.append({'solved': right, 'flop': np.full(N, float(flop))})
        models[name] = runs

    u = lambda m, k: opcount.unit(m, 'sudoku-extreme', k)
    fixed('HRM', [str(T / f'hrm/sudoku-e40000-s{s}/score.right.npy') for s in range(3)], 16 * u('HRM', 'per_step'))
    fixed('TRM', [str(T / f'trm/sudoku-s{s}/score.right.npy') for s in range(3)], 16 * u('TRM', 'per_step'))
    fixed('FRM', [str(T / f'frm-compiled{"" if s == 0 else f"-s{s}"}/sudoku/score-unseen.solved.npz') for s in range(3)], 4097 * u('FRM', 'per_call'))
    for D, B, f in ((16, 1, 'score-unseen-D16-B1'), (64, 1, 'score-unseen-D64-B1'), (64, 128, 'score-first10k-D64-B128')):
        fixed(f'EqR D{D} B{B}', [str(T / f'eqr/sudoku-s{s}/{f}.npz') for s in range(3)], D * B * u('EqR', 'per_nfe') + (u('EqR', 'select_over_B') * B / 128 if B > 1 else 0), key='top1_conv')

    # TRMP: predictions and per-rollout steps, full-all rows 10,000 onward in 8,192-row blocks
    sol = fa['solutions'][rows]
    tp = opcount.components()['baselines']['TRMP']['sudoku-extreme']
    models['TRMP'] = []
    for s in range(3):
        files = sorted(glob.glob(str(T / f'trmp/sudoku-extreme-s{s}/score/test/*.npz')))
        pred = np.concatenate([np.load(f)['prediction'] for f in files])[:N]
        steps = np.concatenate([np.load(f)['steps'] for f in files])[:N]
        shift = 0 if pred.min() == sol.min() else int(sol.min() - pred.min())
        models['TRMP'].append({'solved': np.all(pred + shift == sol, 1), 'flop': tp['input_check'] + 64 * tp['per_rollout'] + steps.sum(1) * tp['per_step'] + (steps.sum(1) - 64) * tp['noise_per_continuing_step']})

    # EBR: per-puzzle records of its most accurate compute-sweep setting, official test order
    models['EBR'] = []
    for s in range(3):
        z = np.load(T / f'ebr/sudoku-s{s}/report-fc2.flags.npz')
        models['EBR'].append({'solved': z[EBR_KEY].astype(bool)[official], 'flop': z[f'flop_{EBR_KEY}'].astype(float)[official]})
        sweep = json.loads((T / f'ebr/sudoku-s{s}/report-fc2.json').read_text())['sweep10k'][EBR_KEY]
        assert models['EBR'][-1]['solved'].sum() == sweep['solved'], 'EBR rows differ from its sweep10k subset'

    out = {'subset': {'full_all_rows': [LO, LO + N], 'n': N, 'rating_mean': float(rating.mean())},
           'bins': [[BINS[b], BINS[b + 1]] for b in range(len(BINS) - 1)],
           'bin_counts': [int(((rating >= BINS[b]) & (rating < BINS[b + 1])).sum()) for b in range(len(BINS) - 1)],
           'budgets': BUDGETS.tolist(), 'models': {}}
    for name, runs in models.items():
        rec = {'seeds': len(runs),
               'solved_pct': [100 * r['solved'].mean() for r in runs],
               'mean_flop': [float(r['flop'].mean()) for r in runs], 'bins': []}
        for b in range(len(BINS) - 1):
            m = (rating >= BINS[b]) & (rating < BINS[b + 1])
            fl = np.concatenate([r['flop'][m] for r in runs])
            rec['bins'].append({'solved_pct': [100 * r['solved'][m].mean() for r in runs] if m.any() else None,
                                'flop_median': float(np.median(fl)) if m.any() else None,
                                'flop_p10': float(np.percentile(fl, 10)) if m.any() else None,
                                'flop_p90': float(np.percentile(fl, 90)) if m.any() else None,
                                'flop_mean': float(fl.mean()) if m.any() else None})
        # solved within a per-puzzle budget: a run that stopped correctly at flop f is solved under any budget >= f
        rec['solved_within_budget_pct'] = [[100 * float((r['solved'] & (r['flop'] <= b)).mean()) for b in BUDGETS] for r in runs]
        out['models'][name] = rec
    dest = CODE / 'results/difficulty/sudoku-difficulty.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + '\n')
    for name, rec in out['models'].items():
        print(f"{name:14s} solved {np.mean(rec['solved_pct']):7.3f}%  mean flop {np.mean(rec['mean_flop']):.3g}")
    print('bin counts', out['bin_counts'])


if __name__ == '__main__':
    main()
