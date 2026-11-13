"""Add the results of nqueens_offset.py (the offset-shared N-Queens energy) to results/nqueens-results.json as
`offset_energy`, keeping every existing key; no training or search. Run from experiments/ after nqueens_export.py and
nqueens_free_export.py, and after the `final` runs of every (method, arm) with a choice:

  python nqueens_offset_export.py

Per size: the energy and optimizer description, the selected (method, arm) (key `headline`) and its selection rule,
every (method, arm) choice with per-seed test metrics and GPU fit times, and the selected pair's learned couplings per
offset (seed 0). When their runs exist: the selected pair's one-thread CPU fit times (`cpu` stage) and the ablation
without per-offset preconditioning (`ablation` stage); otherwise these keys are null.
"""
import json
from pathlib import Path

import nqueens_free_export as X

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs' / 'NQ' / 'nq11'
OUT = HERE.parent / 'results' / 'nqueens-results.json'
METHODS = ('pl', 'mpl', 'spl', 'mpf', 'dsm')
ARMS = ('ising', 'table')
KEYS = ('acc_n1', 'acc_majority20', 'cov20', 'cov20_group_mean', 'first_stop_valid', 'flop_n1', 'flop_n20',
        'wrong_stops_20chains', 'cap_failures_20chains', 'fit_gpu_cold_s', 'fit_gpu_warm_s')


def seeds_of(folder, names):
    """Per-seed rows of the runs folder/<name> (one name per seed, in seed order): test metrics, fit times and model
    statistics. The fit times come from fit-dense.json (nqueens_offset.py --mode retime) when it exists."""
    out = []
    for s, name in enumerate(names):
        f = X.read(folder / name / 'final.json')
        fit = f['fit']
        dense = folder / name / 'fit-dense.json'  # the same fit timed again with the dense pair-matrix map (--mode retime)
        t = X.read(dense) if dense.exists() else dict(cold_seconds=fit.get('cold_seconds', fit.get('fit_seconds')),
                                                       warm_seconds=fit.get('warm_seconds'), warm_repeats=fit.get('warm_repeats'))
        out.append(dict(seed=s, **X.test_row(f['test']), fit_gpu_cold_s=t['cold_seconds'],
                        fit_gpu_warm_s=t['warm_seconds'], fit_gpu_warm_repeats=t['warm_repeats'],
                        fit_route='dense pair-matrix map' if dense.exists() else 'index route',
                        fit_gpu_cold_s_index_route=fit.get('cold_seconds', fit.get('fit_seconds')),
                        cuda_context_s=fit.get('cuda_context_seconds'), gpu_hardware=fit.get('hardware'),
                        params_learned=fit['params_learned'], params_nonzero=fit['params_nonzero'],
                        diverged=fit['diverged'], updates=fit['updates'], tau=fit['tau'], delta=fit['delta'], unit=fit['unit']))
    return out


def size_block(n):
    """The `offset_energy` results of board size n (see the module docstring)."""
    head = X.read(ROOT / f'n{n}' / 'selected-headline.json')
    rows = {}
    for m in METHODS:
        for arm in ARMS:
            sel = X.read(ROOT / f'n{n}' / f'selected-{m}-{arm}.json')
            if sel['choice'] is None:
                continue
            seeds = seeds_of(ROOT / f'n{n}' / 'final', [f'{m}-{arm}-s{s}' for s in range(3)])
            rows[f'{m}-{arm}'] = dict(method=m, arm=arm, headline=(m == head['method'] and arm == head['arm']),
                                      choice=sel['choice'], rule=sel['rule'], **X.summary(seeds, keys=KEYS),
                                      params_learned=seeds[0]['params_learned'], per_seed=seeds)
    hm, ha = head['method'], head['arm']
    cpu = ROOT / f'n{n}' / 'final' / f'{hm}-{ha}-cpu-s0.json'
    abl = ROOT / f'n{n}' / 'ablation'
    abl_seeds = seeds_of(abl, [f'{hm}-{ha}-noprecond-s{s}' for s in range(3)]) if (abl / f'{hm}-{ha}-noprecond-s2' / 'final.json').exists() else None
    fit0 = X.read(ROOT / f'n{n}' / 'final' / f'{hm}-{ha}-s0' / 'fit.json')
    return dict(n=n, headline=dict(method=hm, arm=ha, rule=head['rule']), rows=rows,
                headline_cpu=X.read(cpu) if cpu.exists() else None,
                ablation_no_precond=None if abl_seeds is None else dict(**X.summary(abl_seeds, keys=KEYS), per_seed=abl_seeds),
                headline_couplings_seed0=fit0.get('offsets'))


def main():
    """Add `offset_energy` to results/nqueens-results.json."""
    export = json.loads(OUT.read_text())
    export['offset_energy'] = dict(
        pilot='code experiments/nqueens_offset.py',
        energy=('n^2 binary cells (0 empty, 1 queen); a pair term that depends only on the two states and the offset '
                '(dr, dc) of the cells (dr = r_j - r_i >= 0, dc > 0 when dr = 0; 2n(n-1) offsets); one unary cost shared '
                'by every cell; given queens clamped. table: a free 2x2 table per offset and a free 2-vector unary; '
                'ising: J_d x_i x_j per offset and h x_i (no term on empty cells). Nothing says which offsets attack.'),
        prior='matched to EBR: 2D RoPE attention without an absolute position embedding (pair interactions depend on relative position)',
        optimizer=('full-batch SGD on all training rows, momentum 0.5, cosine decay from lr to 0, zero start, 4,000 updates, '
                   'no L1; per-offset preconditioning: the gradient of offset d is divided by m_d / mean(m), m_d its number of cell pairs'),
        inference='focused heat-bath search (cram/cram_nqueens.solve), cap 20,000 moves, 20 chains; stop and T chosen on validation',
        sizes={str(n): size_block(n) for n in (8, 10)})
    OUT.write_text(json.dumps(export, indent=1) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
