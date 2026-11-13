"""Add the results of nqueens_free.py to results/nqueens-results.json, keeping every existing key; no training or
search. Run from experiments/ after nqueens_export.py (which creates the file) and after the `hand` and `final`
stages of nqueens_free.py have finished for every method and both sizes:

  python nqueens_free_export.py

Adds: `hand_set` (the hand-set attack-pair energy per size: per-seed test metrics) and `training_methods` (per size
and method: the validation choice, per-seed test metrics, fit times, and the flag `selected` for the
validation-selected method). nqueens_export.py keeps these keys when it rewrites the file.
Metric names: acc_n1 the first chain's board is valid; acc_majority20 the majority vote of 20 chains is valid; cov20
the coverage of a row's solutions by 20 chains (Cov@20), cov20_group_mean its mean over groups of rows with equal
solution count; all in %.
"""
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs' / 'NQ' / 'alt'
OUT = HERE.parent / 'results' / 'nqueens-results.json'
METHODS = ('pl', 'mpl', 'mpf', 'dsm', 'cd', 'pcd', 'chl')
SIZES = (8, 10)


def read(path):
    """The parsed JSON file at path."""
    return json.loads(Path(path).read_text())


def stats(values):
    """{mean, sd (sample SD; None for one value), per_seed} of the values that are not None; None when there is none."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    return dict(mean=mean(values), sd=stdev(values) if len(values) > 1 else None, per_seed=values)


def test_row(t):
    """The per-seed test metrics of one nqueens_free.score result t, in % where they are fractions."""
    pct = lambda k: None if t.get(k) is None else 100 * t[k]  # noqa: E731
    return dict(acc_n1=pct('acc_n1'), acc_majority20=pct('acc_majority20'), cov20=pct('cov20'),
                cov20_group_mean=pct('cov20_group_mean'), pass20=pct('pass20'), first_stop_valid=pct('first_stop_valid'),
                wrong_stops_20chains=t.get('wrong_stops'), cap_failures_20chains=t.get('budget_failures'),
                flop_n1=t.get('flop_n1'), flop_n20=t.get('flop_all_chains'), rows=t.get('rows'),
                unusable=t.get('unusable', False))


def summary(seeds, keys=('acc_n1', 'acc_majority20', 'cov20', 'cov20_group_mean', 'first_stop_valid', 'flop_n1',
                         'flop_n20', 'wrong_stops_20chains', 'cap_failures_20chains')):
    """stats() of each key over the per-seed rows `seeds`."""
    return {k: stats([s[k] for s in seeds]) for k in keys}


def method_block(n, m, winner):
    """The results of method m at board size n (seeds 0-2); `selected` is true for the method `winner`."""
    sel = read(ROOT / f'n{n}' / f'selected-{m}.json')
    seeds = []
    for s in range(3):
        f = read(ROOT / f'n{n}' / 'final' / f'{m}-s{s}' / 'final.json')
        fit = f['fit']
        seeds.append(dict(seed=s, **test_row(f['test']), fit_gpu_cold_s=fit.get('cold_seconds'),
                          fit_gpu_warm_s=fit.get('warm_seconds'), fit_gpu_warm_repeats=fit.get('warm_repeats'),
                          cuda_context_s=fit.get('cuda_context_seconds'), gpu_hardware=fit.get('hardware'),
                          params_learned=fit['params_learned'], params_nonzero=fit['params_nonzero'],
                          diverged=fit['diverged'], steps=fit['steps'], tau=fit['tau'], delta=fit['delta'],
                          unit=fit['unit']))
    return dict(method=m, selected=(m == winner), choice=sel['choice'], rule=sel['rule'],
                **summary(seeds, keys=('acc_n1', 'acc_majority20', 'cov20', 'cov20_group_mean', 'first_stop_valid',
                                       'flop_n1', 'flop_n20', 'wrong_stops_20chains', 'cap_failures_20chains',
                                       'fit_gpu_cold_s', 'fit_gpu_warm_s', 'params_nonzero')),
                params_learned=seeds[0]['params_learned'], per_seed=seeds)


def hand_block(n):
    """The results of the hand-set energy at board size n: the validation searches, the chosen temperature and the
    per-seed test metrics."""
    dev = read(ROOT / f'n{n}' / 'hand' / 'dev.json')
    seeds = []
    for s in range(3):
        f = read(ROOT / f'n{n}' / 'hand' / f'test-s{s}' / 'final.json')
        seeds.append(dict(seed=s, **test_row(f['test']), params_nonzero=f['params_nonzero']))
    return dict(energy='E = 1 x (attacking queen pairs) - 0.5 x (queens): W_p[1,1] = 1 on every attack pair (row, '
                       'column, diagonal), B_i[1] = -0.5; nothing learned; energy test exact (tau = -n/2)',
                hand=dev['hand'], choice=dev['choice'], validation=dev['searches'], params_learned=0,
                params_nonzero=seeds[0]['params_nonzero'], fit_s=0.0, **summary(seeds), per_seed=seeds)


def main():
    """Add `hand_set` and `training_methods` to results/nqueens-results.json and print the main numbers."""
    export = read(OUT)
    hand = {str(n): hand_block(n) for n in SIZES}
    methods = {}
    for n in SIZES:
        win = read(ROOT / f'n{n}' / 'selected-method.json')
        methods[str(n)] = dict(selected_method=win['method'], selection=win,
                               rows={m: method_block(n, m, win['method']) for m in METHODS})
    export['hand_set'] = dict(pilot='reference row (uses the rules); same search as the learned free-table energy; '
                                    'temperature chosen on validation', sizes=hand)
    export['training_methods'] = dict(
        pilot='the free-table energy of cram/cram_nqueens.py trained by pl, mpl, mpf, dsm (10,000 updates) and cd, pcd, chl '
              '(2,000 updates), none differentiating through a solver; per method the validation choice of lr, knob, '
              'lam, alpha, stopping test and T; 3 seeds, test once, 20 chains; fit times on one H200 (cold, warm median '
              'of 7), CUDA context apart; Cov@20 as row mean (cov20) and GRAM\'s group mean (cov20_group_mean)',
        sizes=methods)
    sources = [HERE / p for p in ('nqueens_free.py', 'training_methods.py', 'nqueens_free_export.py')]
    export.setdefault('source_sha256', {}).update(
        {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    OUT.write_text(json.dumps(export, indent=1) + '\n')
    print(OUT)
    for n in SIZES:
        h = hand[str(n)]
        print(f'n {n} hand: N=1 {h["acc_n1"]["mean"]:.2f} Cov@20 {h["cov20"]["mean"]:.2f} / group {h["cov20_group_mean"]["mean"]:.2f}')
        for m, r in methods[str(n)]['rows'].items():
            g = lambda k: r[k]['mean'] if r[k] else float('nan')  # noqa: E731
            print(f'n {n} {m:4s}{"*" if r["selected"] else " "} val {100 * r["choice"]["val_acc_n1"]:.2f} | test N=1 '
                  f'{g("acc_n1"):.2f} maj20 {g("acc_majority20"):.2f} Cov@20 {g("cov20"):.2f} / group '
                  f'{g("cov20_group_mean"):.2f} | wrong {g("wrong_stops_20chains"):.0f} cap {g("cap_failures_20chains"):.0f} '
                  f'| fit cold {g("fit_gpu_cold_s"):.1f} s warm {g("fit_gpu_warm_s"):.1f} s')


if __name__ == '__main__':
    main()
