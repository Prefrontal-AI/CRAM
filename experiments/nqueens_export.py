"""Write results/nqueens-results.json with the results of the free-table N-Queens energy trained by pseudolikelihood
(PL) plus L1 (cram/cram_nqueens.py); no training or search. Run from experiments/ (it loads the fitted models).

Reads runs/NQ/n<n>/selected.json (the selected recipe), the per-seed test stages and the timed fits on one GPU and on
one CPU thread. This script creates the results file; nqueens_free_export.py and nqueens_offset_export.py then add
their sections (`hand_set`, `training_methods`, `offset_energy`). When the file exists, `hand_set` and
`training_methods` are kept; run nqueens_offset_export.py again afterwards.

GRAM's published numbers (Baek et al. 2026, arXiv 2605.19376 v5, Table 1 and Table 26; mean +- SD over 3 runs, in %):
Acc. is the fraction of test rows whose output is a valid board (GRAM's verify_nqueens), N = 1 a single sample and
N = 20 the majority vote of 20; Cov@20 the coverage of the (at most 20) enumerated completions by 20 samples.
"""
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from cram import cram_nqueens as Q

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs' / 'NQ'
OUT = HERE.parent / 'results' / 'nqueens-results.json'

PUBLISHED = {  # arXiv 2605.19376 v5, Table 1 (GRAM, TRM, HRM, AR, MDLM); 10x10 GRAM also Table 26
    'source': 'Baek et al. 2026, Generative Recursive Reasoning, arXiv 2605.19376 v5, Table 1 (Sec. 4.2)',
    'rows': {
        'GRAM': {'params': '10M', '8': {'acc_n1': [99.5, 0.3], 'acc_n20': [100.0, 0.0], 'cov20': [89.7, 1.7]},
                 '10': {'acc_n1': [89.7, 2.7], 'acc_n20': [100.0, 0.0], 'cov20': [57.5, 3.4]}},
        'TRM': {'params': '7M', '8': {'acc_n1': [66.8, 5.6], 'cov20': [22.6, 0.8]},
                '10': {'acc_n1': [17.5, 11.2], 'cov20': [2.0, 1.3]}},
        'HRM': {'params': '27M', '8': {'acc_n1': [78.7, 2.9], 'cov20': [26.7, 1.3]},
                '10': {'acc_n1': [37.8, 0.7], 'cov20': [4.7, 0.1]}},
        'AR': {'params': '10.6M', '8': {'acc_n1': [96.3, 1.0], 'acc_n20': [98.3, 0.7], 'cov20': [84.8, 0.8]},
               '10': {'acc_n1': [90.0, 2.2], 'acc_n20': [95.1, 1.5], 'cov20': [53.2, 0.8]}},
        'MDLM': {'params': '12.6M', '8': {'acc_n1': [96.1, 1.5], 'acc_n20': [100.0, 0.0], 'cov20': [87.2, 0.6]},
                 '10': {'acc_n1': [74.3, 6.6], 'acc_n20': [98.6, 0.8], 'cov20': [47.4, 2.2]}},
    },
    'gram_training': 'Table 7: 8 RTX 4090 GPUs, about 1 h (8x8, 3K epochs) and 3 h (10x10, 1K epochs)',
}


def read(path):
    """The parsed JSON file at path."""
    return json.loads(Path(path).read_text())


def sha(path):
    """Hex SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    """The key of a source file in `source_sha256`: its path relative to experiments/ for a file in experiments/, else
    relative to the repository root."""
    path = Path(path)
    return str(path.relative_to(HERE) if path.is_relative_to(HERE) else path.relative_to(HERE.parent))


def stats(values):
    """{mean, sd (sample SD; None for one value), per_seed} of a list of per-seed values."""
    return dict(mean=mean(values), sd=stdev(values) if len(values) > 1 else None, per_seed=values)


def figure_data(model, n):
    """The couplings of a fitted model {W (p, 2, 2), B (c, 2)} for a figure: J (c x c), h (c), rounded to 1e-4, and
    every pair's attack class (none, row, column, diagonal, anti-diagonal; used only to label the figure)."""
    W, B = model['W'].double().numpy(), model['B'].double().numpy()
    c = n * n
    i, j = np.triu_indices(c, 1)
    Jp = W[:, 1, 1] - W[:, 1, 0] - W[:, 0, 1] + W[:, 0, 0]
    J = np.zeros((c, c))
    J[i, j] = Jp
    J = J + J.T
    h = B[:, 1] - B[:, 0]
    np.add.at(h, i, W[:, 1, 0] - W[:, 0, 0])
    np.add.at(h, j, W[:, 0, 1] - W[:, 0, 0])
    cls = np.zeros((c, c), np.int8)
    cls[i, j] = Q.attack_classes(n)
    cls = cls + cls.T
    return dict(n=n, cells='row-major r * n + col', J=np.round(J, 4).tolist(), h=np.round(h, 4).tolist(),
                pair_class=cls.tolist(), class_names=list(Q.CLASS_NAMES),
                definition='J_ij = W[1,1] - W[1,0] - W[0,1] + W[0,0] (queen-queen interaction of the pair); '
                           'h_i = E(only cell i holds a queen) - E(empty board); E = const + sum h_i x_i + sum J_ij x_i x_j')


def group_mean_coverage(records):
    """GRAM's printed form of Cov@20: rows grouped by their completion count capped at 20 (find_all_solutions with
    max_solutions=20, evaluators/nqueens.py), the mean coverage of each group, then the unweighted mean over the groups
    present. Our runs of GRAM's code reproduce GRAM's published 8 x 8 value with this form, so it is most likely the
    definition behind GRAM's table. records: the arrays of records.npz (coverage20 and n_solutions per test row)."""
    groups = np.minimum(records['n_solutions'], 20).astype(int)
    return float(np.mean([records['coverage20'][groups == k].mean() for k in np.unique(groups)]))


def size_block(n):
    """The results of board size n: the recipe, per-seed test metrics and fit times, their means and SDs (accuracies
    and coverages in %), parameter counts and the seed-0 couplings."""
    sel = read(ROOT / f'n{n}' / 'selected.json')
    fk = f"all-u{sel['steps']}-lr{sel['lr']:g}-lam{sel['lam']:g}-m{sel['momentum']:g}"
    sk = f"t{sel['temperature']:g}-c{sel['chains']}-m{sel['cap']}" + ('' if sel['stop'] == 'energy' else f"-{sel['stop']}")
    seeds = []
    for s in range(3):
        test = read(ROOT / f'n{n}' / f'test-{fk}-{sk}-s{s}' / 'final.json')
        cov_group = group_mean_coverage(np.load(ROOT / f'n{n}' / f'test-{fk}-{sk}-s{s}' / 'records.npz'))
        gpu = read(ROOT / f'n{n}' / f'{fk}-cuda-s{s}' / 'final.json')
        cpu = read(ROOT / f'n{n}' / f'{fk}-cpu-s{s}' / 'final.json')
        mg = torch.load(ROOT / f'n{n}' / f'{fk}-cuda-s{s}' / 'model.pt', weights_only=True)
        mc = torch.load(ROOT / f'n{n}' / f'{fk}-cpu-s{s}' / 'model.pt', weights_only=True)
        seeds.append(dict(
            seed=s, rows=test['rows'], acc_n1=test['acc_n1'], acc_majority20=test['acc_majority20'], cov20=test['cov20'],
            cov20_group_mean=cov_group,
            pass20=test['pass20'], cov_all20=test['cov_all20'], distinct_valid20=test['distinct_valid20'],
            first_stop_valid=test['first_stop_valid'], wrong_stops_20chains=test['wrong_stops'],
            budget_failures_20chains=test['budget_failures'], chains_stopped=test['chains_stopped'],
            flop_n1=test['flop_n1'], flop_n20=test['flop_all_chains'], flop_n1_p99=test['flop_n1_p99'],
            flop_n1_max=test['flop_n1_max'], search_seconds=test['search_seconds'],
            gpu_cold_s=gpu['cold_seconds'], gpu_warm_s=gpu['warm_seconds'], gpu_warm_repeats=gpu['warm_repeats'],
            cuda_context_s=test['training'].get('cuda_context_seconds'), gpu_hardware=gpu['hardware'],
            cpu_cold_s=cpu['cold_seconds'], cpu_warm_s=cpu['warm_seconds'], cpu_warm_repeats=cpu['warm_repeats'],
            cpu_hardware=cpu['hardware'], cpu_threads=cpu['threads'],
            cpu_gpu_max_abs_diff=float(max((mg[k] - mc[k]).abs().max() for k in ('W', 'B'))),
            params_learned=gpu['params_learned'], params_nonzero=gpu['params_nonzero'], tau=gpu['tau'],
            delta=gpu['delta'], unit=gpu['unit'], pl=gpu['pl'], coupling=gpu['coupling']))
    pct = lambda key: stats([100 * r[key] for r in seeds])  # noqa: E731
    m0 = torch.load(ROOT / f'n{n}' / f'{fk}-cuda-s0' / 'model.pt', weights_only=True)
    return dict(
        n=n, recipe=sel, test_rows=seeds[0]['rows'],
        acc_n1=pct('acc_n1'), acc_majority20=pct('acc_majority20'), cov20=pct('cov20'),
        cov20_group_mean=pct('cov20_group_mean'), pass20=pct('pass20'),
        cov_all20=pct('cov_all20'), first_stop_valid=pct('first_stop_valid'),
        flop_n1=stats([r['flop_n1'] for r in seeds]), flop_n20=stats([r['flop_n20'] for r in seeds]),
        fit_gpu_cold_s_seed0=seeds[0]['gpu_cold_s'], fit_cpu_cold_s_seed0=seeds[0]['cpu_cold_s'],
        fit_gpu_cold_s=stats([r['gpu_cold_s'] for r in seeds]), fit_gpu_warm_s=stats([r['gpu_warm_s'] for r in seeds]),
        fit_cpu_cold_s=stats([r['cpu_cold_s'] for r in seeds]), fit_cpu_warm_s=stats([r['cpu_warm_s'] for r in seeds]),
        params_learned=seeds[0]['params_learned'], params_nonzero=[r['params_nonzero'] for r in seeds],
        per_seed=seeds, couplings_seed0=figure_data(m0, n))


def main():
    """Write results/nqueens-results.json and print the main numbers."""
    sizes = {str(n): size_block(n) for n in (8, 10)}
    sources = [HERE.parent / 'cram/cram_nqueens.py', HERE.parent / 'cram/nqueens_data.py', HERE / 'nqueens_export.py']
    sources += [ROOT / f'n{n}' / 'selected.json' for n in (8, 10)]
    export = dict(
        task='N-Queens, GRAM\'s sets (8x8: 7,224 train / 1,240 test rows; 10x10: 108,234 / 18,466)',
        data='GRAM dataset/build_nqueens_dataset.py at gram-release f2e32b9, --n 8 --remove-counts 5 6 7 and --n 10 '
             '--remove-counts 7 8 9 (seed 42, test ratio 0.15, no augmentation); rows split by distinct input',
        model='n^2 binary cells; one free 2x2 table W_p per unordered cell pair (all pairs) and a free input cost '
              'B_i in R^2 per cell; given queens clamped; no attack lines, rows, columns, diagonals or queen count supplied',
        training='PL of the free cells of every training row at its labeled solution + lam L1 (proximal), zero start, '
                 'full-batch SGD with momentum and cosine decay; deterministic (seeds set the search and each seed '
                 'has its own timed fit); no solver, no differentiation through inference',
        metrics='GRAM\'s: acc_n1 = first chain\'s board valid; acc_majority20 = majority vote of 20 independent chains '
                '(ties to the lower mean energy) valid; cov20 = Cov@20 with GRAM\'s cap of 20 completions, mean over '
                'test rows; cov20_group_mean = the same per-row coverage averaged within groups of rows with equal '
                'completion count (capped at 20), then unweighted over the groups, as GRAM\'s evaluator prints it '
                '(most likely the Cov@20 of GRAM\'s table); accuracies over test rows (an input with s '
                'solutions counts s times). first_stop_valid = the first chain stopped '
                '(test passed) on a valid board holding the givens (rule checker, not label match).',
        flop='per test row, mean over rows: flop_n1 one chain; flop_n20 all 20 chains; per chain 2c^2 + 2c + 1 to '
             'initialize and 5c + 14 per move (cram_nqueens.move_flop); RNG, comparisons, indexing not counted',
        timing='cold: the first fit in a fresh process (one per seed); warm: median of 7 further fits; from GRAM\'s '
               'token arrays in host memory to the inference-ready model (token conversion, transfer, reduction to '
               'distinct solutions, initialization, all updates, S, tau, delta and the temperature unit); CUDA context '
               'apart; GPU one H200; CPU one thread in a CPU-only job',
        sizes=sizes, published=PUBLISHED,
        source_sha256={rel(p): sha(p) for p in sources})
    if OUT.exists():  # keep the sections that nqueens_free_export.py adds and their source hashes
        old = read(OUT)
        for key in ('hand_set', 'training_methods'):
            if key in old:
                export[key] = old[key]
        for p, h in old.get('source_sha256', {}).items():
            export['source_sha256'].setdefault(p, h)
    OUT.write_text(json.dumps(export, indent=1) + '\n')
    print(OUT)
    for n, b in sizes.items():
        print(n, {k: b[k] for k in ('acc_n1', 'acc_majority20', 'cov20', 'cov20_group_mean', 'first_stop_valid', 'flop_n1', 'flop_n20',
                                     'fit_gpu_cold_s_seed0', 'fit_cpu_cold_s_seed0', 'params_learned', 'params_nonzero')})


if __name__ == '__main__':
    main()
