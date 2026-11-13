"""Export the graph-coloring CRAM results of the paper (arm `full`); no training or evaluation. Run from experiments/.

Reads runs/CRAM-GC/selected.json, the final fits (runs/CRAM-GC/final/<recipe>/{cpu,cuda}/) and the test records of
graph_coloring.py, and writes results/graph-coloring-results.json (per size: per-seed metrics, mean and SD, flop, fit
times, parameter counts, what the pair tables K learned) and one table record per size in runs/table/
(graph-coloring-<n>__cram-full.json). Recipes of the other arms in selected.json (`pos`, `shared`) are not exported:
the paper does not report them. GRAM's published numbers (arXiv 2605.19376, Table 1) are copied beside the results for
comparison.
"""
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from cram import cram_graph_coloring as G
from cram import table

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs' / 'CRAM-GC'
OUT = HERE.parent / 'results' / 'graph-coloring-results.json'
METRICS = ('proper_1', 'proper_canonical_1', 'conflicts_1', 'conflicts_majority_20', 'coverage_20',
           'coverage_canonicalized_20', 'proper_any_20', 'flop_1', 'flop_20', 'wrong_stops_1', 'budget_failures_1',
           'wrong_stops_all', 'budget_failures_all', 'moves_mean', 'moves_max')
GRAM = dict(source='Baek et al. 2026, Generative Recursive Reasoning, arXiv 2605.19376, Table 1 (Sec. 4.2); mean +- SD '
                   'over runs; conflicts summed over the test graphs; Cov@20 in percent',
            rows={'GRAM': {'params': '10M', 'N1': {8: dict(conflict=(8.3, 2.5)), 10: dict(conflict=(31.3, 15.0))},
                           'N20': {8: dict(conflict=(2.7, 2.1), cov20=(85.8, 0.5)),
                                   10: dict(conflict=(3.3, 1.5), cov20=(51.3, 2.8))}},
                  'MDLM': {'params': '12.6M', 'N1': {8: dict(conflict=(16.3, 2.5)), 10: dict(conflict=(57.3, 4.7))},
                           'N20': {8: dict(conflict=(2.7, 0.6), cov20=(84.5, 4.0)),
                                   10: dict(conflict=(12.0, 7.0), cov20=(48.2, 1.4))}},
                  'AR': {'params': '10.6M', 'N1': {8: dict(conflict=(37.3, 6.1)), 10: dict(conflict=(101.7, 4.0))},
                         'N20': {8: dict(conflict=(19.0, 11.3), cov20=(83.0, 0.7)),
                                 10: dict(conflict=(61.3, 8.3), cov20=(40.0, 0.3))}},
                  'TRM': {'params': '7M', 'N1': {8: dict(conflict=(109.3, 3.1), cov20=(22.3, 0.6)),
                                                 10: dict(conflict=(170.7, 17.9), cov20=(6.8, 0.3))}},
                  'HRM': {'params': '27M', 'N1': {8: dict(conflict=(109.7, 1.5), cov20=(21.8, 0.3)),
                                                  10: dict(conflict=(164.3, 21.6), cov20=(8.9, 1.7))}}},
            training='5,000 epochs, global batch 768, 8 RTX 4090 GPUs: 1.5 h (8 vertices), 6 h (10 vertices) (App. B, Table 7)')


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


def agg(values):
    """{mean, sd (sample SD; None for one value), per_seed} of a list of per-seed values."""
    return dict(mean=mean(values), sd=stdev(values) if len(values) > 1 else None, per_seed=values)


def main():
    """Write results/graph-coloring-results.json and the table records, and print the main numbers."""
    selected = read(ROOT / 'selected.json')
    rows, records = [], []
    for key, sel in selected.items():
        if key.startswith('_') or sel['arm'] != 'full':  # keys with '_' are notes; the paper reports the arm `full`
            continue
        n, arm = sel['n'], sel['arm']
        stem = ROOT / 'final' / G_tag(sel)
        cpu, gpu = read(stem / 'cpu' / 'final.json'), read(stem / 'cuda' / 'final.json')
        with np.load(stem / 'cpu' / 'model.npz') as a, np.load(stem / 'cuda' / 'model.npz') as b:
            max_diff = max(float(np.abs(a[k].astype(np.float64) - b[k]).max()) for k in ('W0', 'D', 'B'))
            same_support = all(np.array_equal(a[k] != 0, b[k] != 0) for k in ('W0', 'D', 'B'))
        tests = [read(stem / f"test-{sel['stop']}-T{sel['temperature']:g}-S{sel['samples']}-cap{sel['cap']}-s{s}" / 'final.json')
                 for s in (0, 1, 2)]
        summ = cpu['summary']
        with np.load(stem / 'cpu' / 'model.npz') as f:
            fig = figure_data({k: f[k] for k in ('W0', 'D', 'B')}, n)
        row = dict(benchmark=f'graph-coloring-{n}', row=f'cram-{arm}', n=n, arm=arm, n_test=tests[0]['n_test'],
                   solutions_total=tests[0]['solutions_total'],
                   metrics={m: agg([t[m] for t in tests]) for m in METRICS},
                   params=summ['params'], params_nonzero=summ['params_nonzero'],
                   fit=dict(gpu_cold_s=gpu['cold_seconds'], gpu_warm_s=gpu['warm_seconds'],
                            cpu_cold_s=cpu['cold_seconds'], cpu_warm_s=cpu['warm_seconds'],
                            cuda_context_s=gpu['cuda_context_seconds'], gpu_hardware=gpu['hardware'],
                            cpu_hardware=cpu['hardware'], cpu_threads=cpu['torch_threads'],
                            gpu_warm_repeats=gpu['warm_repeats'], cpu_warm_repeats=cpu['warm_repeats'],
                            cpu_gpu_max_abs_diff=max_diff, cpu_gpu_same_support=same_support,
                            training_deterministic='full batch; one fit per device serves seeds 0-2 (seeds set the search)'),
                   learned=summ, figure=fig, recipe=sel)
        rows.append(row)
        records.append(table.write_row(
            f'graph-coloring-{n}', f'cram-{arm}', label=f'CRAM ({arm} pair tables, PL + L1)',
            per_seed=[t['proper_1'] for t in tests], n_test=tests[0]['n_test'],
            test_set=f"GRAM's graph-coloring-{n} test split (CUN-bjy/graph-coloring 08d5f1f8), {tests[0]['n_test']} graphs",
            params=summ['params'], flop_per_puzzle=mean(t['flop_1'] for t in tests), train_gpu_s=gpu['cold_seconds'],
            train_cpu_s=cpu['cold_seconds'], provenance='experiments/graph_coloring.py; '
            f'cram/cram_graph_coloring.py; runs/CRAM-GC/final/{G_tag(sel)}',
            notes='per_seed is the fraction of test graphs whose single output (chain 0) is a proper coloring; GRAM\'s '
                  'metrics are in extra. Training deterministic (full batch); seeds 0-2 set the search. Fit times: seed-0 '
                  'cold fits (one fit per device).',
            extra=dict(metrics=row['metrics'], params_nonzero=summ['params_nonzero'], fit=row['fit'], recipe=sel)))
    export = dict(rows=rows, gram_published=GRAM, data=read(ROOT / 'data.json'),
                  metric_definitions=dict(
                      proper_1='fraction of test graphs whose single output (chain 0, its first stop or its lowest-energy '
                               'coloring at the cap) has no conflict, any color order',
                      proper_canonical_1="GRAM's validity of the single output: proper and in canonical color order",
                      conflicts_1="GRAM's Conflict at N = 1: conflict edges of the single output, summed over test graphs",
                      conflicts_majority_20="GRAM's Conflict at N = 20: conflict edges of the most frequent of 20 outputs "
                                            "(ties: lower mean energy), summed over test graphs",
                      coverage_20="GRAM's Cov@20: distinct proper canonical outputs of 20 chains, summed over graphs, over "
                                  "all canonical proper colorings of the test graphs",
                      coverage_canonicalized_20='as coverage_20 after relabeling each proper output into canonical order '
                                                '(uses color-permutation symmetry, which GRAM does not receive)',
                      flop_1='coupling assembly of the graph plus chain 0 (mean over test graphs)',
                      flop_20='coupling assembly plus all 20 chains (mean over test graphs)'),
                  source_sha256={rel(p): sha(p) for p in (
                      HERE.parent / 'cram/cram_graph_coloring.py', HERE.parent / 'cram/graph_coloring_data.py',
                      HERE / 'graph_coloring.py', ROOT / 'selected.json')},
                  records=[rel(p) for p in records])
    OUT.write_text(json.dumps(export, indent=1) + '\n')
    for r in rows:
        m = r['metrics']
        print(r['benchmark'], r['arm'], {k: (round(m[k]['mean'], 4), None if m[k]['sd'] is None else round(m[k]['sd'], 4))
                                         for k in ('proper_1', 'conflicts_1', 'conflicts_majority_20', 'coverage_20',
                                                   'coverage_canonicalized_20', 'flop_1', 'flop_20')},
              r['params'], r['params_nonzero'], r['fit']['gpu_cold_s'], r['fit']['cpu_cold_s'])


def figure_data(m, n):
    """Small arrays for a figure of what K learned (CPU fit). m: the model arrays {W0, D, B}; n: vertices.
    Pairs p in row-major (i, j) order, i < j.
    edge_table[p] = W0_p + D_pp (own token = edge), non_edge_table[p] = W0_p; for `full`, cross_token_max_abs[p][t] =
    max |D_{p,t}| over the 3 x 3 entries (the diagonal t = p is the own token)."""
    W0, D, B = (m[k].astype(np.float64) for k in ('W0', 'D', 'B'))
    P = n * (n - 1) // 2
    I, J = G.pairs(n)
    if D.ndim == 4:
        own = D[np.arange(P), np.arange(P)]
        cross = np.abs(D).max((2, 3))
    else:
        own, cross = (D if D.ndim == 3 else np.broadcast_to(D, (P, 3, 3))), None
    W0 = W0 if W0.ndim == 3 else np.broadcast_to(W0, (P, 3, 3))
    r = lambda v: np.round(np.asarray(v), 5).tolist()
    out = dict(pairs=[[int(i), int(j)] for i, j in zip(I, J)], edge_table=r(W0 + own), non_edge_table=r(W0),
               bias=r(B if B.ndim == 2 else np.broadcast_to(B, (n, 3))))
    if cross is not None:
        out['cross_token_max_abs'] = r(cross)
        own_abs = np.abs(own).sum()
        out['own_token_share_of_abs_D'] = float(own_abs / np.abs(D).sum())
        out['cross_token_nonzero'] = int((D != 0).sum() - (own != 0).sum())
        out['cross_token_entries'] = int(D.size - own.size)
    return out


def G_tag(sel):
    """The folder name of a selected recipe (as graph_coloring.tag_of)."""
    return f"n{sel['n']}-{sel['arm']}-u{sel['updates']}-lr{sel['lr']:g}-lam{sel['lam']:g}"


if __name__ == '__main__':
    main()
