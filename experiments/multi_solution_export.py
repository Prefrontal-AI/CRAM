"""Every cell of the paper's N-Queens and graph-coloring comparison (Table 3; results/multi-solution-table.json), from
the runs of this repository; read-only (no training or search). Run from experiments/ (the 10 x 10 board enumerates the
solutions of every test puzzle; 4 CPU cores, 32 GB of memory and 40 minutes are sufficient):
  python multi_solution_export.py
after nqueens_export.py, nqueens_free_export.py, nqueens_offset_export.py, graph_coloring_export.py, the baselines'
exports (results/ebr-comparison.json) and nqueens_coverage.py --mode ebr (EBR's N-Queens coverage, which needs EBR's
environment). A cell whose runs have not finished is null. The paper reads the key `rows` of the output.

Methods and sources (seeds 0-2; each mean +- sample SD over seeds):
- TRM, HRM: trained with GRAM's code, each architecture's config and GRAM's per-task training settings (baselines/gram).
- GRAM: scored by baselines/gram/score_multi.py at 16 steps (GRAM's setting), 20 samples per test row.
- EBR: the authors' code at commit 8c5271c, scored at its |grad_M E| < 0.2 stop (results/ebr-comparison.json).
- CRAM rows: the offset-shared energy (nqueens_offset.py: the recipe selected at 10 x 10, at both sizes), the free pair
  tables (N-Queens: the validation-selected training method of nqueens_free.py; graph coloring: the `full` arm of
  graph_coloring.py, pseudolikelihood + L1), and the hand-set energies (nqueens_free.py --mode hand,
  graph_coloring_hand_set.py).
Metrics:
- N-Queens Acc: test rows (a puzzle with S solutions occurs S times) whose answer is a valid board holding every given
  queen; N = 1 the first sample; N = 20 GRAM, TRM, HRM the majority vote of 20 (GRAM's rule), EBR and CRAM the
  lowest-energy of 20 (EBR's rule). Cov@20: nqueens_coverage.py's any-valid rule, per puzzle (main) and per row.
- Graph coloring Conflict: conflicting edges of the answer, summed over the test graphs (255 and 192); N = 20 as above.
  Cov@20: distinct proper colorings in canonical color order found by the 20 samples, over all canonical proper
  colorings, summed over graphs (GRAM's definition, color tokens only).
- Params: learned parameters of the networks; CRAM: nonzero contributing parameters (learned in `params_learned`).
- Fit: wall clock of seed 0 on one H200 GPU, seconds. Baselines: one cold fit (the first fit in a fresh process).
  CRAM: the warm fit, the median of 7 complete fits after the cold fit in its process, because the one-time start-up
  of a process is a large part of a fit of seconds and a negligible part of a baseline fit of hours. For the
  offset-shared CRAM runs, the fit timed again with the dense pair-matrix map (nqueens_offset.py --mode retime,
  fit-dense.json).
Dimension names: r test rows, k samples per row, c cells, u distinct puzzles.
"""
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

import nqueens_coverage as C
from cram import cram_nqueens as Q
from cram import graph_coloring_data as G

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
TIM = HERE.parent / 'baselines' / 'runs' / 'gram'
NQ = HERE / 'runs' / 'NQ'
GC = HERE / 'runs' / 'CRAM-GC' / 'final'
OUT = RES / 'multi-solution-table.json'
SEEDS = (0, 1, 2)
GC_RUN = {8: 'n8-full-u1024-lr10-lam0.0001/test-terms-T2-S20-cap10000-s{}', 10: 'n10-full-u256-lr10-lam0.001/test-terms-T3-S20-cap10000-s{}'}
GC_FIT = {8: 'n8-full-u1024-lr10-lam0.0001', 10: 'n10-full-u256-lr10-lam0.001'}


def read(p):
    """The parsed JSON file at p."""
    return json.loads(Path(p).read_text())


def agg(values):
    """{mean, sd (sample SD), per_seed} of per-seed values; None when a value is missing."""
    if values is None or any(v is None for v in values):
        return None
    return dict(mean=float(mean(values)), sd=float(stdev(values)) if len(values) > 1 else None, per_seed=[float(v) for v in values])


def nq_cells(n, acc1, acc20, covs):
    """The N-Queens cells of one method: acc1, acc20 per-seed accuracies in % (N = 1, N = 20); covs per seed
    (per_row, per_puzzle) Cov@20 in %. None for runs that have not finished."""
    return dict(acc_n1=agg(acc1), acc_n20=agg(acc20), cov20_per_puzzle=agg(None if covs is None else [c[1] for c in covs]),
                cov20_per_row=agg(None if covs is None else [c[0] for c in covs]))


# --------------------------------------------------------------------------------------------- GRAM, TRM, HRM

def gram_family(prefix, n_nq, te, gc_n):
    """N-Queens (board size n_nq) and graph-coloring (gc_n vertices) cells of one model trained with GRAM's code
    (run-name prefix '' for GRAM, 'trm_' or 'hrm_'), from baselines/runs/gram/score/. te: (inputs (r, c), inv (r,),
    S (u,)) of the N-Queens test rows (nqueens_coverage.puzzles). Returns {'nq': cells, 'gc': cells}."""
    out = {}
    inputs, inv, S = te
    acc1, acc20, covs = [], [], []
    for s in SEEDS:
        d = TIM / 'score' / f'{prefix}nqueens{n_nq}-s{s}'
        if not (d / 'score.json').exists():
            acc1 = acc20 = covs = None
            break
        t = read(d / 'score.json')['table']
        acc1.append(100 * t['steps16-n1']['acc_completion'])
        acc20.append(100 * t['steps16-n20']['acc_completion'])
        z = np.load(d / 'shard-0-20.npz')
        P = z['preds'][:, :, list(z['steps']).index(16)]
        boards = (P == 2) & np.isin(P, (1, 2)).all(-1, keepdims=True)  # a pad or other token makes a sample count for nothing
        covs.append(C.pooled(inv, *C.row_coverage(inputs, boards, n_nq, S[inv])))
    out['nq'] = nq_cells(n_nq, acc1, acc20, covs)
    c1, c20, cov = [], [], []
    for s in SEEDS:
        f = TIM / 'score' / f'{prefix}gc{gc_n}-s{s}' / 'score.json'
        if not f.exists():
            c1 = c20 = cov = None
            break
        t = read(f)['table']
        c1.append(t['steps16-n1']['conflicts'])
        c20.append(t['steps16-n20']['conflicts'])
        cov.append(100 * t['steps16-n20']['coverage_strict'])
    out['gc'] = dict(conflicts_n1=agg(c1), conflicts_n20=agg(c20), cov20=agg(cov))
    return out


def gram_fit(prefix, task):
    """(fit seconds, parameter count) of the seed-0 fit of one model and task, or (None, None) when it has not finished."""
    f = TIM / f'{prefix}{task}-s0' / 'fit.json'
    if not f.exists():
        return None, None
    d = read(f)
    return d['fit_seconds'], d['params']


# ------------------------------------------------------------------------------------------------------- CRAM

def cram_nq(n, te, paths, fits):
    """N-Queens cells of a CRAM row from the per-seed records.npz files `paths` (nqueens_free.score: boards (r, k, c)
    packed bits and energy (r, k)). N = 20 is the lowest-energy board of the 20 chains. fits: path of a fit.json, or
    None. Returns (cells, the parsed fit.json or None)."""
    inputs, inv, S = te
    if not all(Path(p).exists() for p in paths):
        return nq_cells(n, None, None, None), None
    acc1, acc20, covs = [], [], []
    for p in paths:
        z = np.load(p)
        boards = np.unpackbits(z['boards'], -1)[..., :n * n].astype(bool)
        ok = Q.valid(boards, n) & (boards >= inputs[:, None, :]).all(-1)
        acc1.append(100 * ok[:, 0].mean())
        pick = np.argmin(z['energy'], 1)
        acc20.append(100 * ok[np.arange(len(ok)), pick].mean())
        covs.append(C.pooled(inv, *C.row_coverage(inputs, boards, n, S[inv])))
    fit = read(fits) if fits and Path(fits).exists() else None
    return nq_cells(n, acc1, acc20, covs), fit


def cram_gc(n, run=None):
    """Graph-coloring cells from the test outputs of graph_coloring.py (outputs.npz, final.json) for n vertices.
    run: a folder pattern with {} for the seed (None: the `full` arm's test folders, GC_RUN)."""
    x, y = G.load_split(n, 'test')
    a, _ = G.convert(x, y, n)
    A = G.adjacency(a, n)
    c1, c20, cov = [], [], []
    for s in SEEDS:
        d = (GC / GC_RUN[n].format(s)) if run is None else Path(run.format(s))
        z = np.load(d / 'outputs.npz')
        conf = G.conflicts(A, z['boards'])  # (graphs, 20)
        c1.append(int(conf[:, 0].sum()))
        c20.append(int(conf[np.arange(len(conf)), np.argmin(z['energy'], 1)].sum()))
        cov.append(100 * read(d / 'final.json')['coverage_20'])
    return dict(conflicts_n1=agg(c1), conflicts_n20=agg(c20), cov20=agg(cov))


def main():
    """Write results/multi-solution-table.json and print the means of every cell."""
    gc_res = read(RES / 'graph-coloring-results.json')
    nq_res = read(RES / 'nqueens-results.json')
    ebr_cmp = read(RES / 'ebr-comparison.json')
    ebr_cov = read(RES / 'nqueens-coverage-ebr.json')['sizes']
    te = {}
    for n in (8, 10):
        inputs = Q.load_split(n, 'test')[0]
        uniq, inv, S = C.puzzles(inputs)
        te[n] = (inputs, inv, S)
    rows = []
    for label, prefix in (('TRM', 'trm_'), ('HRM', 'hrm_'), ('GRAM', '')):
        cells = {}
        fits, params = {}, {}
        for n in (8, 10):
            g = gram_family(prefix, n, te[n], n)
            cells[f'nq{n}'], cells[f'gc{n}'] = g['nq'], g['gc']
            for task in (f'nqueens{n}', f'gc{n}'):
                fits[task], params[task] = gram_fit(prefix, task)
                if params[task] is None:  # the fit has not finished: the architecture's size on the other task of the family
                    params[task] = next((v for k, v in params.items() if v is not None and k[:2] == task[:2]), None)
        rows.append(dict(method=label, kind='network', deterministic=label != 'GRAM', params=params, fit_s=fits, cells=cells,
                         n20_rule='majority vote of 20 (GRAM)'))
    # EBR
    e = ebr_cmp['ebr']
    cells, fits = {}, {}
    for n in (8, 10):
        q = e[f'nqueens{n}']
        cells[f'nq{n}'] = dict(acc_n1=dict(mean=q['n1']['mean'], sd=q['n1']['sd'], per_seed=q['n1'].get('per_seed')),
                               acc_n20=dict(mean=q['n20']['mean'], sd=q['n20']['sd'], per_seed=q['n20'].get('per_seed')),
                               cov20_per_puzzle=dict(mean=ebr_cov[str(n)]['per_puzzle']['mean'], sd=ebr_cov[str(n)]['per_puzzle']['sd']),
                               cov20_per_row=dict(mean=ebr_cov[str(n)]['per_row']['mean'], sd=ebr_cov[str(n)]['per_row']['sd']))
        g = e[f'gc{n}']
        cells[f'gc{n}'] = dict(conflicts_n1=dict(mean=g['n1']['mean'], sd=g['n1']['sd'], per_seed=g['n1'].get('per_seed')),
                               conflicts_n20=dict(mean=g['n20']['mean'], sd=g['n20']['sd'], per_seed=g['n20'].get('per_seed')),
                               cov20=dict(mean=g['cov20_gram_uncapped']['mean'], sd=g['cov20_gram_uncapped']['sd']))
        fits[f'nqueens{n}'] = q['fit_s']['per_seed'][0]
        fits[f'gc{n}'] = g['fit_s']['per_seed'][0]
    params = {f'{t}{n}': e[f'{k}{n}']['params'] for n in (8, 10) for t, k in (('nqueens', 'nqueens'), ('gc', 'gc'))}
    rows.append(dict(method='EBR', kind='network', deterministic=False, params=params, fit_s=fits, cells=cells,
                     n20_rule='lowest energy of 20 (EBR)'))
    # CRAM, offset-shared: one recipe at both sizes, the recipe selected at 10 x 10, unchanged;
    # at 8 x 8 its runs are in nq11/n8/shared/, at 10 x 10 they are the selected pair's final runs
    cells, fits, params = {}, {}, {}
    for n in (8, 10):
        hf = NQ / 'nq11' / 'n10' / 'selected-headline.json'
        stage = 'final' if n == 10 else 'shared'
        if hf.exists():
            h = read(hf)
            name = f"{h['method']}-{h['arm']}"
            paths = [NQ / 'nq11' / f'n{n}' / stage / f'{name}-s{s}' / 'records.npz' for s in SEEDS]
            cells[f'nq{n}'], fit = cram_nq(n, te[n], paths, NQ / 'nq11' / f'n{n}' / stage / f'{name}-s0' / 'fit.json')
            dense = NQ / 'nq11' / f'n{n}' / stage / f'{name}-s0' / 'fit-dense.json'  # the fit timed again with the dense pair-matrix map (--mode retime)
            fits[f'nqueens{n}'] = read(dense)['warm_seconds'] if dense.exists() else (None if fit is None else fit.get('warm_seconds'))
            params[f'nqueens{n}'] = None if fit is None else fit['params_nonzero']
            cells[f'nq{n}']['config'] = h['choice']
        else:
            cells[f'nq{n}'] = nq_cells(n, None, None, None)
            fits[f'nqueens{n}'] = params[f'nqueens{n}'] = None
    rows.append(dict(method='CRAM (offset-shared pair tables)', kind='cram', deterministic=False, params=params, fit_s=fits,
                     cells=cells, n20_rule='lowest energy of 20', pilot='nqueens_offset.py: the recipe selected at 10 x 10, at both sizes'))
    # CRAM, free pair tables (N-Queens: the selected training methods of nqueens_free.py; graph coloring: the `full` arm)
    cells, fits, params = {}, {}, {}
    tm = nq_res['training_methods']['sizes']
    for n in (8, 10):
        m = tm[str(n)]['selected_method']
        paths = [NQ / 'alt' / f'n{n}' / 'final' / f'{m}-s{s}' / 'records.npz' for s in SEEDS]
        cells[f'nq{n}'], _ = cram_nq(n, te[n], paths, None)
        cells[f'nq{n}']['method'] = m
        r = tm[str(n)]['rows'][m]
        fits[f'nqueens{n}'] = r['fit_gpu_warm_s']['per_seed'][0]
        params[f'nqueens{n}'] = max(r['params_nonzero']['per_seed'])
        cells[f'gc{n}'] = cram_gc(n)
        g = next(x for x in gc_res['rows'] if x['benchmark'] == f'graph-coloring-{n}' and x['row'] == 'cram-full')
        fits[f'gc{n}'] = read(GC / GC_FIT[n] / 'cuda' / 'final.json').get('warm_seconds')
        params[f'gc{n}'] = g.get('params_nonzero', g.get('params'))
    rows.append(dict(method='CRAM (free pair tables)', kind='cram', deterministic=False, params=params, fit_s=fits,
                     cells=cells, n20_rule='lowest energy of 20', pilot='nqueens_free.py, selected method (N-Queens); graph_coloring.py, arm full (graph coloring)'))
    # Hand-set energies
    cells = {}
    for n in (8, 10):
        paths = [NQ / 'alt' / f'n{n}' / 'hand' / f'test-s{s}' / 'records.npz' for s in SEEDS]
        cells[f'nq{n}'], _ = cram_nq(n, te[n], paths, None)
        cells[f'gc{n}'] = cram_gc(n, str(HERE / 'runs' / 'CRAM-GC' / 'hand' / f'n{n}' / 'test-s{}'))
    rows.append(dict(method='CRAM (hand-set)', kind='reference', deterministic=False,
                     params={'nqueens8': 2, 'nqueens10': 2, 'gc8': 1, 'gc10': 1}, fit_s=None, cells=cells,
                     n20_rule='lowest energy of 20', pilot='nqueens_free.py --mode hand (N-Queens), graph_coloring_hand_set.py (graph coloring)',
                     note=('N-Queens: 1 per attacking queen pair, -0.5 per queen. Graph coloring: 1 per edge whose vertices share '
                           'a color, outputs relabeled to canonical color order. No learned parameters; params count the '
                           'distinct nonzero values, as Table 2 counts its hand-set rows (N-Queens: 1 and -0.5; graph '
                           'coloring: 1). Per-pair nonzero entries: 728 + 64 (8x8), 1,470 + 100 (10x10).')))
    ceilings = {}
    for n in (8, 10):
        inputs, inv, S = te[n]
        r, p = C.ceiling(inv, S)
        ceilings[f'nq{n}'] = dict(per_row=r, per_puzzle=p)
    OUT.write_text(json.dumps(dict(description=__doc__, rows=rows, cov20_ceiling_uniform=ceilings,
                                   published=dict(source='EBR Table 2 (arXiv 2611.08977) = GRAM Table 1 for TRM, HRM, GRAM',
                                                  note='EBR published per-puzzle any-valid Cov@20; GRAM, TRM, HRM per-row first-20 Cov@20')),
                              indent=1) + '\n')
    for r in rows:
        print(r['method'], {k: (None if v is None else {m: (None if x is None or not isinstance(x, dict) else round(x['mean'], 2)) for m, x in v.items() if m not in ('config', 'method')}) for k, v in r['cells'].items()})
    print('ceilings', ceilings)


if __name__ == '__main__':
    main()
