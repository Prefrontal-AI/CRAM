"""EBR (Energy-Based Reasoner, arXiv 2611.08977) against CRAM on the tasks of the paper that EBR was trained on here:
Sudoku-Extreme, Maze-Unique, N-Queens (8, 10) and graph coloring (8, 10). CPU only; reads finished records, runs nothing.

  python ebr/comparison.py      # writes results/ebr-comparison.json

EBR: three cold fits per task with the authors' code (github.com/wwillchen/ebr at 8c5271c, unmodified; ebr/fit.py),
scored by the authors' evaluators (ebr/score.sbatch) and aggregated by ebr/report.py (each run's report-fc2.json, written
by ebr/report_fc2.sbatch). Fit seconds come from each run's fit.json (baselines/clock.py: evaluation, checkpoints and disk
reads excluded). Flop (floating-point operations) per puzzle comes from the per-component counts of `ebr/flops.py --full`
(runs/ebr/flops-full/<bench>.json: all arithmetic, the operation-count convention of baselines/flops/fullcount.py; math
attention, batch 1, float32). Published values: the project page energy-based-reasoners.github.io, retrieved 2026-11-08.
CRAM: the table records of the paper's rows (experiments/runs/table/<benchmark>__<row>.json) for Sudoku-Extreme,
Maze-Unique and Maze-Hard, and the selected N-Queens and graph-coloring energies (results/nqueens-results.json,
results/graph-coloring-results.json); operation counts from results/op-counts.json.
Each task keeps one metric definition for both methods; the definitions are in the output's "definitions".
"""
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
RUNS = HERE.parent / 'runs/ebr'
REPORT = 'report-fc2.json'  # counts all arithmetic (baselines/flops/fullcount.py); the file name is only a label
TABLE = CODE / 'experiments/runs/table'  # one record per paper row of CRAM

PUBLISHED = {  # energy-based-reasoners.github.io, retrieved 2026-11-08 (percent)
    'sudoku': dict(n1=99.4, n5=99.9),
    'nqueens8': dict(cov20=98.1), 'nqueens10': dict(cov20=92.0), 'gc8': dict(cov20=88.6), 'gc10': dict(cov20=54.5),
}


def read(p):
    """Parse the JSON file at path p."""
    return json.loads(Path(p).read_text())


def stats(v):
    """Mean, sample standard deviation (0 for one value) and the values themselves, of a list of numbers."""
    v = [float(x) for x in v]
    return dict(mean=statistics.fmean(v), sd=statistics.stdev(v) if len(v) > 1 else 0.0, per_seed=v)


def ebr_fit(bench):
    """Fit seconds of the three seeds of one benchmark; fails if a fit was resumed (not a cold fit)."""
    fits = [read(RUNS / f'{bench}-s{k}/fit.json') for k in range(3)]
    assert not any(f['resumed'] for f in fits)
    return stats([f['fit_seconds'] for f in fits])


def ebr_params(bench):
    """Parameter count of EBR's model for one benchmark."""
    return read(RUNS / f'flops/{bench}.json')['params']


def ebr_sudoku():
    """EBR on Sudoku-Extreme: percent solved and flop per puzzle at four inference settings, over three seeds."""
    reps = [read(RUNS / f'sudoku-s{k}/{REPORT}') for k in range(3)]
    cell = lambda key, f: stats([f(r['unseen'][key]) for r in reps])
    rate = lambda c: 100 * c['solved'] / c['n']
    return dict(
        params=ebr_params('sudoku'), fit_s=ebr_fit('sudoku'), test='412,786 unseen Sudoku-Extreme test puzzles',
        published_setting=dict(setting='lowest energy of 5 samples, 64 segments', solved=cell('D64-N5', rate), flop=cell('D64-N5', lambda c: c['mean_flop'])),
        n1_d64=dict(solved=cell('D64-N1', rate), flop=cell('D64-N1', lambda c: c['mean_flop'])),
        adaptive_n5=dict(setting='|grad_M E| < 0.2, at most 64 segments, N=5', solved=cell('gtan0.2-cap64-N5', rate), flop=cell('gtan0.2-cap64-N5', lambda c: c['mean_flop'])),
        adaptive_n1=dict(setting='|grad_M E| < 0.2, at most 64 segments, N=1', solved=cell('gtan0.2-cap64-N1', rate), flop=cell('gtan0.2-cap64-N1', lambda c: c['mean_flop'])),
        published=PUBLISHED['sudoku'])


def ebr_maze(bench, recipe):
    """EBR on a maze benchmark: percent solved (exact label, and any shortest path) and flop per maze, over three seeds."""
    reps = [read(RUNS / f'{bench}-s{k}/{REPORT}') for k in range(3)]
    out = dict(params=ebr_params(bench), recipe=recipe, test='1,000 test mazes')
    out['fit_s'] = ebr_fit(bench)
    for metric in ('exact', 'shortest'):
        for key in ('D16-N5', 'D16-N1', 'gtan0.2-cap16-N5', 'gtan0.2-cap16-N1'):
            out[f'{metric} {key}'] = dict(solved=stats([100 * r[metric][key]['solved'] / r[metric][key]['n'] for r in reps]), flop=stats([r[metric][key]['mean_flop'] for r in reps]))
    return out


def ebr_gram(bench):
    """EBR on N-Queens or graph coloring at its own stopping rule: every metric of ebr/report.py, over three seeds."""
    reps = [read(RUNS / f'{bench}-s{k}/{REPORT}') for k in range(3)]
    rule = 'gtan<0.2'  # EBR's own protocol for these tasks (evaluators/gram.py)
    keys = [k for k in reps[0][rule] if k != 'segments']
    out = dict(params=ebr_params(bench), fit_s=ebr_fit(bench), stop_rule='|grad_M E| < 0.2 at a segment end, else the trained depth; 20 samples per input',
               **{k: stats([r[rule][k] for r in reps]) for k in keys}, segments=stats([r[rule]['segments'] for r in reps]))
    if bench in PUBLISHED:
        out['published'] = PUBLISHED[bench]
    return out


def cram():
    """CRAM's cells for the same tasks, from its table records, the tracked results and results/op-counts.json."""
    fc2 = {(c['benchmark'], c['row']): c['new'] for c in read(CODE / 'results/op-counts.json')['cells']}  # flop per paper cell
    su, mu, mh = (read(TABLE / f'{name}.json') for name in ('sudoku-extreme__cram-0-fit', 'maze-unique__cram-sgd-fg', 'maze-hard__cram-latent-zmu'))
    nq = read(CODE / 'results/nqueens-results.json')['training_methods']['sizes']
    gc = {r['benchmark']: r for r in read(CODE / 'results/graph-coloring-results.json')['rows'] if r['row'] == 'cram-full'}
    out = dict(
        sudoku=dict(solved=100 * su['solved_mean'], flop=fc2['sudoku-extreme', 'cram-0-fit'], params_nonzero=su['extra']['nonzero_params'], fit_s=su['fit_gpu_warm_s'],
                    source='experiments/runs/table/sudoku-extreme__cram-0-fit.json (fit_s: fit_gpu_warm_s; params_nonzero: per training split)'),
        mazeunique=dict(solved=100 * mu['solved_mean'], flop=fc2['maze-unique', 'cram-sgd-fg'], params=mu['params'], fit_s=mu['fit_gpu_warm_s'],
                        source='experiments/runs/table/maze-unique__cram-sgd-fg.json (fit_s: fit_gpu_warm_s)'),
        mazehard=dict(solved=100 * mh['solved_mean'], flop=fc2['maze-hard', 'cram-latent-zmu'], params=mh['params'], fit_s=mh['train_gpu_s'],
                      metric='exact match to the labeled path', source='experiments/runs/table/maze-hard__cram-latent-zmu.json (fit_s: train_gpu_s, one cold fit)'))
    for n in ('8', '10'):
        r = nq[n]['rows'][nq[n]['selected_method']]
        row = f"cram-{r['method']}"
        out[f'nqueens{n}'] = dict(method=r['method'], **{k: r[k] for k in ('acc_n1', 'acc_majority20', 'cov20', 'cov20_group_mean', 'fit_gpu_cold_s')}, params_nonzero=r['params_learned'],
                                 flop_n1=fc2[f'nqueens-{n}', f'{row}-n1'], flop_n20=fc2[f'nqueens-{n}', row],
                                 source='results/nqueens-results.json training_methods, selected method')
    for n in ('8', '10'):
        r = gc[f'graph-coloring-{n}']
        out[f'gc{n}'] = dict(**{k: r['metrics'][k] for k in ('conflicts_1', 'conflicts_majority_20', 'coverage_20')}, flop_1=fc2[f'graph-coloring-{n}', 'cram-full-n1'], flop_20=fc2[f'graph-coloring-{n}', 'cram-full'], params=r['params'], params_nonzero=r['params_nonzero'], fit_s=r['fit']['gpu_cold_s'],
                             source='results/graph-coloring-results.json row cram-full')
    return out


def main():
    """Assemble both methods' cells and write results/ebr-comparison.json."""
    rec = dict(
        description=__doc__, status='In the paper: Table 2, the compute figure, and the N-Queens and graph-coloring table.',
        ebr=dict(sudoku=ebr_sudoku(), mazeunique=ebr_maze('mazeunique', 'maze_hard_fast, unchanged (EBR has no Maze-Unique recipe)'),
                 **{b: ebr_gram(b) for b in ('nqueens8', 'nqueens10', 'gc8', 'gc10')}),
        cram=cram(),
        definitions=dict(
            sudoku_maze='solved = exact match to the label; Maze "shortest" = any valid shortest path (a Maze-Unique maze has one shortest path).',
            nqueens_n1='N=1: percent of test rows (an input with s solutions occurs s times) whose first sample is a valid completion keeping the given queens.',
            nqueens_cov20='cov20 (EBR): distinct valid completions found by 20 samples, at most 20 per input, pooled over inputs over the sum of min(|S(x)|, 20); the published EBR value.',
            nqueens_cov20_group_mean='cov20_group_mean (the form GRAM\'s evaluator prints; the paper uses it for CRAM): per row, distinct samples equal to one of the input\'s first 20 completions over their number; mean within groups of equal min(s, 20), then over groups.',
            gc_cov20='Graph coloring Cov@20 (GRAM, uncapped): distinct proper colorings in canonical color order found by 20 samples over all canonical proper colorings, summed over graphs. EBR: cov20_gram_uncapped; CRAM: coverage_20.',
            gc_conflicts='conflicts: total conflicting edges of sample 0 (N=1) or of the selected sample (N=20) over the test graphs.',
            flop='EBR: all arithmetic (the operation-count convention of baselines/flops/fullcount.py), math attention, batch 1; adaptive settings add one gradient measure per segment. CRAM: the same convention applied to its search kernels (results/op-counts.json), every chain included.'))
    out = CODE / 'results/ebr-comparison.json'
    out.write_text(json.dumps(rec, indent=1) + '\n')
    print(out)


if __name__ == '__main__':
    main()
