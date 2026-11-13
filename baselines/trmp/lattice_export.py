"""Export the TRMP lattice ("ten-minute") Sudoku-Extreme results for the paper: results/trmp-lattice-sudoku.json.

Inputs: seed 0 is the cold fit of the warm set runs/trmp-lattice/warm-s0 (cold/ and
repeats.json); seeds 1 and 2 are single cold fits runs/trmp-lattice/sudoku-extreme-s<seed>. Each run folder holds
fit.json, score/test.json (412,786 test puzzles) and score/flops.json (operation count by the convention of baselines/flops/fullcount.py, written by flops/trmp_lattice_count.py).
Accuracy: mean and sample SD over the three seeds. FLOPs per puzzle: mean over all test puzzles of the three seeds.
Fit time: seed 0, cold and warm (median of 7 further fits in the process of the cold fit), one H200.

  python trmp/lattice_export.py        # from baselines/
"""
import hashlib
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINES = HERE.parent
ROOT = BASELINES.parent
RUNS = BASELINES / 'runs/trmp-lattice'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    folders = [RUNS / 'warm-s0/cold', RUNS / 'sudoku-extreme-s1', RUNS / 'sudoku-extreme-s2']
    seeds, sources = [], {}
    for seed, f in enumerate(folders):
        fit = json.loads((f / 'fit.json').read_text())
        test = json.loads((f / 'score/test.json').read_text())
        flops = json.loads((f / 'score/flops.json').read_text())
        assert fit['config']['seed'] == seed and not fit['resumed'] and test['count'] == 412786
        assert test['model_sha256'] == fit['model_sha256'] == flops['model_sha256']
        seeds.append(dict(seed=seed, solved=test['solved'], count=test['count'], solve_rate=test['solve_rate'],
                          timeouts=test['timeouts'], wrong=test['wrong'], mean_steps=test['mean_steps'],
                          flop_per_puzzle=flops['flop_per_puzzle_mean'], fit_seconds_cold=fit['fit_seconds'],
                          selected_step=fit['selected_step'], selected_dev=fit['selected_dev'], gpu=fit['gpu'],
                          slurm_job=fit['slurm_job'], params=fit['params']))
        for p in (f / 'fit.json', f / 'score/test.json', f / 'score/flops.json'):
            sources[str(p.relative_to(ROOT))] = sha(p)
    rep = json.loads((RUNS / 'warm-s0/repeats.json').read_text())
    sources[str((RUNS / 'warm-s0/repeats.json').relative_to(ROOT))] = sha(RUNS / 'warm-s0/repeats.json')
    assert abs(rep['cold_seconds'] - seeds[0]['fit_seconds_cold']) < 1e-6
    rates = [s['solve_rate'] for s in seeds]
    out = dict(
        method='TRMP lattice model (factor graph, 16k), the ten-minute recipe of arXiv 2610.18649 (Table 4, "Full gradient, factor graph")',
        benchmark='sudoku-extreme', status='3-seed scored clean-room TRMP lattice reimplementation', published_accuracy_used=False,
        parameters=seeds[0]['params'], solved_mean=statistics.fmean(rates), solved_sd=statistics.stdev(rates),
        per_seed=rates, n_test=412786, flop_per_puzzle=statistics.fmean(s['flop_per_puzzle'] for s in seeds),
        fit_cold_seconds_seed0=rep['cold_seconds'], fit_warm_median_seconds_seed0=rep['warm_median_seconds'],
        fit_warm_seconds_seed0=rep['warm_seconds'], device='one H200',
        inference='64 chains per puzzle, budget 1,000 deep-supervision steps, check every step, conflict threshold 0.8',
        flop_convention='FC2 (baselines/flops/fullcount.py), whole chain search counted; mean over all test puzzles of the three seeds',
        published=dict(abstract='about 99% with 16k parameters, about 10 minutes on an H100',
                       appendix_b6='a 20,000-step Sudoku lattice run at 16k parameters takes about 15 minutes (H100 often shared by two jobs)',
                       appendix_b4='with full gradients, the 16k factor graph exceeds 99% (1,000 test puzzles)'),
        seeds=seeds, sources=sources, code='baselines/trmp/lattice*.py, baselines/flops/trmp_lattice_count.py')
    path = ROOT / 'results/trmp-lattice-sudoku.json'
    path.write_text(json.dumps(out, indent=1) + '\n')
    print(path, json.dumps({k: out[k] for k in ('solved_mean', 'solved_sd', 'flop_per_puzzle', 'fit_cold_seconds_seed0', 'fit_warm_median_seconds_seed0')}))


if __name__ == '__main__':
    main()
