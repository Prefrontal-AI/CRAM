"""Copy the trained CRAM models that the paper reports from the run folders into checkpoints/, and write the manifest.

The run folders (experiments/runs/) are not tracked; checkpoints/ is. Each file is copied byte for byte, so its SHA-256 equals
that of the run file. Run from the repository root, after a model that the paper reports has been retrained:

    uv run python scripts/collect_checkpoints.py

`checkpoints/load.py` reads the manifest; `checkpoints/README.md` describes the tensors.
The run-folder names in MODELS (CRAM-S, CU13, NQ/nq11, CRAM-GC) are only labels of the folders that the experiment
scripts write to.
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'experiments' / 'runs'
OUT = ROOT / 'checkpoints'
SEEDS = (0, 1, 2)

# name -> the paper's table, the run folder of each training seed and the files to copy from it
MODELS = {
    'sudoku-extreme': dict(
        table='Table 2', record='experiments/runs/table/sudoku-extreme__cram-0-fit.json',
        runs=[f'CRAM-S/full/full-aug-s{s}' for s in SEEDS], files=['model.pt', 'final.json'],
        script='experiments/sudoku.py', module='cram/cram_sudoku_general.py'),
    'maze-unique': dict(
        table='Table 2', record='experiments/runs/table/maze-unique__cram-sgd-fg.json',
        runs=[f'CU13/final/fg-s{s}' for s in SEEDS], files=['model.pt', 'config.json', 'final.json'],
        script='experiments/maze_unique.py', module='cram/cram_maze.py (CRAMMazeFreeEnergy)'),
    'nqueens-8': dict(
        table='Table 3', record='results/multi-solution-table.json',
        runs=[f'NQ/nq11/n8/shared/pl-ising-s{s}' for s in SEEDS], files=['model.pt', 'final.json', 'fit.json'],
        script='experiments/nqueens_offset.py', module='cram/cram_nqueens.py'),
    'nqueens-10': dict(
        table='Table 3', record='results/multi-solution-table.json',
        runs=[f'NQ/nq11/n10/final/pl-ising-s{s}' for s in SEEDS], files=['model.pt', 'final.json', 'fit.json'],
        script='experiments/nqueens_offset.py', module='cram/cram_nqueens.py'),
    # Graph coloring: the fit is deterministic, so there is one model; the three seeds of Table 3 are seeds of the search.
    'graph-coloring-8': dict(
        table='Table 3', record='results/multi-solution-table.json',
        runs=['CRAM-GC/final/n8-full-u1024-lr10-lam0.0001/cuda'], files=['model.npz', 'model.json', 'final.json'],
        script='experiments/graph_coloring.py', module='cram/cram_graph_coloring.py'),
    'graph-coloring-10': dict(
        table='Table 3', record='results/multi-solution-table.json',
        runs=['CRAM-GC/final/n10-full-u256-lr10-lam0.001/cuda'], files=['model.npz', 'model.json', 'final.json'],
        script='experiments/graph_coloring.py', module='cram/cram_graph_coloring.py'),
}
# The Maze-Hard models (checkpoints/maze-hard/) are the files final-s<seed>.pt and final-s<seed>.json of
# experiments/runs/CRAM-HE/final-triton/CHE8-mu, stored as model.pt and final.json; their manifest entry holds the
# scores of that folder's test-scores.json. This script keeps an existing 'maze-hard' entry unchanged.


def sha256(path):
    """SHA-256 of a file, as a hexadecimal string."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected(name, src, seed):
    """The recorded test score of one model, which checkpoints/evaluate.py compares its own score with.

    name: a key of MODELS; src: the run folder of the model; seed: the training seed (index into the model's runs).
    Returns a dict (a list of dicts for graph coloring, one per search seed) with the record path, the search settings
    and the recorded counts."""
    read = lambda p: json.loads(p.read_text())
    if name == 'sudoku-extreme':  # the test record is 20 MB (per-puzzle results); keep its summary and settings
        r = read(RUNS / 'CRAM-S/full/test' / f'cram0-hb-fit-s{seed}.json')
        dev = read(RUNS / 'CRAM-S/full/dev' / f'cram0-hb-fit-s{seed}.json')  # the 10,000 development puzzles: rerun exactly
        keys = ('n', 'solved', 'wrong_stops', 'budget_failures', 'mean_flop')
        return dict(record=f'experiments/runs/CRAM-S/full/test/cram0-hb-fit-s{seed}.json', rel=r['signature']['rel'],
                    **{k: r['summary'][k] for k in keys},
                    dev=dict(record=f'experiments/runs/CRAM-S/full/dev/cram0-hb-fit-s{seed}.json', **{k: dev['summary'][k] for k in keys}))
    if name == 'maze-unique':
        r = read(src / 'score-test-gibbs.json')
        return dict(record=f'{src.relative_to(ROOT)}/score-test-gibbs.json', search=r['search'],
                    **{k: r[k] for k in ('n', 'solved', 'wrong_stops', 'budget_failures', 'mean_flop')})
    if name.startswith('nqueens'):
        r = read(src / 'final.json')
        return dict(record=f'{src.relative_to(ROOT)}/final.json',
                    search={k: r['selection'][k] for k in ('stop', 'temperature', 'cap')},
                    **{k: r['test'][k] for k in ('rows', 'acc_n1', 'acc_majority20', 'cov20', 'wrong_stops')})
    out = []  # graph coloring: one fit, three seeds of the search
    for d in sorted(src.parent.glob('test-*-s[0-9]')):
        r = read(d / 'final.json')
        out.append(dict(record=f'{d.relative_to(ROOT)}/final.json', seed=r['seed'],
                        search={k: r['recipe'][k] for k in ('stop', 'temperature', 'samples', 'cap')},
                        **{k: r[k] for k in ('graphs', 'conflicts_1', 'conflicts_majority_20', 'coverage_20', 'wrong_stops_all')}))
    return out


def main():
    """Copy every file of MODELS to checkpoints/<name>/seed<i>/ (or model/ for a single run) and write manifest.json."""
    old = json.loads((OUT / 'manifest.json').read_text()) if (OUT / 'manifest.json').exists() else {}
    manifest = {}
    for name, m in MODELS.items():
        entries = []
        for i, run in enumerate(m['runs']):
            src = RUNS / run
            dest = OUT / name / (f'seed{i}' if len(m['runs']) > 1 else 'model')
            dest.mkdir(parents=True, exist_ok=True)
            files = {}
            for f in m['files']:
                shutil.copyfile(src / f, dest / f)
                files[f] = sha256(dest / f)
            entries.append(dict(folder=str(dest.relative_to(ROOT)), run=f'experiments/runs/{run}', sha256=files,
                                expected=expected(name, src, i)))
        manifest[name] = dict(table=m['table'], record=m['record'], script=m['script'],
                              module=m['module'], weights=m['files'][0], models=entries)
    if 'maze-hard' in old:
        manifest['maze-hard'] = old['maze-hard']
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'wrote {OUT / "manifest.json"}: ' + ', '.join(f'{k} ({len(v["models"])})' for k, v in manifest.items()))


if __name__ == '__main__':
    main()
