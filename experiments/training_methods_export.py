"""Export the probe curves of the densely probed runs of the training-method study; no training or search.

  python training_methods_export.py     # run from experiments/

Writes ../results/alt-training/alt-dense.json from the runs <root>/dense/<task>-<method>-s<seed>/ that
slurm/training_methods/dense.sbatch trains: the selected setting of each method of the figure on seeds 0-2, with a
probe (solve rate on held-out puzzles, training_methods.probe) after every update up to the 64th and 8 probes per
doubling after that. A run that has not finished is left out and listed under "missing".

<root> is training_methods.root(task): runs/ALT for Sudoku; for the maze the folder that the environment variables
ALT_MAZE_KERNEL and ALT_MAZE_OPT select (see training_methods.py). The output records the roots under "roots".
The tracked alt-dense.json has the maze root runs/ALT-free (ALT_MAZE_KERNEL=free, ALT_MAZE_OPT unset).

Output: {"note", "tasks": {task: {method: {seed: {"curve": [[fit_s, solved %, update], ...], "t99", "clock_from"}}}},
"missing", "maze_kernel", "roots"}. fit_s is the fit time in seconds, t99 the first fit time at which the probe solve
rate is at least 99% (None: never).
"""
import json
import re
from pathlib import Path

import training_methods as A  # reads ALT_MAZE_KERNEL and ALT_MAZE_OPT at import: they select the maze run folder

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / 'results/alt-training'


def clock(name, task):
    """Fit time after each update of the run with the same task, method and seed that was not probed densely.

    name: run folder name '<task>-<method>-s<seed>'. task: 'sudoku' or 'maze'.
    The source is <root>/retime/<name> when it exists (a repeat of the final run that training_methods.retime_check
    verified), else <root>/final/<name>. Its log records every update. The dense run's own clock also runs between
    frequent probes, so the figure takes the time axis from this run; the table's time to 99% and the curves then share
    one clock.
    Returns ({update: fit seconds}, the source folder relative to experiments/), or ({}, None) when neither exists."""
    for d in (A.root(task) / 'retime' / name, A.root(task) / 'final' / name):
        if (d / 'log.jsonl').exists():
            recs = [json.loads(l) for l in (d / 'log.jsonl').read_text().splitlines()]
            return {r['update']: r['fit_s'] for r in recs if 'update' in r}, str(d.relative_to(HERE))
    return {}, None


def curve(run, task):
    """The probe curve of one dense run.

    run: the run folder (Path). task: 'sudoku' or 'maze'.
    Returns (points, t99, source): points is [[fit_s, solved %, update], ...] at every probe, with fit_s from clock()
    (the dense run's own fit time for an update that clock() does not have); t99 is the first fit_s with at least 99%
    solved (None: never); source is the folder that clock() read."""
    fit, src = clock(run.name, task)
    pts, t99 = [], None
    for line in (run / 'log.jsonl').read_text().splitlines():
        r = json.loads(line)
        if 'probe' not in r:
            continue
        pct = 100.0 * r['probe']['solved'] / r['probe']['n']
        t = fit.get(r['update'], r['fit_s'])
        pts.append([t, pct, r['update']])
        if t99 is None and pct >= 99.0:
            t99 = t
    return pts, t99, src


def dense():
    """The alt-dense.json object: the probe curve of every finished dense run of both tasks (see the module docstring)."""
    tasks, missing = {}, []
    for run in sorted(r for t in ('sudoku', 'maze') for r in (A.root(t) / 'dense').glob(f'{t}-*-s[0-9]')):
        task, method, seed = re.fullmatch(r'(\w+)-(\w+)-s(\d)', run.name).groups()
        if not (run / 'final.json').exists():
            missing.append(run.name)
            continue
        pts, t99, src = curve(run, task)
        tasks.setdefault(task, {}).setdefault(method, {})[seed] = {'curve': pts, 't99': t99, 'clock_from': src}
    return {'note': 'The selected setting of each method trained again with dense probes (every update to 64, then 8 per doubling); '
                    'curve rows are [fit_s, probe solved %, update]; fit_s is the fit clock at that update of the final run (or of its verified repeat in retime/) that was not probed densely.', 'tasks': tasks, 'missing': missing,
            'maze_kernel': A.MAZE_KERNEL, 'roots': A.REL}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    obj = dense()
    (OUT / 'alt-dense.json').write_text(json.dumps(obj, indent=1) + '\n')
    print('alt-dense.json', 'missing:', len(obj['missing']))
