"""Collect every timed fit into tracked records: baselines/records/<method>-<task>.json and baselines/records/table.md.

Reads the run folders under baselines/runs/ (gitignored) and copies, per cell, the fit records, the test scores and the
SHA-256 of each source file into one small JSON file that git tracks. Python standard library only.
A cell is one (method, task, seed) fit. A cell is `done` when its fit records and its scores exist; the table shows
the state of every expected cell. The fit records come from the wrappers `<baseline>/fit.py` (clock: baselines/clock.py)
and the scores from each baseline's scorer. The script also writes results/baseline-fit-times.json, a summary of the
records. It calls no baseline code.
Usage: python3 baselines/collect.py   (safe to run while jobs are running; it only reads runs/ and rewrites records/)
"""

import glob
import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS, RECORDS = HERE / 'runs', HERE / 'records'

# method, task, run folder (relative to runs/), fit files (relative to the run folder), score globs, Slurm job name
CELLS = [
    ('TRM', 'Sudoku-Extreme', 'trm/sudoku-s0', ['fit.json'], ['score.json'], 'time-trm-sudoku'),
    ('TRM', 'Maze-Unique', 'trm/mazeunique-s0', ['fit.json'], ['score.json'], 'time-trm-mazeunique'),
    ('FRM', 'Sudoku-Extreme', 'frm-compiled/sudoku', ['timing/fit-A.json', 'timing/fit-B.json'], ['runs/s0/result.json', 'runs/s0/*.json'], 'time-frm-sudoku'),
    ('FRM', 'Maze-Unique', 'frm-compiled/mazeunique', ['timing/fit-A.json', 'timing/fit-B.json'], ['runs/paper1000/s0/result.json'], 'time-frm-mazeunique'),
    # reference fits without the added torch.compile (the authors' code compiles only a helper); the paper reports the compiled fits
    ('FRM-nocompile', 'Sudoku-Extreme', 'frm/sudoku', ['timing/fit-A.json', 'timing/fit-B.json'], ['runs/s0/result.json', 'runs/s0/*.json'], 'time-frm-sudoku-nocompile'),
    ('FRM-nocompile', 'Maze-Unique', 'frm/mazeunique', ['timing/fit-A.json', 'timing/fit-B.json'], ['runs/paper1000/s0/result.json'], 'time-frm-mazeunique-nocompile'),
    ('EqR', 'Sudoku-Extreme', 'eqr/sudoku-s0', ['fit.json'], ['score-*.json'], 'time-eqr-sudoku'),
    ('EqR', 'Maze-Unique', 'eqr/maze-repo-s0', ['fit.json'], ['score-*.json'], 'time-eqr-maze-repo'),
    # the EqR paper's Maze-Unique configuration (hidden 128), timed to pair with EqR's published 93.0%
    ('EqR-paper', 'Maze-Unique', 'eqr/maze-paper-s0', ['fit.json'], ['score-*.json'], 'time-eqr-maze-paper'),
    # HRM Sudoku: the schedule of the released checkpoint (40,000 epochs)
    ('HRM', 'Sudoku-Extreme', 'hrm/sudoku-e40000-s0', ['fit.json'], ['score.json'], 'time-hrm-sudoku-e40000'),
    ('HRM', 'Maze-Unique', 'hrm/mazeunique-s0', ['fit.json'], ['score.json'], 'time-hrm-mazeunique'),
    ('TRMP', 'Sudoku-Extreme', 'trmp/sudoku-extreme-s0', ['fit.json'], ['score/test.json'], 'time-trmp-sudoku'),
    ('TRMP', 'Maze-Unique', 'trmp/maze-unique-s0', ['fit.json'], ['score/test.json'], 'time-trmp-mazeunique'),
    # GRAM (official code f2e32b9): scores in runs/gram/score/<task>-s<k>/ (gram/score.py, gram/unseen.py)
    ('GRAM', 'Sudoku-Extreme', 'gram/sudoku-s0', ['fit.json'], ['../score/sudoku-s0/score-unseen.json'], 'time-gram-sudoku-s0'),
    ('GRAM', 'Maze-Unique', 'gram/mazeunique-s0', ['fit.json'], ['../score/mazeunique-s0/score.json'], 'time-gram-mazeunique-s0'),
    # EBR (wwillchen/ebr 8c5271c; baselines/ebr/): ebr/report.py summaries of EBR's own evaluate.py outputs. Sudoku and
    # Maze-Unique are rows of Table 2; N-Queens and graph coloring are rows of Table 3
    ('EBR', 'Sudoku-Extreme', 'ebr/sudoku-s0', ['fit.json'], ['report.json'], 'time-ebr-sudoku-s0'),
    ('EBR', 'Maze-Unique', 'ebr/mazeunique-s0', ['fit.json'], ['report.json'], 'time-ebr-mazeunique-s0'),
    ('EBR', 'N-Queens-8', 'ebr/nqueens8-s0', ['fit.json'], ['report.json'], 'time-ebr-nqueens8-s0'),
    ('EBR', 'N-Queens-10', 'ebr/nqueens10-s0', ['fit.json'], ['report.json'], 'time-ebr-nqueens10-s0'),
    ('EBR', 'Graph-Coloring-8', 'ebr/gc8-s0', ['fit.json'], ['report.json'], 'time-ebr-gc8-s0'),
    ('EBR', 'Graph-Coloring-10', 'ebr/gc10-s0', ['fit.json'], ['report.json'], 'time-ebr-gc10-s0'),
]


def seeded(cells, seeds=(1, 2)):
    """Seed 0 as listed, plus seeds 1 and 2 (run folders and job names with the seed; FRM: one campaign folder per seed)."""
    out = [(*c, 0) for c in cells]
    for m, t, folder, fits, globs, job in cells:
        if m in ('FRM-nocompile', 'EqR-paper'):
            continue
        for k in seeds:
            if m == 'FRM':
                f = folder.replace('frm-compiled/', f'frm-compiled-s{k}/')
                g = [x.replace('/s0/', f'/s{k}/') for x in globs]
            elif m == 'GRAM':
                f, g = folder.replace('-s0', f'-s{k}'), [x.replace('-s0/', f'-s{k}/') for x in globs]
            else:
                f, g = folder.replace('-s0', f'-s{k}'), globs
            out.append((m, t, f, fits, g, job.replace('-s0', f'-s{k}') if m in ('GRAM', 'EBR') else f'{job}-s{k}', k))
    return out


CELLS = seeded(CELLS)


def sha256(path):
    """SHA-256 hex digest of the file at `path`."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def queue():
    """The user's Slurm jobs as {job name: [job id, state, elapsed]}; empty when `squeue` is not available."""
    try:
        out = subprocess.run(['squeue', '--me', '-h', '-o', '%j %i %T %M'], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return {}
    return {line.split()[0]: line.split()[1:] for line in out.splitlines() if line.strip()}


def accuracy(method, path, d):
    """A short accuracy summary for the table; the full score record is kept as it is."""
    try:
        if method == 'TRM':
            return f"{d['solved']}/{d['n']}"
        if method == 'FRM' and 'test' in d:
            return f"{d['test']['solved']}/{d['test']['count']}"
        if method == 'EBR':  # the authors' headline setting: lowest energy of 5 samples; Sudoku on the unseen puzzles
            if 'unseen' in d:
                r = d['unseen']['D64-N5']
                return f"D64-N5 unseen: {r['solved']}/{r['n']}"
            if 'exact' in d:
                r = d['exact']['D16-N5']
                return f"D16-N5: {r['solved']}/{r['n']}"
            return None
        if method == 'EqR':
            # EqR's selection rule at B > 1 is top-1 convergence (the authors' headline); at B = 1 every key is the one trajectory
            solved = d['solved'] if not isinstance(d['solved'], dict) else d['solved'].get('top1_conv', next(iter(d['solved'].values())))
            return f"{Path(path).stem.removeprefix('score-')}: {solved}/{d['n']}"
        for k in ('solved', 'correct'):
            if k in d and ('n' in d or 'count' in d):
                return f"{d[k]}/{d.get('n', d.get('count'))}"
    except Exception:
        pass
    return None


def source_hashes():
    """SHA-256 of the timing code: top-level scripts and the method folders (never runs/, which holds caches)."""
    files = [*HERE.glob('*.py'), *HERE.glob('*.sbatch')]
    for d in HERE.iterdir():
        if d.is_dir() and d.name not in ('runs', 'records', 'logs', '__pycache__'):
            files += [*d.glob('*.py'), *d.glob('*.sbatch'), *d.glob('*.sh')]
    return {str(p.relative_to(HERE)): sha256(p) for p in sorted(files)}


def collect():
    """Write records/<method>-<task>-s<seed>.json, records/table.md and results/baseline-fit-times.json.

    Returns True when every expected cell is `done`.
    """
    global SOURCE
    SOURCE = source_hashes()
    RECORDS.mkdir(exist_ok=True)
    jobs = queue()
    rows = []
    for method, task, folder, fits, score_globs, job, seed in CELLS:
        run = RUNS / folder
        fit_records = {f: json.loads((run / f).read_text()) for f in fits if (run / f).exists()}
        scores = {}
        for pattern in score_globs:
            for p in sorted(glob.glob(str(run / pattern))):
                if p.endswith(('fit.json', 'config.json', 'final.json', 'selection.json')) or p in scores:
                    continue
                try:
                    scores[p] = json.loads(Path(p).read_text())
                except Exception:
                    pass
        fit_done = len(fit_records) == len(fits)
        state = 'done' if fit_done and (scores or not score_globs) else 'scoring' if fit_done else (
            'running' if job in jobs else 'not submitted')
        fit_seconds = sum(r['fit_seconds'] for r in fit_records.values()) if fit_done else None  # FRM: two timed stages
        rec = dict(method=method, task=task, seed=seed, state=state, fit_seconds=fit_seconds,
                   cold=fit_done and not any(r.get('resumed') for r in fit_records.values()),
                   gpu=sorted({r.get('gpu') for r in fit_records.values()}), fit_records=fit_records,
                   scores={str(Path(p).relative_to(HERE)): s for p, s in scores.items()},
                   source_sha256=SOURCE,
                   slurm=jobs.get(job), collected_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        slug = f"{method.lower()}-{task.lower().replace('-', '')}-s{seed}"
        if state != 'not submitted':
            (RECORDS / f'{slug}.json').write_text(json.dumps(rec, indent=1, default=str) + '\n')
        acc = '; '.join(a for p, s in scores.items() if (a := accuracy(method, p, s))) or '-'
        fit = f'{fit_seconds:,.0f} s ({fit_seconds / 3600:.2f} h)' if fit_seconds is not None else '-'
        flags = [] if rec['cold'] or not fit_done else ['RESUMED, not cold']
        slurm = ' '.join(jobs[job][:2]) if job in jobs else ''
        rows.append(f'| {method} | {task} | {seed} | {state} | {fit} | {acc} | {slurm} | {", ".join(flags)} |')
    table = ['# Timed cold fits (one seed, one H200)', '',
             f'Collected {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())} by baselines/collect.py. Protocol: '
             'baselines/clock.py.', '',
             '| Method | Task | Seed | State | Fit time | Test solved | Slurm | Flags |', '|---|---|---|---|---|---|---|---|', *rows]
    (RECORDS / 'table.md').write_text('\n'.join(table) + '\n')
    export_cells = []
    for f in sorted(RECORDS.glob('*.json')):
        r = json.loads(f.read_text())
        export_cells.append({k: r.get(k, 0) if k == 'seed' else r[k] for k in ('method', 'task', 'seed', 'state', 'fit_seconds', 'cold', 'gpu')}
                            | dict(record=f'baselines/records/{f.name}',
                                   test_scores={p: accuracy(r['method'], p, s) for p, s in r['scores'].items()},
                                   numerics=sorted({x.get('numerics') for x in r['fit_records'].values()} - {None}),
                                   gpus=max((x.get('gpus', 1) for x in r['fit_records'].values()), default=1)))  # 1 for every reported fit; >1 only for a data-parallel fit
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=HERE, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=HERE, capture_output=True,
                                text=True).stdout.strip())
    out = dict(description='Cold complete fits of the baselines, one seed, one H200 (baselines/clock.py).',
               writer='baselines/collect.py', code_commit=commit, code_dirty=dirty, cells=export_cells)
    export = HERE.parent / 'results' / 'baseline-fit-times.json'
    text = json.dumps(out, indent=1) + '\n'
    if not export.exists() or json.loads(export.read_text()) != out:
        export.write_text(text)
    return all(r.split('|')[4].strip() == 'done' for r in rows)


if __name__ == '__main__':
    print('all done' if collect() else 'not all done')
