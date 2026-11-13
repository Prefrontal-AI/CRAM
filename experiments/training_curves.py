"""Training-method study: the probe curves of the figure (solve rate on held-out puzzles against training time) on
20 seeds, and the exports of the curves behind the table's time to 99%. Run from experiments/.

  python training_curves.py maze-levels                 # GPU: score the dense maze checkpoints again, per maze
  python training_curves.py splits                      # CPU: check the Sudoku training splits train-s0..s19
  python training_curves.py plan                        # the job lines of slurm/training_methods/many.sbatch
  python training_curves.py train --task sudoku --method cd --seed 3   # GPU, alone on it: train, save checkpoints
  python training_curves.py probe --runs runs/ALT/many/sudoku-cd-s3    # GPU, may share: score the saved checkpoints
  python training_curves.py check-probe                 # GPU: the later scoring reproduces the dense runs' in-loop probes
  python training_curves.py export                      # ../results/alt-training/alt-many.json
  python training_curves.py plan --long                 # job lines of the long runs (<root>/long/)
  python training_curves.py export-table                # ../results/alt-training/alt-t99.json (no GPU)
  python training_curves.py export-long                 # ../results/alt-training/alt-long.json (no GPU)

Terms. A probe is the solve rate of the current model on a fixed held-out set (training_methods.probe). A dense run
is a run of slurm/training_methods/dense.sbatch: the selected setting of a method, probed after every update up to the
64th and 8 times per doubling after that. <root> is training_methods.root(task): runs/ALT for Sudoku and, for the maze,
the folder that the environment variables ALT_MAZE_KERNEL and ALT_MAZE_OPT select (see training_methods.py). The folder
names are only labels. The tracked alt-dense.json and alt-many.json have the maze root runs/ALT-free
(ALT_MAZE_KERNEL=free, ALT_MAZE_OPT unset); alt-t99.json has the maze root runs/ALT-sgd (the setting of the table).

maze-levels loads every checkpoint that a dense maze run saved at a probe (<root>/dense/maze-<method>-s<seed>/
model-u<k>.pt), scores it with the probe's search (maze_unique.score, focused Gibbs at temperature 0, seed 0) on the
100 development mazes and on the 900 training mazes, and writes <root>/curves/maze-levels.json: per run and update, the
indices of the solved mazes of each split.

`train` trains the figure's five methods (FIG) on seeds 3-19 with the dense runs' settings (<root>/plan-dense-*, seed
and output changed), so that the figure can show the mean over 20 runs (seeds 0-2 are the dense runs). It runs
training_methods.train unchanged except that the probe is deferred: the checkpoint is still saved at every probe point
(every update up to the 64th, then 8 per doubling), outside the fit clock, and nothing else runs between updates, so
the run's own log gives an undisturbed fit clock when the run is alone on its GPU. `probe` then scores each saved
checkpoint with training_methods.probe and writes probes.json in the run folder. Sudoku seed s trains on train-s<s>,
the balanced 1,000-puzzle subset of FRM's loader with subsample seed s (scripts/prepare_data.py sudoku writes the
splits); `splits` checks that no training puzzle is a probe (dev10k) or test (full-all) puzzle.

`train --long` runs two of the figure's settings longer, on seeds 0-19 (LONG; <root>/long/): Sudoku full BPTT
(backpropagation through time) with a cosine schedule of 60,000 updates instead of 2,000 (about 250 s instead of 8 s),
so that the figure shows whether it trains given more time, and Maze-Unique pseudolikelihood with 33,000 updates at a
constant learning rate (the default Adam setting; about 245 s; its first 100 updates are those of the 100-update
setting), so that the figure shows whether the runs that stop below 99% recover. `export` takes these two methods from
<root>/long/.
"""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import tyro

import training_methods as A  # reads ALT_MAZE_KERNEL and ALT_MAZE_OPT at import: they select the maze run folder

HERE = Path(__file__).resolve().parent
MANY = HERE / 'runs/ALT/many'
from cram.paths import DATA  # noqa: E402

SPLIT = DATA / 'frm/split'  # the Sudoku training splits of seeds 0-19
FIG = ('pl', 'mpf', 'cd', 'bptt1', 'bptt')  # the figure's methods
SEEDS = tuple(range(20))  # seeds 0-2: the dense runs; 3-19: the runs of `train`
CAP_S = 310.0  # fit-time cap of the runs of `train`: the figure's x axis ends 300 s after the start-up
LONG = {('sudoku', 'bptt'): 60000, ('maze', 'pl'): 33000}  # updates of the long runs (about 250 s each)


def maze_levels(splits: tuple[str, ...] = ('dev100', 'train900')):
    """Score every saved checkpoint of the dense maze runs on each split and write <root>/curves/maze-levels.json:
    {run name: {update: {split: [indices of the solved mazes]}}}."""
    import maze_unique as U
    from cram import cram_maze as CM
    out = {}
    for run in sorted((A.root('maze') / 'dense').glob('maze-*-s[0-9]')):
        ks = sorted(int(re.fullmatch(r'model-u(\d+)\.pt', p.name).group(1)) for p in run.glob('model-u*.pt'))
        free = json.loads((run / 'config.json').read_text()).get('maze_kernel', 'separable') == 'free'
        rows = {}
        for k in ks:
            m = CM.CRAMMazeFreeEnergy(window=1) if free else CM.CRAMMazeEnergy(R=1, de=2, window=1)
            m.load_state_dict(torch.load(run / f'model-u{k}.pt', map_location='cpu'))
            rows[k] = {}
            for split in splits:
                r = U.score(m, split, U.Search('gibbs'))
                rows[k][split] = [i for i, ok in enumerate(r['per_maze']['right']) if ok]
        out[run.name] = rows
        print(run.name, len(ks), 'checkpoints', flush=True)
    out_dir = A.root('maze') / 'curves'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'maze-levels.json').write_text(json.dumps(out) + '\n')


# ------------------------------------------------------------------------------------------- the 20-seed runs
def sha256(path):
    """Hex SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def puzzles(path, rows=None):
    """The set of puzzle identities (clue digits, zero at blanks) of an npz split, optionally a row range."""
    with np.load(path) as f:
        sol, clue = f['solutions'], f['clue_mask']
    if rows is not None:
        sol, clue = sol[rows[0]:rows[1]], clue[rows[0]:rows[1]]
    return {r.tobytes() for r in (sol * clue).astype(np.uint8)}


def splits(n: int = 20):
    """Check that train-s0..s<n-1> (scripts/prepare_data.py sudoku) hold 1,000 distinct puzzles each and share none with
    the development and test puzzles. The splits are FRM's loader output (balanced by rating, subsample seed s)."""
    held = puzzles(DATA / 'frm/dev10k/dev.npz') | puzzles(DATA / 'full-test/full-all.npz') | puzzles(DATA / 'frm/split/test.npz')
    for s in range(n):
        ids = puzzles(SPLIT / f'train-s{s}.npz')
        assert len(ids) == 1000, f'seed {s}: duplicate puzzles'
        assert not ids & held, f'seed {s}: shares puzzles with dev10k or the test set'
        print(f'train-s{s}.npz', sha256(SPLIT / f'train-s{s}.npz')[:12], flush=True)


def recipe(task, method):
    """training_methods.Run fields (dict) of the method's dense-run setting: the first line (seed 0) of
    <root>/plan-dense-<task>-<method>.txt, a `train --flag value ...` command line."""
    line = (A.root(task) / f'plan-dense-{task}-{method}.txt').read_text().splitlines()[0].split()
    assert line[0] == 'train'
    kv = dict(zip(line[1::2], line[2::2]))
    out = {k[2:].replace('-', '_'): v for k, v in kv.items()}
    conv = {'lr': float, 'knob': float, 'l1': float, 'reg': float, 'cap_s': float, 'seed': int, 'probe_per_octave': int, 'probe_every_until': int}
    return {k: conv.get(k, str)(v) for k, v in out.items()}


def run_dir(task, method, seed, long=False):
    """The run folder (str, relative to experiments/) of one seed: <root>/many/ or, for a long run, <root>/long/."""
    return f'{A.REL[task]}/{"long" if long else "many"}/{task}-{method}-s{seed}'


def train(task: str, method: str, seed: int, long: bool = False):
    """One run of the 20-seed curves: the dense run's setting on this seed with the probe deferred (checkpoints only),
    fit cap CAP_S. With `long`, the schedule has LONG[(task, method)] updates and the run goes to <root>/long/."""
    os.environ['FRM_DATA'] = str(SPLIT)
    A.probe = lambda *a, **k: {'solved': 0, 'n': 1, 'deferred': True}  # training_methods.train saves model-u<k>.pt before this call
    cfg = recipe(task, method) | {'seed': seed, 'out': run_dir(task, method, seed, long), 'cap_s': CAP_S}
    if long:
        cfg['updates'] = LONG[(task, method)]
    A.train(A.Run(**cfg))


def probe(runs: tuple[str, ...], updates: tuple[int, ...] = (), out: str = ''):
    """Score the saved checkpoints of each run with training_methods.probe. runs: run folders relative to experiments/.
    updates: score only these updates (empty: all). Results go to <run>/probes.json ({update: {solved, n}}), or with
    `out` to <out>/<run name>.json (the check of the dense runs, whose folders this does not write to). Resumes."""
    os.environ['FRM_DATA'] = str(SPLIT)
    dev = torch.device('cuda')
    ratio = None
    for run in runs:
        rd = HERE / run
        cfg = json.loads((rd / 'config.json').read_text())
        dest = (HERE / out / f'{rd.name}.json') if out else rd / 'probes.json'
        dest.parent.mkdir(parents=True, exist_ok=True)
        done = json.loads(dest.read_text()) if dest.exists() else {}
        ks = sorted(int(re.fullmatch(r'model-u(\d+)\.pt', p.name).group(1)) for p in rd.glob('model-u*.pt'))
        ks = [k for k in ks if (not updates or k in updates) and str(k) not in done]
        if not ks:
            continue
        if cfg['task'] == 'sudoku' and ratio is None:
            ratio = A.hbar_ratio()[0]
        task = A.TASKS[cfg['task']](cfg['seed'], dev, kernel=cfg.get('maze_kernel', 'separable'))
        task.setup()
        for k in ks:
            task.load({n: v.to(dev) for n, v in torch.load(rd / f'model-u{k}.pt', map_location='cpu').items()})
            r = A.probe(cfg['task'], task, cfg['seed'], ratio)
            done[str(k)] = {'solved': r['solved'], 'n': r['n']}
            A.dump(done, dest)
        print(run, len(done), 'probes', flush=True)


def check_probe(updates: tuple[int, ...] = (1, 8, 12, 14, 64, 76, 83, 256, 1024, 2000)):
    """Rescore some checkpoints of the dense seed-0 runs and compare with their in-loop probes (log.jsonl)."""
    runs = [f'{A.REL[t]}/dense/{t}-{m}-s0' for t in ('sudoku', 'maze') for m in FIG]
    probe(tuple(runs), updates, out='runs/ALT/many/check')
    bad = 0
    for run in runs:
        logged = {r['update']: r['probe']['solved'] for r in map(json.loads, (HERE / run / 'log.jsonl').read_text().splitlines()) if 'probe' in r}
        got = json.loads((MANY / 'check' / f'{Path(run).name}.json').read_text())
        for k, v in got.items():
            same = v['solved'] == logged[int(k)]
            bad += not same
            print(run, k, v['solved'], logged[int(k)], '' if same else 'DIFFERENT')
    print('rescored probes that differ from the in-loop probe:', bad)


def plan(gpus: int = 8, long: bool = False, tasks: tuple[str, ...] = ('sudoku', 'maze')):
    """Job lines for the runs of `train` on seeds 3-19: <root>/many/plan-train-g<i>.txt, one list per GPU (run one
    after another, each alone on its GPU), longest runs first and balanced by the fit times of the seed-0 final runs;
    and plan-probe.txt (the runs). With `long`: the LONG runs on seeds 0-19, into <root>/long/.
    gpus: the number of lists. tasks: tasks with the same root (plan the maze alone when its root differs)."""
    cost = {}
    roots = {A.REL[t] for t in tasks}
    assert len(roots) == 1, f'one plan directory per root: {roots}'  # plan the maze alone when its root differs
    if long:
        cost = {k: CAP_S + 30.0 for k in LONG if k[0] in tasks}
        seeds = SEEDS
    else:
        seeds = SEEDS[3:]
        for t in tasks:
            for m in FIG:
                src = 'retime' if (A.root(t) / f'retime/{t}-{m}-s0').exists() else 'final'
                fin = json.loads((A.root(t) / f'{src}/{t}-{m}-s0/final.json').read_text())
                cost[(t, m)] = min(fin['fit_s'], CAP_S) + 30.0  # plus about 30 s of process start, data and checkpoints
    jobs = sorted(((cost[(t, m)], t, m, s) for (t, m) in cost for s in seeds), reverse=True)
    todo = [j for j in jobs if not (HERE / run_dir(*j[1:], long) / 'final.json').exists()]  # finished runs are skipped
    queues = [[0.0, []] for _ in range(gpus)]
    for c, t, m, s in todo:
        q = min(queues, key=lambda q: q[0])
        q[0] += c
        q[1].append(f'train --task {t} --method {m} --seed {s}' + (' --long' if long else ''))
    d = A.root(tasks[0]) / ('long' if long else 'many')
    d.mkdir(parents=True, exist_ok=True)
    for i, (c, lines) in enumerate(queues):
        (d / f'plan-train-g{i}.txt').write_text('\n'.join(lines) + '\n')
        print(f'GPU {i}: {len(lines)} runs, about {c / 60:.0f} min')
    (d / 'plan-probe.txt').write_text('\n'.join(run_dir(t, m, s, long) for _, t, m, s in jobs) + '\n')


def ready_s(task, method, seed):
    """Seconds to make the fitted energy inference-ready: on Sudoku the stopping threshold, the focus statistic and the
    search unit, fitted from the training solutions (training_methods.sudoku_ready, timed once per final run in a fresh
    process by `training_methods.py test`); 0 on the maze, whose search stops at a fixed point and fits nothing."""
    if task != 'sudoku':
        return 0.0
    return json.loads((A.root(task) / 'final' / f'{task}-{method}-s{seed}' / 'test.json').read_text())['ready_s']


def export(out: str = '../results/alt-training/alt-many.json'):
    """Probe curves of the figure's 20 runs per method: seeds 0-2 from alt-dense.json (the dense runs on the clock of
    the final runs or of their verified repeats, training_methods_export.clock), seeds 3-19 from <root>/many
    (probes.json on the run's own clock). Rows are [fit_s, probe solved %, update]; runs not yet scored are listed
    under "missing". out: output path relative to experiments/."""
    dense = json.loads((HERE.parent / 'results/alt-training/alt-dense.json').read_text())['tasks']
    tasks, missing = {}, []
    for t in ('sudoku', 'maze'):
        for m in FIG:
            runs = {}
            long = (t, m) in LONG
            for s in SEEDS:
                if s < 3 and not long:
                    sd = dense[t][m][str(s)]
                    runs[str(s)] = {'curve': sd['curve'], 'clock_from': sd['clock_from'], 'probe_from': f'{A.REL[t]}/dense/{t}-{m}-s{s}'}
                    continue
                rd = HERE / run_dir(t, m, s, long)
                if not (rd / 'final.json').exists() or not (rd / 'probes.json').exists():
                    missing.append(rd.name)
                    continue
                pr = json.loads((rd / 'probes.json').read_text())
                fit = {r['update']: r['fit_s'] for r in map(json.loads, (rd / 'log.jsonl').read_text().splitlines()) if 'update' in r}
                pts = sorted(int(k) for k in pr)
                saved = sorted(int(p.name[7:-3]) for p in rd.glob('model-u*.pt'))
                if saved != pts:  # a checkpoint not yet scored
                    missing.append(rd.name)
                    continue
                runs[str(s)] = {'curve': [[fit[k], 100.0 * pr[str(k)]['solved'] / pr[str(k)]['n'], k] for k in pts],
                                'clock_from': run_dir(t, m, s, long), 'probe_from': run_dir(t, m, s, long) + '/probes.json'}
            tasks.setdefault(t, {})[m] = runs
    obj = {'note': 'The figure methods on seeds 0-19 (0-2: the dense runs, on the clock of the final run or of its verified '
                   'repeat; 3-19: the same settings, each run alone on its GPU, checkpoints scored afterwards with the '
                   'same probe). Rows are [fit_s, probe solved %, update]; fit_s is the run\'s fit clock after that update. '
                   'sudoku bptt and maze pl are the long runs of the long/ folder on seeds 0-19 (updates in "long").',
           'ready_s': {t: {m: [ready_s(t, m, s) for s in range(3)] for m in tasks[t]} for t in tasks},
           'ready_note': 'ready_s: seconds to fit the stopping threshold, focus statistic and search unit from the training '
                         'solutions (Sudoku; 0 on the maze), measured on the final runs of seeds 0-2. Training time to an '
                         'inference-ready model = fit_s - start-up + ready_s.',
           'long': {f'{t}/{m}': u for (t, m), u in LONG.items()}, 'tasks': tasks, 'missing': missing,
           'maze_kernel': A.MAZE_KERNEL, 'roots': A.REL}
    path = HERE / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + '\n')
    print(path, 'missing:', len(missing))


def export_table(out: str = '../results/alt-training/alt-t99.json', later: int = 20):
    """Per final run of the training-method table (every method, seeds 0-2), the probe curve and the one-time start-up
    on the clock the table uses (the verified repeat in <root>/retime/ where it exists, else the final run), so that
    the paper can compute the time to 99% as the figure does. Start-up: fit time after update 1 minus the median time
    of updates 2..later+1 (as in the figure). Rows of "curve" are [update, fit_s, probe solved %].
    out: output path relative to experiments/. later: the number of updates behind the start-up estimate."""
    tasks = {}
    for t in ('sudoku', 'maze'):
        for d in sorted((A.root(t) / 'final').glob(f'{t}-*-s[0-2]')):
            m, s = re.fullmatch(rf'{t}-(\w+)-s(\d)', d.name).groups()
            src = A.root(t) / 'retime' / d.name
            src = src if (d / 'retime.json').exists() and (src / 'log.jsonl').exists() else d
            recs = [r for r in map(json.loads, (src / 'log.jsonl').read_text().splitlines()) if 'update' in r]
            fit = {r['update']: r['fit_s'] for r in recs}
            n = min(later, max(fit) - 1)
            start_up = fit[1] - float(np.median([fit[k + 1] - fit[k] for k in range(1, n + 1)]))
            curve = [[r['update'], r['fit_s'], 100.0 * r['probe']['solved'] / r['probe']['n']] for r in recs if 'probe' in r]
            tasks.setdefault(t, {}).setdefault(m, {})[s] = {'start_up_s': start_up, 'updates_for_start_up': n, 'ready_s': ready_s(t, m, int(s)),
                                                             'clock_from': str(src.relative_to(HERE)), 'curve': curve}
    obj = {'note': 'Probe curves and one-time start-up of the final runs of the training-method table (seeds 0-2), on the '
                   'table\'s clock (the verified repeats in retime/ where they exist). Rows are [update, fit_s, probe solved %].', 'tasks': tasks,
           'maze_kernel': A.MAZE_KERNEL, 'roots': A.REL}
    path = HERE / out
    path.write_text(json.dumps(obj, indent=1) + '\n')
    print(path, {t: len(v) for t, v in tasks.items()})


LONG_TABLE = [  # (method, run prefix without -s<seed>, label): the Sudoku class D arms run with longer schedules.
    # The label is written to the field "amendment" of alt-long.json; it identifies the job script that trained the
    # runs (13: long_d.sbatch, 15: long_d2.sbatch, 16: long_d3.sbatch in slurm/training_methods/).
    ('bptt5', 'runs/ALT/longD/sudoku-bptt5-lr6', 13),
    ('eqp', 'runs/ALT/longD2/sudoku-eqp-lr6', 15),
    ('bptt', 'runs/ALT/longD2/sudoku-bptt-lr2', 15),
    ('bptt', 'runs/ALT/longD3/sudoku-bptt-lr2', 16),
]


def export_long(out: str = '../results/alt-training/alt-long.json', later: int = 20):
    """The tested long-schedule runs of the Sudoku class D arms (LONG_TABLE, seeds 0-2), with the fields that the
    training-method table takes from a final run: the probe curve and one-time start-up on the run's own clock (as
    export_table), the schedule, the learning rate and the test record. Rows of "curve" are [update, fit_s, probe
    solved %]. out: output path relative to experiments/. later: as export_table."""
    arms = []
    for m, prefix, amendment in LONG_TABLE:
        seeds = {}
        for s in range(3):
            d = HERE / f'{prefix}-s{s}'
            cfg, fin, test = (json.loads((d / f).read_text()) for f in ('config.json', 'final.json', 'test.json'))
            fit = {}
            with open(d / 'log.jsonl') as f:  # only the first updates: a long run logs millions of lines
                for line in f:
                    r = json.loads(line)
                    if 'update' in r:
                        fit[r['update']] = r['fit_s']
                    if len(fit) > later + 1:
                        break
            start_up = fit[1] - float(np.median([fit[k + 1] - fit[k] for k in range(1, later + 1)]))
            seeds[str(s)] = {'start_up_s': start_up, 'ready_s': test['ready_s'], 'curve': [[r['update'], r['fit_s'], 100.0 * r['solved'] / r['n']] for r in fin['curve']],
                             'fit_s': fin['fit_s'], 'updates_done': fin['updates'], 'selected_update': test['selected_update'],
                             'test': {k: test['test'][k] for k in ('n', 'solved', 'wrong_stops', 'budget_failures', 'mean_flop')}}
            assert (cfg['method'], cfg['seed'], fin['updates']) == (m, s, cfg['updates_resolved']), d
        arms.append({'method': m, 'runs': prefix, 'amendment': amendment, 'updates': cfg['updates_resolved'], 'lr': cfg['lr'],
                     'knob': cfg['knob_resolved'], 'l1': cfg['l1'], 'seeds': seeds})
    obj = {'note': 'The tested long-schedule runs of the Sudoku class D arms (seeds 0-2); '
                   'each run ended at its schedule. Rows of "curve" are [update, fit_s, probe solved %].', 'arms': arms}
    path = HERE / out
    path.write_text(json.dumps(obj, indent=1) + '\n')
    print(path, [(a['method'], a['updates']) for a in arms])


if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict({'maze-levels': maze_levels, 'splits': splits, 'train': train, 'probe': probe,
                                          'check-probe': check_probe, 'plan': plan, 'export': export, 'export-table': export_table,
                                          'export-long': export_long})
