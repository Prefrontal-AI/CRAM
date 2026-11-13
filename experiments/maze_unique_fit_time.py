"""The fit time of the Maze-Unique row of Table 2 under the paper's fit-time protocol.

  python maze_unique_fit_time.py run --row maze-unique__cram-sgd-fg --seed 0 --device cuda   # one fresh process: cold + 7 warm
  python maze_unique_fit_time.py run --row maze-unique__cram-sgd-fg --seed 0 --device cpu    # CPU-only job, one thread
  python maze_unique_fit_time.py summary --rows maze-unique__cram-sgd-fg --write-records     # adds fit_* to the table record

Each `run` is one fresh Python process. It reads the row's training data from disk into host memory (not timed),
creates the CUDA context (timed, reported separately), then times the whole fit `1 + repeats` times: the first is the
cold fit, the others are the warm repeats (the warm time is their median). A fit is everything from the raw host arrays
to the inference-ready model (TIMED_OPS); it calls `maze_unique.train` with the config.json of the row's saved run.
Every timed fit is compared with the saved model of that run, so the run must exist first (`maze_unique.py train`).
The Sudoku-Extreme fit time comes from `sudoku.py time`, and the Maze-Hard one from `maze_hard_latent.py fit`.

Dimension names: b mazes, k updates.
"""
import time

T_IMPORT0 = time.perf_counter()  # process start, to report the (untimed) import time

import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import tyro  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs' / 'CRAM-TIME'

# row (the name of its table record): the pipeline kind and the run folder of the saved model of each seed
ROWS = {
    'maze-unique__cram-sgd-fg': dict(kind='maze_unique', run='runs/CU13/final/fg-s{seed}'),
}

DEFINITION = (
    'Fit time: synchronized wall clock from the training data in host memory (as the row\'s loader reads them from '
    'disk) to the inference-ready model and its inference statistics; includes format conversion, device transfer, '
    'initialization, augmentation, every forward, backward and optimizer step, compilation and CUDA-graph capture with '
    'their warm-up steps, post-fit compression, stopping thresholds and focus statistics fitted from training '
    'solutions, support estimation and latent caching; excludes reading files, evaluation and saving. Cold: the first '
    'fit in a fresh process (CUDA context creation reported separately); warm: median of 7 further full fits in the same '
    'process. GPU: one H200. CPU: one thread (torch.set_num_threads(1)) in a CPU-only job.')


def sha256(path):
    """SHA-256 of a file, as a hexadecimal string."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(obj, path):
    """Write `obj` as JSON through a temporary file, so that an interrupted write leaves no partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1) + '\n')
    os.replace(tmp, path)


class Clock:
    """Synchronized laps: `lap(name)` adds the time since the previous lap to parts[name]. On a GPU every reading
    waits for the queued GPU work first (torch.cuda.synchronize). `total()` is the time from construction to the
    last lap."""

    def __init__(self, dev):
        self.sync = torch.cuda.synchronize if dev.type == 'cuda' else (lambda: None)
        self.sync()
        self.t0 = self.last = time.perf_counter()
        self.parts = {}

    def lap(self, name):
        self.sync()
        t = time.perf_counter()
        self.parts[name] = self.parts.get(name, 0.0) + t - self.last
        self.last = t

    def total(self):
        return self.last - self.t0


@contextlib.contextmanager
def patched(obj, name, value):
    """Set obj.<name> to `value` inside the `with` block and restore it afterwards."""
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


def max_diff(a, b):
    """Largest absolute difference of two tensors of the same shape (0 for empty tensors)."""
    return float((a.double() - b.double()).abs().max()) if a.numel() else 0.0


# ---------------------------------------------------------------- Maze-Unique (maze_unique.py)
class MazeUnique:
    """The fit of one Maze-Unique run, split into `load` (reading from disk, not timed), `fit` (timed) and `compare`
    (the fitted weights against the saved model, not timed)."""
    TIMED_OPS = [
        'maze_unique.load on the raw .npy arrays (int64 casts, tokens, path targets, free mask)',
        'transfer of the 1,000 training mazes (tokens, targets, free mask) to the device',
        'torch.manual_seed; CRAMMazeFreeEnergy (zero start) construction and transfer; SGD construction',
        'batch-index draws (numpy, seeded) for every update and their transfer',
        'every update: pseudo-likelihood loss, backward, the cosine learning rate, SGD step with momentum (with l1 > 0, the soft-threshold)',
        'maze_unique.train bookkeeping: per-update losses copied to the host, count of entries with a gradient',
        'no post-fit compression and no inference statistics (the stopping test is a fixed point; start and step '
        'size are constants)',
    ]

    def __init__(self, spec, seed):
        import maze_unique as U
        from cram import maze_unique as M
        self.U, self.M, self.seed = U, M, seed
        self.run = HERE / spec['run'].format(seed=seed)
        self.cfg = U.Train(**json.loads((self.run / 'config.json').read_text()))
        assert self.cfg.split == 'train', self.cfg.split
        self.modules = [U, M]

    def load(self, device):
        """Read the raw training arrays and the saved model into host memory."""
        # the files maze_unique.load reads (with its sha256 check): this is the read from disk
        raw = {rel: self.M._read(rel) for rel in ('train/all__inputs.npy', 'train/all__labels.npy')}
        ref = torch.load(self.run / 'model.pt', map_location='cpu')
        return {'raw': raw, 'ref': ref}

    def fit(self, host, dev, clock):
        """One complete fit on device `dev`. Returns (state dict on the CPU, statistics of the fit)."""
        cfg = dataclasses.replace(self.cfg, device=dev.type)
        with patched(self.M, '_read', lambda rel: host['raw'][rel]):  # maze_unique.load gets the arrays from host memory
            model, info = self.U.train(cfg)
        clock.lap('fit')
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return state, {'update_loop_s': info['loop_s'], 'final_loss': info['loss'][-1]}

    def compare(self, state, stats, host, dev):
        """Compare the fitted weights with the saved model: bitwise, and within 1e-3 of the largest weight."""
        ref = host['ref']
        bitwise = all(torch.equal(state[k], ref[k]) for k in ref) and state.keys() == ref.keys()
        diff = max(max_diff(state[k], ref[k]) for k in ref)
        scale = max(float(v.abs().max()) for v in ref.values())
        return {'bitwise_equal': bool(bitwise), 'max_abs_diff': diff, 'max_abs_weight': scale,
                'same_model': bool(bitwise or diff <= 1e-3 * scale),
                'tolerance': 'bitwise, or max |difference| <= 1e-3 x the largest |weight| (18 continuous weights, no '
                             'support or levels)'}


KINDS = {'maze_unique': MazeUnique}


# ---------------------------------------------------------------- one process: cold + warm
@dataclasses.dataclass
class Run:
    row: str  # key of ROWS: maze-unique__cram-sgd-fg
    seed: int = 0  # training seed
    device: str = 'cuda'  # cuda | cpu (cpu: run in a CPU-only job; one thread)
    repeats: int = 7  # warm repeats after the cold fit
    tag: str = ''  # output file suffix
    force: bool = False  # overwrite an existing result


def cmd_run(a: Run):
    """One fresh process: the cold fit and `repeats` warm fits of one row, seed and device.
    Writes runs/CRAM-TIME/<row>/s<seed>-<device><tag>.json (the folder name CRAM-TIME is only a label)."""
    t_main = time.perf_counter()
    spec = ROWS[a.row]
    out = OUT / a.row / f's{a.seed}-{a.device}{a.tag}.json'
    if out.exists() and not a.force:
        print(out, 'exists; skip', flush=True)
        return
    dev = torch.device(a.device)
    if dev.type == 'cpu':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        assert os.environ.get('CUDA_VISIBLE_DEVICES', '') in ('', 'NoDevFiles') or torch.cuda.device_count() == 0, \
            'run the CPU timing in a CPU-only job'
    pipe = KINDS[spec['kind']]({**spec, 'row': a.row}, a.seed)
    src = {m.__name__: str(Path(m.__file__).resolve()) for m in pipe.modules}
    for name, path in src.items():  # the timed code must be the code of this repository, not another installed copy
        assert path.startswith(str(HERE.parent)), (name, path, HERE)
    t_load = time.perf_counter()
    host = pipe.load(a.device)
    load_s = time.perf_counter() - t_load
    ctx_s = None
    if dev.type == 'cuda':
        t0 = time.perf_counter()
        torch.cuda.init()
        torch.empty(1, device=dev)
        torch.cuda.synchronize()
        ctx_s = time.perf_counter() - t0
    fits = []
    for rep in range(1 + a.repeats):
        clock = Clock(dev)
        state, stats = pipe.fit(host, dev, clock)
        total = clock.total()
        check = pipe.compare(state, stats, host, dev)
        keep = {k: v for k, v in stats.items() if isinstance(v, (int, float, str, list, dict))}
        fits.append({'kind': 'cold' if rep == 0 else 'warm', 'seconds': total, 'parts': clock.parts, 'check': check, 'stats': keep})
        print(json.dumps({'rep': rep, 'seconds': round(total, 6), 'parts': {k: round(v, 6) for k, v in clock.parts.items()},
                          'same_model': check['same_model'], 'bitwise': check.get('bitwise_equal')}), flush=True)
        del state, stats
    warm = [f['seconds'] for f in fits[1:]]
    part_names = list(fits[0]['parts'])
    res = {
        'row': a.row, 'seed': a.seed, 'device': a.device, 'definition': DEFINITION, 'timed_ops': pipe.TIMED_OPS,
        'excluded': ['process start and imports', 'reading the training data from disk into host memory', 'evaluation', 'saving'],
        'cold_s': fits[0]['seconds'], 'warm_median_s': statistics.median(warm) if warm else None, 'warm_all_s': warm,
        'cuda_context_s': ctx_s, 'cold_parts_s': fits[0]['parts'],
        'warm_parts_median_s': {k: statistics.median(f['parts'][k] for f in fits[1:]) for k in part_names} if warm else None,
        'all_same_model': all(f['check']['same_model'] for f in fits),
        'all_bitwise': all(f['check'].get('bitwise_equal', False) for f in fits),
        'fits': fits, 'load_s_untimed': load_s, 'imports_s_untimed': t_main - T_IMPORT0,
        'code_root': str(HERE.parent), 'modules': src,
        'harness_sha256': sha256(__file__), 'threads': torch.get_num_threads(),
        'interop_threads': torch.get_num_interop_threads(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name() if dev.type == 'cuda' else None, 'node': os.uname().nodename,
        'slurm_job': os.environ.get('SLURM_JOB_ID'), 'visible_gpus': torch.cuda.device_count(), 'cpus_allowed': len(os.sched_getaffinity(0)),
        'written_utc': time.strftime('%Y-%m-%d %H:%M', time.gmtime()),
    }
    save(res, out)
    print(json.dumps({k: res[k] for k in ('row', 'seed', 'device', 'cold_s', 'warm_median_s', 'cuda_context_s',
                                          'all_same_model', 'all_bitwise')}), flush=True)


# ---------------------------------------------------------------- summary and table records
@dataclasses.dataclass
class Summary:
    rows: tuple[str, ...] = ()  # rows to summarize (default: every row with results)
    write_records: bool = False  # add the fit_* fields to runs/table/<row>.json (never changes an existing other field)


NEW_FIELDS = ('fit_gpu_cold_s', 'fit_gpu_warm_s', 'fit_cpu_cold_s', 'fit_cpu_warm_s', 'fit_cuda_context_s', 'fit_timing')


def cmd_summary(a: Summary):
    """Collect the timing records of each row (3 seeds x 2 devices) into runs/CRAM-TIME/summary.json and print a table.
    With --write-records, add the fit_* fields (means over the seeds) to the row's table record; this needs all six
    records, each with every timed fit equal to the saved model."""
    table = HERE / 'runs' / 'table'
    out, lines = {}, []
    for row in a.rows or ROWS:
        res = {(d, s): json.loads(p.read_text()) for d in ('cuda', 'cpu') for s in range(3)
               if (p := OUT / row / f's{s}-{d}.json').exists()}
        if not res:
            continue
        rec = json.loads((table / f'{row}.json').read_text()) if (table / f'{row}.json').exists() else {}  # {}: the table record is not written yet
        mean = lambda d, key: statistics.fmean(r[key] for (dd, _), r in res.items() if dd == d) \
            if any(dd == d for dd, _ in res) else None
        per = lambda d, key: [res[(d, s)][key] if (d, s) in res else None for s in range(3)]
        o = {'gpu_cold_s': mean('cuda', 'cold_s'), 'gpu_warm_s': mean('cuda', 'warm_median_s'),
             'cpu_cold_s': mean('cpu', 'cold_s'), 'cpu_warm_s': mean('cpu', 'warm_median_s'),
             'cuda_context_s': mean('cuda', 'cuda_context_s'),
             'per_seed': {'gpu_cold_s': per('cuda', 'cold_s'), 'gpu_warm_s': per('cuda', 'warm_median_s'),
                          'cpu_cold_s': per('cpu', 'cold_s'), 'cpu_warm_s': per('cpu', 'warm_median_s'),
                          'cuda_context_s': per('cuda', 'cuda_context_s')},
             'parts_gpu_cold_s': {k: statistics.fmean(r['cold_parts_s'][k] for (d, _), r in res.items() if d == 'cuda')
                                  for k in next((r['cold_parts_s'] for (d, _), r in res.items() if d == 'cuda'), {})},
             'parts_gpu_warm_s': {k: statistics.fmean(r['warm_parts_median_s'][k] for (d, _), r in res.items() if d == 'cuda')
                                  for k in next((r['warm_parts_median_s'] for (d, _), r in res.items() if d == 'cuda'), {})},
             'parts_cpu_cold_s': {k: statistics.fmean(r['cold_parts_s'][k] for (d, _), r in res.items() if d == 'cpu')
                                  for k in next((r['cold_parts_s'] for (d, _), r in res.items() if d == 'cpu'), {})},
             'parts_cpu_warm_s': {k: statistics.fmean(r['warm_parts_median_s'][k] for (d, _), r in res.items() if d == 'cpu')
                                  for k in next((r['warm_parts_median_s'] for (d, _), r in res.items() if d == 'cpu'), {})},
             'same_model': {f'{d}-s{s}': r['all_same_model'] for (d, s), r in res.items()},
             'bitwise': {f'{d}-s{s}': r['all_bitwise'] for (d, s), r in res.items()},
             'old_train_gpu_s': rec.get('train_gpu_s'), 'old_train_cpu_s': rec.get('train_cpu_s'),
             'timed_ops': next(iter(res.values()))['timed_ops'], 'definition': DEFINITION,
             'complete': len(res) == 6 and all(r['all_same_model'] for r in res.values()),
             'sources': [f'runs/CRAM-TIME/{row}/s{s}-{d}.json' for d, s in sorted(res)]}
        out[row] = o
        f = lambda v: '-' if v is None else (f'{v * 1e3:.2f} ms' if v < 1 else f'{v:.3f} s')
        lines.append(f"| {row} | {f(o['gpu_cold_s'])} | {f(o['gpu_warm_s'])} | {f(o['cpu_cold_s'])} | {f(o['cpu_warm_s'])} | "
                     f"{f(o['cuda_context_s'])} | {f(o['old_train_gpu_s'])} / {f(o['old_train_cpu_s'])} |")
        if a.write_records:
            if not o['complete']:
                print(row, 'incomplete or not the frozen model: record not written', flush=True)
                continue
            path = table / f'{row}.json'
            text = path.read_text()
            rec = json.loads(text)
            old = {k: v for k, v in rec.items() if k not in NEW_FIELDS}
            new = {'fit_gpu_cold_s': o['gpu_cold_s'], 'fit_gpu_warm_s': o['gpu_warm_s'], 'fit_cpu_cold_s': o['cpu_cold_s'],
                   'fit_cpu_warm_s': o['cpu_warm_s'], 'fit_cuda_context_s': o['cuda_context_s'],
                   'fit_timing': {'definition': DEFINITION, 'mean_over': 'training seeds 0, 1, 2',
                                  'per_seed': o['per_seed'], 'parts_gpu_cold_s': o['parts_gpu_cold_s'],
                                  'parts_gpu_warm_s': o['parts_gpu_warm_s'], 'parts_cpu_cold_s': o['parts_cpu_cold_s'],
                                  'parts_cpu_warm_s': o['parts_cpu_warm_s'], 'timed_ops': o['timed_ops'],
                                  'same_model_check': o['same_model'],
                                  'sources': o['sources'], 'written_utc': time.strftime('%Y-%m-%d %H:%M', time.gmtime())}}
            rec.update(new)
            assert {k: v for k, v in rec.items() if k not in NEW_FIELDS} == old  # no other field changed
            tmp = path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(rec, indent=1) + '\n')
            os.replace(tmp, path)
            print('wrote', path.name, flush=True)
    old = json.loads((OUT / 'summary.json').read_text()) if (OUT / 'summary.json').exists() else {}
    save(old | out, OUT / 'summary.json')  # --rows updates those rows and keeps the others
    print('| row | GPU cold | GPU warm (median of 7) | CPU cold | CPU warm (median of 7) | CUDA context | old GPU / CPU |')
    print('| --- | --- | --- | --- | --- | --- | --- |')
    print('\n'.join(lines))


if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict({'run': cmd_run, 'summary': cmd_summary}, config=(tyro.conf.OmitArgPrefixes,))
