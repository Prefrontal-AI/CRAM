"""CRAM on Maze-Unique: a local pair energy on the 30 x 30 grid, fitted by pseudolikelihood (model: cram/cram_maze.py).

The Table 2 row (record runs/table/maze-unique__cram-sgd-fg.json), for the training seeds 0, 1 and 2:

  M="--kernel free --free-init 0 --opt sgd --momentum 0.5 --cosine --lr 1 --updates 20 --l1 0"
  python maze_unique.py train $M --split train900 --seed 0 --out runs/<name>/fg-s0     # development: fit on 900 mazes, score dev100
  python maze_unique.py train $M --split train --seed 0 --out runs/CU13/final/fg-s0 --score   # final fit on all 1,000 training mazes
  python maze_unique.py score --run runs/CU13/final/fg-s0 --split test --search gibbs  # the 1,000 test mazes, once
  python maze_unique.py time $M --split train --seed 0 --out runs/CU13/final/fg-s0     # time of the update loop on the GPU (gpu.json)
  python maze_unique.py cpu $M --split train --seed 0 --out runs/CU13/final/fg-s0      # the same on one CPU thread (cpu.json)
  python maze_unique.py rows --rows cram-sgd-fg                                         # the table record

The folder name CU13 is only a label. `maze_unique_fit_time.py` measures the complete fit time that the paper reports.

Energy (cram_maze.CRAMMazeFreeEnergy, `--kernel free`). Tokens x_i: wall, open, S (start), G (goal). Each cell has a
two-state answer s_i = (off, on); on means that the cell is on the path. Walls are clamped off, S and G are clamped on,
and the open cells are free. With one free 2 x 2 cost table T_d per offset d of the 3 x 3 window (4 offsets; the
opposite offset uses T_d^T) and a bias b (2,) at the open cells,
    E(s) = sum_i sum_d s_i^T T_d s_(i+d) - sum_(open i) b^T s_i.
The tables are the same at every cell: 4 x 4 + 2 = 18 parameters, all zero at the start.

Fit (`train`). The loss is the pseudolikelihood of the labeled answer of each free cell, given the labeled answers of
all other cells (cram.maze_unique.pl_loss):
    loss = -mean_(free cells i) log softmax_a( -dE/ds_i[a] )[y_i],
evaluated at the labeled board, so the fit runs no search. The paper's recipe: 20 updates of SGD with momentum 0.5 on
batches of 64 mazes, learning rate 1 with cosine decay, lr_k = 1/2 (1 + cos(pi k / 20)), no L1 step, no gradient clip.
There is no augmentation.

Search (`score`, cram.maze_unique.gibbs). Focused Gibbs at temperature 0 from the board with every open cell on. Each
move picks, uniformly at random, one free cell whose answer is not its lowest-energy answer and sets it to that answer.
A maze stops at a fixed point, when no such cell is left. The budget is 20,000 moves. `--search descend` runs damped
descent of the continuous answers instead; the development runs score both.

A run directory holds config.json, model.pt, train.json, log.jsonl and final.json (written last). A run whose final.json
exists is not trained again. Training takes under 10 s, so no rolling checkpoint is written: a run stopped before
final.json restarts from its seed, and a seeded run is deterministic.
Dimension names: b mazes, h and w grid rows and columns.
"""
import dataclasses
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo  # noqa: F401  (the first torch.optim construction imports it lazily, about 3 s; import it here with the other modules)
import tyro

from cram import cram_maze as C
from cram import maze_unique as M

HERE = Path(__file__).resolve().parent


@dataclasses.dataclass
class Train:
    """The settings of one fit. Each run folder stores them in config.json, and `load_run` and the checkpoint loader
    rebuild the model from that file, so the field names are part of the record format.
    The paper's recipe sets kernel free, opt sgd, momentum 0.5, cosine, lr 1, updates 20, l1 0 (module docstring)."""
    split: str = 'train900'  # train900 (fit on 900 mazes; the other 100 are dev100) | train (all 1,000 training mazes: final runs)
    seed: int = 0  # torch seed of the initialization and numpy seed of the batch order
    R: int = 1  # separable kernel: relations
    de: int = 2  # separable kernel: width of the state-kernel factors P, Q
    window: int = 1  # offsets within +-window (1: the 3x3 window)
    updates: int = 100  # optimizer updates
    batch: int = 64  # mazes per update
    lr: float = 0.03  # learning rate (adam: constant; sgd: the initial rate)
    clip: float = 1.0  # adam: gradient-norm clip
    legacy: bool = False  # train maze_unique.MazeEnergy (34 stored floats, 16 of them trained) instead of CRAMMazeEnergy
    kernel: str = 'separable'  # separable: CRAMMazeEnergy (16 parameters) | free: CRAMMazeFreeEnergy, a free 2x2 table per window offset (18; the paper's)
    free_init: float = 0.0  # free kernel: standard deviation of the initial tables (0: zero start)
    opt: str = 'adam'  # adam (constant lr, gradient-norm clip) | sgd (momentum, no clip; the paper's)
    momentum: float = 0.5  # sgd momentum
    cosine: bool = False  # sgd: cosine decay of the lr to 0 over the updates
    l1: float = 0.0  # L1 strength: proximal soft-threshold lr_k x l1 after each step (0: none)
    n_train: int = 0  # train on this many mazes of the split, a nested subset in a seeded order (0: all)
    device: str = 'cuda'  # cuda | cpu
    out: str = ''  # run directory, relative to experiments/
    score: tuple[str, ...] = ('dev100',)  # splits scored after training, with gibbs (T 0) and descend


def batches(n, size, rng):
    """An endless stream of index arrays (size,): each pass draws a new permutation of 0..n-1 from `rng` and yields
    its consecutive blocks of `size` (a last partial block is dropped)."""
    while True:
        order = rng.permutation(n)
        for lo in range(0, n - size + 1, size):
            yield order[lo:lo + size]


def make_model(cfg):
    """The untrained model of a config, on the CPU. It seeds torch with cfg.seed first: the separable kernel starts
    from random factors, the free kernel (free_init 0) from zero."""
    torch.manual_seed(cfg.seed)
    if cfg.kernel == 'free':
        return C.CRAMMazeFreeEnergy(window=cfg.window, init=cfg.free_init)
    assert cfg.kernel == 'separable', cfg.kernel
    cls = M.MazeEnergy if cfg.legacy else C.CRAMMazeEnergy
    return cls(R=cfg.R, de=cfg.de, window=cfg.window)


def train(cfg: Train):
    """Fit one model by pseudolikelihood (cram.maze_unique.pl_loss) and return (model, info).

    The batch indices of every update are drawn before the timed loop, from a numpy generator seeded with cfg.seed.
    Each update: the loss on a batch of mazes, backward, then one optimizer step. With opt sgd the learning rate of
    update k is set to lr_k (cosine decay when cfg.cosine); with opt adam the gradient norm is clipped. With l1 > 0 a
    proximal step shrinks every parameter toward 0 by lr_k x l1.
    info: 'loop_s' (seconds of the synchronized update loop), 'loss' (one value per update), 'stored' (stored
    parameter count), 'entries_with_gradient' (parameter entries with a nonzero gradient at the last update)."""
    dev = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    data = M.load(cfg.split)
    if cfg.n_train:  # a nested subset of the split: the first n of a seeded order (as training_methods.Maze)
        keep = np.random.default_rng(1000 + cfg.seed).permutation(len(data['inputs']))[:cfg.n_train]
        data = {k: v[keep] for k, v in data.items()}
    X = torch.as_tensor(data['inputs'], device=dev).view(-1, M.SIDE, M.SIDE)
    Y = torch.as_tensor(data['targets'], device=dev).view(-1, M.SIDE, M.SIDE)
    FR = torch.as_tensor(data['free'], device=dev).view(-1, M.SIDE, M.SIDE)
    model = make_model(cfg).to(dev)
    if cfg.opt == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    else:
        assert cfg.opt == 'adam', cfg.opt
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(cfg.seed)
    stream = batches(len(X), min(cfg.batch, len(X)), rng)
    idxs = [torch.as_tensor(next(stream), device=dev) for _ in range(cfg.updates)]
    losses = []
    sync = (lambda: torch.cuda.synchronize()) if dev.type == 'cuda' else (lambda: None)
    sync()
    t0 = time.perf_counter()
    for k in range(cfg.updates):
        i = idxs[k]
        loss = M.pl_loss(model, X[i], Y[i], FR[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        lr_k = cfg.lr * 0.5 * (1 + math.cos(math.pi * k / cfg.updates)) if cfg.cosine else cfg.lr
        if cfg.opt == 'adam':
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        else:
            for g in opt.param_groups:
                g['lr'] = lr_k
        opt.step()
        if cfg.l1:
            with torch.no_grad():
                for p in model.parameters():
                    p.copy_(p.sign() * (p.abs() - lr_k * cfg.l1).clamp_min(0))
        losses.append(loss.detach())
    sync()
    secs = time.perf_counter() - t0
    trained = sum(int((p.grad != 0).sum()) for p in model.parameters() if p.grad is not None)  # entries, last update
    return model, {'loop_s': secs, 'updates': cfg.updates, 'loss': [float(l) for l in losses], 'device': str(dev),
                   'threads': torch.get_num_threads(), 'stored': model.n_params(), 'entries_with_gradient': trained}


def load_run(run):
    """The trained model of a run directory (relative to experiments/), on the CPU."""
    run = HERE / run
    cfg = Train(**json.loads((run / 'config.json').read_text()))
    m = make_model(cfg)
    m.load_state_dict(torch.load(run / 'model.pt', map_location='cpu'))
    return m


def learned_params(model):
    """Parameters that receive a training gradient and enter the energy: all of CRAMMazeEnergy; for MazeEnergy,
    Fp, Gp, P, Q and the open-token bias row."""
    if isinstance(model, (C.CRAMMazeEnergy, C.CRAMMazeFreeEnergy)):
        return model.n_params()
    return model.Fp.numel() + model.Gp.numel() + model.P.numel() + model.Q.numel() + model.B.shape[1]


def tails(f):
    """Mean, 99th and 99.9th percentile and maximum of the per-maze operation counts `f`."""
    f = np.asarray(f, dtype=np.float64)
    return {'mean_flop': float(f.mean()), 'p99_flop': float(np.quantile(f, 0.99)), 'p999_flop': float(np.quantile(f, 0.999)),
            'max_flop': float(f.max())}


def summarize(right, stopped, flop, steps, extra=None):
    """The score record of one split from per-maze arrays: right (the board equals the label), stopped (the search
    reached its stopping test), flop (operation count) and steps (moves or descent steps).
    'wrong_stops' counts mazes that stopped at a wrong board; 'budget_failures' counts mazes that did not stop."""
    right, stopped = np.asarray(right, bool), np.asarray(stopped, bool)
    out = {'n': int(len(right)), 'solved': int(right.sum()), 'stopped': int(stopped.sum()),
           'wrong_stops': int((stopped & ~right).sum()), 'budget_failures': int((~stopped).sum()),
           'budget_failures_right': int((~stopped & right).sum()), **tails(flop),
           'mean_steps': float(np.mean(steps)), 'max_steps': int(np.max(steps))}
    out.update(extra or {})
    out['per_maze'] = {'right': right.tolist(), 'steps': np.asarray(steps).tolist(), 'flop': np.asarray(flop, float).tolist()}
    return out


@dataclasses.dataclass
class Search:
    """The settings of one search. The paper's search is the default: focused Gibbs at temperature 0."""
    search: str = 'gibbs'  # gibbs (focused Gibbs, cram.maze_unique.gibbs; the paper's) | descend (damped descent, cram.maze_unique.descend)
    T: float = 0.0  # gibbs temperature
    focus: str = 'random'  # gibbs: random among the cells not at their lowest-energy answer
    max_moves: int = 20000  # gibbs move budget per maze
    eta: float = 0.5  # descend step size
    tol: float = 1e-3  # descend fixed-point residual
    margin: float = 3.0  # descend start logits (0, margin) at open cells
    max_steps: int = 6000  # descend step budget per maze
    seed: int = 0  # generator of the gibbs focus


def score(model, split, sc: Search, device='cuda', batch=1000):
    """Search every maze of `split` (train | train900 | dev100 | test) with `model` and return the score record.

    gibbs: from every open cell on; the operation count is cram.maze_unique.gibbs_flop of the moves made.
    descend: Euler steps of the continuous answers from logits (0, margin) at the open cells; the operation count is
    (steps + checks) x one evaluation of the field over the grid.
    A maze is solved when its final board equals the label at every free cell. The generator of the Gibbs focus is
    seeded with sc.seed. Returns `summarize`'s dict plus the split, the search settings and the parameter counts."""
    data = M.load(split)
    dev = torch.device(device)
    model = model.to(dev).eval()
    gen = torch.Generator(device=dev).manual_seed(sc.seed)
    R, S, Fl, St = [], [], [], []
    t0 = time.perf_counter()
    for lo in range(0, len(data['inputs']), batch):
        idx = np.arange(lo, min(lo + batch, len(data['inputs'])))
        x, y, fr = M.to_device(data, idx, dev)
        if sc.search == 'gibbs':
            r = M.gibbs(model, x, y, fr, T=sc.T, max_moves=sc.max_moves, focus=sc.focus, gen=gen)
            R.append(r['right']); S.append(r['stopped']); St.append(r['moves'])
            Fl.append(M.gibbs_flop(model, r['moves'].double()))
        else:
            r = M.descend(model, x, y, fr, eta=sc.eta, max_steps=sc.max_steps, tol=sc.tol, margin=sc.margin)
            R.append(r['right']); S.append(r['stop_kind'] > 0); St.append(r['steps'])
            Fl.append((r['steps'] + r['checks']).double() * M.flop_per_step(model))
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    secs = time.perf_counter() - t0
    R, S, Fl, St = (torch.cat(a).cpu().numpy() for a in (R, S, Fl, St))
    return summarize(R, S, Fl, St, {'split': split, 'search': dataclasses.asdict(sc), 'params': learned_params(model),
                                    'stored': model.n_params(), 'seconds': secs})


def save(obj, path):
    """Write `obj` as JSON through a temporary file, so that an interrupted write leaves no partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def short(r):
    """The summary fields of a score record, without the per-maze lists."""
    return {k: r[k] for k in ('n', 'solved', 'stopped', 'wrong_stops', 'budget_failures', 'mean_flop', 'p99_flop', 'max_flop', 'mean_steps') if k in r}


ROW_SEARCH = {'gibbs': Search('gibbs'), 'descend': Search('descend')}


def main_train(cfg: Train, searches: tuple[str, ...] = ('gibbs', 'descend')):
    """Fit one run, save it, and score the splits of cfg.score with each search of `searches`."""
    out = HERE / cfg.out
    if (out / 'final.json').exists():
        print(out, 'finished; nothing to do', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    model, info = train(cfg)
    save(dataclasses.asdict(cfg), out / 'config.json')
    torch.save(model.state_dict(), out / 'model.pt')
    (out / 'log.jsonl').write_text(''.join(json.dumps({'update': k, 'loss': l}) + '\n' for k, l in enumerate(info['loss'])))
    save({k: v for k, v in info.items() if k != 'loss'} | {'final_loss': info['loss'][-1]}, out / 'train.json')
    print({k: v for k, v in info.items() if k != 'loss'}, 'final loss', info['loss'][-1], flush=True)
    scores = {}
    for split in cfg.score:
        for s in searches:
            r = score(model, split, ROW_SEARCH[s], cfg.device)
            save(r, out / f'score-{split}-{s}.json')
            scores[f'{split}-{s}'] = short(r)
            print(split, s, short(r), flush=True)
    save({'train': {k: v for k, v in info.items() if k != 'loss'}, 'scores': scores}, out / 'final.json')


def main_score(run: str, split: str = 'dev100', sc: Search = Search(), tag: str = '', device: str = 'cuda'):
    """Score one run. A test score is refused if the file exists (the test is scored once per final run)."""
    path = HERE / run / f"score-{split}-{sc.search}{('-' + tag) if tag else ''}.json"
    if split == 'test' and path.exists():
        print(path, 'exists: the test is scored once', flush=True)
        return
    r = score(load_run(run), split, sc, device)
    save(r, path)
    print(split, sc.search, tag, short(r), flush=True)


# ---- timing --------------------------------------------------------------------------------------------------------

def main_time(cfg: Train, repeats: int = 5, warm: int = 20):
    """GPU training time without scoring: `warm` updates first, then the median of `repeats` timed trainings from the
    seed's start (synchronized loop: the updates only; the data are on the GPU before it starts). <out>/gpu.json."""
    train(dataclasses.replace(cfg, updates=min(warm, cfg.updates)))
    secs = [train(cfg)[1]['loop_s'] for _ in range(repeats)]
    res = {'gpu_loop_s': float(np.median(secs)), 'all': secs, 'warm_updates': warm, 'updates': cfg.updates,
           'device': torch.cuda.get_device_name(), 'config': dataclasses.asdict(cfg),
           'includes': 'the update loop only (pseudolikelihood loss, backward, optimizer step), synchronized; no capture, no compression, no data loading or evaluation'}
    save(res, HERE / cfg.out / 'gpu.json')
    print(json.dumps(res), flush=True)


def main_cpu(cfg: Train, repeats: int = 3):
    """The same loop on one CPU thread, median of `repeats`; <out>/cpu.json."""
    torch.set_num_threads(1)
    c = dataclasses.replace(cfg, device='cpu')
    secs = [train(c)[1]['loop_s'] for _ in range(repeats)]
    res = {'cpu_thread_loop_s': float(np.median(secs)), 'timed_s': secs, 'updates': cfg.updates,
           'note': 'CPU: the whole loop on one CPU thread (median of %d).' % repeats, 'config': dataclasses.asdict(cfg)}
    save(res, HERE / cfg.out / 'cpu.json')
    print(json.dumps(res), flush=True)


# ---- table records -------------------------------------------------------------------------------------------------

ROWS = {  # row: (run prefix under runs/, search, label, note)
    'cram-sgd-fg': ('CU13/final/fg', 'gibbs', 'CRAM (PL with SGD, free 3x3 kernel, focused Gibbs T=0)',
                    'focused Gibbs at T 0 from every open cell on: each move sets one free cell that is not at its lowest-energy '
                    'answer (chosen uniformly among them) to it; stops at a fixed point of the energy (no such cell left); '
                    'the energy is CRAMMazeFreeEnergy, trained with SGD'),
}
MODEL_NOTE = {
    'cram-sgd-fg': 'cram.cram_maze.CRAMMazeFreeEnergy, 18 stored = 18 learned parameters (a free 2x2 table for each '
                   'of the 4 window offsets, the kernel holding T_d at d and T_d^T at -d, and the open-cell bias 2)',
}
PROVENANCE = {
    'cram-sgd-fg': 'SGD lr 1, 20 updates, chosen on dev100: runs/CU13/final/fg-s*; maze_unique.py, cram/cram_maze.py',
}


def main_rows(rows: tuple[str, ...] = ()):
    """Write the table records of `rows` (keys of ROWS) from the test scores and the gpu.json and cpu.json timings of
    the three seeds. A rewrite drops the fit_* fields that maze_unique_fit_time.py summary --write-records added, so
    name the rows to write; with none named, nothing is written."""
    from cram.table import write_row
    for row, (prefix, search, label, note) in ROWS.items():
        if row not in rows:
            continue
        runs = [HERE / 'runs' / f'{prefix}-s{s}' for s in range(3)]
        if not all((r / f'score-test-{search}.json').exists() for r in runs):
            print(row, 'not ready')
            continue
        sc = [json.loads((r / f'score-test-{search}.json').read_text()) for r in runs]
        gpu = [json.loads((r / 'gpu.json').read_text())['gpu_loop_s'] for r in runs]
        cpu = json.loads((runs[0] / 'cpu.json').read_text())
        cfg = json.loads((runs[0] / 'config.json').read_text())
        flop = np.concatenate([np.asarray(r['per_maze']['flop']) for r in sc])
        p = write_row('maze-unique', f'{row}', label=label, per_seed=[r['solved'] / r['n'] for r in sc], n_test=sc[0]['n'],
                      test_set='EqR maze-30x30-unique-1k test (1,000 mazes), x 3 training seeds',
                      params=sc[0]['params'], flop_per_puzzle=statistics.fmean(r['mean_flop'] for r in sc),
                      train_gpu_s=statistics.fmean(gpu), train_cpu_s=cpu['cpu_thread_loop_s'],
                      provenance=PROVENANCE[row],
                      notes=f"{note}. Model: {MODEL_NOTE[row]}. Recipe: pseudo-likelihood only, "
                            + (f"free kernel, zero start, " if cfg.get('kernel') == 'free' else f"R {cfg['R']}, de {cfg['de']}, ")
                            + (f"window {cfg['window']}, {cfg['updates']} SGD updates (momentum {cfg['momentum']}, cosine decay, no clip), batch {cfg['batch']}, lr {cfg['lr']}, l1 {cfg['l1']}, "
                               if cfg.get('opt') == 'sgd' else f"window {cfg['window']}, {cfg['updates']} Adam updates, batch {cfg['batch']}, lr {cfg['lr']}, clip {cfg['clip']}, ")
                            + f"no post-fit compression or rescaling, T as stated; trained on all 1,000 training mazes. GPU: the "
                            f"synchronized update loop on one H200, warm, median of 5 per seed, mean over seeds. {cpu['note']}",
                      extra={'stopped': [r['stopped'] for r in sc], 'wrong_stops': [r['wrong_stops'] for r in sc],
                             'budget_failures': [r['budget_failures'] for r in sc],
                             'mean_steps': [r['mean_steps'] for r in sc], 'max_steps': [r['max_steps'] for r in sc],
                             'p99_flop': float(np.quantile(flop, 0.99)), 'p999_flop': float(np.quantile(flop, 0.999)),
                             'max_flop': float(flop.max()), 'gpu_s_per_seed': gpu, 'stored_params': sc[0]['stored']})
        print(p.name, json.loads(p.read_text())['solved_mean'])


if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict({'train': main_train, 'score': main_score,
                                          'time': main_time, 'cpu': main_cpu, 'rows': main_rows},
                                         config=(tyro.conf.OmitArgPrefixes,))
