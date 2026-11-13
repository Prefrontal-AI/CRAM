"""N-Queens CRAM with one free table per cell pair: a sweep over training methods, and the hand-set reference energy
(the hand-set N-Queens row of Table 3). Run from experiments/.

  python nqueens_free.py --mode tune --n 8 --method cd             # every tuning config of one method: fit on the fitting rows, search validation
  python nqueens_free.py --mode select --n 8                       # the choice per method and the selected method (validation only)
  python nqueens_free.py --mode final --n 8 --method cd --seed 0   # timed fit on all training rows (cold + warm), test once, 20 chains
  python nqueens_free.py --mode hand --n 8 --split dev             # hand-set attack-pair energy: temperature on validation
  python nqueens_free.py --mode hand --n 8 --split test --seed 0   # its test rows, 20 chains

The energy is the one of cram/cram_nqueens.py: n^2 binary cells, a free 2 x 2 table W_p per unordered cell pair and a
free input cost B_i per cell, zero start, given queens clamped:
    E(x) = 1/2 sum_(i != j) S[i, j, x_i, x_j] + sum_i B[i, x_i],   S[i, j] = W_p, S[j, i] = W_p^T for the pair p = (i < j).
Only the training method changes. The method losses are training_methods.loss_fn on a binary task class: pl
(pseudolikelihood, PL), mpl (masked PL), mpf (minimum probability flow), dsm (denoising score matching), cd
(contrastive divergence, CD, from the label), pcd (persistent CD, one chain per training row, started at its label),
chl (contrastive Hebbian learning, free mean-field relaxation from the uniform state). None differentiates through a
solver: the negatives of cd, pcd and chl (class S) are drawn without gradient. Every method uses full-batch SGD with
momentum 0.5, cosine decay and a proximal L1 step after each update, and the same inference-ready model (the stopping
threshold tau, the temperature unit and the local margin delta, fitted from the training solutions inside the fit
clock). The search is cram_nqueens.solve (focused heat-bath Gibbs from the board that holds only the given queens).

The hand-set energy (--mode hand) is E = 1 x (attacking queen pairs) - 0.5 x (queens); it uses the rules and learns
nothing. nqueens_offset.py (the learned N-Queens row of Table 3) imports ready, score, data and the io helpers.

Outputs: runs/NQ/alt/n<n>/ (the folder name is only a label). Every stage writes final.json last (atomic rename) and
is skipped when that exists. Untimed fits (tuning) keep a rolling checkpoint ckpt.pt every 10 minutes; a timed fit is
never resumed (it restarts, so its clock covers one process).
Dimension names: b rows (boards), i and j cells (n * n), a and s cell states (0 empty, 1 queen), p cell pairs,
k chains per row.
"""
import dataclasses
import json
import math
import os
import platform
import statistics
import time
import uuid
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from einops import rearrange

import training_methods as AT
from cram import cram_nqueens as Q
from cram.nqueens_data import gram_metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs' / 'NQ' / 'alt'

METHODS = ('pl', 'mpl', 'mpf', 'dsm', 'cd', 'pcd', 'chl')
CLASS = {m: AT.CLASS[m] for m in METHODS}
LRS = {'L': (0.1, 1.0, 10.0, 100.0), 'S': (1.0, 10.0, 100.0)}  # learning rates per method class; mpf and dsm diverge at 10
LAMS = (0.0, 3e-6, 1e-5)  # L1 weights: the useful range (at 1e-4 and above, delta < 0 or the stopping test accepts invalid boards)
REGS = {'L': (0.0,), 'S': (0.0, 1.0)}  # class S: weight of the energy-magnitude regularizer of Du & Mordatch (training_methods.contrast)
STEPS = {'L': 10_000, 'S': 2_000}  # updates per method class (class S: the count of the Sudoku study; a 10,000-update cd/pcd grid costs more than 25 GPU-hours)
SEARCH = tuple((stop, t) for stop in ('local', 'energy') for t in (0.3, 1.0, 3.0))  # validation search grid: (stopping test, temperature)
CAP = 20_000  # moves per chain before the search returns its lowest-energy board
CKPT_S = 600.0  # seconds between rolling checkpoints of an untimed fit


@torch.no_grad()
def heat_bath(task, ctx, board, moves, gen):
    """training_methods.heat_bath's move rule (one free cell drawn uniformly per board, its state drawn from the T 1 heat-bath
    conditional exp(-field)), with the fields updated incrementally (the change of every cell's field when one cell
    changes state) instead of recomputed from all cells after every move. Installed as training_methods.heat_bath below, so
    training_methods.loss_fn's cd and pcd use it."""
    c = task.c
    Sg = rearrange(Q.full_S(task.W.detach(), task.iu, c), 'i j a s -> j s i a').contiguous()
    g = task.field(AT.onehot(board, 2), ctx).detach()
    x = board.clone()
    ar = torch.arange(len(x), device=x.device)
    free = ctx['free']
    for _ in range(moves):
        cell = torch.where(free, torch.rand(x.shape, device=x.device, generator=gen), -1.0).argmax(1)
        L = g[ar, cell]  # (b, a)
        new = (torch.rand(len(x), device=x.device, generator=gen) < torch.sigmoid(L[:, 0] - L[:, 1])).long()
        old = x[ar, cell]
        g += Sg[cell, new] - Sg[cell, old]
        x[ar, cell] = new
    return x


AT.heat_bath = heat_bath  # loss_fn looks heat_bath up in training_methods's namespace at call time


def knobs(method, n):
    """The tuning values of the method's own parameter (training_methods.loss_fn's knob) at board size n."""
    c = n * n
    return {'pl': (None,), 'mpf': (None,), 'mpl': (0.1, 0.3), 'dsm': (0.5, 1.5), 'cd': (c, 4 * c), 'pcd': (c, 4 * c),
            'chl': (10, 30)}[method]


def configs(method, n):
    """The tuning grid of one method: a list of dicts {method, lr, knob, lam (L1 weight), reg}."""
    return [dict(method=method, lr=lr, knob=k, lam=lam, reg=reg) for lr in LRS[CLASS[method]] for k in knobs(method, n) for lam in LAMS
            for reg in REGS[CLASS[method]]]


def config_name(cfg):
    """The folder name of a tuning config."""
    return f"{cfg['method']}-lr{cfg['lr']:g}-k{cfg['knob']}-lam{cfg['lam']:g}-a{cfg['reg']:g}"


@dataclasses.dataclass
class Args:
    mode: Literal['tune', 'select', 'final', 'hand']  # stage to run
    n: int = 8  # board size (8 or 10)
    method: str = 'pl'  # training method (tune, final)
    seed: int = 0  # final and hand test: training generator and search seed
    split: Literal['dev', 'test'] = 'dev'  # hand mode: validation temperature choice or the test rows
    repeats: int = 7  # warm fit repeats after the cold fit (final)
    lr: float = 0.0  # tune: run only the configs with this learning rate (0: every learning rate)


# --------------------------------------------------------------------------------------------------- io helpers

def save(path, value):
    """Write value as JSON to path (a Path) through a temporary file and a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    os.replace(tmp, path)


def save_torch(path, value):
    """torch.save value to path (a Path) through a temporary file and a rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def read(path):
    """The parsed JSON file at path."""
    return json.loads(Path(path).read_text())


def hardware():
    """The name of the GPU, or of the processor when there is no GPU."""
    return torch.cuda.get_device_name() if torch.cuda.is_available() else platform.processor()


def cuda_context():
    """Create the CUDA context and return the seconds it took (reported apart from the fit time)."""
    t = time.perf_counter()
    torch.cuda.init()
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    return time.perf_counter() - t


# ------------------------------------------------------------------------------------------- the binary task class

class NQTask:
    """The task interface of training_methods.loss_fn for the free-table energy: states v (b, i, a) one-hot or
    relaxed, dE/dv = v S + B. x_in: (b, i) bool, the given queens. y: (b, i) int labels (1 queen). n: board size.
    Learned: W (p, 2, 2) and B (i, 2), zero start."""
    C = 2

    def __init__(self, x_in, y, n):
        self.dev = y.device
        self.c = n * n
        self.iu = Q.pair_index(self.c, self.dev)
        self.W = torch.zeros(self.iu.shape[1], 2, 2, device=self.dev, requires_grad=True)
        self.B = torch.zeros(self.c, 2, device=self.dev, requires_grad=True)
        self.Y = y  # (b, i) labels; pcd's persistent chains start here
        self.ctx = {'y': y, 'free': ~x_in, 'idx': torch.arange(len(y), device=self.dev)}

    def begin(self):
        """Build the pair matrix M (2 c, 2 c) from W; call once per update, before field or energy."""
        self.M = rearrange(Q.full_S(self.W, self.iu, self.c), 'i j a s -> (i a) (j s)')  # symmetric

    def field(self, v, ctx):
        """dE/dv (b, i, a) = sum_j S[i, j] v_j + B_i, for states v (b, i, a)."""
        return rearrange(rearrange(v, 'b i a -> b (i a)') @ self.M, 'b (i a) -> b i a', a=2) + self.B[None]

    def energy(self, v, ctx):
        """E(v) (b,) = 1/2 sum_(i, j) v_i^T S[i, j] v_j + sum_i B_i^T v_i, for states v (b, i, a)."""
        pair = rearrange(rearrange(v, 'b i a -> b (i a)') @ self.M, 'b (i a) -> b i a', a=2)
        return 0.5 * (v * pair).sum((1, 2)) + (v * self.B[None]).sum((1, 2))


def ready(W, B, y):
    """The inference-ready model, as cram_nqueens.fit computes it. W: (p, 2, 2). B: (c, 2). y: (b, c) int labels.
    Returns (S (c, c, 2, 2), tau, unit, delta): tau is the highest energy of a training solution, unit the temperature
    unit (mean |field difference| over the cells of the training rows' solutions) and delta the smallest local margin
    of a training cell (the field of the other state minus the field of the cell's own state)."""
    c = B.shape[0]
    sol, which = torch.unique(y, dim=0, return_inverse=True)
    S = Q.full_S(W, Q.pair_index(c, W.device), c)
    g = Q.fields(S, B, sol)
    tau = Q.board_energy(S, B, sol).max()
    rows = torch.bincount(which, minlength=len(sol)).float()
    unit = ((g[..., 1] - g[..., 0]).abs().mean(1) * rows).sum() / rows.sum()
    own = g.gather(2, sol[..., None])[..., 0]
    other = g.gather(2, (1 - sol)[..., None])[..., 0]
    delta = (other - own).min()
    return S, float(tau), float(unit), float(delta)


def fit(cfg, inputs, labels, n, seed, steps=None, ckpt_dir=None, log_every=500):
    """The complete fit, timed from GRAM's token arrays in host memory to the inference-ready model: transfer to the
    GPU, zero start, `steps` full-batch SGD updates (momentum, cosine decay, proximal L1), then ready().

    cfg: {method, lr, knob, lam, reg, momentum}. inputs, labels: (b, c) token arrays (2 = queen). n: board size.
    seed: the generator of the method's random draws. steps: updates (None: STEPS of the method's class).
    With ckpt_dir (untimed tuning fits only): a rolling checkpoint every CKPT_S seconds and a log.jsonl of the loss.
    Returns (model, info): model = {W, B, S, tau, unit, delta} on the CPU; info = the config, fit_seconds, parameter
    counts and `diverged`."""
    steps = steps or STEPS[CLASS[cfg['method']]]
    dev = torch.device('cuda')
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    x_in = torch.as_tensor(inputs, device=dev) == 2
    y = (torch.as_tensor(labels, device=dev) == 2).long()
    task = NQTask(x_in, y, n)
    gen = torch.Generator(device=dev).manual_seed(seed)
    mstate = {'_reg': cfg['reg']}
    bufs = [torch.zeros_like(task.W), torch.zeros_like(task.B)]
    knob = cfg['knob']
    start, spent, diverged, loss = 0, 0.0, False, None
    ck = ckpt_dir / 'ckpt.pt' if ckpt_dir else None
    log = ckpt_dir / 'log.jsonl' if ckpt_dir else None
    if ck and ck.exists():
        st = torch.load(ck, weights_only=False)
        with torch.no_grad():
            task.W.copy_(st['W']); task.B.copy_(st['B'])
            bufs[0].copy_(st['bufs'][0]); bufs[1].copy_(st['bufs'][1])
        gen.set_state(st['gen'])
        mstate = st['mstate']
        start, spent = st['step'], st['seconds']
        if log.exists():
            keep = [l for l in log.read_text().splitlines() if 'step' in json.loads(l) and json.loads(l)['step'] <= start]
            log.write_text(''.join(l + '\n' for l in keep) + json.dumps({'resume': start}) + '\n')
    last_ck = time.perf_counter()
    for k in range(start, steps):
        task.begin()
        loss = AT.loss_fn(cfg['method'], knob, task, task.ctx, gen, mstate)
        gW, gB = torch.autograd.grad(loss, (task.W, task.B))
        lr_k = cfg['lr'] * 0.5 * (1 + math.cos(math.pi * k / steps))
        with torch.no_grad():
            for p, grad, buf in zip((task.W, task.B), (gW, gB), bufs):
                buf.mul_(cfg['momentum'] if 'momentum' in cfg else 0.5).add_(grad)
                p.sub_(lr_k * buf)
                if cfg['lam']:
                    p.copy_(p.sign() * (p.abs() - lr_k * cfg['lam']).clamp_min(0))
        if ckpt_dir and (k + 1) % log_every == 0:
            lv = float(loss)
            with log.open('a') as f:
                f.write(json.dumps({'step': k + 1, 'loss': lv}) + '\n')
            if not math.isfinite(lv) or not bool(torch.isfinite(task.W).all()):
                diverged = True
                break
            if time.perf_counter() - last_ck > CKPT_S:
                save_torch(ck, dict(W=task.W.detach(), B=task.B.detach(), bufs=bufs, gen=gen.get_state(),
                                    mstate=mstate, step=k + 1, seconds=spent + time.perf_counter() - t0))
                last_ck = time.perf_counter()
    with torch.no_grad():
        W, B = task.W.detach(), task.B.detach()
        diverged = diverged or not bool(torch.isfinite(W).all() and torch.isfinite(B).all())
        S, tau, unit, delta = ready(W, B, y) if not diverged else (None, math.nan, math.nan, math.nan)
    torch.cuda.synchronize()
    seconds = spent + time.perf_counter() - t0
    model = dict(W=W.cpu(), B=B.cpu(), S=None if S is None else S.cpu(), tau=tau, unit=unit, delta=delta)
    info = dict(cfg, fit_seconds=seconds, steps=steps, seed=seed, rows=len(y), diverged=diverged,
                final_loss=None if loss is None else float(loss.detach()), tau=tau, unit=unit, delta=delta,
                params_learned=int(W.numel() + B.numel()), params_nonzero=int((W != 0).sum() + (B != 0).sum()),
                resumed_from=start)
    if not diverged:
        info['coupling'] = Q.coupling_summary(W.cpu().numpy(), B.cpu().numpy(), n)
    if ck and ck.exists():
        ck.unlink()
    return model, info


def usable(model):
    """True when the model is finite and has a positive temperature unit, so that the search can run."""
    return model['S'] is not None and math.isfinite(model['unit']) and model['unit'] > 1e-8 and bool(
        torch.isfinite(model['S']).all())


def score(model, inputs, labels, n, *, chains, temperature, cap, seed, stop, records=None):
    """Search `chains` chains per row (cram_nqueens.solve) and score the boards with GRAM's metrics.

    model: {S, B, tau, unit, delta}. inputs, labels: (b, c) bool (given queens, labeled solution). temperature: in
    units of model['unit']. cap: moves per chain. stop: the stopping test, 'energy' (E <= tau) or 'local' (every free
    cell's margin >= delta). records: a path for the per-row arrays (boards, energies, moves, flop), saved when
    chains >= 20.
    Returns a dict: acc_n1 (the first chain's board is valid), first_stop_valid, wrong_stops, budget_failures (chains
    that reached the cap), flop_n1, flop_all_chains, and with 20 chains acc_majority20, pass20, cov20 (coverage of the
    row's solutions by the 20 chains), cov20_group_mean."""
    t0 = time.perf_counter()
    if not usable(model):
        r = len(inputs)
        return dict(rows=r, chains=chains, unusable=True, acc_n1=0.0, first_stop_valid=0.0, wrong_stops=0,
                    budget_failures=r * chains, flop_n1=0.0, flop_all_chains=0.0, search_seconds=0.0,
                    **({'acc_majority20': 0.0, 'cov20': 0.0, 'cov20_group_mean': 0.0, 'pass20': 0.0} if chains >= 20 else {}))
    boards, stopped, energy, moves, flop = Q.solve(model, inputs, chains=chains, temperature=temperature, cap=cap,
                                                  seed=seed, device='cuda', stop=stop)
    seconds = time.perf_counter() - t0
    ok = Q.valid(boards, n)
    contains = (boards >= inputs[:, None, :]).all(-1)
    res = dict(rows=len(inputs), chains=chains, search_seconds=seconds, acc_n1=float(ok[:, 0].mean()),
               first_stop_valid=float((stopped[:, 0] & ok[:, 0] & contains[:, 0]).mean()),
               label_match_n1=float((boards[:, 0] == labels).all(-1).mean()),
               chains_stopped=float(stopped.mean()), wrong_stops=int((stopped & ~ok).sum()),
               budget_failures=int((~stopped).sum()), unstopped_valid=int((~stopped & ok).sum()),
               flop_n1=float(flop[:, 0].mean()), flop_all_chains=float(flop.sum(1).mean()),
               moves_n1_mean=float(moves[:, 0].mean()))
    if chains >= 20:
        m = gram_metrics(inputs, boards, n, tiebreak=-energy)
        groups = np.minimum(m['n_solutions'], 20).astype(int)
        res.update(acc_majority20=float(m['majority'].mean()), pass20=float(m['pass@20'].mean()),
                   cov20=float(m['coverage20'].mean()),
                   cov20_group_mean=float(np.mean([m['coverage20'][groups == k].mean() for k in np.unique(groups)])),
                   cov_all20=float(m['coverage_all'].mean()))
        if records is not None:
            records.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(records, boards=np.packbits(boards, -1), stopped=stopped, energy=energy, moves=moves,
                                flop=flop, majority=m['majority'], coverage20=m['coverage20'],
                                n_solutions=m['n_solutions'])
    return res


def data(n):
    """GRAM's training tokens of board size n and the validation split: (inputs (b, c), labels (b, c), fit_rows,
    val_rows), the last two index arrays of the rows for the tuning fits and for validation (split by distinct input)."""
    tr_tok, lab_tok = Q.load_tokens(n, 'train')
    fit_rows, val_rows = Q.carve_validation(tr_tok == 2)
    return tr_tok, lab_tok, fit_rows, val_rows


# ------------------------------------------------------------------------------------------------------- stages

def tune(a):
    """Every tuning config of a.method: fit on the fitting rows (seed 0), then search the validation rows with one
    chain at every (stopping test, temperature) of SEARCH. Writes runs/NQ/alt/n<n>/tune/<config>/final.json."""
    tr_tok, lab_tok, fit_rows, val_rows = data(a.n)
    val_in, val_lab = tr_tok[val_rows] == 2, lab_tok[val_rows] == 2
    for cfg in configs(a.method, a.n):
        dest = ROOT / f'n{a.n}' / 'tune' / config_name(cfg)
        if (dest / 'final.json').exists() or (a.lr and cfg['lr'] != a.lr):
            continue
        dest.mkdir(parents=True, exist_ok=True)
        if (dest / 'fit.json').exists():
            info = read(dest / 'fit.json')
            model = torch.load(dest / 'model.pt', weights_only=False)
        else:
            model, info = fit(dict(cfg, momentum=0.5), tr_tok[fit_rows], lab_tok[fit_rows], a.n, seed=0, ckpt_dir=dest)
            info.update(hostname=platform.node(), slurm_job=os.environ.get('SLURM_JOB_ID'), hardware=hardware())
            save_torch(dest / 'model.pt', model)
            save(dest / 'fit.json', info)
        searches = []
        for stop, t in SEARCH:
            r = score(model, val_in, val_lab, a.n, chains=1, temperature=t, cap=CAP, seed=0, stop=stop)
            searches.append(dict(stop=stop, temperature=t, **r))
            print(config_name(cfg), stop, t, f"val N=1 {r['acc_n1']:.4f} wrong {r['wrong_stops']} cap {r['budget_failures']}",
                  flush=True)
        save(dest / 'final.json', dict(config=cfg, fit=info, searches=searches))


def choose(n, method):
    """The choice of one method's config and search setting: highest validation accuracy of one chain (N = 1); ties to
    the shortest fit time, then the lower mean flop. Returns a dict {config, stop, temperature, cap, val_*}."""
    best = None
    for cfg in configs(method, n):
        f = ROOT / f'n{n}' / 'tune' / config_name(cfg) / 'final.json'
        if not f.exists():
            raise FileNotFoundError(f)
        r = read(f)
        for s in r['searches']:
            key = (-s['acc_n1'], r['fit']['fit_seconds'], s['flop_n1'])
            if best is None or key < best[0]:
                best = (key, dict(config=cfg, stop=s['stop'], temperature=s['temperature'], cap=CAP,
                                  val_acc_n1=s['acc_n1'], val_wrong_stops=s['wrong_stops'],
                                  val_budget_failures=s['budget_failures'], val_flop_n1=s['flop_n1'],
                                  tune_fit_seconds=r['fit']['fit_seconds'], val_rows=s['rows']))
    return best[1]


def select(a):
    """Write the choice of every method (selected-<method>.json) and the selected method (selected-method.json: the
    highest validation N = 1 over the methods' choices). Existing files are kept, so a choice is made once."""
    out = {}
    for m in METHODS:
        dest = ROOT / f'n{a.n}' / f'selected-{m}.json'
        if not dest.exists():
            save(dest, dict(choice=choose(a.n, m), rule='highest validation N=1 (seed 0, 1 chain); ties: shortest '
                            'tuning fit time, then lower mean flop', at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        out[m] = read(dest)['choice']
    dest = ROOT / f'n{a.n}' / 'selected-method.json'
    if not dest.exists():
        win = min(METHODS, key=lambda m: (-out[m]['val_acc_n1'], out[m]['tune_fit_seconds']))
        save(dest, dict(method=win, rule='highest validation N=1 over the methods\' choices; ties: shortest fit time',
                        val_acc_n1={m: out[m]['val_acc_n1'] for m in METHODS}))
    print(json.dumps(read(dest), indent=1))


def final(a):
    """The chosen config of a.method on all training rows: one cold fit and a.repeats warm fits in this process (the
    fit times), then the test rows once with 20 chains. Writes runs/NQ/alt/n<n>/final/<method>-s<seed>/."""
    dest = ROOT / f'n{a.n}' / 'final' / f'{a.method}-s{a.seed}'
    if (dest / 'final.json').exists():
        print((dest / 'final.json').read_text())
        return
    sel = read(ROOT / f'n{a.n}' / f'selected-{a.method}.json')['choice']
    context = cuda_context()
    tr_tok, lab_tok = Q.load_tokens(a.n, 'train')  # file reading: outside the fit clock
    cfg = dict(sel['config'], momentum=0.5)
    if (dest / 'fit.json').exists():  # the timed fits finished in an earlier process; only the test remains
        info = read(dest / 'fit.json')
        model = torch.load(dest / 'model.pt', weights_only=False)
    else:
        model, info = fit(cfg, tr_tok, lab_tok, a.n, seed=a.seed)
        warm = [fit(cfg, tr_tok, lab_tok, a.n, seed=a.seed)[1]['fit_seconds'] for _ in range(a.repeats)]
        info.update(cold_seconds=info['fit_seconds'], warm_seconds=statistics.median(warm), warm_repeats=warm,
                    cuda_context_seconds=context, hostname=platform.node(), hardware=hardware(),
                    slurm_job=os.environ.get('SLURM_JOB_ID'))
        save_torch(dest / 'model.pt', model)
        save(dest / 'fit.json', info)
    te_in, te_lab, _ = Q.load_split(a.n, 'test')
    res = score(model, te_in, te_lab, a.n, chains=20, temperature=sel['temperature'], cap=sel['cap'], seed=a.seed,
                stop=sel['stop'], records=dest / 'records.npz')
    save(dest / 'final.json', dict(method=a.method, seed=a.seed, selection=sel, fit=info, test=res))
    print(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------------------- hand-set reference

HAND = dict(attack=1.0, queen=0.5)  # E = 1.0 x (attacking queen pairs) - 0.5 x (queens)
HAND_T = (0.1, 0.3, 1.0, 3.0)


def hand_model(n, labels):
    """W_p[1,1] = 1 on every attack pair (row, column, diagonal), B_i[1] = -0.5. For 0 < queen < attack, E <= -queen * n
    holds exactly on the valid boards (an independent set has at most n queens; each attack costs more than a queen
    gains), so the energy test with tau = the training solutions' energy (all -n/2) is exact."""
    c = n * n
    W = torch.zeros(c * (c - 1) // 2, 2, 2)
    W[torch.as_tensor(Q.attack_classes(n) > 0), 1, 1] = HAND['attack']
    B = torch.zeros(c, 2)
    B[:, 1] = -HAND['queen']
    S, tau, unit, delta = ready(W.cuda(), B.cuda(), torch.as_tensor(labels, device='cuda').long())
    return dict(W=W, B=B, S=S.cpu(), tau=tau, unit=unit, delta=delta)


def hand(a):
    """The hand-set energy. --split dev: choose the temperature on the validation rows (one chain, seed 0) and write
    hand/dev.json. --split test: the test rows once with 20 chains and that temperature, into hand/test-s<seed>/."""
    tr_tok, lab_tok, fit_rows, val_rows = data(a.n)
    model = hand_model(a.n, lab_tok == 2)
    root = ROOT / f'n{a.n}' / 'hand'
    if a.split == 'dev':
        dest = root / 'dev.json'
        if dest.exists():
            print(dest.read_text())
            return
        rows = []
        for t in HAND_T:
            r = score(model, tr_tok[val_rows] == 2, lab_tok[val_rows] == 2, a.n, chains=1, temperature=t, cap=CAP,
                      seed=0, stop='energy')
            rows.append(dict(temperature=t, **r))
            print(t, r['acc_n1'], r['wrong_stops'], r['budget_failures'], flush=True)
        best = min(rows, key=lambda r: (-r['acc_n1'], r['flop_n1']))
        save(dest, dict(hand=HAND, tau=model['tau'], unit=model['unit'], delta=model['delta'], searches=rows,
                        choice=dict(temperature=best['temperature'], stop='energy', cap=CAP),
                        rule='highest validation N=1 (seed 0, 1 chain); ties to lower mean flop'))
        return
    sel = read(root / 'dev.json')['choice']
    dest = root / f'test-s{a.seed}'
    if (dest / 'final.json').exists():
        print((dest / 'final.json').read_text())
        return
    te_in, te_lab, _ = Q.load_split(a.n, 'test')
    res = score(model, te_in, te_lab, a.n, chains=20, temperature=sel['temperature'], cap=sel['cap'], seed=a.seed,
                stop=sel['stop'], records=dest / 'records.npz')
    save(dest / 'final.json', dict(hand=HAND, choice=sel, seed=a.seed, test=res,
                                   params_nonzero=int((model['W'] != 0).sum() + (model['B'] != 0).sum())))
    print(json.dumps(res, indent=1))


def main(a: Args):
    """Run the stage a.mode. TF32 matrix products are off, so that float32 products are exact IEEE products."""
    if a.mode != 'select':
        torch.backends.cuda.matmul.allow_tf32 = False
    {'tune': tune, 'select': select, 'final': final, 'hand': hand}[a.mode](a)


if __name__ == '__main__':
    main(tyro.cli(Args))
