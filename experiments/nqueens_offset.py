"""N-Queens CRAM with an offset-shared energy (the learned N-Queens row of Table 3), trained on GRAM's N-Queens sets
with a local objective at the labeled solutions. Run from experiments/.

  python nqueens_offset.py --mode tune --n 10 --method pl --arm ising  # every grid config of one method and arm: fit on the fitting rows, search validation
  python nqueens_offset.py --mode select --n 10                      # the choice per method and arm, and the selected (method, arm) (validation only)
  python nqueens_offset.py --mode final --n 10 --method pl --arm ising --seed 0  # timed fits (GPU cold + 7 warm), test once, 20 chains
  python nqueens_offset.py --mode shared --n 8 --seed 0             # the selected 10 x 10 recipe unchanged on another size (timed fits, test once)
  python nqueens_offset.py --mode cpu --n 10                        # the selected fit on one CPU thread (cold + 7 warm, seed 0)
  python nqueens_offset.py --mode shared-cpu --n 8                  # the same for the shared recipe
  python nqueens_offset.py --mode ablation --n 10 --seed 0          # the selected recipe without per-offset preconditioning
  python nqueens_offset.py --mode retime --n 10 --stage final --method pl --arm ising --seed 0  # that run's timed fits again with the dense pair-matrix map (the reported fit times)

The paper reports one recipe at both board sizes: the (method, arm) that `select` chooses at 10 x 10 (pseudolikelihood,
arm ising; recipes/nqueens-offset-n10.json), run at 10 x 10 by `final` and at 8 x 8 by `shared`.

Energy. A pair's term depends only on the two cells' states and their offset (dr, dc), and every cell has the same
unary cost. Nothing says which offsets attack. This prior matches the one of the baseline EBR, whose energy network is
attention with 2D rotary position embeddings (RoPE) over the board and no absolute position embedding. For cells i < j
in row-major order, dr = r_j - r_i >= 0 (dc > 0 when dr = 0), so each unordered pair has one offset in this half-plane:
2 n (n - 1) offsets (112 at n = 8, 180 at n = 10). Arms:
  table: E(x) = sum_(i<j) A_d(i,j)[x_i, x_j] + sum_i b[x_i], a free 2 x 2 table A_d per offset d and a free unary
         table b (2,): 4 x 2n(n-1) + 2 parameters.
  ising: E(x) = sum_d J_d sum_(pairs at d) x_i x_j + h sum_i x_i (x the queen indicator): 2n(n-1) + 1 parameters.
Given queens are clamped. The per-pair tables W_p = A_d(p) feed the free-table code unchanged (nqueens_free.ready,
nqueens_free.score, cram/cram_nqueens.solve).
Methods (local objectives at the labeled boards; no model samples, no solver in training): pl (pseudolikelihood, PL),
mpl (masked PL), mpf (minimum probability flow), dsm (denoising score matching) from training_methods.loss_fn, and spl
(PL of the label at a board with some free cells set to the start state, nqueens_emptied.spl_loss). Optimizer:
full-batch SGD, momentum 0.5, cosine decay, zero start, per-offset preconditioning; no L1.

Outputs: runs/NQ/nq11/n<n>/ (the folder name `nq11` is only a label).
Dimension names: b rows (boards), i and j cells (c = n * n), a and s cell states (0 empty, 1 queen), p cell pairs,
d offsets, k parameters.
"""
import dataclasses
import itertools
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from einops import rearrange

import training_methods as AT
import nqueens_free as N
import nqueens_emptied as SN  # noqa: F401  (installs its loss_fn as AT.loss_fn: adds the method spl; other methods pass through)
from cram import cram_nqueens as Q

ROOT = N.HERE / 'runs' / 'NQ' / 'nq11'
METHODS = ('pl', 'mpl', 'mpf', 'dsm', 'spl')
ARMS = ('table', 'ising')
LRS = (0.3, 1.0, 3.0, 10.0, 30.0)  # learning-rate grid
UPDATES = 4000  # full-batch updates of every fit
SEARCH = tuple((stop, t) for stop in ('local', 'energy') for t in (1.0, 3.0))  # validation search grid: (stopping test, temperature)
CHAINS = 20  # chains per row in validation and test
N1_SLACK = 0.005  # selection: highest validation group-mean Cov@20 among settings within 0.5 points of the best validation N = 1


def knobs(method):
    """The tuning values of the method's own parameter (the knob of training_methods.loss_fn; spl: {'rate'})."""
    return {'pl': (None,), 'mpf': (None,), 'mpl': (0.1, 0.3, 0.5), 'dsm': (0.5, 1.5),
            'spl': ({'rate': 0.5}, {'rate': 1.0})}[method]


def configs(method, arm, precond=True):
    """The tuning grid of one method and arm: every learning rate and knob, per-offset preconditioning, no L1.
    Returns a list of dicts {method, arm, lr, knob, lam, precond}."""
    return [dict(method=method, arm=arm, lr=lr, knob=k, lam=0.0, precond=precond) for lr in LRS for k in knobs(method)]


def config_name(cfg):
    """The folder name of a config."""
    k = cfg['knob']
    k = k['rate'] if isinstance(k, dict) else k
    return f"{cfg['method']}-{cfg['arm']}-lr{cfg['lr']:g}-k{k}" + (f"-lam{cfg['lam']:g}" if cfg.get('lam') else '') + ('-pre' if cfg.get('precond') else '')


@dataclasses.dataclass
class Args:
    mode: Literal['tune', 'select', 'final', 'cpu', 'ablation', 'shared', 'shared-cpu', 'retime']  # stage to run
    n: int = 8  # board size (8 or 10)
    method: str = 'pl'  # training method (tune, final)
    arm: str = 'table'  # energy arm (tune, final): table or ising
    seed: int = 0  # final: training generator and search seed
    repeats: int = 7  # final: warm fits after each cold fit
    stage: Literal['final', 'shared', 'ablation'] = 'final'  # retime: which runs' fit to time again (shared: the 10 x 10 recipe at this size; ablation: the selected recipe without preconditioning)
    retime_name: str = 'fit-dense.json'  # retime: record name in the run folder (fit-dense-warm.json: a cold + warm retime of the ablation)


# ---------------------------------------------------------------------------------------------------------- energy

_OFFSETS = {}


def offsets(n, device):
    """(p,) offset index of every unordered cell pair in Q.pair_index order, and (d, 2) the offsets (dr, dc).
    Cached per (n, device)."""
    key = (n, str(device))
    if key not in _OFFSETS:
        iu = Q.pair_index(n * n, device)
        r, c = iu // n, iu % n
        dr, dc = r[1] - r[0], c[1] - c[0]
        assert bool(((dr > 0) | ((dr == 0) & (dc > 0))).all())
        code = dr * (2 * n - 1) + dc + n - 1
        uniq, idx = torch.unique(code, return_inverse=True)
        _OFFSETS[key] = (idx, torch.stack((uniq // (2 * n - 1), uniq % (2 * n - 1) - n + 1), 1))
    return _OFFSETS[key]


def pair_tables(A, b, arm, n):
    """W (p, 2, 2) per cell pair and B (c, 2) per cell from the offset parameters A and b.
    arm 'table': A (d, 2, 2), b (2,). arm 'ising': A (d,) the couplings J_d, b (1,) the unary cost h."""
    idx, _ = offsets(n, A.device)
    if arm == 'table':
        Wd, u = A, b
    else:
        z = torch.zeros_like(A)
        Wd = torch.stack((torch.stack((z, z), -1), torch.stack((z, A), -1)), -2)  # only the queen-queen entry
        u = torch.cat((torch.zeros_like(b), b))
    return Wd[idx], u.expand(n * n, 2).contiguous()


def dense_map(n, arm, device):
    """(4 c^2, k) 0/1 matrix P with M = P A.flatten(), M = rearrange(full_S(pair_tables(A)), 'i j a s -> (i a) (j s)').
    Every entry of M is one parameter or zero, so each row of P has at most one 1 and the product is exact. With
    fit(dense=True), one matrix product and its transpose replace the gather and scatter kernels of the index route
    (OffsetTask.begin), with the same M and the same gradient up to the order of a float32 sum."""
    c = n * n
    idx, _ = offsets(n, device)
    i, j = Q.pair_index(c, device)
    if arm == 'table':
        a, s = [t.flatten() for t in torch.meshgrid(torch.arange(2, device=device), torch.arange(2, device=device), indexing='ij')]
        col = (idx[:, None] * 4 + a * 2 + s).flatten()  # A_d[a, s] at d * 4 + a * 2 + s
        i, j, a, s = i[:, None].expand(-1, 4).flatten(), j[:, None].expand(-1, 4).flatten(), a.repeat(len(idx)), s.repeat(len(idx))
        k = 4 * len(torch.unique(idx))
    else:
        col, a, s = idx, torch.ones_like(i), torch.ones_like(i)  # only the queen-queen entry
        k = len(torch.unique(idx))
    P = torch.zeros(4 * c * c, k, device=device)
    P[(i * 2 + a) * 2 * c + j * 2 + s, col] = 1.0  # S[i, j, a, s] = W_p[a, s]
    P[(j * 2 + s) * 2 * c + i * 2 + a, col] = 1.0  # S[j, i, s, a] = W_p[a, s]
    return P


class OffsetTask:
    """The task interface of training_methods.loss_fn for the offset-shared energy: states v (b, i, a).
    x_in: (b, i) bool, the given queens. y: (b, i) int labels (1 queen). n: board size. arm: 'table' or 'ising'.
    dense: build the pair matrix with dense_map. Learned: A and b (shapes in pair_tables), zero start."""
    C = 2

    def __init__(self, x_in, y, n, arm, dense=False):
        self.dev, self.n, self.arm, self.c = y.device, n, arm, n * n
        self.iu = Q.pair_index(self.c, self.dev)
        d = 2 * n * (n - 1)
        self.A = torch.zeros(*((d, 2, 2) if arm == 'table' else (d,)), device=self.dev, requires_grad=True)
        self.b = torch.zeros(2 if arm == 'table' else 1, device=self.dev, requires_grad=True)
        self.Y = y
        self.ctx = {'y': y, 'free': ~x_in, 'idx': torch.arange(len(y), device=self.dev)}
        self.P = dense_map(n, arm, self.dev) if dense else None

    def params(self):
        """The learned tensors [A, b]."""
        return [self.A, self.b]

    def begin(self):
        """Build the pair matrix M (2 c, 2 c) and the unary table B (c, 2) from A and b; call once per update."""
        if self.P is not None:  # the same M, as one matrix product (dense_map)
            self.M = (self.P @ self.A.flatten()).view(2 * self.c, 2 * self.c)
            self.B = (self.b if self.arm == 'table' else torch.cat((torch.zeros_like(self.b), self.b))).expand(self.c, 2).contiguous()
            return
        W, self.B = pair_tables(self.A, self.b, self.arm, self.n)
        self.M = rearrange(Q.full_S(W, self.iu, self.c), 'i j a s -> (i a) (j s)')

    def field(self, v, ctx):
        """dE/dv (b, i, a) = sum_j S[i, j] v_j + B_i, for states v (b, i, a)."""
        return rearrange(rearrange(v, 'b i a -> b (i a)') @ self.M, 'b (i a) -> b i a', a=2) + self.B[None]

    def energy(self, v, ctx):
        """E(v) (b,) = 1/2 sum_(i, j) v_i^T S[i, j] v_j + sum_i B_i^T v_i, for states v (b, i, a)."""
        pair = rearrange(rearrange(v, 'b i a -> b (i a)') @ self.M, 'b (i a) -> b i a', a=2)
        return 0.5 * (v * pair).sum((1, 2)) + (v * self.B[None]).sum((1, 2))


def offset_summary(A, b, arm, n):
    """What the fit learned, per offset (dr, dc): the interaction contrast W11 - W10 - W01 + W00 (ising: J_d), and
    its mean over the offsets on a row, a column, a diagonal, or on no attack line (used only for this summary).
    A, b: the parameters on the CPU. Returns a JSON-serializable dict."""
    _, off = offsets(n, 'cpu')
    J = (A[:, 1, 1] - A[:, 1, 0] - A[:, 0, 1] + A[:, 0, 0]) if arm == 'table' else A
    dr, dc = off[:, 0], off[:, 1]
    line = np.where(dr == 0, 'row', np.where(dc == 0, 'col', np.where(dr.abs() == dc.abs(), 'diag', 'none')))
    J = J.cpu().numpy()
    return dict(offsets=off.tolist(), contrast=J.tolist(), unary=b.cpu().tolist(),
                contrast_mean_by_line={k: float(J[line == k].mean()) for k in ('row', 'col', 'diag', 'none')},
                contrast_min_attack=float(J[line != 'none'].min()), contrast_max_nonattack=float(J[line == 'none'].max()))


def fit(cfg, inputs, labels, n, seed, device='cuda', dense=False):
    """The complete fit, timed from GRAM's token arrays in host memory to the inference-ready model: transfer, zero
    start, UPDATES full-batch SGD updates (momentum 0.5, cosine decay), then tau, the temperature unit and delta
    (nqueens_free.ready).

    cfg: {method, arm, lr, knob, lam, precond}. inputs, labels: (b, c) token arrays (2 = queen). n: board size.
    seed: the generator of the method's random draws. device: 'cuda' or 'cpu'.
    dense: build the pair matrix M of every update as one matrix product with a fixed 0/1 matrix (dense_map) instead
    of by gather and scatter. M is the same; the gradient differs only in the order of a float32 sum. The loss, the
    rows it sums over and the optimizer are unchanged. False reproduces the fits of `final`, `shared` and `ablation`
    (the scored models); True is the route of `retime` (the reported fit times).
    Returns (model, info): model = {W, B, S, tau, unit, delta, A, b, arm} on the CPU; info = the config, fit_seconds,
    parameter counts, `diverged` and offset_summary."""
    dev = torch.device(device)
    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    x_in = torch.as_tensor(inputs, device=dev) == 2
    y = (torch.as_tensor(labels, device=dev) == 2).long()
    task = OffsetTask(x_in, y, n, cfg['arm'], dense=dense)
    gen = torch.Generator(device=dev).manual_seed(seed)
    mstate, bufs = {}, [torch.zeros_like(p) for p in task.params()]
    diverged, loss = False, None
    cnt = torch.bincount(offsets(n, dev)[0]).float()
    cnt = cnt / cnt.mean()  # pairs per offset over the mean: the step and L1 scale of each offset
    pre = cnt
    for k in range(UPDATES):
        task.begin()
        loss = AT.loss_fn(cfg['method'], cfg['knob'], task, task.ctx, gen, mstate)
        grads = torch.autograd.grad(loss, task.params())
        lr_k = cfg['lr'] * 0.5 * (1 + math.cos(math.pi * k / UPDATES))
        with torch.no_grad():
            if cfg.get('precond'):  # per-offset step: the gradient of A_d sums over its pairs, so divide by its pair count / mean
                grads = (grads[0] / pre.view(-1, *[1] * (grads[0].dim() - 1)), grads[1])
            for p, g, buf in zip(task.params(), grads, bufs):
                buf.mul_(0.5).add_(g)
                p.sub_(lr_k * buf)
            if cfg.get('lam'):
                # proximal L1 on the per-pair couplings (sum over pairs of |W_p|, as the L1 term of sudoku.py), i.e. lam * count_d
                # per offset parameter; with precond the threshold is lr_k * lam for every offset
                thr = lr_k * cfg['lam'] * (torch.ones_like(cnt) if cfg.get('precond') else cnt).view(-1, *[1] * (task.A.dim() - 1))
                task.A.copy_(task.A.sign() * (task.A.abs() - thr).clamp_min(0))
        if (k + 1) % 500 == 0 and not all(bool(torch.isfinite(p).all()) for p in task.params()):
            diverged = True
            break
    with torch.no_grad():
        A, b = task.A.detach(), task.b.detach()
        diverged = diverged or not (bool(torch.isfinite(A).all()) and bool(torch.isfinite(b).all()))
        W, B = pair_tables(A, b, cfg['arm'], n)
        S, tau, unit, delta = N.ready(W, B, y) if not diverged else (None, math.nan, math.nan, math.nan)
    if device == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    model = dict(W=W.cpu(), B=B.cpu(), S=None if S is None else S.cpu(), tau=tau, unit=unit, delta=delta,
                 A=A.cpu(), b=b.cpu(), arm=cfg['arm'])
    info = dict(cfg, fit_seconds=seconds, updates=UPDATES, seed=seed, rows=len(y), diverged=diverged, device=device, dense=dense,
                final_loss=None if loss is None else float(loss.detach()), tau=tau, unit=unit, delta=delta,
                params_learned=int(A.numel() + b.numel()), params_nonzero=int((A != 0).sum() + (b != 0).sum()))
    if not diverged:
        info['offsets'] = offset_summary(A.cpu(), b.cpu(), cfg['arm'], n)
    return model, info


# ---------------------------------------------------------------------------------------------------------- stages

def validate(model, val_in, val_lab, n, settings=SEARCH):
    """Search the validation rows (val_in, val_lab: (b, c) bool) with CHAINS chains at every (stopping test,
    temperature) of settings. Returns a list of nqueens_free.score results with `stop` and `temperature` added."""
    out = []
    for stop, t in settings:
        r = N.score(model, val_in, val_lab, n, chains=CHAINS, temperature=t, cap=N.CAP, seed=0, stop=stop)
        out.append(dict(stop=stop, temperature=t, **r))
    return out


def brief(name, info, searches):
    """Print one line for a fit: per search setting, validation N = 1 accuracy / group-mean Cov@20 in %, wrong stops
    (w), chains at the cap (c) and seconds."""
    s = ' | '.join(f"{r['stop'][0]}{r['temperature']:g}: {100 * r['acc_n1']:.2f}/{100 * r.get('cov20_group_mean', 0):.2f} "
                   f"w{r['wrong_stops']} c{r['budget_failures']} {r['search_seconds']:.0f}s" for r in searches)
    print(f"{name} fit {info['fit_seconds']:.1f}s{' DIVERGED' if info['diverged'] else ''} | {s}", flush=True)


def tune(a):
    """Every grid config of a.method and a.arm: fit on the fitting rows (seed 0), then validate().
    Writes runs/NQ/nq11/n<n>/tune/<config>/{model.pt, final.json}."""
    tr_tok, lab_tok, fit_rows, val_rows = N.data(a.n)
    val_in, val_lab = tr_tok[val_rows] == 2, lab_tok[val_rows] == 2
    for cfg in configs(a.method, a.arm):
        dest = ROOT / f'n{a.n}' / 'tune' / config_name(cfg)
        if (dest / 'final.json').exists():
            continue
        model, info = fit(cfg, tr_tok[fit_rows], lab_tok[fit_rows], a.n, seed=0)
        info.update(hostname=platform.node(), slurm_job=os.environ.get('SLURM_JOB_ID'), hardware=N.hardware())
        searches = validate(model, val_in, val_lab, a.n) if N.usable(model) else []
        N.save_torch(dest / 'model.pt', model)
        N.save(dest / 'final.json', dict(config=cfg, fit=info, searches=searches))
        brief(config_name(cfg), info, searches)


def choose(n, method, arm):
    """Selection rule: per method and arm, among the (fit, stop, T) within 0.5 points of the best validation
    N = 1, the highest validation group-mean Cov@20; ties to the shorter tuning fit, then the lower 20-chain flop."""
    cands = []
    for cfg in configs(method, arm):
        r = N.read(ROOT / f'n{n}' / 'tune' / config_name(cfg) / 'final.json')
        for s in r['searches']:
            cands.append(dict(config=cfg, stop=s['stop'], temperature=s['temperature'], cap=N.CAP, val_acc_n1=s['acc_n1'],
                              val_cov20_group_mean=s.get('cov20_group_mean', 0.0), val_cov20=s.get('cov20', 0.0),
                              val_wrong_stops=s['wrong_stops'], val_budget_failures=s['budget_failures'],
                              val_flop_all_chains=s['flop_all_chains'], tune_fit_seconds=r['fit']['fit_seconds']))
    if not cands:
        return None
    best = max(c['val_acc_n1'] for c in cands)
    near = [c for c in cands if c['val_acc_n1'] >= best - N1_SLACK]
    return dict(min(near, key=lambda c: (-c['val_cov20_group_mean'], c['tune_fit_seconds'], c['val_flop_all_chains'])),
                val_best_acc_n1=best)


def select(a):
    """Write the choice of every (method, arm) (selected-<method>-<arm>.json; its `rule` field is the docstring of
    choose) and the selected pair (selected-headline.json): among the pairs' choices within 0.5 points of the best
    validation N = 1, the highest validation group-mean Cov@20. Existing files are kept, so a choice is made once."""
    for m, arm in itertools.product(METHODS, ARMS):
        dest = ROOT / f'n{a.n}' / f'selected-{m}-{arm}.json'
        if not dest.exists():
            N.save(dest, dict(choice=choose(a.n, m, arm), rule=choose.__doc__.strip(),
                              at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
        c = N.read(dest)['choice']
        print(m, arm, None if c is None else f"{config_name(c['config'])} {c['stop']} T{c['temperature']:g} "
              f"val N=1 {100 * c['val_acc_n1']:.2f} group Cov@20 {100 * c['val_cov20_group_mean']:.2f}")
    dest = ROOT / f'n{a.n}' / 'selected-headline.json'
    if not dest.exists():
        ch = {(m, arm): N.read(ROOT / f'n{a.n}' / f'selected-{m}-{arm}.json')['choice'] for m, arm in itertools.product(METHODS, ARMS)}
        ch = {k: v for k, v in ch.items() if v is not None}
        best = max(v['val_acc_n1'] for v in ch.values())
        near = {k: v for k, v in ch.items() if v['val_acc_n1'] >= best - N1_SLACK}
        m, arm = min(near, key=lambda k: (-near[k]['val_cov20_group_mean'], near[k]['tune_fit_seconds']))
        N.save(dest, dict(method=m, arm=arm, choice=ch[(m, arm)], rule='among the methods\' and arms\' choices within 0.5 points '
                          'of the best validation N = 1, the highest validation group-mean Cov@20; ties to the shorter tuning fit',
                          at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print('headline', json.dumps({k: v for k, v in N.read(dest).items() if k != 'choice'}))


def final(a):
    """run_final with the choice of (a.method, a.arm) at this board size."""
    dest = ROOT / f'n{a.n}' / 'final' / f'{a.method}-{a.arm}-s{a.seed}'
    if (dest / 'final.json').exists():
        print((dest / 'final.json').read_text())
        return
    sel = N.read(ROOT / f'n{a.n}' / f'selected-{a.method}-{a.arm}.json')['choice']
    if sel is None:  # every tuning fit of this method and arm diverged: nothing to test (recorded, not an error)
        print(f'{a.method}-{a.arm}: no usable tuning fit (all diverged); no test run')
        return
    run_final(a, sel, dest, a.method, a.arm)


def shared_recipe():
    """One recipe for both board sizes: the method, arm, training settings, stopping test, temperature and cap that
    `select` chose at 10 x 10, unchanged; only the training data differ. Returns (method, arm, choice)."""
    h = N.read(ROOT / 'n10' / 'selected-headline.json')
    return h['method'], h['arm'], dict(h['choice'], source='n10 selected-headline.json (the 10 x 10 recipe, unchanged)')


def shared(a):
    """run_final with the 10 x 10 recipe (shared_recipe) at board size a.n; outputs in runs/NQ/nq11/n<n>/shared/."""
    m, arm, sel = shared_recipe()
    dest = ROOT / f'n{a.n}' / 'shared' / f'{m}-{arm}-s{a.seed}'
    if (dest / 'final.json').exists():
        print((dest / 'final.json').read_text())
        return
    run_final(a, sel, dest, m, arm)


def run_final(a, sel, dest, method, arm):
    """Timed fits on all training rows (one cold fit and a.repeats warm fits on the GPU, in this fresh process), then
    the test split once with 20 chains. sel: the choice {config, stop, temperature, cap}. dest: the run folder; it
    receives model.pt, fit.json, records.npz and final.json."""
    context = N.cuda_context()
    tr_tok, lab_tok = Q.load_tokens(a.n, 'train')  # file reading: outside the fit clock
    if (dest / 'fit.json').exists():  # the timed fits finished in an earlier process; only the test remains
        info = N.read(dest / 'fit.json')
        model = torch.load(dest / 'model.pt', weights_only=False)
    else:
        model, info = fit(sel['config'], tr_tok, lab_tok, a.n, seed=a.seed)
        warm = [fit(sel['config'], tr_tok, lab_tok, a.n, seed=a.seed)[1]['fit_seconds'] for _ in range(a.repeats)]
        info.update(cold_seconds=info['fit_seconds'], warm_seconds=statistics.median(warm), warm_repeats=warm,
                    cuda_context_seconds=context, hostname=platform.node(), hardware=N.hardware(),
                    slurm_job=os.environ.get('SLURM_JOB_ID'))
        N.save_torch(dest / 'model.pt', model)
        N.save(dest / 'fit.json', info)
    te_in, te_lab, _ = Q.load_split(a.n, 'test')
    res = N.score(model, te_in, te_lab, a.n, chains=CHAINS, temperature=sel['temperature'], cap=sel['cap'],
                  seed=a.seed, stop=sel['stop'], records=dest / 'records.npz')
    N.save(dest / 'final.json', dict(method=method, arm=arm, seed=a.seed, selection=sel, fit=info, test=res))
    print(json.dumps(res, indent=1))


def cpu(a):
    """The fit time on one CPU thread for the selected (method, arm): a cold fit and a.repeats warm fits in a fresh
    CPU-only process, seed 0, all training rows; the model is compared with the GPU fit of the same seed."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    h = N.read(ROOT / f'n{a.n}' / 'selected-headline.json')
    run_cpu(a, h['method'], h['arm'], h['choice']['config'], ROOT / f'n{a.n}' / 'final' / f"{h['method']}-{h['arm']}-cpu-s0.json",
            ROOT / f'n{a.n}' / 'final' / f"{h['method']}-{h['arm']}-s0" / 'model.pt')


def shared_cpu(a):
    """The shared recipe's fit on one CPU thread (cold + a.repeats warm, seed 0), compared with its GPU fit."""
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m, arm, sel = shared_recipe()
    run_cpu(a, m, arm, sel['config'], ROOT / f'n{a.n}' / 'shared' / f'{m}-{arm}-cpu-s0.json',
            ROOT / f'n{a.n}' / 'shared' / f'{m}-{arm}-s0' / 'model.pt')


def run_cpu(a, method, arm, config, dest, gpu_model):
    """1 + a.repeats fits of config on the CPU; writes dest (JSON: cold and warm seconds, hardware, and the largest
    parameter difference from the GPU-fitted model at the path gpu_model)."""
    if dest.exists():
        print(dest.read_text())
        return
    tr_tok, lab_tok = Q.load_tokens(a.n, 'train')
    runs = [fit(config, tr_tok, lab_tok, a.n, seed=0, device='cpu') for _ in range(1 + a.repeats)]
    model, info = runs[0]
    gpu = torch.load(gpu_model, weights_only=False)
    N.save(dest, dict(method=method, arm=arm, cold_seconds=info['fit_seconds'],
                      warm_seconds=statistics.median([r[1]['fit_seconds'] for r in runs[1:]]),
                      warm_repeats=[r[1]['fit_seconds'] for r in runs[1:]], threads=torch.get_num_threads(),
                      hostname=platform.node(), hardware=N.hardware() if torch.cuda.is_available() else platform.processor(),
                      cpu_model=next((l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines() if l.startswith('model name')), ''),
                      slurm_job=os.environ.get('SLURM_JOB_ID'),
                      max_abs_diff_vs_h200=dict(A=float((model['A'] - gpu['A']).abs().max()), b=float((model['b'] - gpu['b']).abs().max()))))
    print(dest.read_text())


def ablation(a):
    """The selected configuration without per-offset preconditioning: one cold fit on the GPU, then the test split
    once (seed a.seed) with the selected stopping test and temperature."""
    h = N.read(ROOT / f'n{a.n}' / 'selected-headline.json')
    sel = dict(h['choice'], config=dict(h['choice']['config'], precond=False))
    dest = ROOT / f'n{a.n}' / 'ablation' / f"{h['method']}-{h['arm']}-noprecond-s{a.seed}"
    if (dest / 'final.json').exists():
        print((dest / 'final.json').read_text())
        return
    N.cuda_context()
    tr_tok, lab_tok = Q.load_tokens(a.n, 'train')
    model, info = fit(sel['config'], tr_tok, lab_tok, a.n, seed=a.seed)
    te_in, te_lab, _ = Q.load_split(a.n, 'test')
    res = N.score(model, te_in, te_lab, a.n, chains=CHAINS, temperature=sel['temperature'], cap=sel['cap'],
                  seed=a.seed, stop=sel['stop'], records=dest / 'records.npz')
    N.save(dest / 'final.json', dict(method=h['method'], arm=h['arm'], seed=a.seed, selection=sel, fit=info, test=res))
    print(json.dumps(res, indent=1))


def retime(a):
    """The timed fits of one finished run (stage final, shared or ablation; a.method, a.arm, a.seed; the method and
    arm are the selected pair's for ablation and the 10 x 10 recipe's for shared) again with fit(dense=True), in a
    fresh process: the cold fit and a.repeats warm fits on the GPU (the ablation's record has a cold fit only: run it
    with --repeats 0). The config is the run's own; the test result stays the run's, and fit-dense.json records the
    largest parameter difference from the scored model.pt. These are the fit times that the paper reports."""
    if a.stage == 'final':
        name = f'{a.method}-{a.arm}'
    elif a.stage == 'shared':
        m, arm, _ = shared_recipe()
        name = f'{m}-{arm}'
    else:
        h = N.read(ROOT / f'n{a.n}' / 'selected-headline.json')
        name = f"{h['method']}-{h['arm']}-noprecond"
    run = ROOT / f'n{a.n}' / a.stage / f'{name}-s{a.seed}'
    dest = run / a.retime_name
    if dest.exists():
        print(dest.read_text())
        return
    if not (run / 'final.json').exists():  # a (method, arm) without a usable tuning fit has no run
        print(f'{run.relative_to(N.HERE)}: no run; nothing to time')
        return
    cfg = N.read(run / 'final.json')['selection']['config']
    context = N.cuda_context()
    tr_tok, lab_tok = Q.load_tokens(a.n, 'train')  # file reading: outside the fit clock
    runs = [fit(cfg, tr_tok, lab_tok, a.n, seed=a.seed, dense=True) for _ in range(1 + a.repeats)]
    model, info = runs[0]
    old = N.read(run / 'fit.json') if (run / 'fit.json').exists() else N.read(run / 'final.json')['fit']
    if (run / 'model.pt').exists():
        ref = torch.load(run / 'model.pt', weights_only=False)
    else:  # the ablation stage saved no model: its record keeps tau, unit, delta and the per-offset contrast (ising: A itself)
        ref = {k: old[k] for k in ('tau', 'unit', 'delta')}

    def diff(x, y):
        x, y = x.double(), y.double()
        both = torch.isfinite(x) & torch.isfinite(y)
        return dict(max_abs=float((x - y)[both].abs().max()) if both.any() else None, ref_max_abs=float(y[both].abs().max()) if both.any() else None,
                    nonfinite_match=bool(torch.equal(torch.isfinite(x), torch.isfinite(y))))

    out = dict(config=cfg, stage=a.stage, run=str(run.relative_to(N.HERE)), seed=a.seed, dense=True,
               cold_seconds=info['fit_seconds'], warm_seconds=statistics.median([r[1]['fit_seconds'] for r in runs[1:]]) if a.repeats else None,
               warm_repeats=[r[1]['fit_seconds'] for r in runs[1:]], cuda_context_seconds=context, diverged=info['diverged'],
               hostname=platform.node(), hardware=N.hardware(), slurm_job=os.environ.get('SLURM_JOB_ID'),
               index_route=dict(cold_seconds=old.get('cold_seconds', old.get('fit_seconds')), warm_seconds=old.get('warm_seconds'),
                                hostname=old.get('hostname'), slurm_job=old.get('slurm_job')),
               vs_scored_model=dict(ready={k: [model[k], ref[k]] for k in ('tau', 'unit', 'delta')}, diverged_scored=old.get('diverged')))
    if 'A' in ref:
        out['vs_scored_model'].update(A=diff(model['A'], ref['A']), b=diff(model['b'], ref['b']))
    elif old.get('offsets') and info.get('offsets'):
        out['vs_scored_model'].update(source='final.json fit.offsets (no model.pt)',
                                      contrast=diff(torch.tensor(info['offsets']['contrast']), torch.tensor(old['offsets']['contrast'])),
                                      unary=diff(torch.tensor(info['offsets']['unary']), torch.tensor(old['offsets']['unary'])))
    N.save(dest, out)
    print(json.dumps(out, indent=1))


def main(a: Args):
    """Run the stage a.mode. TF32 matrix products are off (plain float32 products)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    {'tune': tune, 'select': select, 'final': final, 'cpu': cpu, 'ablation': ablation, 'shared': shared,
     'shared-cpu': shared_cpu, 'retime': retime}[a.mode](a)


if __name__ == '__main__':
    main(tyro.cli(Args))
