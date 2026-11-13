"""Training-method study: the same CRAM energies trained by different methods (the paper's training-method figure and
its appendix table). Run from experiments/.

  python training_methods.py train --task sudoku --method cd --lr 180 --knob 81 --seed 0 --out runs/ALT/tune/...  # one run
  python training_methods.py select --task sudoku --method cd                                            # tuning choice
  python training_methods.py test --run runs/ALT/final/sudoku-cd-s0                                     # search tuning, test
  python training_methods.py plan --stage tune                                                          # job lines
  python training_methods.py count --tasks sudoku                                                       # arithmetic per update
  python training_methods.py export                                                                     # ../results/alt-training/alt-training.json

Every arm trains the energy of the task's CRAM row in Table 2 and changes only the training method:
- sudoku (Sudoku-Extreme): the energy of sudoku.py, a free 9 x 9 block W_p per unordered cell pair and a free input
  cost B_i per cell (263,169 learned parameters),
      E(v) = 1/2 sum_(n != m) v_n^T S[n, m] v_m + sum_n B_n^T v_n,   S[n, m] = W_p, S[m, n] = W_p^T for the pair p = (n < m).
  Zero start, clues clamped, a fresh digit relabeling at every update, full batch of the 1,000 puzzles of
  train-s<seed>, SGD with momentum 0.5 and cosine decay, and the L1 term of sudoku.py as a proximal soft-threshold
  after each update (every arm).
- maze (Maze-Unique): a translation-invariant kernel K over the two cell states (off path, on path) in a 3 x 3
  window and a bias table b per input token (cram/cram_maze.py),
      E(v) = 1/2 sum_n v_n^T (K * v)_n - sum_n b[x_n]^T v_n,   * the 2D convolution over the 30 x 30 grid.
  Walls clamped off path, S and G clamped on path, batches of 64 of the 900 training mazes.

The Maze-Unique setting is selected by two environment variables, read at import:
  ALT_MAZE_KERNEL  free: the free 3 x 3 kernel (CRAMMazeFreeEnergy, 18 parameters, zero start).
                   separable (default): the separable kernel (CRAMMazeEnergy, 16 parameters, seeded start).
  ALT_MAZE_OPT     sgd-cosine: SGD, momentum 0.5, cosine decay, no gradient clip (needs ALT_MAZE_KERNEL=free).
                   adam (default): Adam, constant learning rate, gradient-norm clip 1.0.
The paper reports ALT_MAZE_KERNEL=free ALT_MAZE_OPT=sgd-cosine (the kernel and optimizer of the Table 2 row
cram-sgd-fg); slurm/training_methods/sgd.sbatch sets both. The defaults select an earlier setting that the paper does
not report. They stay the defaults so that an existing command line computes what it computed before. Each setting
has its own run folder (runs/ALT, runs/ALT-free, runs/ALT-sgd), so the settings never share a folder; Sudoku runs are
always under runs/ALT. The folder names are only labels.

Dimension names: b boards, n and m cells (81 or 900), a and c answer states (9 digits or 2 path states),
p unordered cell pairs, h and w grid rows and columns (maze).
All states are (b, n, a): v_n on the simplex at free cells, the one-hot label at clamped cells.

Methods (class L: a local objective at the label, no inference; class S: negatives drawn from the model, not
differentiated; class D: needs relaxations of the mean-field flow u <- u + eta (-dE/dv - u), v = softmax(u), during
training). PL is pseudolikelihood, CD contrastive divergence, BPTT backpropagation through time. `knob` is the one
method parameter that the tuning grid varies:
  pl    L  pseudolikelihood: -log softmax(-L_i)[y_i], L_i the local energies of cell i at the labeled board
  mpl   L  masked pseudolikelihood: each other cell is hidden (state 0) with probability knob, PL on the shown cells
  mpf   L  minimum probability flow: sum over single-cell changes of exp((E(y) - E(y'))/2)
  dsm   L  denoising score matching: -dE/dv at v = softmax(u* + knob xi) matches the clean logits u* = 3 (y - 1/C)
  cd    S  contrastive divergence: E(y) - E(y-), y- after knob heat-bath moves from the label
  pcd   S  persistent CD: one persistent chain per training board, started at its label, knob moves per update
  chl   S  contrastive Hebbian learning: E(y) - E(v-), v- the free relaxation (knob steps) from the uniform state
  eqp   D  equilibrium propagation: symmetric nudges +-0.01 of the free relaxation (knob steps), cross-entropy cost
  imp   D  implicit differentiation: cross-entropy at the relaxed state (knob steps), adjoint by fixed-point iteration
  tbptt D  truncated BPTT: 30 relaxation steps, backpropagation through the last knob of them (Sudoku only)
  bptt1 D  one-step backpropagation: knob relaxation steps, backpropagation through the last one (HRM's one-step gradient)
  bptt5 D  truncated BPTT through the last 5 of knob relaxation steps
  bptt  D  full BPTT: knob relaxation steps, backpropagation through all of them
"""
import dataclasses
import json
import math
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tyro
from einops import rearrange

HERE = Path(__file__).resolve().parent
# Maze energy. ALT_MAZE_KERNEL=free trains the free 3x3 kernel (CRAMMazeFreeEnergy, 18 parameters, zero start) instead
# of the separable kernel (CRAMMazeEnergy, 16 parameters), and puts every maze run under runs/ALT-free/ (same layout as
# runs/ALT/), so that no separable-kernel run is overwritten. Sudoku runs always stay under runs/ALT/. The exports
# combine the Sudoku rows of runs/ALT/ with the maze rows of the chosen root.
MAZE_KERNEL = os.environ.get('ALT_MAZE_KERNEL', 'separable')
assert MAZE_KERNEL in ('separable', 'free'), MAZE_KERNEL
# Maze optimizer. ALT_MAZE_OPT=sgd-cosine (with ALT_MAZE_KERNEL=free) trains every maze arm with SGD, momentum 0.5 and
# cosine decay, as the Sudoku arms, with its own grids (MAZE_SGD below) and every maze run under runs/ALT-sgd/;
# runs/ALT/ and runs/ALT-free/ are not touched. This is the setting that the paper reports. Default: adam.
MAZE_OPT = os.environ.get('ALT_MAZE_OPT', 'adam')
assert MAZE_OPT in ('adam', 'sgd-cosine'), MAZE_OPT
assert MAZE_OPT == 'adam' or MAZE_KERNEL == 'free', 'ALT_MAZE_OPT=sgd-cosine needs ALT_MAZE_KERNEL=free'
REL = {'sudoku': 'runs/ALT',
       'maze': 'runs/ALT-sgd' if MAZE_OPT == 'sgd-cosine' else 'runs/ALT' if MAZE_KERNEL == 'separable' else 'runs/ALT-free'}


def root(task):
    """The run folder (Path) of a task ('sudoku' or 'maze'): runs/ALT; for the maze runs/ALT-free
    (ALT_MAZE_KERNEL=free) or runs/ALT-sgd (ALT_MAZE_KERNEL=free and ALT_MAZE_OPT=sgd-cosine)."""
    return HERE / REL[task]


MAZE_FLOP_SUFFIX = '-maze-sgd' if MAZE_OPT == 'sgd-cosine' else '-maze-free'  # suffix of the maze operation-count files: -maze-sgd for the paper's setting


def maze_root_of(kernel, opt):
    """The run folder prefix (str) that a maze run of this kernel and optimizer must be under (train refuses any other)."""
    if opt == 'sgd-cosine':
        return 'runs/ALT-sgd/'
    return 'runs/ALT-free/' if kernel == 'free' else 'runs/ALT/'
ETA = 0.5  # Euler step size eta of the mean-field flow

METHODS = ('pl', 'mpl', 'mpf', 'dsm', 'cd', 'pcd', 'chl', 'eqp', 'imp', 'tbptt', 'bptt1', 'bptt5', 'bptt')
CLASS = dict(pl='L', mpl='L', mpf='L', dsm='L', cd='S', pcd='S', chl='S', eqp='D', imp='D', tbptt='D', bptt1='D', bptt5='D', bptt='D')
# Tuning grids: learning rates x two values of the method's own parameter (knob). Sudoku: the SGD learning rate of
# sudoku.py (540) and factors 1/9 and 1/90 (MPF and DSM diverge at 540); maze with the default Adam setting: 0.03 and a
# factor 3 either side. ALT_MAZE_OPT=sgd-cosine replaces the maze entries of LRS, L1S and REF with MAZE_SGD (below).
KNOBS = {
    'sudoku': dict(pl=(None,), mpl=(0.1, 0.3), mpf=(None,), dsm=(0.5, 1.5), cd=(81, 324), pcd=(81, 324), chl=(10, 30),
                   eqp=(10, 30), imp=(10, 30), tbptt=(1, 5), bptt1=(10, 30), bptt5=(10, 30), bptt=(10, 30)),
    'maze': dict(pl=(None,), mpl=(0.1, 0.3), mpf=(None,), dsm=(0.5, 1.5), cd=(256, 1024), pcd=(256, 1024),
                 chl=(30, 100), eqp=(30, 100), imp=(30, 100), tbptt=(1, 5), bptt1=(10, 30), bptt5=(10, 30), bptt=(10, 30)),
}
LRS = {'sudoku': (6.0, 60.0, 540.0), 'maze': (0.01, 0.03, 0.1)}
UPDATES = {  # updates of the full schedule per class; the reference recipes keep their own counts
    'sudoku': dict(L=2000, S=2000, D=2000), 'maze': dict(L=2000, S=1500, D=2000)}  # each fits in TUNE_CAP_S
REF = {'sudoku': dict(lr=540.0, updates=32), 'maze': dict(lr=0.03, updates=100)}  # pl keeps the learning rate and update count of the task's own recipe (Sudoku: sudoku.py)
TUNE_CAP_S = 7200.0  # fit seconds of a tuning run (a safety cap: up to 6 runs share a GPU, so the slowest schedule takes about 3,000 s)
FINAL_CAP_S = 3600.0  # fit seconds of a final run (one H200-hour)


# ---------------------------------------------------------------------------------------------------------- tasks
class Sudoku:
    """The Sudoku energy as a task for loss_fn and train: W (p, 9, 9) and B (81, 9), states v (b, 81, 9).

    seed: the training split train-s<seed>. device: torch device. n_train: train on a seeded subset of this many
    puzzles (0: all). aug: a fresh digit relabeling at every update. pool: an .npz of training puzzles to use
    instead of the split. kernel: ignored (maze only)."""
    C = 9

    def __init__(self, seed, device, n_train=0, aug=True, pool='', kernel='separable'):
        from cram import cram_sudoku_search as SS
        import sudoku as FS
        self.FS, self.SS = FS, SS
        self.dev = device
        self.aug = aug
        if pool:  # another training pool, stored in the format of the train-s<k> files; file reading: outside the fit clock
            with np.load(HERE / pool) as f:
                self.host = (f['solutions'].astype(np.int64) - 1, ~f['clue_mask'].astype(bool))
        else:
            self.host = SS.load_puzzles(f'train-s{seed}')[:2]  # file reading: outside the fit clock
        if n_train:  # nested subsets of the split, in a seeded order
            keep = np.random.default_rng(1000 + seed).permutation(len(self.host[0]))[:n_train]
            self.host = tuple(a[keep] for a in self.host)

    def setup(self):
        """Transfer and initialization: inside the fit clock."""
        device = self.dev
        sol, free = self.host
        self.sol, self.free = torch.as_tensor(sol, device=device), torch.as_tensor(free, device=device)
        self.iu = self.FS.pair_index(device)
        self.W = torch.zeros(self.iu.shape[1], 9, 9, device=device, requires_grad=True)
        self.B = torch.zeros(81, 9, device=device, requires_grad=True)
        self.lam = 3e-3 / 9  # the L1 weight per entry of sudoku.py
        self.n_train = len(self.sol)

    def params(self):
        """The learned tensors [W (p, 9, 9), B (81, 9)]."""
        return [self.W, self.B]

    def batch(self, gen):
        """Full batch with a fresh digit relabeling. The relabeling is drawn from the default CUDA generator, which
        train seeds as sudoku.py does (so the pl arm repeats the relabelings of sudoku.py); the methods' own draws use
        `gen`. Without augmentation the relabeling is the identity, and no draw is made.
        Returns the batch context: 'y' (b, 81) labels 0-8, 'free' (b, 81) bool (not a clue), 'perm' (b, 9) the
        relabeling, 'idx' None."""
        if not self.aug:
            perm = torch.arange(9, device=self.dev).expand(self.n_train, 9)
            return {'y': self.sol, 'free': self.free, 'perm': perm, 'idx': None}
        perm = torch.rand(self.n_train, 9, device=self.dev).argsort(1)
        return {'y': perm.gather(1, self.sol), 'free': self.free, 'perm': perm, 'idx': None}

    def begin(self):
        """Build the symmetric pair matrix M (81 * 9, 81 * 9) from W; call once per update, before field or energy."""
        self.M = rearrange(self.FS.full_S(self.W, self.iu), 'n m a c -> (n a) (m c)')

    def field(self, v, ctx):
        """dE/dv (b, n, a) = sum_m S[n, m] v_m + B_n, for states v (b, n, a)."""
        return rearrange(rearrange(v, 'b n a -> b (n a)') @ self.M, 'b (n a) -> b n a', a=9) + self.B[None]

    def energy(self, v, ctx):
        """E(v) (b,) = 1/2 sum_(n, m) v_n^T S[n, m] v_m + sum_n B_n^T v_n, for states v (b, n, a)."""
        pair = rearrange(rearrange(v, 'b n a -> b (n a)') @ self.M, 'b (n a) -> b n a', a=9)
        return 0.5 * (v * pair).sum((1, 2)) + (v * self.B[None]).sum((1, 2))

    @torch.no_grad()
    def step(self, lr, bufs, momentum=0.5):
        """One SGD update with momentum from the gradients in p.grad, then the proximal L1 step (soft-threshold
        lr * lam). bufs: the momentum buffers, one per parameter, updated in place."""
        for p, buf in zip(self.params(), bufs):
            buf.mul_(momentum).add_(p.grad)
            p.sub_(lr * buf)
            p.copy_(p.sign() * (p.abs() - lr * self.lam).clamp_min(0))
            p.grad = None

    def state(self):
        """A copy of the parameters, {'W', 'B'}, as saved in model-u<k>.pt."""
        return {'W': self.W.detach().clone(), 'B': self.B.detach().clone()}

    def load(self, sd):
        """Copy the parameters from a dict that state() returned."""
        with torch.no_grad():
            self.W.copy_(sd['W']); self.B.copy_(sd['B'])


class Maze:
    """The Maze-Unique energy as a task for loss_fn and train: states v (b, 900, 2) over the 30 x 30 grid.

    seed: seeds the model start and the batch order. device: torch device. n_train: train on a seeded subset of this
    many mazes (0: all 900). kernel: 'free' (CRAMMazeFreeEnergy, 18 parameters, zero start) or 'separable'
    (CRAMMazeEnergy, 16 parameters, seeded start)."""
    C = 2

    def __init__(self, seed, device, n_train=0, aug=True, pool='', kernel='separable'):  # aug and pool: sudoku only
        from cram import cram_maze as CM
        assert kernel in ('separable', 'free'), kernel
        self.kernel = kernel
        from cram import maze_unique as M
        import maze_unique as U
        self.M, self.U, self.CM = M, U, CM
        self.dev, self.seed = device, seed
        self.host = M.load('train900')  # file reading: outside the fit clock
        if n_train:  # nested subsets of train900, in a seeded order
            keep = np.random.default_rng(1000 + seed).permutation(len(self.host['inputs']))[:n_train]
            self.host = {k: v[keep] for k, v in self.host.items()}

    def setup(self):
        """Transfer and initialization: inside the fit clock."""
        M, U, CM, device, seed, d = self.M, self.U, self.CM, self.dev, self.seed, self.host
        self.X = torch.as_tensor(d['inputs'], device=device).view(-1, M.SIDE * M.SIDE)
        self.Y = torch.as_tensor(d['targets'], device=device).view(-1, M.SIDE * M.SIDE)
        self.FR = torch.as_tensor(d['free'], device=device).view(-1, M.SIDE * M.SIDE)
        torch.manual_seed(seed)  # the seeded start of the separable kernel, as maze_unique.make_model
        if self.kernel == 'free':  # zero start; the seed still sets the batch order and every training generator
            self.model = CM.CRAMMazeFreeEnergy(window=1).to(device)
        else:
            self.model = CM.CRAMMazeEnergy(R=1, de=2, window=1).to(device)
        self.n_train = len(self.X)
        self.rng = np.random.default_rng(seed)  # the batch order of maze_unique.py
        self.stream = U.batches(self.n_train, min(64, self.n_train), self.rng)

    def params(self):
        """The learned tensors of the model (kernel and bias parameters)."""
        return list(self.model.parameters())

    def batch(self, gen):
        """The next batch of 64 mazes. Returns the batch context: 'y' (b, 900) labels (1 on the path), 'free' (b, 900)
        bool (not a wall, S or G), 'x' (b, 30, 30) input tokens, 'idx' (b,) the mazes' indices in the training set."""
        idx = torch.as_tensor(next(self.stream), device=self.dev)
        return {'y': self.Y[idx], 'free': self.FR[idx], 'x': self.X[idx].view(-1, self.M.SIDE, self.M.SIDE), 'idx': idx}

    def begin(self):
        """Build the kernel K (2, 2, 3, 3) from the parameters; call once per update, before field or energy."""
        self.K = self.model.kernel()

    def _s(self, v):
        return rearrange(v, 'b (h w) a -> b a h w', h=self.M.SIDE)

    def field(self, v, ctx):
        """dE/dv (b, n, a) for states v (b, n, a) and the inputs ctx['x'] (the model's `current`)."""
        return rearrange(self.model.current(self._s(v), ctx['x'], self.K), 'b a h w -> b (h w) a')

    def energy(self, v, ctx):
        """E(v) (b,) = 1/2 sum_n v_n^T (K * v)_n - sum_n b[x_n]^T v_n, for states v (b, n, a)."""
        s = self._s(v)
        pair = 0.5 * (F.conv2d(s, self.K, padding=self.model.w) * s).sum((1, 2, 3))
        bias = (s * rearrange(self.model.bias_table()[ctx['x']], 'b h w a -> b a h w')).sum((1, 2, 3))
        return pair - bias

    def state(self):
        """A copy of the model's state dict, as saved in model-u<k>.pt."""
        return {k: v.detach().clone() for k, v in self.model.state_dict().items()}

    def load(self, sd):
        """Load a state dict that state() returned."""
        self.model.load_state_dict(sd)


TASKS = {'sudoku': Sudoku, 'maze': Maze}


# -------------------------------------------------------------------------------------------------------- methods
def onehot(y, C):
    """Labels y (b, n) int -> one-hot float states (b, n, C)."""
    return F.one_hot(y, C).float()


def clamp(v, ctx, C):
    """States v (b, n, C) with every clamped (not free) cell set to its one-hot label."""
    return torch.where(ctx['free'][..., None], v, onehot(ctx['y'], C))


def free_mean(x, free):
    """Mean of x (b, n) over the cells where free (b, n) is true; a scalar."""
    return (x * free).sum() / free.sum()


def relax(task, ctx, u, steps, nudge=0.0, cost='ce'):
    """`steps` Euler steps of the mean-field flow u <- u + ETA (-dE/dv - u), v = softmax(u), at the free cells.

    task: Sudoku or Maze. ctx: the batch context. u: logits (b, n, a). nudge: adds nudge * dl/dv to dE/dv at the free
    cells (equilibrium propagation), l the cross-entropy -sum y log v ('ce', the cost of imp and the BPTT methods) or
    1/2 |v - y|^2 ('mse'). Returns the logits u (b, n, a) after the steps. Differentiable in the parameters."""
    C = task.C
    Y = onehot(ctx['y'], C)
    fr = ctx['free'][..., None]
    for _ in range(steps):
        v = clamp(torch.softmax(u, -1), ctx, C)
        g = task.field(v, ctx)
        if nudge:
            dl = -Y / v.clamp_min(1e-3) if cost == 'ce' else v - Y
            g = g + nudge * dl * fr
        u = torch.where(fr, u + ETA * (-g - u), u)
    return u


@torch.no_grad()
def heat_bath(task, ctx, board, moves, gen):
    """`moves` random-scan heat-bath moves at temperature 1 per board: each move draws one free cell uniformly and
    then its state from softmax(-local energies). board: (b, n) int states. gen: the generator of the draws.
    Returns the boards (b, n) after the moves."""
    C = task.C
    fr = ctx['free'].float()
    ar = torch.arange(len(board), device=board.device)
    for _ in range(moves):
        cell = torch.multinomial(fr, 1, generator=gen).squeeze(1)
        L = task.field(onehot(board, C), ctx)[ar, cell]
        new = torch.multinomial(torch.softmax(-L, -1), 1, generator=gen).squeeze(1)
        board = board.clone()
        board[ar, cell] = new
    return board


def contrast(ep, en, free, norm, mstate):
    """The contrastive loss sum(E(data) - E(negative)) / norm, plus the energy-magnitude regularizer of Du & Mordatch
    (2019), alpha mean((E+/n)^2 + (E-/n)^2), n the free cells of each board (alpha = mstate['_reg'], 0: none).

    ep, en: energies (b,) of the data and the negative boards. free: (b, n) bool. norm: a scalar, the number of free
    cells. Without the regularizer the contrastive objective is unbounded below; on the maze with Adam and the
    separable kernel the couplings grew to 1e4."""
    loss = (ep - en).sum() / norm
    a = mstate.get('_reg', 0.0)
    if a:
        nb = free.sum(1).clamp_min(1)
        loss = loss + a * ((ep / nb) ** 2 + (en / nb) ** 2).mean()
    return loss


def loss_fn(method, knob, task, ctx, gen, mstate):
    """A scalar whose gradient with respect to the parameters is the method's update direction.

    method: one of METHODS (see the module docstring). knob: the method's own parameter. task: Sudoku, Maze or
    another object with C, field and energy. ctx: the batch context (task.batch). gen: the generator of the method's
    random draws. mstate: a dict that persists across updates ('_reg' the regularizer weight; pcd keeps its chains
    in it)."""
    C, y, free = task.C, ctx['y'], ctx['free']
    Y = onehot(y, C)
    if method == 'pl':
        L = task.field(Y, ctx)
        return free_mean(-torch.log_softmax(-L, -1).gather(-1, y[..., None]).squeeze(-1), free)
    if method == 'mpl':
        shown = torch.rand(free.shape, device=y.device, generator=gen) >= knob
        L = task.field(Y * shown[..., None], ctx)
        nll = -torch.log_softmax(-L, -1).gather(-1, y[..., None]).squeeze(-1)
        return free_mean(nll, free & shown)
    if method == 'mpf':
        L = task.field(Y, ctx)
        gap = L - L.gather(-1, y[..., None])  # E(y with i <- a) - E(y)
        k = torch.exp(-gap / 2) * (1 - Y)
        return free_mean(k.sum(-1), free)
    if method == 'dsm':
        ustar = 3.0 * (Y - 1.0 / C)
        u = ustar + knob * torch.randn(Y.shape, device=y.device, generator=gen)
        g = task.field(clamp(torch.softmax(u, -1), ctx, C), ctx)
        d = -(g - g.mean(-1, keepdim=True)) - ustar
        return free_mean(d.pow(2).sum(-1), free)
    norm = free.sum()  # energy differences per free cell, so that the SGD scale matches PL's
    if method in ('cd', 'pcd'):
        if method == 'cd':
            start = torch.where(free, y, y)
        else:  # persistent chains of the training boards, kept in the original digit labels (sudoku) or per maze
            key = 'chains'
            if key not in mstate:  # one chain per training board, started at its label (from random boards the
                # maze chains, each moved only when its maze is in the batch, stayed near random)
                mstate[key] = (task.sol if 'perm' in ctx else task.Y).clone()
            ch = mstate[key]
            start = ctx['perm'].gather(1, ch) if 'perm' in ctx else ch[ctx['idx']]
            start = torch.where(free, start, y)
        neg = heat_bath(task, ctx, start, knob, gen)
        if method == 'pcd':
            if 'perm' in ctx:
                mstate['chains'] = ctx['perm'].argsort(1).gather(1, neg)
            else:
                mstate['chains'][ctx['idx']] = neg
        return contrast(task.energy(Y, ctx), task.energy(onehot(neg, C), ctx), free, norm, mstate)
    zero = torch.zeros(Y.shape, device=y.device)
    if method == 'chl':
        with torch.no_grad():
            vneg = clamp(torch.softmax(relax(task, ctx, zero, knob), -1), ctx, C)
        return contrast(task.energy(Y, ctx), task.energy(vneg, ctx), free, norm, mstate)
    if method == 'eqp':
        beta = 0.01  # the strength at which the cross-entropy nudge matches the implicit-differentiation gradient (0.5 diverged)
        with torch.no_grad():
            u0 = relax(task, ctx, zero, knob)
            vp = clamp(torch.softmax(relax(task, ctx, u0, knob, nudge=beta), -1), ctx, C)
            vm = clamp(torch.softmax(relax(task, ctx, u0, knob, nudge=-beta), -1), ctx, C)
        return (task.energy(vp, ctx) - task.energy(vm, ctx)).sum() / (2 * beta * norm)
    if method == 'imp':
        with torch.no_grad():
            us = relax(task, ctx, zero, knob)
        us = us.detach().requires_grad_(True)
        f = lambda u: -task.field(clamp(torch.softmax(u, -1), ctx, C), ctx) * ctx['free'][..., None]
        fu = f(us)  # the fixed-point map u = f(u); its graph reaches the parameters and us
        ell = free_mean(-torch.log_softmax(us, -1).gather(-1, y[..., None]).squeeze(-1), free)
        gl = torch.autograd.grad(ell, us)[0]
        lam = gl.clone()
        for _ in range(knob):  # damped iteration of lam = gl + J^T lam (the forward relaxation's damping)
            jt = torch.autograd.grad(fu, us, lam, retain_graph=True)[0]
            lam = (1 - ETA) * lam + ETA * (gl + jt)
        return (lam.detach() * fu).sum()
    if method in ('tbptt', 'bptt1', 'bptt5', 'bptt'):  # T total steps, backpropagation through the last K
        T, K = (30, knob) if method == 'tbptt' else (knob, {'bptt1': 1, 'bptt5': 5}.get(method, knob))
        with torch.no_grad():
            u = relax(task, ctx, zero, T - K)
        u = relax(task, ctx, u, K)
        return free_mean(-torch.log_softmax(u, -1).gather(-1, y[..., None]).squeeze(-1), free)
    raise ValueError(method)


# ---------------------------------------------------------------------------------------------- probe and search
SUDOKU_SEARCH = dict(C=4, check=4096, tol=1e-3, period=524288, hi=0.40, lo=0.31, theta=0.45)  # the search settings of sudoku.py; temperatures in units of hbar, the mean positive coupling
PROBE = dict(sudoku=dict(n=512, cap=4e10), maze=dict(split='dev100'))  # the probe: held-out puzzles scored during training; cap in flop per puzzle
TEST = dict(sudoku=dict(lo=10000, n=50000, cap=4e11), maze=dict(split='test'))  # cap 4e11 flop per puzzle, as in Table 2


def margin_scale(S, B, sol):
    """The search unit of an energy: the mean, over the training solutions' free and clamped cells, of the gap between
    the lowest local energy of another digit and the local energy of the cell's own digit (clamped below at 1e-4).
    For the energy that sudoku.py fits it is a fixed multiple of hbar, the mean positive coupling.
    S: (81, 81, 9, 9) pair blocks. B: (81, 9). sol: (b, 81) solutions, digits 0-8. Returns a float."""
    from cram import cram_sudoku_general as GS
    L = GS.local_energies(S, B, sol)
    own = L.gather(2, sol[..., None].long())
    other = torch.where(F.one_hot(sol.long(), 9).bool(), torch.inf, L).min(2, keepdim=True).values
    return max(float((other - own).mean()), 1e-4)


def hbar_ratio():
    """hbar / margin_scale on the three Sudoku fits of sudoku.py (runs/CRAM-S/full/full-aug-s*), so that the probe's
    temperature schedule on those fits is the tuned schedule of sudoku.py; one constant for every arm.
    Returns (the mean over the three fits, the three values)."""
    from cram import cram_sudoku_search as SS
    import sudoku as FS
    r = []
    for s in range(3):
        W, B = FS.load_model(s)
        S = FS.full_S(W, FS.pair_index('cuda'))
        sol = torch.as_tensor(SS.load_puzzles(f'train-s{s}')[0], device='cuda')
        r.append(float(W[W > 0].mean()) / margin_scale(S, B, sol))
    return statistics.fmean(r), r


def sudoku_ready(task, sol):
    """The inference-ready model as a dict: S (81, 81, 9, 9), B (81, 9), 'unit' (margin_scale), 'ebar' (the mean
    local energy of a cell's own digit, used by the focused search) and the stopping threshold 'tau' (the highest energy of a training solution), all fitted from the
    training solutions sol (b, 81)."""
    from cram import cram_sudoku_general as GS
    W, B = task.W.detach(), task.B.detach()
    S = task.FS.full_S(W, task.iu)
    tau, ebar = GS.solution_stats(S, B, sol)
    return {'S': S, 'B': B, 'unit': margin_scale(S, B, sol), 'tau': tau, 'ebar': ebar}


def sudoku_search(m, sol, free, cap, seed, mult=1.0, ratio=None):
    """The focused Gibbs search of sudoku.py with the model m (sudoku_ready) on puzzles sol (b, 81), free (b, 81).

    cap: flop per puzzle before the search gives up. mult: factor on the temperature schedule. ratio: hbar_ratio()[0].
    Returns (summary dict: n, solved, stopped, wrong_stops, budget_failures, mean_flop; the raw result with the
    per-puzzle 'flops'; {'D', 'fl_move'}: the number of coupled neighbour cells per cell and the flop per move)."""
    from cram import cram_sudoku_general as GS
    from cram import cram_sudoku_search as SS
    c = SUDOKU_SEARCH
    if not bool((m['S'] != 0).any()):  # no coupled pair (the L1 step zeroed every cost): nothing to search with
        n = len(sol)
        return ({'n': n, 'solved': 0, 'stopped': 0, 'wrong_stops': 0, 'budget_failures': n, 'mean_flop': 0.0,
                 'empty_support': True}, {'flops': [0.0] * n}, {'D': 0, 'fl_move': 0.0})
    unit = m['unit'] * ratio
    ss = GS.GeneralSearch(m['S'], m['B'], sol, free, temp=c['hi'] * mult * unit, theta=c['theta'] * unit, focused=True,
                          C=c['C'], tau=m['tau'], tol=c['tol'], ebar=m['ebar'], cap=cap, check=c['check'], seed=seed,
                          anneal=(c['hi'] * mult * unit, c['lo'] * mult * unit, c['period']))
    res = ss.run()
    return SS.summarize(res), res, {'D': ss.D, 'fl_move': ss.fl_move}


def probe(task_name, task, seed, ratio):
    """Solve rate of the fixed probe (Sudoku: 512 development puzzles; maze: the 100 development mazes).
    The random-number states of the training are restored afterwards, so a probe does not change the training.
    Returns a dict with n, solved, stopped, wrong_stops, budget_failures and mean_flop."""
    st = (torch.get_rng_state(), torch.cuda.get_rng_state())
    with torch.no_grad():
        if task_name == 'sudoku':
            from cram import cram_sudoku_search as SS
            sol, free, _ = SS.load_puzzles('dev10k', 0, PROBE['sudoku']['n'])
            m = sudoku_ready(task, task.sol)
            sm, _, info = sudoku_search(m, sol, free, PROBE['sudoku']['cap'], seed, ratio=ratio)
            out = {**sm, **info, 'unit': m['unit'], 'tau': m['tau']}
        else:
            r = task.U.score(task.model, PROBE['maze']['split'], task.U.Search('gibbs'))
            out = {k: r[k] for k in ('n', 'solved', 'stopped', 'wrong_stops', 'budget_failures', 'mean_flop')}
    torch.set_rng_state(st[0]); torch.cuda.set_rng_state(st[1])
    return out


# ------------------------------------------------------------------------------------------------------- training
@dataclasses.dataclass
class Run:
    task: str = 'sudoku'  # sudoku | maze
    method: str = 'pl'  # one of METHODS
    lr: float = 540.0  # initial learning rate
    knob: float | None = None  # the method's own parameter (see the module docstring)
    reg: float = 0.0  # class S: weight of the energy-magnitude regularizer (Du & Mordatch 2019); 0: none
    l1: float = 1.0  # L1 strength: sudoku, a multiple of 3e-3/9 per entry (the weight of sudoku.py); maze, the soft-threshold is lr x l1 per update
    updates: int = 0  # updates of the cosine (sudoku) schedule; 0: the class default (UPDATES)
    cap_s: float = FINAL_CAP_S  # fit-time cap in seconds; the run stops at the cap
    seed: int = 0  # training split (sudoku train-s<seed>) and every training generator
    out: str = ''  # run directory, relative to experiments/
    stop_after: int = 0  # for a test of resuming: save the checkpoint and exit after this many updates
    probe_per_octave: int = 2  # probe points per doubling of the update count (the figure runs use more)
    probe_every_until: int = 0  # also probe after every update up to this one
    n_train: int = 0  # train on this many puzzles of the split, a seeded nested subset (0: all)
    aug: bool = True  # sudoku: a fresh digit relabeling of every puzzle at every update (as sudoku.py); False: the labels as given
    pool: str = ''  # sudoku: train on this .npz (relative to experiments/) instead of train-s<seed>; n_train takes a seeded nested subset
    maze_opt: str = 'adam'  # maze: sgd-cosine (momentum 0.5, cosine decay as Sudoku, no clip; the paper's setting, needs ALT_MAZE_OPT=sgd-cosine) | adam (constant lr, gradient-norm clip 1) | sgd (momentum 0.5, constant lr, no clip)
    maze_kernel: str = 'separable'  # maze: free (CRAMMazeFreeEnergy, 18 parameters; the paper's setting) | separable (CRAMMazeEnergy, 16 parameters, runs/ALT)


def dump(obj, path):
    """Write obj as JSON to path through a temporary file and a rename, so that a reader never sees a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def probe_points(updates, per_octave=2, every_until=0):
    """Updates after which the probe is scored: per_octave log-spaced points per doubling (default 1, 1.4, 2, 2.8, ...),
    every update up to every_until, and the last."""
    pts = {updates} | set(range(1, min(every_until, updates) + 1))
    k = 1.0
    while k < updates:
        pts.add(int(round(k)))
        k *= 2 ** (1 / per_octave)
    return sorted(pts)


def train(cfg: Run):
    """One training run. Writes into <cfg.out>: config.json, log.jsonl (one line per update: loss and fit seconds),
    model-u<k>.pt at every probe point, a rolling checkpoint ckpt.pt every 10 minutes, and final.json last (the
    selected checkpoint: the highest probe solve rate, the earliest on ties; the probe curve; T99_s, the first fit
    time with at least 99% of the probe solved). A run with final.json is skipped; a run with ckpt.pt resumes.
    The fit clock (fit_s) covers the transfer to the device, the initialization and every update; it excludes file
    reading, the probes and saving."""
    if cfg.task == 'maze':  # a maze run lives under the root of its kernel and optimizer, so they never share a folder
        want = maze_root_of(cfg.maze_kernel, cfg.maze_opt)
        assert cfg.out.startswith(want) and all(not cfg.out.startswith(r) for r in ('runs/ALT-free/', 'runs/ALT-sgd/') if r != want), (cfg.out, cfg.maze_kernel, cfg.maze_opt)
        assert cfg.maze_opt != 'sgd-cosine' or MAZE_OPT == 'sgd-cosine', 'sgd-cosine runs read their pl schedule from the SGD grids (MAZE_SGD): set ALT_MAZE_OPT=sgd-cosine'
    out = HERE / cfg.out
    if (out / 'final.json').exists():
        print(out, 'finished; nothing to do', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device('cuda')
    method, knob = cfg.method, cfg.knob
    if knob is not None and knob == int(knob):
        knob = int(knob)
    updates = cfg.updates or (REF[cfg.task]['updates'] if method == 'pl' else UPDATES[cfg.task][CLASS[method]])
    ratio = hbar_ratio()[0] if cfg.task == 'sudoku' else None
    task = TASKS[cfg.task](cfg.seed, dev, cfg.n_train, cfg.aug, cfg.pool, cfg.maze_kernel)
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    task.setup()
    if cfg.task == 'sudoku':
        task.lam *= cfg.l1
    gen = torch.Generator(device=dev).manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)  # the relabeling draws of sudoku.py (sudoku); the maze model was seeded in Maze.setup
    mstate = {'_reg': cfg.reg}
    if cfg.task == 'sudoku':
        bufs = [torch.zeros_like(p) for p in task.params()]
        opt = None
    elif cfg.maze_opt in ('sgd', 'sgd-cosine'):
        opt = torch.optim.SGD(task.params(), lr=cfg.lr, momentum=0.5)
    else:
        assert cfg.maze_opt == 'adam', cfg.maze_opt
        opt = torch.optim.Adam(task.params(), lr=cfg.lr)
    k0, fit_s, log = 0, 0.0, []
    ck = out / 'ckpt.pt'
    if ck.exists():
        sd = torch.load(ck, map_location='cpu', weights_only=False)  # RNG states stay on the CPU
        to = lambda x: {k: (to(v) if isinstance(v, dict) else v.to(dev) if torch.is_tensor(v) else v) for k, v in x.items()}
        task.load(to(sd['params']))
        if opt is None:
            bufs = [b.to(dev) for b in sd['bufs']]
        else:  # the maze batch stream is replayed from its seed
            opt.load_state_dict(sd['opt'])
            task.rng = np.random.default_rng(cfg.seed)
            task.stream = task.U.batches(task.n_train, min(64, task.n_train), task.rng)
            for _ in range(sd['step']):
                next(task.stream)
        gen.set_state(sd['gen'])
        torch.set_rng_state(sd['cpu_rng']); torch.cuda.set_rng_state(sd['cuda_rng'])
        mstate = to(sd['mstate'])
        k0, fit_s, log = sd['step'], sd['fit_s'], sd['log']
        lines = [l for l in (out / 'log.jsonl').read_text().splitlines() if 'resume' in json.loads(l) or json.loads(l)['update'] <= k0]
        (out / 'log.jsonl').write_text(''.join(l + '\n' for l in lines) + json.dumps({'resume': k0}) + '\n')
        print(f'resume at update {k0}, fit {fit_s:.1f} s', flush=True)
    else:
        (out / 'log.jsonl').write_text('')
        dump(dataclasses.asdict(cfg) | {'updates_resolved': updates, 'knob_resolved': knob, 'hbar_ratio': ratio,
                                        'class': CLASS[method]}, out / 'config.json')
        fit_s = 0.0
    pts = set(probe_points(updates, cfg.probe_per_octave, cfg.probe_every_until))
    torch.cuda.synchronize()
    if k0 == 0:
        fit_s = time.perf_counter() - t_start  # setup (transfer, initialization) is part of the fit
    t_ck = time.monotonic()
    stop = 'schedule'
    for k in range(k0, updates):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        task.begin()
        ctx = task.batch(gen)
        loss = loss_fn(method, knob, task, ctx, gen, mstate)
        if cfg.task == 'sudoku':
            lr = cfg.lr * 0.5 * (1 + math.cos(math.pi * k / updates))
            gs = torch.autograd.grad(loss, task.params())
            for p, g in zip(task.params(), gs):
                p.grad = g
            task.step(lr, bufs)
        else:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            lr_k = cfg.lr * 0.5 * (1 + math.cos(math.pi * k / updates)) if cfg.maze_opt == 'sgd-cosine' else cfg.lr
            if cfg.maze_opt == 'adam':
                torch.nn.utils.clip_grad_norm_(task.params(), 1.0)
            for g in opt.param_groups:
                g['lr'] = lr_k
            opt.step()
            with torch.no_grad():  # L1 as a proximal soft-threshold, as on Sudoku
                for p in task.params():
                    p.copy_(p.sign() * (p.abs() - lr_k * cfg.l1).clamp_min(0))
        torch.cuda.synchronize()
        fit_s += time.perf_counter() - t0
        rec = {'update': k + 1, 'loss': float(loss), 'fit_s': fit_s}
        finite = all(torch.isfinite(p).all() for p in task.params())
        if (k + 1) in pts or fit_s >= cfg.cap_s or not finite:
            if not finite:
                rec['probe'] = {'solved': 0, 'n': 1, 'nonfinite': True}
            else:
                torch.save(task.state(), out / f'model-u{k + 1}.pt')
                rec['probe'] = probe(cfg.task, task, cfg.seed, ratio)
            log.append(rec)
            print(json.dumps({k2: v for k2, v in rec.items() if k2 != 'probe'} | {'probe_solved': rec['probe']['solved']}), flush=True)
        with open(out / 'log.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        if not finite:
            stop = 'nonfinite'
            break
        if fit_s >= cfg.cap_s:
            stop = 'cap'
            break
        if time.monotonic() - t_ck >= 600 or (cfg.stop_after and k + 1 == cfg.stop_after):
            sd = {'params': task.state(), 'step': k + 1, 'fit_s': fit_s, 'log': log, 'mstate': mstate,
                  'gen': gen.get_state(), 'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}
            if opt is None:
                sd['bufs'] = bufs
            else:
                sd['opt'] = opt.state_dict()
            torch.save(sd, ck.with_suffix('.tmp'))
            os.replace(ck.with_suffix('.tmp'), ck)
            t_ck = time.monotonic()
            if cfg.stop_after and k + 1 == cfg.stop_after:
                print('stopped for the resume test', flush=True)
                return
    # the selected checkpoint: the highest probe solve rate, the earliest on ties
    best = max(log, key=lambda r: (r['probe']['solved'] / r['probe']['n'], -r['update']))
    hit = next((r for r in log if r['probe']['solved'] >= 0.99 * r['probe']['n']), None)
    final = {'config': dataclasses.asdict(cfg), 'updates': updates, 'stop': stop, 'fit_s': fit_s,
             'selected': {'update': best['update'], 'fit_s': best['fit_s'], 'probe': best['probe']},
             'T99_s': hit['fit_s'] if hit else None, 'T99_update': hit['update'] if hit else None,
             'curve': [{'update': r['update'], 'fit_s': r['fit_s'], 'solved': r['probe']['solved'], 'n': r['probe']['n']} for r in log],
             'gpu': torch.cuda.get_device_name(), 'node': os.uname().nodename, 'job': os.environ.get('SLURM_JOB_ID')}
    dump(final, out / 'final.json')
    ck.unlink(missing_ok=True)
    print(json.dumps({k: final[k] for k in ('stop', 'fit_s', 'T99_s')} | {'selected': best['update'], 'solved': best['probe']['solved']}), flush=True)


# ---------------------------------------------------------------------------------------------- selection and test
def tune_dir(task, method):
    """The folder (Path) of the tuning runs of one task and method."""
    return root(task) / 'tune' / f'{task}-{method}'


def grid(task, method):
    """The tuning grid of one task and method: a list of dicts {lr, knob, l1} (class S also {reg}), the product of the
    learning rates, the method's knob values and the L1 strengths. The L1 strength is tuned for every method, pl
    included; it is never 0, except in the maze SGD grid (MAZE_SGD). pl keeps the learning rate and update count of
    REF."""
    lrs = (REF[task]['lr'],) if method == 'pl' else LRS[task]
    g = [dict(lr=lr, knob=k, l1=l) for lr in lrs for k in KNOBS[task][method] for l in L1S[task]]
    if CLASS[method] == 'S':  # the energy-magnitude regularizer, off or at Du & Mordatch's weight 1
        g = [dict(c, reg=r) for c in g for r in REGS]
    return g


# L1 strength. Sudoku: multiples of the weight of sudoku.py, 3e-3/9 per entry (at x1 the class D arms stay at their
# zero start). Maze: the soft-threshold after each update is lr x l1 (an Adam step is about lr).
L1S = {'sudoku': (1.0, 0.1, 0.01), 'maze': (0.003, 0.03, 0.3)}

# The maze grids for SGD with cosine decay (ALT_MAZE_OPT=sgd-cosine, the paper's setting). Learning rates 0.1, 0.3 and
# 1 (the range in which the loss decreases smoothly); L1 strengths 0 (as the Table 2 row cram-sgd-fg), 0.003 and 0.03;
# pl keeps the recipe of the Table 2 row (learning rate 1, 20 updates) and tunes only the L1 strength.
# The knobs and the other update counts are those of the default setting.
MAZE_SGD = dict(lrs=(0.1, 0.3, 1.0), l1s=(0.0, 0.003, 0.03), ref=dict(lr=1.0, updates=20))
if MAZE_OPT == 'sgd-cosine':
    LRS['maze'], L1S['maze'], REF['maze'] = MAZE_SGD['lrs'], MAZE_SGD['l1s'], MAZE_SGD['ref']


REGS = (0.0, 1.0)


def tag(c):
    """The folder name of a grid entry c (a dict of grid)."""
    return f"lr{c['lr']:g}-k{c['knob']}-l{c['l1']:g}" + (f"-r{c['reg']:g}" if 'reg' in c else '')


def flags(c, task='sudoku'):
    """The command-line flags (str) of a grid entry c for `train`, with the maze kernel and optimizer flags of the
    setting that the environment variables select."""
    return (('' if c['knob'] is None else f" --knob {c['knob']}") + f" --l1 {c['l1']}" + (f" --reg {c['reg']}" if 'reg' in c else '')
            + (' --maze-kernel free' if task == 'maze' and MAZE_KERNEL == 'free' else '')
            + (' --maze-opt sgd-cosine' if task == 'maze' and MAZE_OPT == 'sgd-cosine' else ''))


def select(task: str, method: str):
    """The tuning choice: the highest probe solve rate at the selected checkpoint, then the earliest fit time to it.
    Writes <root>/tune/<task>-<method>/choice.json: {"choice": {lr, knob, l1, reg}, "rows": every grid entry's result}."""
    rows = []
    for c in grid(task, method):
        f = tune_dir(task, method) / tag(c) / 'final.json'
        if not f.exists():
            raise SystemExit(f'missing {f}')
        r = json.loads(f.read_text())
        rows.append({**c, 'solved': r['selected']['probe']['solved'], 'n': r['selected']['probe']['n'],
                     'fit_s': r['selected']['fit_s'], 'T99_s': r['T99_s']})
    best = max(rows, key=lambda r: (r['solved'] / r['n'], -r['fit_s']))
    dump({'choice': {k: best[k] for k in ('lr', 'knob', 'l1', 'reg') if k in best}, 'rows': rows}, tune_dir(task, method) / 'choice.json')
    print(json.dumps(best), flush=True)


def test(run: str, n: int = 0):
    """For the run's selected checkpoint: sudoku: the search temperature (x0.8, x1, x1.25 of the probe schedule) chosen on
    dev10k puzzles 512-2,559 (most solved, then lowest mean flop), then the test subset once; maze: the search of
    maze_unique.py (focused Gibbs at temperature 0, stop at a fixed point) on the 1,000 test mazes once.
    run: the run folder relative to experiments/. n: a smaller Sudoku test subset (0: the 50,000 puzzles of TEST).
    Writes <run>/test.json; a run that has test.json is skipped."""
    run = HERE / run
    out = run / 'test.json'
    if out.exists():
        print(out, 'exists: the test is scored once', flush=True)
        return
    fin = json.loads((run / 'final.json').read_text())
    cfg = Run(**fin['config'])
    dev = torch.device('cuda')
    task = TASKS[cfg.task](cfg.seed, dev, cfg.n_train, cfg.aug, cfg.pool, cfg.maze_kernel)
    task.setup()
    task.load(torch.load(run / f"model-u{fin['selected']['update']}.pt", map_location=dev, weights_only=True))
    if cfg.task == 'sudoku':
        from cram import cram_sudoku_search as SS
        ratio = hbar_ratio()[0]
        t0 = time.perf_counter()
        m = sudoku_ready(task, task.sol)
        torch.cuda.synchronize()
        ready_s = time.perf_counter() - t0
        sol, free, _ = SS.load_puzzles('dev10k', 512, 2048)
        screen = []
        for mult in (0.8, 1.0, 1.25):
            sm, _, info = sudoku_search(m, sol, free, PROBE['sudoku']['cap'], cfg.seed, mult, ratio)
            screen.append({'mult': mult, **sm, **info})
            print('screen', mult, sm['solved'], sm['mean_flop'], flush=True)
        pick = max(screen, key=lambda r: (r['solved'], -r['mean_flop']))
        t = TEST['sudoku']
        sol, free, idx = SS.load_puzzles('full-all', t['lo'], n or t['n'])  # n: a smaller test subset
        sm, res, info = sudoku_search(m, sol, free, t['cap'], cfg.seed, pick['mult'], ratio)
        rec = {'test': sm, 'search': {**SUDOKU_SEARCH, 'mult': pick['mult'], 'unit': m['unit'], 'hbar_ratio': ratio,
                                      'tau': m['tau'], 'ebar': m['ebar'], 'cap': t['cap'], **info},
               'screen': screen, 'ready_s': ready_s, 'puzzles': [int(idx[0]), int(idx[-1])],
               'nonzero': int((task.W != 0).sum() + (task.B != 0).sum()), 'flops': res['flops']}
    else:
        r = task.U.score(task.model, 'test', task.U.Search('gibbs'))
        rec = {'test': {k: r[k] for k in ('n', 'solved', 'stopped', 'wrong_stops', 'budget_failures', 'mean_flop')},
               'search': r['search'], 'ready_s': 0.0}
    rec |= {'fit_s': fin['selected']['fit_s'] + rec['ready_s'], 'selected_update': fin['selected']['update'],
            'gpu': torch.cuda.get_device_name(), 'job': os.environ.get('SLURM_JOB_ID')}
    dump(rec, out)
    print(json.dumps(rec['test']), flush=True)


def plan(stage: str = 'tune', tasks: tuple[str, ...] = ('sudoku', 'maze'), methods: tuple[str, ...] = METHODS):
    """Print one command line per run of the stage (tune: the grid on seed 0 with the tuning cap; final: the choice on
    seeds 0-2 with the final cap; test: each final run)."""
    for t in tasks:
        for m in methods:
            if stage == 'tune':
                for c in grid(t, m):
                    k = flags(c, t)
                    print(f"train --task {t} --method {m} --lr {c['lr']}{k} --seed 0 --cap-s {TUNE_CAP_S} --out {REL[t]}/tune/{t}-{m}/{tag(c)}")
            elif stage == 'final':
                c = json.loads((tune_dir(t, m) / 'choice.json').read_text())['choice']
                k = flags(c, t)
                for s in range(3):
                    print(f"train --task {t} --method {m} --lr {c['lr']}{k} --seed {s} --cap-s {FINAL_CAP_S} --out {REL[t]}/final/{t}-{m}-s{s}")
            else:
                for s in range(3):
                    print(f"test --run {REL[t]}/final/{t}-{m}-s{s}")


def count(out: str = '', full: bool = False, tasks: tuple[str, ...] = ('sudoku', 'maze')):
    """Training arithmetic per update of each (task, method) at its tuned choice: torch FlopCounterMode (matrix products
    and convolutions, 2 flop per multiply-add; elementwise operations not counted, as for the FRM and TRM rows) over one
    update after one warm-up update: the loss, its gradient and the parameter step, including every heat-bath move and
    relaxation step. The batch size is fixed, so the cost per update is constant. Training flop to T99 = per-update flop
    x the update at which the probe first reached 99% (median over seeds; None if at least two seeds never did).
    --full: the operation-count convention of baselines/flops/fullcount.py (every arithmetic operation), with its
    FullCounter in place of FlopCounterMode (every training update is PyTorch: heat_bath and relax; the Triton search
    runs only in the probe and test); writes results/alt-training/train-flop-fc2.json unless --out is given; also
    records the FlopCounterMode count of the same update, which equals the counter's matrix-product class.
    out: output path relative to experiments/ ('': the default file). tasks: the tasks to count."""
    from torch.utils.flop_counter import FlopCounterMode
    if full:
        import sys
        sys.path.insert(0, str(HERE.parent / 'baselines/flops'))
        from fullcount import CONVENTION, FullCounter
    # default outputs: train-flop.json / train-flop-fc2.json. With ALT_MAZE_KERNEL=free the maze rows only, into
    # train-flop<MAZE_FLOP_SUFFIX>.json / train-flop-fc2<MAZE_FLOP_SUFFIX>.json, so that the first two files keep
    # their rows. The paper's setting (ALT_MAZE_OPT=sgd-cosine) writes train-flop-maze-sgd.json / train-flop-fc2-maze-sgd.json.
    if not out:
        free = MAZE_KERNEL == 'free'
        if free:
            assert tuple(tasks) == ('maze',), 'with ALT_MAZE_KERNEL=free, count the maze alone (--tasks maze)'
        out = '../results/alt-training/train-flop' + ('-fc2' if full else '') + (MAZE_FLOP_SUFFIX if free else '') + '.json'
    dev = torch.device('cuda')
    rec = {'convention': (CONVENTION if full else 'torch.utils.flop_counter.FlopCounterMode') + ', one update (loss, gradient, step) after one warm-up update',
           'rows': {}, 'maze_kernel': MAZE_KERNEL, 'maze_opt': MAZE_OPT, 'roots': {t: REL[t] for t in tasks}}
    for t in tasks:
        for m in METHODS:
            if not (tune_dir(t, m) / 'choice.json').exists():
                continue
            ch = json.loads((tune_dir(t, m) / 'choice.json').read_text())['choice']
            knob = ch.get('knob')
            knob = int(knob) if knob is not None and knob == int(knob) else knob
            task = TASKS[t](0, dev, kernel=MAZE_KERNEL if t == 'maze' else 'separable'); task.setup()
            gen = torch.Generator(device=dev).manual_seed(0)
            torch.manual_seed(0)
            mstate = {'_reg': ch.get('reg', 0.0)}
            opt = (None if t == 'sudoku' else torch.optim.SGD(task.params(), lr=ch['lr'], momentum=0.5) if MAZE_OPT == 'sgd-cosine'
                   else torch.optim.Adam(task.params(), lr=ch['lr']))
            bufs = [torch.zeros_like(p) for p in task.params()] if t == 'sudoku' else None

            def update():
                task.begin()
                ctx = task.batch(gen)
                loss = loss_fn(m, knob, task, ctx, gen, mstate)
                if t == 'sudoku':
                    for p, g in zip(task.params(), torch.autograd.grad(loss, task.params())):
                        p.grad = g
                    task.step(1e-9, bufs)
                else:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            update()
            fc = FullCounter() if full else FlopCounterMode(display=False)
            with fc:
                update()
            per = int(fc.get_total_flops())
            extra = {}
            if full:
                ref = FlopCounterMode(display=False)
                with ref:
                    update()
                extra = {'classes': fc.classes(), 'total_with_compare': fc.total_with_compare(), 'flopcountermode_next_update': int(ref.get_total_flops()),
                         'matmul_class_equals_flopcountermode': fc.classes()['matmul'] == int(ref.get_total_flops())}
            ups = []
            for s in range(3):
                f = root(t) / 'final' / f'{t}-{m}-s{s}' / 'final.json'
                ups.append(json.loads(f.read_text()).get('T99_update') if f.exists() else 'missing')
            if 'missing' in ups:
                med = None
            else:
                med = sorted(u if u is not None else math.inf for u in ups)[1]
                med = None if med == math.inf else med
            rec['rows'][f'{t}/{m}'] = {'choice': ch, 'flop_per_update': per, 't99_update_per_seed': ups,
                                       't99_update': med, 't99_flop': None if med is None else per * med, **extra}
            print(t, m, f'{per:.3e}', ups, flush=True)
    dump(rec, HERE / out)


def export(out: str = '../results/alt-training/alt-training.json'):
    """The data of the paper's training-method table, from the final runs (<root>/final, seeds 0-2) and the tuning
    choices. Per row '<task>/<method>': test solve rate (mean, sample SD, %), T99 (median over seeds; a seed that
    never reached 99% counts as never, so the median is None when two or more never did), fit (mean s), inference flop
    (mean), the median probe curve (per probe index: median fit s, median solved %), and `unverified` when every tuning
    config scored 0 on the probe. Set ALT_MAZE_KERNEL=free ALT_MAZE_OPT=sgd-cosine for the paper's maze rows.
    out: output path relative to experiments/."""
    rows = {}
    for t in ('sudoku', 'maze'):
        for m in METHODS:
            if not (tune_dir(t, m) / 'choice.json').exists():
                continue
            if not all((root(t) / 'final' / f'{t}-{m}-s{s}' / 'test.json').exists() for s in range(3)):
                continue  # runs not finished: no row (the paper prints it as pending)
            ch = json.loads((tune_dir(t, m) / 'choice.json').read_text())
            fins, tests = [], []
            for s in range(3):
                d = root(t) / 'final' / f'{t}-{m}-s{s}'
                fin, tst = json.loads((d / 'final.json').read_text()), json.loads((d / 'test.json').read_text())
                if (d / 'retime.json').exists():  # retime_check: the clock values of the repeated run, same selected model
                    rt = json.loads((d / 'retime.json').read_text())
                    fin = fin | {'T99_s': rt['T99_s'], 'curve': rt['curve']}
                    tst = tst | {'fit_s': rt['selected_fit_s'] + tst['ready_s']}
                fins.append(fin)
                tests.append(tst)
            sv = [100 * x['test']['solved'] / x['test']['n'] for x in tests]
            t99 = sorted((f['T99_s'] if f['T99_s'] is not None else math.inf) for f in fins)[1]
            curves = [f['curve'] for f in fins]
            k = min(len(c) for c in curves)
            curve = [[statistics.median(c[i]['fit_s'] for c in curves), statistics.median(100 * c[i]['solved'] / c[i]['n'] for c in curves)]
                     for i in range(k)]
            rows[f'{t}/{m}'] = {
                'solved': [statistics.fmean(sv), statistics.stdev(sv)], 'solved_per_seed': sv,
                't99': None if t99 == math.inf else t99, 't99_per_seed': [f['T99_s'] for f in fins],
                'fit': statistics.fmean(x['fit_s'] for x in tests), 'fit_per_seed': [x['fit_s'] for x in tests],
                'flop': statistics.fmean(x['test']['mean_flop'] for x in tests),
                'wrong_stops': sum(x['test']['wrong_stops'] for x in tests),
                'budget_failures': sum(x['test']['budget_failures'] for x in tests),
                'n_test': tests[0]['test']['n'], 'choice': ch['choice'],
                'unverified': max(r['solved'] for r in ch['rows']) == 0, 'curve': curve}
    tf = HERE / '../results/alt-training/train-flop.json'  # training_methods.py count: training arithmetic per update
    tfs = [tf] + ([HERE / f'../results/alt-training/train-flop{MAZE_FLOP_SUFFIX}.json'] if MAZE_KERNEL == 'free' else [])
    for tf in tfs:  # with the free kernel, the maze counts of the chosen setting replace the separable-kernel maze rows
        if not tf.exists():
            continue
        for key, c in json.loads(tf.read_text())['rows'].items():
            if key in rows and (MAZE_KERNEL == 'separable' or key.startswith('sudoku/') or tf.name.endswith(f'{MAZE_FLOP_SUFFIX}.json')):
                rows[key].update(train_flop_per_update=c['flop_per_update'], t99_update=c['t99_update'], t99_train_flop=c['t99_flop'])
    rec = {'source': 'experiments/training_methods.py export; '
                     f"sudoku: {REL['sudoku']}, maze: {REL['maze']} (final/*-s{{0,1,2}}/{{final,test}}.json and tune/*/choice.json)",
           'maze_kernel': MAZE_KERNEL, 'maze_opt': MAZE_OPT,
           'test_sets': {'sudoku': 'full-all puzzles 10,000-59,999 (50,000), cap 4e11 flop', 'maze': '1,000 EqR test mazes'},
           'probe': {'sudoku': 'dev10k puzzles 0-511, cap 4e10 flop', 'maze': 'dev100'},
           'rows': rows}
    dump(rec, HERE / out)
    print(HERE / out)


def retime_check(task: str = 'maze'):
    """Take the fit times of final runs from repeats of those runs. The first construction of a torch optimizer
    imports torch._dynamo (about 3 s). In earlier final runs this import was inside the fit clock; a repeat (the same
    command with --out <root>/retime/<name>) imports it before the clock starts. For each final run with a repeat in <root>/retime/, check that the repeat selected the same update and that
    its selected model equals the final run's bitwise; then write <final run>/retime.json with the repeat's T99,
    selected fit time and probe curve. A run that fails either check gets no retime.json and is listed."""
    bad = []
    for d in sorted((root(task) / 'retime').glob(f'{task}-*-s*')):
        f, r = root(task) / 'final' / d.name, d
        if not (r / 'final.json').exists():
            bad.append((d.name, 'rerun not finished'))
            continue
        a, b = json.loads((f / 'final.json').read_text()), json.loads((r / 'final.json').read_text())
        u = a['selected']['update']
        ma = torch.load(f / f'model-u{u}.pt', map_location='cpu', weights_only=True)
        mb = torch.load(r / f'model-u{u}.pt', map_location='cpu', weights_only=True)
        same = b['selected']['update'] == u and ma.keys() == mb.keys() and all(torch.equal(ma[k], mb[k]) for k in ma)
        if not same:
            bad.append((d.name, 'selected model differs'))
            continue
        dump({'source': str(r.relative_to(HERE)), 'selected_update': u, 'selected_fit_s': b['selected']['fit_s'],
              'T99_s': b['T99_s'], 'curve': b['curve'], 'old_selected_fit_s': a['selected']['fit_s'], 'old_T99_s': a['T99_s'],
              'bitwise_equal_selected_model': True, 'why': 'torch._dynamo imported before the fit clock'},
             f / 'retime.json')
        print(d.name, f"fit {a['selected']['fit_s']:.2f} -> {b['selected']['fit_s']:.2f} s", flush=True)
    for name, why in bad:
        print('NOT RETIMED', name, why, flush=True)


if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict({'train': train, 'select': select, 'test': test, 'plan': plan, 'export': export, 'retime-check': retime_check, 'count': count},
                                         config=(tyro.conf.OmitArgPrefixes,))
