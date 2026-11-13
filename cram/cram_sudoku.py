"""Sudoku with the same-digit pair energy: the energy, its pseudolikelihood (PL) fit with an optional L1 proximal
term, weight sharing after the fit, the stopping tests, and the digit-relabeling augmentation.

This energy has one scalar cost per cell pair, paid when the two cells hold the same digit. It is the form of the
hand-set Sudoku energy (`experiments/hand_set.py`), which `cram_sudoku_search.Search` and
`cram_sudoku_sparse.SparseSearch` search. The learned Sudoku model of the paper is more general (a free 9 x 9 cost
table per cell pair, `cram_sudoku_general`, fitted by `experiments/sudoku.py`); from this file it uses the augmentation
`augment_digit` and, for diagnostics, `kmeans_levels_values` and `conflict_graph`. The fit `train` of this file is used
by the tests of the search kernels.

Dimension names: b puzzles (or boards), n = 81 cells, m = 81 cells (second index of a pair), d = 9 digits,
t = 10 input tokens (0 blank, 1-9 clue digit), k levels of shared weights.

Energy on a board y (one digit y_i per cell), with the input tokens x, the directed pair table A (n, n; zero diagonal)
and the input biases B (t, d):
    E(y) = sum_(i != j) A_ij [y_i = y_j] - sum_i B[x_i, y_i] = 1/2 sum_(i != j) S_ij [y_i = y_j] - sum_i B[x_i, y_i],
with S = A + A^T and [.] equal to 1 when the condition holds and 0 otherwise. In the notation of the general energy,
the cost table of the pair (i, j) is S_ij times the 9 x 9 identity. P(y) = y.A y, the first sum, is the pair energy.
Clue cells are clamped; only the blank-token row B[0] receives a gradient, so the learned parameters are the 6,480
off-diagonal entries of A and the 9 entries of B[0].

Training loss: the pseudolikelihood, the mean over free cells of the cross-entropy of softmax_a(-beta E_i(a)) against
the solution digit, where E_i(a) is the energy of the solution with cell i set to digit a. An optional term
lam_l1 * sum_(i != j) |A_ij| is applied as a proximal step after each optimizer update:
A <- sign(A) * max(|A| - lr * lam_l1, 0). The update needs no search and no differentiation through a solver.

Augmentation: `augment_digit` draws a random permutation of the 9 digits for each training board in every update.
`augment_sudoku` also permutes rows, columns, bands and stacks and transposes the grid; it uses the Sudoku layout, so
the paper's model does not use it.

Sign property used by the per-term stopping test: a conflict pair (i, j) never holds the same digit in a training
solution, so its PL gradient d/dA_ij = beta * (0 - p_i(y_j)) - ... is <= 0 in every update; SGD with momentum (a positive
combination of past gradients), Adam (m has the sign of every gradient, v > 0) and the proximal shrink toward 0 all
keep such an A_ij >= 0 when it starts at 0. Spatial augmentation maps conflict pairs to conflict pairs, so it keeps the
property. Weight sharing after the fit (`postfit`) does not; the per-term test is sound only without it.
"""
import dataclasses
import math
import time

import torch
import torch.nn.functional as F

N, V, T_IN = 81, 9, 10
TERM_TOL = 1e-6  # per-term test: a same-digit pair passes only if S_ij < -TERM_TOL


@dataclasses.dataclass
class FitConfig:
    """The settings of `train`."""
    aug: str = 'digit'  # digit: digit relabeling only (cell IDs); sudoku: also band/stack/row/column/transpose symmetries
    updates: int = 8  # full-batch updates
    opt: str = 'sgd'  # sgd | adam
    lr: float = 30.0  # initial learning rate (cosine decay to 0)
    momentum: float = 0.5  # sgd momentum
    cosine: bool = True  # cosine learning-rate schedule
    l1: float = 0.0  # lam_l1 of the proximal L1 term on the directed couplings
    bias: bool = True  # learn the input biases B; False keeps B = 0 (no input cost; clue clamps carry the input)
    sym: bool = False  # learn one cost W_ij = S_ij per unordered pair (3,240) instead of the directed A (6,480), at learning rate 2 lr (keeps S = 2 A identical)
    beta: float = 1.0  # inverse temperature inside the PL softmax
    batch: int = 1000  # >= training set size: every puzzle every update, batch // P copies
    split: int = 0  # training split train-s<split> (cram_sudoku_search.load_puzzles)
    seed: int = 0  # RNG seed of the augmentation stream
    device: str = 'cuda'  # device of the fit


# ---------------------------------------------------------------- augmentation
def augment_digit(sol, free):
    """Digit relabeling only: one random permutation of the 9 digits per board; cell positions are not changed, so the
    augmentation uses nothing about the grid. sol (b, n) digits 0-8, free (b, n) bool. Returns (relabeled sol, free).
    Draws from the default generator of sol's device."""
    digits = torch.rand(len(sol), V, device=sol.device).argsort(1)
    return digits.gather(1, sol), free


def augment_sudoku(sol, free):
    """The symmetry group of Sudoku: band and stack order, rows within a band, columns within a stack, transpose, and
    digit relabeling, drawn per board. sol (b, n), free (b, n). Returns (sol, free) with the cells permuted alike."""
    P, dev = sol.shape[0], sol.device

    def lines():
        band = torch.rand(P, 3, device=dev).argsort(1)
        within = torch.rand(P, 3, 3, device=dev).argsort(2)
        return (3 * band[:, :, None] + within).reshape(P, 9)

    rows, cols = lines(), lines()
    idx = rows[:, :, None] * 9 + cols[:, None, :]
    idx = torch.where((torch.rand(P, device=dev) < 0.5)[:, None, None], idx.transpose(1, 2), idx).reshape(P, N)
    digits = torch.rand(P, V, device=dev).argsort(1)
    return digits.gather(1, sol.gather(1, idx)), free.gather(1, idx)


AUGMENT = {'digit': augment_digit, 'sudoku': augment_sudoku}


# ---------------------------------------------------------------- loss
def digit_energies(A, B, sol, free, S=None):
    """(b, n, d): E_i(a) = (S y)_ia - B[x_i, a], the energy of the solution with cell i set to a, less a constant.
    S = A + A^T unless given."""
    y = F.one_hot(sol, V).float()
    x = F.one_hot(torch.where(free, 0, sol + 1), T_IN).float()
    return torch.einsum('nm,bmd->bnd', A + A.T if S is None else S, y) - x @ B


def pl_loss(A, B, sol, free, beta=1.0, S=None):
    """The pseudolikelihood loss (scalar): the mean over the free cells of -log softmax_a(-beta E_i(a)) at the solution
    digit. A (n, m), B (t, d), sol (b, n) digits 0-8, free (b, n) bool; S = A + A^T unless given."""
    logits = -beta * digit_energies(A, B, sol, free, S)
    nll = -torch.log_softmax(logits, -1).gather(-1, sol.unsqueeze(-1)).squeeze(-1)
    return (nll * free).sum() / free.sum()


def prox_l1(A, threshold):
    """Soft-thresholding, in place: the proximal map of threshold * |A|."""
    A.copy_(A.sign() * (A.abs() - threshold).clamp_min(0))


def _opt_update(p, g, st, lr, k, kind, momentum, betas=(0.9, 0.999), eps=1e-8):
    """One in-place optimizer update of p with gradient g and state st: Adam with bias correction at update k (a
    tensor), or SGD with momentum."""
    if kind == 'adam':
        st['m'].mul_(betas[0]).add_(g, alpha=1 - betas[0])
        st['v'].mul_(betas[1]).addcmul_(g, g, value=1 - betas[1])
        t = (k + 1).double()
        bc1, bc2 = 1 - torch.pow(betas[0], t), 1 - torch.pow(betas[1], t)
        p.sub_((lr / bc1).float() * st['m'] / (st['v'].sqrt() / bc2.sqrt().float() + eps))
    else:
        st['buf'].mul_(momentum).add_(g)
        p.sub_(lr * st['buf'])


def train(cfg, sol, free, graph=None):
    """Fit the directed couplings A (n, n; zero diagonal) and the biases B (t, d) from a zero start by full-batch
    updates on the solutions sol (b, n) (digits 0-8) with free cells free (b, n) bool. Returns (A, B, seconds of the
    update loop). graph (default: True on CUDA) captures one update as a CUDA graph and replays it; graph=False runs the
    updates eagerly. The random stream of the augmentation is the default generator of cfg.device seeded with cfg.seed."""
    dev = cfg.device
    graph = (dev == 'cuda') if graph is None else graph
    P = len(sol)
    reps = max(1, cfg.batch // P)
    assert cfg.batch >= P, 'full-batch training only'
    aug = AUGMENT[cfg.aug]
    iu = torch.triu_indices(N, N, 1, device=dev)
    # the pair parameters: directed A (n, n) or, with cfg.sym, one W per unordered pair (3,240), S = W on both halves
    A = torch.zeros(iu.shape[1] if cfg.sym else (N, N), device=dev, requires_grad=True)
    B = torch.zeros(T_IN, V, device=dev, requires_grad=True)
    off = 1 - torch.eye(N, device=dev)
    pair_scale = 2.0 if cfg.sym else 1.0  # W = S = 2 A: twice the step keeps S identical (exact in floating point)

    def coupling(p):
        if not cfg.sym:
            return p * off, None
        M = torch.zeros(N, N, device=dev).index_put((iu[0], iu[1]), p)
        return None, M + M.T
    zeros = lambda p: {'m': torch.zeros_like(p), 'v': torch.zeros_like(p)} if cfg.opt == 'adam' else {'buf': torch.zeros_like(p)}
    st = [zeros(A), zeros(B)]
    ks = torch.arange(cfg.updates, device=dev, dtype=torch.float32)
    lrs = cfg.lr * 0.5 * (1 + torch.cos(math.pi * ks / cfg.updates)) if cfg.cosine else torch.full_like(ks, cfg.lr)
    k = torch.zeros((), dtype=torch.long, device=dev)
    torch.manual_seed(cfg.seed)
    s_all, f_all = sol.repeat(reps, 1), free.repeat(reps, 1)

    def step():
        s, f = aug(s_all, f_all)
        Ad, Sd = coupling(A)
        gA, gB = torch.autograd.grad(pl_loss(Ad, B, s, f, cfg.beta, Sd), (A, B))
        lr = lrs.index_select(0, k.clamp(max=cfg.updates - 1).view(1)).squeeze(0)
        with torch.no_grad():
            _opt_update(A, gA, st[0], pair_scale * lr, k, cfg.opt, cfg.momentum)
            if cfg.bias:
                _opt_update(B, gB, st[1], lr, k, cfg.opt, cfg.momentum)
            if cfg.l1 > 0:
                prox_l1(A, pair_scale * lr * cfg.l1)
            k.add_(1)

    def directed(p):
        """The directed table the rest of the code reads: A itself, or S / 2 (exact) for the symmetric W."""
        with torch.no_grad():
            return p.detach() * off if not cfg.sym else coupling(p.detach())[1] * 0.5

    def reset():
        with torch.no_grad():
            A.zero_(); B.zero_(); k.zero_()
            for d in st:
                for t in d.values():
                    t.zero_()

    if not graph:
        if dev == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(cfg.updates):
            step()
        if dev == 'cuda':
            torch.cuda.synchronize()
        return directed(A), B.detach(), time.perf_counter() - t0
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(cfg.updates):
        g.replay()
    torch.cuda.synchronize()
    return directed(A), B.detach(), time.perf_counter() - t0


# ---------------------------------------------------------------- post-fit compression (learned weights only)
def kmeans_levels(A, levels, sym=False):
    """Weight sharing after the fit: Lloyd's k-means on the scalar values of the off-diagonal entries of A (n, n) with
    `levels` centroids (initialized evenly between the smallest and the largest value, at most 64 iterations); each
    entry is replaced by its centroid. It reads the learned weights only, no cell positions.
    sym: cluster the 3,240 unordered pair values (the learned W) instead of the 6,480 directed entries.
    Returns (the shared-weight matrix (n, n), the list of centroids)."""
    if sym:
        iu = torch.triu_indices(N, N, 1, device=A.device)
        Aq, c = kmeans_levels_values(A[iu[0], iu[1]], levels)
        M = A.new_zeros(A.shape).index_put((iu[0], iu[1]), Aq)
        return M + M.T, c
    off = ~torch.eye(len(A), dtype=torch.bool, device=A.device)
    w = A[off]
    c = torch.linspace(w.min().item(), w.max().item(), levels, device=A.device)
    for _ in range(64):
        code = (w[:, None] - c[None, :]).square().argmin(1)
        new = torch.stack([w[code == j].mean() if (code == j).any() else c[j] for j in range(levels)])
        if torch.equal(new, c):
            break
        c = new
    return A.new_zeros(A.shape).masked_scatter(off, c[code]), c.tolist()


def kmeans_levels_values(w, levels):
    """Lloyd's k-means on the values w (any 1-D tensor), as `kmeans_levels`. Returns (w with each value replaced by its
    centroid, the list of centroids)."""
    c = torch.linspace(w.min().item(), w.max().item(), levels, device=w.device)
    for _ in range(64):
        code = (w[:, None] - c[None, :]).square().argmin(1)
        new = torch.stack([w[code == j].mean() if (code == j).any() else c[j] for j in range(levels)])
        if torch.equal(new, c):
            break
        c = new
    return c[code], c.tolist()


def postfit(A, kind, sym=False):
    """Processing of the fitted A (n, n) after the fit. kind: 'none' or 'k<levels>' (`kmeans_levels` with that number
    of levels). sym: A = S / 2 from a symmetric fit; k-means then clusters the unordered pair values.
    Returns (A, info); info['centroids'] holds the levels."""
    info = {}
    for part in kind.split('-'):
        if part == 'none':
            continue
        if part.startswith('k'):
            A, info['centroids'] = kmeans_levels(A, int(part[1:]), sym)
        else:
            raise ValueError(kind)
    return A, info


# ---------------------------------------------------------------- stopping tests
def pair_energy(A, sol):
    """(b,): y.A y for boards sol (b, n) of digits 0-8."""
    y = F.one_hot(sol, V).float()
    return torch.einsum('bnd,nm,bmd->b', y, A, y)


def blocking(A):
    """(n, m) float: 1 where a same-digit pair blocks the per-term test (S_ij >= -TERM_TOL), off the diagonal."""
    S = A + A.T
    off = ~torch.eye(N, dtype=torch.bool, device=A.device)
    return ((S >= -TERM_TOL) & off).float()


def term_violations(A, sol):
    """(b,): the number of unordered same-digit pairs that block the per-term test; a board passes iff it is 0."""
    y = F.one_hot(sol, V).float()
    return 0.5 * torch.einsum('bnd,nm,bmd->b', y, blocking(A), y)


def conflict_graph(device='cpu'):
    """(n, m) bool: True for two different cells that share a row, a column or a 3 x 3 box (the Sudoku rules).
    For diagnostics of a fitted model and for test fixtures only; never used by training or search of a learned model."""
    i = torch.arange(N, device=device)
    r, c = i // 9, i % 9
    same = lambda a: a[:, None] == a[None, :]
    return (same(r) | same(c) | same(r // 3 * 3 + c // 3)) & ~torch.eye(N, dtype=torch.bool, device=device)
