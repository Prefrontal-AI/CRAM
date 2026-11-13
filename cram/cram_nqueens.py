"""N-Queens CRAM: n^2 binary cells, a free 2 x 2 cost table for every unordered cell pair and a free input cost per cell.

No attack lines, rows, columns, diagonals or queen count enter fitting or search; only the scoring functions in
cram/nqueens_data.py know the rules. Data: GRAM's N-Queens sets (`nqueens_data`, whose docstring gives the source, the
encoding and the splits). The paper's N-Queens rows use the offset-shared energy of `experiments/nqueens_offset.py`,
which shares one table among the pairs with the same offset and searches with `solve` of this file.

Energy of a board x (x_i = 1 where cell i holds a queen), the same form as the Sudoku model (`cram_sudoku_general`):
    E(x) = 1/2 sum_(i != j) S[i, j, x_i, x_j] + sum_i B[i, x_i],  S[i, j] = W_p, S[j, i] = W_p^T for the pair p = (i < j),
with W (p, 2, 2) a free cost table per unordered cell pair and B (c, 2) an input cost per cell and state.
Training: pseudolikelihood (PL) of the free (non-given) cells of every training row at its labeled solution, plus
lam * (|W|_1 + |B|_1) as a proximal soft-threshold after each update; zero start; full-batch SGD with momentum and
cosine decay. The PL of a cell depends only on the row's label, so the loss is computed on the distinct training
solutions, each cell weighted by the number of rows that hold that solution with that cell free (the same objective,
reduced to sufficient statistics inside the timed fit). The fit needs no solver.
Inference-ready model: S, B, the stopping threshold tau (highest energy of a training solution) and the temperature
unit (mean |field difference| of the cells of the training rows' solutions); all fitted inside the timed fit.
Search: focused heat-bath Gibbs from the board that holds only the given queens (clamped); see `solve`.

Dimension names: r data rows, u distinct training solutions, b chains, c cells (n * n), p unordered cell pairs,
a and s cell states (0 empty, 1 queen).
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

# data, rule checker and GRAM's metrics (shared with the scoring of the baselines); re-exported for the run scripts
from cram.nqueens_data import (DATA, carve_validation, completes, completions, gram_metrics, load_split,  # noqa: F401
                                     load_tokens, valid)


# ----------------------------------------------------------------------------------------------------- the energy

def pair_index(c, device):
    """(2, p) int64: the cells (i, j), i < j, of the p = c (c - 1) / 2 unordered pairs, in row-major order."""
    return torch.triu_indices(c, c, 1, device=device)


def full_S(W, iu, c):
    """(c, c, a, s) from W (p, a, s): S[i, j] = W_p, S[j, i] = W_p^T, zero on the diagonal."""
    S = torch.zeros(c, c, 2, 2, device=W.device, dtype=W.dtype)
    S = S.index_put((iu[0], iu[1]), W)
    return S.index_put((iu[1], iu[0]), W.transpose(1, 2))


def fields(S, B, x):
    """(b, c, a): energy of the board with cell i set to a, less a constant: sum_j S[i, j, a, x_j] + B[i, a]."""
    return torch.einsum('ijas,bjs->bia', S, F.one_hot(x.long(), 2).to(S.dtype)) + B[None]


def board_energy(S, B, x):
    """(b,): E(x)."""
    g = fields(S, B, x)
    own = g.gather(2, x.long()[..., None])[..., 0]
    return 0.5 * own.sum(1) + 0.5 * B[torch.arange(B.shape[0], device=B.device), x.long()].sum(1)


def synchronize(device):
    """Wait for queued GPU work, so that a wall-clock reading covers it."""
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def fit(inputs, labels, *, steps=32, lr=10., lam=0., momentum=.5, cosine=True, device='cpu'):
    """The complete fit, timed: host arrays (inputs, labels: (r, c), bool or GRAM's tokens) to the inference-ready model.

    Inside the clock: transfer, the reduction to distinct solutions and free-cell counts, initialization, every update,
    the assembly of S, tau and the temperature unit. Deterministic (full batch, zero start).
    steps: full-batch updates; lr: initial learning rate (cosine decay to 0 when `cosine`); lam: weight of the L1 term.
    Returns (model, info): model holds W (p, 2, 2), B (c, 2), S (c, c, 2, 2), tau, unit and delta; info holds the fit
    time in seconds, the final loss and the settings."""
    synchronize(device)
    t0 = time.perf_counter()
    dev = torch.device(device)
    if inputs.dtype != bool:  # GRAM's tokens (1 empty, 2 queen): the conversion is part of the timed fit
        inputs, labels = inputs == 2, labels == 2
    x_in = torch.as_tensor(inputs, device=dev).bool()
    y = torch.as_tensor(labels, device=dev).to(torch.uint8)
    sol, which = torch.unique(y, dim=0, return_inverse=True)  # (u, c), (r,)
    u, c = sol.shape
    free = torch.zeros(u, c, device=dev).index_add_(0, which, (~x_in).float())  # rows with that solution and cell free
    weight = free / free.sum()
    iu = pair_index(c, dev)
    W = torch.zeros(iu.shape[1], 2, 2, device=dev, requires_grad=True)
    B = torch.zeros(c, 2, device=dev, requires_grad=True)
    bufs = [torch.zeros_like(W), torch.zeros_like(B)]
    target = sol.long()
    loss = None
    for k in range(steps):
        g = fields(full_S(W, iu, c), B, target)
        nll = -torch.log_softmax(-g, -1).gather(-1, target[..., None])[..., 0]
        loss = (nll * weight).sum()
        gW, gB = torch.autograd.grad(loss, (W, B))
        lr_k = lr * 0.5 * (1 + math.cos(math.pi * k / steps)) if cosine else lr
        with torch.no_grad():
            for p, grad, buf in zip((W, B), (gW, gB), bufs):
                buf.mul_(momentum).add_(grad)
                p.sub_(lr_k * buf)
                if lam:
                    p.copy_(p.sign() * (p.abs() - lr_k * lam).clamp_min(0))
    with torch.no_grad():
        W, B = W.detach(), B.detach()
        S = full_S(W, iu, c)
        g = fields(S, B, target)
        tau = board_energy(S, B, target).max()
        rows = torch.bincount(which, minlength=u).float()
        unit = ((g[..., 1] - g[..., 0]).abs().mean(1) * rows).sum() / rows.sum()
        own = g.gather(2, target[..., None])[..., 0]
        other = g.gather(2, (1 - target)[..., None])[..., 0]
        delta = (other - own).min()  # smallest margin by which a training solution's cell holds the lower field
        loss_value = float(loss) if loss is not None else math.log(2)
    synchronize(device)
    seconds = time.perf_counter() - t0
    model = dict(W=W.cpu(), B=B.cpu(), S=S.cpu(), tau=float(tau), unit=float(unit), delta=float(delta))
    info = dict(fit_seconds=seconds, pl=loss_value, distinct_solutions=u, rows=len(y), steps=steps, lr=lr, lam=lam,
                momentum=momentum, cosine=cosine, device=device,
                params_learned=int(W.numel() + B.numel()), params_nonzero=int((W != 0).sum() + (B != 0).sum()),
                tau=float(tau), unit=float(unit), delta=float(delta))
    return model, info


# ----------------------------------------------------------------------------------------------------- the search

def move_flop(c):
    """Arithmetic per move: c focus differences, 2 x 6 for the two Gumbel heat-bath logits (division, two logs, two
    negations, one addition each), 2 x 2c field update (subtract the old column, add the new), 2 energy update."""
    return 5 * c + 14


def init_flop(c):
    """Per chain: initial fields c additions per (cell, state) = 2c^2 (dense, zeros included); energy 2c + 1."""
    return 2 * c * c + 2 * c + 1


@torch.no_grad()
def solve(model, givens, *, chains=1, temperature=.3, cap=20000, seed=0, device='cpu', margin=1e-3, block=1 << 16,
          stop='energy'):
    """Focused heat-bath Gibbs on the learned energy, `chains` independent chains per row of `givens` ((r, c) bool).

    A chain starts from the board holding only the given queens; given cells are never moved. Each move picks a cell
    uniformly among the free cells whose current state has the higher field (flipping it lowers the energy), or among
    all free cells when there is none; then it draws the cell's state from exp(-field / T), T = temperature * unit.
    Stopping test, both fitted from the training solutions inside the fit: `stop='energy'`: E <= tau + margin (tau: the
    highest training-solution energy); `stop='local'`: every free cell holds the state with the lower field by at
    least delta - margin (delta: the smallest such margin over every cell of every training solution), i.e. the board
    is a fixed point of the learned conditionals at least as sharp as every training solution (it reads the focus
    differences the move computes anyway, so it costs no arithmetic). A chain that reaches `cap` moves returns the
    lowest-energy board it visited. Only the learned energy steers the search.
    Flop (see move_flop, init_flop): every move of every chain, counted per chain; RNG, comparisons and indexing are
    not counted. Returns boards (r, chains, c) bool, stopped (r, chains), energy (r, chains), moves, flop (r, chains)."""
    dev = torch.device(device)
    S = model['S'].to(dev)
    B = model['B'].to(dev)
    c = S.shape[0]
    T = temperature * model['unit']
    thresh = model['tau'] + margin
    if stop not in ('energy', 'local'):
        raise ValueError(stop)
    delta = model.get('delta', 0.) - margin

    def passes(x, g, e, free):
        if stop == 'energy':
            return e <= thresh
        own = g.gather(2, x[..., None])[..., 0]
        other = g.gather(2, (1 - x)[..., None])[..., 0]
        return ((other - own >= delta) | ~free).all(1)
    Sg = S.permute(1, 3, 0, 2).contiguous()  # (j, s, i, a): the field change of every cell when cell j takes state s
    gen = torch.Generator(device=dev).manual_seed(seed)
    giv = torch.as_tensor(givens, device=dev).bool().repeat_interleave(chains, 0)  # (b, c), chain-major per row
    total = giv.shape[0]
    out_x = torch.zeros(total, c, dtype=torch.bool, device=dev)
    out_e = torch.zeros(total, device=dev)
    out_stop = torch.zeros(total, dtype=torch.bool, device=dev)
    out_moves = torch.zeros(total, dtype=torch.long, device=dev)
    for lo in range(0, total, block):
        ids = torch.arange(lo, min(lo + block, total), device=dev)
        x = giv[ids].long()
        free = ~giv[ids]
        g = fields(S, B, x)
        e = 0.5 * g.gather(2, x[..., None])[..., 0].sum(1) + 0.5 * B[torch.arange(c, device=dev), x].sum(1)
        best_x, best_e = x.clone(), e.clone()
        moves = torch.zeros(len(ids), dtype=torch.long, device=dev)
        stop_now = passes(x, g, e, free)
        live = ~stop_now
        out_x[ids[stop_now]] = x[stop_now].bool()
        out_e[ids[stop_now]] = e[stop_now]
        out_stop[ids[stop_now]] = True
        keep = live
        ids, x, free, g, e, best_x, best_e, moves = (t[keep] for t in (ids, x, free, g, e, best_x, best_e, moves))
        step = 0
        while len(ids) and step < cap:
            a = torch.arange(len(ids), device=dev)
            own = g.gather(2, x[..., None])[..., 0]
            other = g.gather(2, (1 - x)[..., None])[..., 0]
            unhappy = (own > other) & free
            score = torch.rand(x.shape, generator=gen, device=dev)
            pool = torch.where(unhappy.any(1, keepdim=True), unhappy, free)
            cell = torch.where(pool, score, torch.full_like(score, -1.)).argmax(1)
            local = g[a, cell]  # (b, a)
            gumbel = -torch.log(-torch.log(torch.rand(local.shape, generator=gen, device=dev).clamp_min(1e-30)))
            new = (-local / T + gumbel).argmax(1)
            old = x[a, cell]
            e = e + local[a, new] - local[a, old]
            g = g + Sg[cell, new] - Sg[cell, old]
            x[a, cell] = new
            moves += 1
            better = e < best_e
            best_x[better], best_e[better] = x[better], e[better]
            step += 1
            done = passes(x, g, e, free)
            if done.any() or step == cap:
                fin = done if step < cap else torch.ones_like(done)
                res_x = torch.where(done[:, None], x, best_x)
                res_e = torch.where(done, e, best_e)
                out_x[ids[fin]] = res_x[fin].bool()
                out_e[ids[fin]] = res_e[fin]
                out_stop[ids[fin]] = done[fin]
                out_moves[ids[fin]] = moves[fin]
                keep = ~fin
                ids, x, free, g, e, best_x, best_e, moves = (t[keep] for t in (ids, x, free, g, e, best_x, best_e,
                                                                               moves))
    r = len(givens)
    moves = out_moves.view(r, chains).cpu().numpy()
    flop = init_flop(c) + moves * move_flop(c)
    return (out_x.view(r, chains, c).cpu().numpy(), out_stop.view(r, chains).cpu().numpy(),
            out_e.view(r, chains).cpu().numpy(), moves, flop.astype(np.float64))


# ---------------------------------------------------------------------------------------- what the couplings learned
# These two functions use the rules of N-Queens. They describe a fitted model and are never used by a fit or a search.

def attack_classes(n):
    """(p,) int8 class of every unordered cell pair (triu order): 0 none, 1 row, 2 column, 3 diagonal, 4 anti-diagonal."""
    i, j = np.triu_indices(n * n, 1)
    ri, ci, rj, cj = i // n, i % n, j // n, j % n
    cls = np.zeros(len(i), np.int8)
    cls[(ri - ci) == (rj - cj)] = 3
    cls[(ri + ci) == (rj + cj)] = 4
    cls[ci == cj] = 2
    cls[ri == rj] = 1
    return cls


CLASS_NAMES = ('none', 'row', 'column', 'diagonal', 'anti-diagonal')


def coupling_summary(W, B, n):
    """What the couplings learned. J_p = W_p[1,1] - W_p[1,0] - W_p[0,1] + W_p[0,0] is the part of W_p that no input cost can absorb
    (the queen-queen interaction; E = const + sum_i h_i x_i + sum_p J_p x_i x_j). Per pair class: raw support (any of
    the four entries nonzero), J support, and the range of J and of each entry.
    W (p, 2, 2), B (c, 2) as arrays; returns a dict of counts and ranges."""
    W = np.asarray(W, np.float64)
    B = np.asarray(B, np.float64)
    cls = attack_classes(n)
    J = W[:, 1, 1] - W[:, 1, 0] - W[:, 0, 1] + W[:, 0, 0]
    raw_nz = (W != 0).any((1, 2))
    attack = cls > 0
    per = {}
    for k, name in enumerate(CLASS_NAMES):
        m = cls == k
        if not m.any():
            continue
        per[name] = dict(pairs=int(m.sum()), raw_nonzero_pairs=int(raw_nz[m].sum()),
                         entries_nonzero=int((W[m] != 0).sum()), J_nonzero=int((J[m] != 0).sum()),
                         J_positive=int((J[m] > 0).sum()), J_negative=int((J[m] < 0).sum()),
                         J_min=float(J[m].min()), J_max=float(J[m].max()), J_mean=float(J[m].mean()),
                         entry_min=[[float(W[m][:, a, s].min()) for s in range(2)] for a in range(2)],
                         entry_max=[[float(W[m][:, a, s].max()) for s in range(2)] for a in range(2)])
    # the field of a queen against an empty board: h_i = B[i,1] - B[i,0] + sum_j (W_ij[1,0] - W_ij[0,0]) (oriented)
    c = n * n
    i, j = np.triu_indices(c, 1)
    h = B[:, 1] - B[:, 0]
    np.add.at(h, i, W[:, 1, 0] - W[:, 0, 0])
    np.add.at(h, j, W[:, 0, 1] - W[:, 0, 0])
    support_is_attack = bool((raw_nz == attack).all())
    J_support_is_attack = bool(((J > 0) == attack).all() and not (J < 0).any())
    return dict(classes=per, raw_support_equals_attack_pairs=support_is_attack,
                positive_J_support_equals_attack_pairs=J_support_is_attack,
                raw_nonzero_pairs=int(raw_nz.sum()), attack_pairs=int(attack.sum()),
                raw_nonzero_outside_attack=int((raw_nz & ~attack).sum()),
                attack_pairs_raw_zero=int((~raw_nz & attack).sum()),
                B_nonzero=int((B != 0).sum()), B_range=[float(B.min()), float(B.max())],
                h_range=[float(h.min()), float(h.max())], h_mean=float(h.mean()),
                params_learned=int(W.size + B.size), params_nonzero=int((W != 0).sum() + (B != 0).sum()))
