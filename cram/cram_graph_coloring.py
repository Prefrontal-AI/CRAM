"""Graph-coloring CRAM on GRAM's data: the energy, its pseudolikelihood fit, the search and the scoring.
`graph_coloring_data` gives the data source, the encoding and the splits.

Dimension names: b graphs (rows), n vertices, p vertex pairs i < j (row-major, as GRAM's upper-triangle tokens),
t input tokens (the p off-diagonal adjacency tokens, edge = 1, no edge = 0), c and d colors (C = 3), r search chains.

The energy of a coloring y of a graph with edge indicators a (one per vertex pair) is
  E(y) = sum_i B_i[y_i] + sum_(p = (i, j)) K_p(a)[y_i, y_j],
where B_i (3,) is the cost of each color at vertex i and K_p(a) is a 3 x 3 table that is an affine function of the
adjacency tokens. Three arms differ only in what K_p may read (no arm is told that adjacent vertices must differ):
  full:   K_p(a) = W0_p + sum_t a_t D_{p,t}   every pair reads every token; nothing routes token p to pair p
  pos:    K_p(a) = W0_p + a_p D_p             pair p reads its own token (a free table per pair for edge and non-edge)
  shared: K_p(a) = W0 + a_p D                 one non-edge table W0 and one edge table W0 + D for all pairs, and one
                                              bias B shared by all vertices (vertex-exchangeable)
Training: pseudolikelihood (PL) at the labeled colorings plus lam * L1 over every parameter, full-batch SGD with
momentum and cosine decay from a zero start, the L1 term as the proximal soft-threshold after each update (the recipe
of the Sudoku model). No solver, no augmentation (the labels are in GRAM's canonical color order, which relabeling
breaks).
Inference: focused heat-bath search on the learned energy from random colorings. A chain stops at the first coloring
that passes the stopping test: stop='tau', E <= tau + 1e-3 with tau the highest energy of a training coloring; or
stop='terms', every pair term and every vertex cost at most the highest value it takes on a training coloring
(`term_thresholds`). Both tests are fitted from the training colorings inside the fit. The in-code operation
estimate counts each scalar +, -, *, /, log as one flop (a multiply-add is two) and excludes random draws,
comparisons and indexing.
"""
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Data loading and the rule checker live in graph_coloring_data (NumPy only), which the GRAM baseline also uses.
from cram.graph_coloring_data import (C, DATA, SOURCE, adjacency, all_colorings, canonical, conflicts,  # noqa: F401
                                            convert, is_canonical, load_split, offdiag_positions, pairs,
                                            proper_canonical_sets, validation_graphs)

ARMS = ('full', 'pos', 'shared')


# ----------------------------------------------------------------------------------------------------------------------
# Energy

def init_params(arm, n, device='cpu', dtype=torch.float32):
    """The parameters of one arm at their zero start, as a dict of tensors: W0 and D (the constant and the
    token-dependent part of K) and B (the vertex costs); the shapes are those of the module docstring."""
    P = n * (n - 1) // 2
    z = lambda *s: torch.zeros(*s, device=device, dtype=dtype)
    if arm == 'full':
        return dict(W0=z(P, C, C), D=z(P, P, C, C), B=z(n, C))
    if arm == 'pos':
        return dict(W0=z(P, C, C), D=z(P, C, C), B=z(n, C))
    if arm == 'shared':
        return dict(W0=z(C, C), D=z(C, C), B=z(C))
    raise ValueError(arm)


def n_params(arm, n):
    """Number of learned values of one arm for graphs of n vertices."""
    return sum(v.numel() for v in init_params(arm, n, 'meta').values())


def assemble(prm, a):
    """K (b, p, c, d) for edge indicators a (b, p)."""
    W0, D = prm['W0'], prm['D']
    if D.dim() == 4:  # full
        return W0 + torch.einsum('bt,ptcd->bpcd', a, D)
    return W0 + a[..., None, None] * D  # pos: per-pair tables; shared: (c, d) tables broadcast over pairs


def bias(prm, n):
    """B (n, c): the vertex costs (the shared arm's (c,) vector broadcast over the vertices)."""
    B = prm['B']
    return B.expand(n, C) if B.dim() == 1 else B


def pair_tensor(K, n):
    """Symmetric S (b, n, n, c, d) with S[i, j] = K_p and S[j, i] = K_p^T; zero diagonal blocks."""
    I, J = pairs(n)
    S = K.new_zeros(K.shape[0], n, n, C, C)
    S[:, I, J] = K
    S[:, J, I] = K.transpose(-1, -2)
    return S


def fields(S, B, y):
    """f (b, n, c): the energy of each color of vertex i with every other vertex at y."""
    return B + torch.einsum('bijcd,bjd->bic', S, F.one_hot(y, C).to(S.dtype))


def energy_t(S, B, y):
    """E (b,) of colorings y (b, n) for the pair tensor S (b, n, n, c, d) and the vertex costs B (n, c)."""
    f = fields(S, B, y)
    own = f.gather(2, y[..., None])[..., 0]
    b_own = B.expand(y.shape[0], -1, -1).gather(2, y[..., None])[..., 0]
    return .5 * (own + b_own).sum(1)


def pl_loss(prm, a, y, n):
    """The pseudolikelihood loss (scalar): the mean over graphs and vertices of the cross-entropy of
    softmax_c(-f[i, c]) against the labeled color y_i, with f the fields at the label. a (b, p), y (b, n)."""
    S = pair_tensor(assemble(prm, a), n)
    f = fields(S, bias(prm, n), y)
    return F.cross_entropy(-f.flatten(0, 1), y.flatten())


def soft_threshold(x, t):
    """The proximal map of t * |x|: shrink every entry toward 0 by t."""
    return x.sign() * (x.abs() - t).clamp_min(0)


def synchronize(device):
    """Wait for queued GPU work, so that a wall-clock reading covers it."""
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()


def fit(x_host, y_host, n, *, arm, updates, lr, lam, momentum=.5, device='cpu', theta=.45, rows=None):
    """One complete fit, timed from GRAM's host arrays to the inference-ready model and its statistics.

    Timed: token conversion, transfer, zero start, `updates` full-batch SGD updates (momentum, cosine lr, proximal
    L1), and the inference statistics fitted from the training colorings: tau (highest energy), ebar (mean own local
    energy) and hbar (mean positive coupling entry, the unit of temperature and focus threshold). `rows` selects
    training rows (the validation screen); None uses all.
    x_host, y_host: GRAM's token arrays of `load_split`; lr: initial learning rate; lam: weight of the L1 term;
    theta: focus threshold in units of hbar, stored in the model for `solve`.
    Returns (model, info): model is a dict of NumPy arrays (W0, D, B, the term thresholds tpair, tvert) and scalars
    (arm, n, theta, tau, ebar, hbar, final_pl); info holds the fit time in seconds and the settings.
    """
    synchronize(device)
    t0 = time.perf_counter()
    a_np, y_np = convert(x_host if rows is None else x_host[rows], y_host if rows is None else y_host[rows], n)
    a = torch.as_tensor(a_np, device=device)
    y = torch.as_tensor(y_np, device=device)
    prm = init_params(arm, n, device)
    for v in prm.values():
        v.requires_grad_(True)
    bufs = {k: torch.zeros_like(v) for k, v in prm.items()}
    loss = None
    for k in range(updates):
        loss = pl_loss(prm, a, y, n)
        grads = torch.autograd.grad(loss, list(prm.values()))
        lr_k = lr * .5 * (1 + math.cos(math.pi * k / updates))
        with torch.no_grad():
            for (name, p), g in zip(prm.items(), grads):
                bufs[name].mul_(momentum).add_(g)
                p.sub_(lr_k * bufs[name])
                if lam:
                    p.copy_(soft_threshold(p, lr_k * lam))
    with torch.no_grad():
        prm = {k: v.detach() for k, v in prm.items()}
        K = assemble(prm, a)
        S = pair_tensor(K, n)
        B = bias(prm, n)
        E = energy_t(S, B, y)
        own = fields(S, B, y).gather(2, y[..., None])
        pos = K[K > 0]
        tpair, tvert = term_thresholds(K, B, a, y, n, shared=arm == 'shared')
        stats = dict(tau=float(E.max()), ebar=float(own.mean()),
                     hbar=float(pos.mean()) if pos.numel() else float('nan'),
                     final_pl=float(pl_loss(prm, a, y, n)) if loss is not None else float(math.log(C)))
        terms = dict(tpair=tpair.cpu().numpy(), tvert=tvert.cpu().numpy())
    synchronize(device)
    seconds = time.perf_counter() - t0
    model = {k: v.cpu().numpy() for k, v in prm.items()} | terms | dict(arm=arm, n=n, theta=theta, **stats)
    return model, dict(fit_seconds=seconds, updates=updates, lr=lr, lam=lam, momentum=momentum, device=str(device),
                       train_rows=int(len(y_np)), **stats)


def term_thresholds(K, B, a, y, n, shared=False):
    """The term test's thresholds, fitted from the training colorings: for each vertex pair p and own-token state s
    (0 no edge, 1 edge), the highest value K_p(a)[y_i, y_j] takes on a training coloring whose graph has a_p = s; for
    each vertex i, the highest B_i[y_i]. `shared` (the vertex-exchangeable arm): one threshold per state over all pairs
    and one over all vertices. A state that never occurs takes the highest value of that state over all pairs."""
    I, J = pairs(n)
    b = torch.arange(len(y), device=K.device)[:, None]
    p = torch.arange(len(I), device=K.device)[None]
    held = K[b, p, y[:, I], y[:, J]]  # (b, p)
    tp = []
    for state in (0, 1):
        v = torch.where(a == state, held, torch.full_like(held, -torch.inf)).amax(0)  # (p,)
        top = v.max()
        v = torch.where(torch.isinf(v), top, v)
        tp.append(top.expand_as(v) if shared else v)
    bv = B[torch.arange(n, device=K.device)[None], y]  # (b, n)
    tv = bv.amax(0)
    return torch.stack(tp, 1), (tv.max().expand_as(tv) if shared else tv)


def accepts_terms(model, a, boards):
    """The term test: every pair term and every input cost at most its threshold (+1e-6). boards (b, n) for graphs a."""
    n = boards.shape[1]
    I, J = pairs(n)
    K = assemble_np(model, a)
    b = np.arange(len(a))[:, None]
    pt = K[b, np.arange(len(I))[None], boards[:, I], boards[:, J]]
    thr = model['tpair'][np.arange(len(I))[None], (a > 0).astype(np.int64)]
    bt = bias_np(model, n)[np.arange(n)[None], boards]
    return (pt <= thr + 1e-6).all(1) & (bt <= model['tvert'][None] + 1e-6).all(1)


# ----------------------------------------------------------------------------------------------------------------------
# NumPy energy and search (float64)

def assemble_np(model, a):
    """`assemble` in NumPy float64: K (b, p, c, d) for edge indicators a (b, p)."""
    W0, D = model['W0'].astype(np.float64), model['D'].astype(np.float64)
    if D.ndim == 4:
        return W0 + np.einsum('bt,ptcd->bpcd', a, D)
    return W0 + a[..., None, None] * D


def pair_tensor_np(K, n):
    """`pair_tensor` in NumPy: S (b, n, n, c, d) with S[i, j] = K_p and S[j, i] = K_p^T."""
    I, J = pairs(n)
    S = np.zeros((K.shape[0], n, n, C, C))
    S[:, I, J] = K
    S[:, J, I] = K.transpose(0, 1, 3, 2)
    return S


def bias_np(model, n):
    """`bias` in NumPy float64: B (n, c)."""
    B = model['B'].astype(np.float64)
    return np.broadcast_to(B, (n, C)) if B.ndim == 1 else B


def energy_np(model, a, y):
    """E (b,) of colorings y (b, n) for graphs a (b, p), as a direct sum over vertices and vertex pairs."""
    n = y.shape[1]
    S = pair_tensor_np(assemble_np(model, a), n)
    B = bias_np(model, n)
    I, J = pairs(n)
    b = np.arange(len(y))[:, None]
    return B[np.arange(n), y].sum(1) + S[b, I, J, y[:, I], y[:, J]].sum(1)


def assembly_flop(model, n):
    """Per graph: K from the tokens (dense, zeros included), multiply-add as 2."""
    P = n * (n - 1) // 2
    if model['D'].ndim == 4:
        return 2 * P * P * C * C + P * C * C
    return 2 * P * C * C


def move_flop(n):
    """flop of one move: focus offsets (n), heat-bath logits (6 per color: a division, two logarithms, two negations
    and an addition), field update 2(n-1)C, energy update 2."""
    return n + 6 * C + 2 * (n - 1) * C + 2


def init_flop(n):
    """flop of the start of one chain: initial fields (n(n-1)C additions plus nC bias additions) and energy (2n)."""
    return n * (n - 1) * C + n * C + 2 * n


def solve(model, a, *, samples, temperature, cap, seed, stop='tau'):
    """`samples` independent chains per graph; each stops at its first coloring that passes the stopping test:
    stop='tau': E <= tau + 1e-3 (one threshold on the whole energy); stop='terms': the term test (accepts_terms: every
    pair term and input cost at most its fitted threshold; comparisons and lookups only, no flop).

    A move picks a vertex uniformly among those whose own-color field exceeds ebar + theta * hbar (among all vertices
    when there is none) and draws its color from softmax(-field / T), T = temperature * hbar.
    model: the dict of `fit`; a (b, p) edge indicators; cap: the move limit per chain; seed: seed of the NumPy generator.

    Returns a dict: boards (b, samples, n) (a chain that reaches `cap` moves returns its lowest-energy coloring), and
    per chain (b, samples): energy, stopped, moves, chain_flop (chain work: initial fields and energy plus moves; the
    graph's coupling assembly is done once per graph and returned apart as assembly_flop).
    """
    if temperature <= 0 or cap < 0 or samples < 1:
        raise ValueError('positive temperature and samples required')
    n = int(round((1 + math.sqrt(1 + 8 * a.shape[1])) / 2))
    rng = np.random.default_rng(seed)
    S = pair_tensor_np(assemble_np(model, a), n)
    B = bias_np(model, n)
    nb = len(a)
    inst = np.repeat(np.arange(nb), samples)
    R = len(inst)
    T = temperature * model['hbar']
    focus_at = model['ebar'] + model['theta'] * model['hbar']
    thr = model['tau'] + 1e-3
    y = rng.integers(C, size=(R, n))
    ar = np.arange(n)
    # g[r, i, c] = B[i, c] + sum_j S[i, j][c, y_j] (advanced indices around a slice: the slice axis goes last)
    g = B[None] + S[inst[:, None, None], ar[None, :, None], ar[None, None, :], :, y[:, None, :]].sum(2)
    E = .5 * (np.take_along_axis(g, y[..., None], 2)[..., 0] + B[ar, y]).sum(1)
    best, bestE = y.copy(), E.copy()
    if stop not in ('tau', 'terms'):
        raise ValueError(stop)
    if stop == 'terms':
        I, J = pairs(n)
        thr_p = model['tpair'][np.arange(len(I))[None], (a > 0).astype(np.int64)]  # (b, p)
        thr_v = model['tvert'] + 1e-6

        def passes(rows):
            si = inst[rows]
            pt = S[si[:, None], I[None], J[None], y[rows][:, I], y[rows][:, J]]
            return (pt <= thr_p[si] + 1e-6).all(1) & (B[ar[None], y[rows]] <= thr_v[None]).all(1)
    else:
        def passes(rows):
            return E[rows] <= thr
    stopped = passes(np.arange(R))
    moves = np.zeros(R, np.int64)
    for _ in range(cap):
        act = np.flatnonzero(~stopped)
        if len(act) == 0:
            break
        e = np.take_along_axis(g[act], y[act, :, None], 2)[..., 0]
        hot = e > focus_at
        score = rng.random(hot.shape)
        score = np.where(hot.any(1)[:, None], np.where(hot, score, -1.), score)
        v = score.argmax(1)
        local = g[act, v]
        u = rng.uniform(np.finfo(float).tiny, 1, size=local.shape)
        new = (-local / T - np.log(-np.log(u))).argmax(1)
        old = y[act, v]
        k = np.arange(len(act))
        E[act] += local[k, new] - local[k, old]
        si = inst[act]
        g[act] += (S[si[:, None, None], ar[None, :, None], v[:, None, None], np.arange(C)[None, None, :], new[:, None, None]]
                   - S[si[:, None, None], ar[None, :, None], v[:, None, None], np.arange(C)[None, None, :], old[:, None, None]])
        y[act, v] = new
        moves[act] += 1
        better = act[E[act] < bestE[act]]
        best[better], bestE[better] = y[better], E[better]
        stopped[act] = passes(act)
    out = np.where(stopped[:, None], y, best)
    outE = np.where(stopped, E, bestE)
    flop = init_flop(n) + moves * move_flop(n)
    return dict(boards=out.reshape(nb, samples, n), energy=outE.reshape(nb, samples),
                stopped=stopped.reshape(nb, samples), moves=moves.reshape(nb, samples),
                chain_flop=flop.reshape(nb, samples), assembly_flop=assembly_flop(model, n))


# ----------------------------------------------------------------------------------------------------------------------
# Scoring (GRAM's Table 1 metrics on CRAM's chains; the checker is graph_coloring_data's)


def score(A, res, ks=(1, 20)):
    """GRAM's Table 1 metrics plus rule-checked solve rates. Sample 0 is the single-output (N = 1) prediction.
    A (b, n, n) adjacency matrices; res: the dict of `solve`; ks: the sample counts to report. Returns a dict of
    scalars: proper_1 (fraction of graphs whose first chain is a proper coloring), coverage_k (distinct proper
    canonical colorings found by the first k chains over all that exist), conflicts, wrong stops, budget failures
    and the mean flop per graph."""
    b = res['boards']
    nb, ns, n = b.shape
    conf = conflicts(A, b)
    proper = conf == 0
    canon = is_canonical(b)
    sols = proper_canonical_sets(A)
    total = sum(len(s) for s in sols)
    out = dict(graphs=nb, samples=ns, solutions_total=total,
               proper_1=float(proper[:, 0].mean()), proper_canonical_1=float((proper & canon)[:, 0].mean()),
               conflicts_1=int(conf[:, 0].sum()), wrong_stops_1=int((res['stopped'][:, 0] & ~proper[:, 0]).sum()),
               budget_failures_1=int((~res['stopped'][:, 0]).sum()),
               flop_1=float(res['assembly_flop'] + res['chain_flop'][:, 0].mean()))
    for k in ks:
        if k > ns:
            continue
        found = found_c = maj_conf = 0
        for g in range(nb):
            bs, es = b[g, :k], res['energy'][g, :k]
            found += len({tuple(r) for r, ok in zip(bs, proper[g, :k] & canon[g, :k]) if ok})
            found_c += len({tuple(r) for r in canonical(bs[proper[g, :k]])})
            keys = [r.tobytes() for r in bs]
            stats = {}
            for s, (key, en) in enumerate(zip(keys, es)):
                cnt, tot, first = stats.get(key, (0, 0., s))
                stats[key] = (cnt + 1, tot + en, first)
            maj = max(stats.values(), key=lambda v: (v[0], -v[1] / v[0]))[2]  # frequency, then lower mean energy
            maj_conf += int(conf[g, maj])
        out[f'coverage_{k}'] = found / total
        out[f'coverage_canonicalized_{k}'] = found_c / total
        out[f'conflicts_majority_{k}'] = maj_conf
        out[f'proper_any_{k}'] = float(proper[:, :k].any(1).mean())
        out[f'flop_{k}'] = float(res['assembly_flop'] + res['chain_flop'][:, :k].sum(1).mean())
    out['wrong_stops_all'] = int((res['stopped'] & ~proper).sum())
    out['budget_failures_all'] = int((~res['stopped']).sum())
    out['moves_mean'] = float(res['moves'].mean())
    out['moves_max'] = int(res['moves'].max())
    return out


# ----------------------------------------------------------------------------------------------------------------------
# What the pair tables learned

def summarize(model, n):
    """Effective edge and non-edge tables per pair (own token only), and for the full arm the cross-token part."""
    W0, D = model['W0'].astype(np.float64), model['D'].astype(np.float64)
    P = n * (n - 1) // 2
    eye = np.eye(C, dtype=bool)
    if D.ndim == 4:
        own = D[np.arange(P), np.arange(P)]
        cross = D.copy()
        cross[np.arange(P), np.arange(P)] = 0
    elif W0.ndim == 3:
        own, cross = D, np.zeros(0)
    else:
        W0, own, cross = np.broadcast_to(W0, (P, C, C)), np.broadcast_to(D, (P, C, C)), np.zeros(0)
    non, edge = W0, W0 + own

    def grp(v):
        v = np.asarray(v, dtype=np.float64).ravel()
        if v.size == 0:
            return dict(n=0)
        return dict(n=int(v.size), nonzero=int((v != 0).sum()), min=float(v.min()), max=float(v.max()),
                    mean=float(v.mean()), mean_abs=float(np.abs(v).mean()))

    params = {k: model[k] for k in ('W0', 'D', 'B')}
    return dict(params=int(sum(v.size for v in params.values())),
                params_nonzero=int(sum((v != 0).sum() for v in params.values())),
                edge_diag=grp(edge[:, eye]), edge_off=grp(edge[:, ~eye]),
                non_diag=grp(non[:, eye]), non_off=grp(non[:, ~eye]), cross_token=grp(cross), bias=grp(model['B']),
                edge_mean_table=edge.mean(0).tolist(), non_mean_table=non.mean(0).tolist(),
                bias_table=np.asarray(model['B']).tolist(), tau=model['tau'], ebar=model['ebar'], hbar=model['hbar'])


def sha256(path):
    """Hex SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    """Write `value` as JSON to `path` through a temporary file, so that a reader never sees a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    os.replace(tmp, path)
