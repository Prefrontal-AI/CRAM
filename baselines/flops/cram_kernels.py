"""CRAM's search kernels transcribed line by line into PyTorch, so that FullCounter counts CRAM under the same rule as
every baseline (the operation-count convention of baselines/flops/fullcount.py). It calls no baseline code.

FullCounter sees aten operators only; a Triton kernel launch is invisible to it. Each function below repeats the loop
body of one Triton move kernel for one chain and one move, statement by statement, on the logical extents (n cells,
d digits, D neighbour slots; the kernels' power-of-two padding lanes are not counted). Each Triton expression maps to
the PyTorch operator with the same arithmetic: tl.where -> torch.where, tl.sum -> sum, tl.argmax -> argmax,
tl.maximum -> maximum, tl.rand -> torch.rand (random-number generation, 0 flop), loads and stores through computed
addresses -> indexing (0 flop). Integer address arithmetic and the random-number keys (sk, base) are data movement and
generator state, 0 flop, as indexing is for the baselines. Arithmetic counts as written in the source (an expression
written twice counts twice), including the mask multiplies (* d) and the Gumbel transform; comparisons, selections and
mask logic are recorded by FullCounter but not counted (the convention of fullcount.py). A digit extraction that the
source writes as a masked sum, tl.sum(tl.where(cols == a, x, 0)) = x[a], is indexing and is written x[a] here; each
such line cites the source expression.

The work outside the kernels (field refresh, restarts, the first board, the per-check bookkeeping) is PyTorch code in
the search classes and is counted by running it under FullCounter (count_general_overheads).

Output: one JSON file with the FullCounter summary of one move of each Sudoku kernel at D = 20, 30 and 31 neighbour
slots, and the out-of-kernel counts of the Sudoku search on the seed-0 model. baselines/flops/fc2_export.py and
cram_kernels_extra.py read it.

  ../../.venv/bin/python cram_kernels.py <out.json>          (from baselines/flops, on a compute node)

Dimension names: n cells, d digits, k neighbour slots (D), c chains.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fullcount import FullCounter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT))  # the cram package
import os  # noqa: E402
CWD = os.getcwd()  # output paths are relative to the caller's folder
os.chdir(ROOT / 'experiments')  # the run scripts resolve runs/ relative to experiments/


def _count(fn):
    """Run fn() under FullCounter and return the counter's summary dict."""
    with FullCounter() as c:
        fn()
    return c.summary()


# ---------------------------------------------------------------------------------------------------------------
# Sudoku, general energy (cram_sudoku_general._moves_general; the CRAM Sudoku-Extreme row of Table 2)
# ---------------------------------------------------------------------------------------------------------------
def move_general(st, n, d, D, focused=True):
    """One move of _moves_general for one chain. st: dict of tensors (G (n, d), y (n,), flag (n,), fmask (n,), free (n,),
    nbr (n, D), R (n, D, d, d), Bc (n, d), E, T, nf, found, thr, tau, move0, k)."""
    rows, cols = torch.arange(n), torch.arange(d)
    rmask = torch.ones(n, dtype=torch.bool)
    cmask = torch.ones(d, dtype=torch.bool)
    G, y, flag = st['G'], st['y'], st['flag']
    u0 = torch.rand(())
    idx = torch.minimum((u0 * st['nf']).to(torch.int32), st['nf'] - 1)
    cell = st['free'][idx]
    if focused:
        fl = flag != 0
        score = torch.where(fl, torch.rand(n), -1.0)
        cc = torch.argmax(score, 0)
        cell = torch.where(torch.max(score, 0).values >= 0.0, cc, cell)
    Li = G[cell] + st['Bc'][cell]
    uu = torch.rand(d)
    gum = -torch.log(-torch.log(torch.maximum(uu, torch.tensor(1e-20))))
    lg = torch.where(cmask, -Li / st['T'] + gum, float('-inf'))
    new = torch.argmax(lg, 0)
    old = y[cell]
    new = torch.where(st['found'] >= 0, old, new)
    dd = (new != old).to(torch.float32)
    nb = st['nbr'][cell].long()
    Rc = st['R'][cell]  # (k, d, d): R[cell, k, a, :]
    delta = (Rc[:, :, new] - Rc[:, :, old]) * dd
    Gn = G[nb] + delta
    G[nb] = Gn
    ynb = y[nb]
    own = Gn[torch.arange(D), ynb] + st['Bc'][nb, ynb]  # source: tl.sum(tl.where(cols[None, :] == ynb[:, None], Gn, 0.0), 1)
    fnb = st['fmask'][nb] != 0
    flag[nb] = ((own > st['thr']) & fnb).to(torch.int32)
    lnew = Li[new]  # source: tl.sum(tl.where(cols == new, Li, 0.0), 0)
    st['E'] = st['E'] + (lnew - Li[old]) * dd  # source: tl.sum(tl.where(cols == old, Li, 0.0), 0)
    y[cell] = new
    flag[cell] = (lnew > st['thr']).to(torch.int32)
    cert = (st['found'] < 0) & (st['E'] <= st['tau'])
    yv = y[rmask & cert]  # masked load
    st['ycert_mask'] = rmask & cert  # masked store (the source writes the mask twice)
    st['found'] = torch.where(cert, st['move0'] + st['k'] + 1, st['found'])
    return yv


def _general_state(n=81, d=9, D=30):
    """A synthetic state for move_general with the shapes of one chain; the counts depend only on the shapes."""
    g = torch.Generator().manual_seed(0)
    return dict(G=torch.randn(n, d, generator=g), y=torch.randint(0, d, (n,), generator=g), flag=torch.randint(0, 2, (n,), generator=g, dtype=torch.int32),
                fmask=torch.ones(n, dtype=torch.int32), free=torch.arange(n, dtype=torch.int32), nbr=torch.randint(0, n, (n, D), generator=g, dtype=torch.int32),
                R=torch.randn(n, D, d, d, generator=g), Bc=torch.randn(n, d, generator=g), E=torch.tensor(0.0), T=torch.tensor(0.3),
                nf=torch.tensor(n, dtype=torch.int32), found=torch.tensor(-1, dtype=torch.int32), thr=torch.tensor(0.1), tau=torch.tensor(-1e9),
                move0=torch.tensor(0, dtype=torch.int32), k=torch.tensor(0, dtype=torch.int32))


# ---------------------------------------------------------------------------------------------------------------
# Sudoku, sparse one-level energy (cram_sudoku_sparse._moves_sparse; the hand-set Sudoku row)
# ---------------------------------------------------------------------------------------------------------------
def move_sparse(st, n, d, D, focused=True, nobias=True):
    """One move of _moves_sparse for one chain. st: dict of tensors (G (n, d), y (n,), flag (n,), fmask (n,), free (n,),
    nbr (n, D), w (n, D), B (10, d), x (n,), P, T, hscale, nf, found, thr, tau, move0, k). nobias: the energy has no
    input costs (the hand-set energy)."""
    rows, cols = torch.arange(n), torch.arange(d)
    rmask = torch.ones(n, dtype=torch.bool)
    cmask = torch.ones(d, dtype=torch.bool)
    kmask = torch.ones(D, dtype=torch.bool)
    G, y, flag = st['G'], st['y'], st['flag']
    u0 = torch.rand(())
    idx = torch.minimum((u0 * st['nf']).to(torch.int32), st['nf'] - 1)
    cell = st['free'][idx]
    if focused:
        fl = flag != 0
        score = torch.where(fl, torch.rand(n), -1.0)
        cc = torch.argmax(score, 0)
        cell = torch.where(torch.max(score, 0).values >= 0.0, cc, cell)
    Gi = G[cell]
    uu = torch.rand(d)
    gum = -torch.log(-torch.log(torch.maximum(uu, torch.tensor(1e-20))))
    if nobias:
        lg = torch.where(cmask, Gi * (-st['hscale'] / st['T']) + gum, float('-inf'))
    else:
        Li = st['hscale'] * Gi - st['B'][st['x'][cell]]
        lg = torch.where(cmask, -Li / st['T'] + gum, float('-inf'))
    new = torch.argmax(lg, 0)
    old = y[cell]
    new = torch.where(st['found'] >= 0, old, new)
    dd = (new != old).to(torch.float32)
    nb = st['nbr'][cell].long()
    wv = st['w'][cell] * dd
    go = G[nb, old] - wv
    G[nb, old] = go
    gn = G[nb, new] + wv
    G[nb, new] = gn
    ynb = y[nb]
    own = torch.where(ynb == old, go, gn)
    changed = kmask & ((ynb == old) | (ynb == new))
    fnb = st['fmask'][nb] != 0
    flag[nb[changed]] = ((own > st['thr']) & fnb).to(torch.int32)[changed]
    gnew = Gi[new]  # source: tl.sum(tl.where(cols == new, Gi, 0.0), 0)
    st['P'] = st['P'] + (gnew - Gi[old]) * dd  # source: tl.sum(tl.where(cols == old, Gi, 0.0), 0)
    y[cell] = new
    flag[cell] = (gnew > st['thr']).to(torch.int32)
    cert = (st['found'] < 0) & (st['hscale'] * st['P'] <= st['tau'])
    yv = y[rmask & cert]
    st['ycert_mask'] = rmask & cert
    st['found'] = torch.where(cert, st['move0'] + st['k'] + 1, st['found'])
    return yv


def _sparse_state(n=81, d=9, D=20):
    """A synthetic state for move_sparse with the shapes of one chain; the counts depend only on the shapes."""
    g = torch.Generator().manual_seed(0)
    return dict(G=torch.randn(n, d, generator=g), y=torch.randint(0, d, (n,), generator=g), flag=torch.randint(0, 2, (n,), generator=g, dtype=torch.int32),
                fmask=torch.ones(n, dtype=torch.int32), free=torch.arange(n, dtype=torch.int32), nbr=torch.randint(0, n, (n, D), generator=g, dtype=torch.int32),
                w=torch.ones(n, D), B=torch.zeros(10, d), x=torch.zeros(n, dtype=torch.long), P=torch.tensor(0.0), T=torch.tensor(0.3),
                hscale=torch.tensor(1.0), nf=torch.tensor(n, dtype=torch.int32), found=torch.tensor(-1, dtype=torch.int32), thr=torch.tensor(0.1),
                tau=torch.tensor(-1e9), move0=torch.tensor(0, dtype=torch.int32), k=torch.tensor(0, dtype=torch.int32))


# ---------------------------------------------------------------------------------------------------------------
# Out-of-kernel PyTorch work of the Sudoku search (GeneralSearch): the first boards, and each check of `check` moves
# (field refresh, lowest-energy bookkeeping, stops), counted on the real search object with the kernel switched off.
# ---------------------------------------------------------------------------------------------------------------
def count_general_overheads(split=0):
    """Count the constructor and one check of GeneralSearch on one test puzzle with the model of seed `split`.

    Returns a dict with the neighbour slots D, the chains C, the moves per check and two FullCounter summaries:
    init_per_puzzle and check_per_puzzle."""
    import numpy as np
    import sudoku as CF
    from cram import cram_sudoku_general as GS
    from cram import cram_sudoku_search as SS
    W, B = CF.load_model(split, device='cpu')
    tsol, _, _ = SS.load_puzzles(f'train-s{split}')
    m = CF.prepare(W, B, torch.as_tensor(tsol))
    sol, free, _ = SS.load_puzzles('full-all', 10000, 1)
    unit = m['hbar']
    hi, lo_t, theta = 0.40, 0.31, 0.45  # the relative temperature schedule (signature 'rel'); costs do not depend on it
    kw = dict(temp=hi * unit, theta=theta * unit, focused=True, C=CF.FROZEN['C'], tau=m['tau_train'], tol=CF.FROZEN['tol'],
              ebar=m['ebar'], cap=CF.FROZEN['cap'], check=CF.FROZEN['check'], seed=split,
              anneal=(hi * unit, lo_t * unit, CF.FROZEN['anneal'][2]), device='cpu')
    holder = {}
    init = _count(lambda: holder.setdefault('ss', GS.GeneralSearch(m['S'], B, sol, free, **kw)))
    ss = holder['ss']
    ss._moves = lambda nmoves: None  # the kernel's moves are counted by move_general
    check = _count(ss.step_check)
    assert ss.t == CF.FROZEN['check'] and ss.t % 4096 == 0  # this check includes the periodic refresh
    return dict(D=ss.D, C=ss.C, check_moves=ss.check, fl_move_old=ss.fl_move, init_per_puzzle=init, check_per_puzzle=check,
                note='one puzzle with C chains; init = the constructor (first boards, their fields and energies); check = '
                     'step_check without the kernel (refresh of every chain at t % 4096 == 0, bookkeeping, stops)')


def main(out):
    """Count every component and write the record to `out` (JSON)."""
    res = {}
    for D in (20, 30, 31):
        st = _general_state(D=D)
        res[f'sudoku_general_move_D{D}'] = _count(lambda: move_general(st, 81, 9, D))
        st = _sparse_state(D=D)
        res[f'sudoku_sparse_move_nobias_D{D}'] = _count(lambda: move_sparse(st, 81, 9, D, nobias=True))
        st = _sparse_state(D=D)
        res[f'sudoku_sparse_move_bias_D{D}'] = _count(lambda: move_sparse(st, 81, 9, D, nobias=False))
    res['sudoku_general_overheads_s0'] = count_general_overheads(0)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1) + '\n')
    for k, v in res.items():
        print(k, json.dumps({kk: (vv['total'] if isinstance(vv, dict) and 'total' in vv else vv) for kk, vv in v.items() if kk not in ('by_op',)} if 'total' not in v else {'total': v['total'], 'classes': v['classes']})[:600])


if __name__ == '__main__':
    main(os.path.join(CWD, sys.argv[1]))
