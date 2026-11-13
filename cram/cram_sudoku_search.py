"""The Sudoku-Extreme data loader (`load_puzzles`) and the focused Gibbs search on the same-digit pair energy of
`cram_sudoku`: asynchronous single-cell heat-bath moves at temperature T, several chains and restarts per puzzle, and a
stopping test from the energy and the training data. `Search` is also the base class of the two searches that the
paper's rows use: `cram_sudoku_general.GeneralSearch` (the learned model) and `cram_sudoku_sparse.SparseSearch` (the
hand-set energy). They replace the move kernel and keep the protocol of `Search` (checks, stops, restarts, results).

Data. Sudoku-Extreme is the Hugging Face dataset `sapientinc/sudoku-extreme`. `scripts/prepare_data.py sudoku`
downloads `train.csv` and `test.csv` at a pinned revision and writes the split files that `load_puzzles` reads; the
docstring of that script describes every split, and `scripts/sudoku_extreme_splits.json` names the official rows of each.
A split file holds `solutions` (N, 81) digits 1-9 in row-major order and `clue_mask` (N, 81), 1 where the digit is given.
`load_puzzles` returns the solutions as digits 0-8 and the mask of the free (not given) cells.
Splits: `train-s<k>` 1,000 training puzzles (the paper uses k = 0, 1, 2 as its three seeds); `dev10k` 10,000 development
puzzles from the official test file; `full-all` the 422,786 puzzles of the official test file, of which rows 10,000 and
up (412,786 puzzles) are the test set of the paper. The loader applies no augmentation; the fit relabels digits
(`cram_sudoku.augment_digit`).
Token encoding inside the search: x_i = 0 at a free cell and y_i + 1 (1-9) at a clue cell; x selects the row of the
input-bias table B (t, d).

Dimension names: q puzzles, c chains per puzzle, h = q * c chains, n = 81 cells, d = 9 digits, t = 10 tokens.

Energy: E(y) = P(y) - sum_i B[x_i, y_i] with the pair energy P(y) = y.A y = 1/2 sum_(i != j) S_ij [y_i = y_j],
S = A + A^T (`cram_sudoku`). Each chain keeps the fields G[i, a] = sum_j S_ij [y_j = a] (h, n, d), so the energy change
of setting cell i to digit a is G[i, a] - B[x_i, a] less a constant.

A move (the Triton kernel `_moves`, one program per chain): pick a cell (focused: a random free cell whose own-digit
field G[i, y_i] exceeds ebar, a training-solution cell's mean own-digit field, by more than theta; else a random free
cell), draw its digit from softmax(-(G[cell] - B[x_cell]) / T), update the fields G = S Y of every cell (one column of
S), the pair energy P = y.A y and, for the per-term test, the count Q of same-digit pairs that block it.
Stopping tests (a chain is frozen at its first passing move, and its puzzle stops at the first check after it):
  pair   P <= tau + tol, tau the highest pair energy of the training solutions (the fitted threshold of the paper).
  term   Q = 0: every same-digit pair on the board has S_ij < -1e-6 (no fitted threshold; sound for PL from a zero
         start without weight sharing, see cram.cram_sudoku).
Restarts: every `restart` moves (times `growth` after each restart), each chain that has not passed draws a new random
board. A puzzle that never passes returns the lowest-P board its chains showed at a check.

Counted floating-point operations (flop; a multiply-add is 2 flop) of this dense kernel: per move 4n (fields) + 4d
(heat bath) + 2n (focus scores) + 2 n^2 d / 4096 (the exact field refresh every 4096 moves), + 4n for Q in the term
test; a restart pays one exact field product 2 n^2 d. A puzzle pays for all its chains up to its stop. The paper's
operation counts follow the convention of `baselines/flops/fullcount.py` and are rebuilt by `cram/opcount.py`; the
`flops` of a search result are this in-kernel estimate.
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

N, V = 81, 9  # cells, digits
FL_PRODUCT = 2 * N * N * V  # flop of one exact field product G = S Y


# The move kernel: one program per chain runs `nmoves` moves. Arguments (flattened tensors): S (n, n) symmetric
# couplings; M (n, n) the blocking mask of the per-term test; B (t, d) input biases; x (h, n) tokens; y (h, n) the
# board; G (h, n, d) fields; P (h,) pair energy; Q (h,) blocking-pair count; free (h, n) the free cells first (a
# stable sort); fmask (h, n) 1 at free cells; nfree (h,) number of free cells; temp, theta (h,) temperature and focus
# threshold; found (h,) the move at which the chain first passed its test, or -1; ycert (h, n) the board at that move;
# seed (1,); move0 the number of moves already made; ebar the focus reference; tau the threshold of the pair test.
# MODE 2: focused cell choice, 0: uniform. TERM: per-term test instead of the pair test. NB, VB: block sizes (powers of
# two that cover n and d). Random numbers come from tl.rand, keyed by (seed, move) and offset by the chain.
@triton.jit
def _moves(S_ptr, M_ptr, B_ptr, x_ptr, y_ptr, G_ptr, P_ptr, Q_ptr, free_ptr, fmask_ptr, nfree_ptr, temp_ptr, theta_ptr,
           found_ptr, ycert_ptr, seed_ptr, nmoves, move0, ebar, tau,
           MODE: tl.constexpr, TERM: tl.constexpr, N: tl.constexpr, V: tl.constexpr, NB: tl.constexpr, VB: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, NB)
    cols = tl.arange(0, VB)
    rmask = rows < N
    cmask = cols < V
    gbase = G_ptr + pid * N * V
    y = tl.load(y_ptr + pid * N + rows, mask=rmask, other=-1)
    fm = tl.load(fmask_ptr + pid * N + rows, mask=rmask, other=0) != 0
    e = tl.load(gbase + rows * V + tl.maximum(y, 0), mask=rmask, other=0.0)
    P = tl.load(P_ptr + pid)
    Q = tl.load(Q_ptr + pid)
    T = tl.load(temp_ptr + pid)
    nf = tl.load(nfree_ptr + pid)
    th = tl.load(theta_ptr + pid)
    found = tl.load(found_ptr + pid)
    sd = tl.load(seed_ptr)
    for k in range(nmoves):
        sk = sd * 65536 + k
        base = (pid * 512).to(tl.int32)
        u0 = tl.rand(sk, base)
        idx = tl.minimum((u0 * nf).to(tl.int32), nf - 1)
        cell = tl.load(free_ptr + pid * N + idx)
        if MODE == 2:
            conf = fm & ((e - ebar) > th)
            score = tl.where(conf, tl.rand(sk, base + 64 + rows), -1.0)
            cc = tl.argmax(score, 0)
            cell = tl.where(tl.max(score, 0) >= 0.0, cc, cell)
        Gi = tl.load(gbase + cell * V + cols, mask=cmask, other=0.0)
        xc = tl.load(x_ptr + pid * N + cell)
        Li = Gi - tl.load(B_ptr + xc * V + cols, mask=cmask, other=0.0)
        uu = tl.rand(sk, base + 16 + cols)
        lg = tl.where(cmask, -Li / T - tl.log(-tl.log(tl.maximum(uu, 1e-20))), float('-inf'))
        new = tl.argmax(lg, 0)
        old = tl.sum(tl.where(rows == cell, y, 0), 0)
        new = tl.where(found >= 0, old, new)
        d = (new != old).to(tl.float32)
        col = tl.load(S_ptr + cell * N + rows, mask=rmask, other=0.0) * d
        po = gbase + rows * V + old
        tl.store(po, tl.load(po, mask=rmask, other=0.0) - col, mask=rmask)
        pn = gbase + rows * V + new
        tl.store(pn, tl.load(pn, mask=rmask, other=0.0) + col, mask=rmask)
        tl.debug_barrier()
        gnew = tl.sum(tl.where(cols == new, Gi, 0.0), 0)
        P = P + (gnew - tl.sum(tl.where(cols == old, Gi, 0.0), 0)) * d
        if TERM:
            mrow = tl.load(M_ptr + cell * N + rows, mask=rmask, other=0.0) * d
            Q = Q + tl.sum(mrow * ((y == new).to(tl.float32) - (y == old).to(tl.float32)), 0)
        e = e - tl.where(y == old, col, 0.0) + tl.where(y == new, col, 0.0)
        y = tl.where(rows == cell, new, y)
        e = tl.where(rows == cell, gnew, e)
        if TERM:
            cert = (found < 0) & (Q < 0.5)
        else:
            cert = (found < 0) & (P <= tau)
        tl.store(ycert_ptr + pid * N + rows, y, mask=rmask & cert)
        found = tl.where(cert, move0 + k + 1, found)
    tl.store(y_ptr + pid * N + rows, y, mask=rmask)
    tl.store(P_ptr + pid, P)
    tl.store(Q_ptr + pid, Q)
    tl.store(found_ptr + pid, found)


def load_puzzles(split, lo=0, n=None):
    """Read rows lo .. lo + n of a Sudoku-Extreme split file (all rows from lo when n is None).
    split: 'dev10k' | 'full-all' | 'train-s<k>' (or another file of the split folder, such as 'dev' or 'test').
    The files are under `data/` (or `CRAM_DATA`); `FRM_DATA` overrides the folder of the `train-s<k>` files.
    Returns (solutions (q, n) int64 digits 0-8, free (q, n) bool: True where the cell is not a clue, the row indices
    (q,) in the split file)."""
    data = Path(os.environ.get('CRAM_DATA', Path(__file__).resolve().parents[1] / 'data'))
    path = {'dev10k': data / 'frm/dev10k/dev.npz', 'full-all': data / 'full-test/full-all.npz'}.get(split)
    if path is None:
        path = Path(os.environ.get('FRM_DATA', data / 'frm/split')) / f'{split}.npz'
    f = np.load(path)
    sol, clue = f['solutions'], f['clue_mask']
    hi = len(sol) if n is None else min(len(sol), lo + n)
    return sol[lo:hi].astype(np.int64) - 1, ~clue[lo:hi].astype(bool), np.arange(lo, hi)


def solution_stats(A, sol):
    """(tau, ebar) as floats: tau the highest pair energy and ebar the mean own-digit field G[i, y_i] over the
    training solutions sol (b, n) (digits 0-8), for the couplings A (n, n)."""
    y = F.one_hot(sol, V).float()
    S = A + A.T
    field = torch.einsum('nm,bmd->bnd', S, y)
    return float((0.5 * (y * field).sum((1, 2))).max()), float((y * field).sum(-1).mean())


class Search:
    """One search over puzzles (sol, free) with C chains each; `run` returns per-puzzle results. The state can be saved
    and restored at a check (`state_dict`/`load_state_dict`), bitwise.

    A (n, n) directed couplings; B (t, d) input biases; sol (q, n) solutions (digits 0-8; they give the clue digits
    and score the result, the search never reads a free cell's solution); free (q, n) bool.
    temp: temperature T; theta: focus threshold; focused: focused or uniform cell choice; C: chains per puzzle;
    test: 'pair' or 'term'; tau, tol: the pair test passes at P <= tau + tol; ebar: the focus reference;
    cap: flop budget per puzzle (it sets the move limit T_cap); check: moves between two checks; restart, growth: moves
    before the first restart (0: no restarts) and the growth factor of the interval; seed: seed of the start boards
    and of the kernel's random numbers; anneal: see the comment in the constructor."""

    def __init__(self, A, B, sol, free, *, temp=1.0, theta=1.5, focused=True, C=1, test='pair', tau=0.0, tol=1e-3,
                 ebar=0.0, cap=4e11, check=1024, restart=0, growth=1.0, seed=0, device='cuda', anneal=None):
        dev = torch.device(device)
        self.dev, self.C, self.check, self.seed = dev, C, check, seed
        self.test, self.focused = test, focused
        self.A = A.to(dev).float()
        self.S = (self.A + self.A.T).contiguous()
        self.Mblk = None
        if test == 'term':
            from .cram_sudoku import blocking
            self.Mblk = blocking(self.A).contiguous()
        self.B = B.to(dev).float().contiguous()
        self.ebar, self.tau, self.tol = float(ebar), float(tau), float(tol)
        self.temp, self.theta = float(temp), float(theta)
        self.restart, self.growth = int(restart), float(growth)
        # anneal: (T_hi, T_lo, period): the temperature falls geometrically from T_hi to T_lo over `period` moves and
        # starts again (a cyclic schedule; one value per check of `check` moves). None: the constant temp.
        self.anneal = None if anneal is None else (float(anneal[0]), float(anneal[1]), int(anneal[2]))
        assert self.anneal is None or self.anneal[2] % check == 0
        assert self.restart % check == 0 and 4096 % check == 0
        self.fl_move = self._fl_move()
        self.cap = float(cap)
        self.T_cap = int(cap // (self.fl_move * C))
        NQ = len(sol)
        self.NQ = NQ
        sol_t = torch.as_tensor(sol, device=dev)
        free_t = torch.as_tensor(free, device=dev)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed)
        rep = lambda t: t.repeat_interleave(C, 0)
        self.tgt, self.fr = rep(sol_t), rep(free_t)
        self.x = torch.where(self.fr, 0, self.tgt + 1).to(torch.int32).contiguous()
        self.freelist = torch.argsort((~self.fr).to(torch.int8), dim=1, stable=True).to(torch.int32).contiguous()
        self.fmask = self.fr.to(torch.int32).contiguous()
        self.nfree = self.fr.sum(1).to(torch.int32).contiguous()
        H = NQ * C
        self.y = torch.empty(H, N, dtype=torch.int32, device=dev)
        self.G = torch.empty(H, N, V, device=dev)
        self.P = torch.empty(H, device=dev)
        self.Q = torch.zeros(H, device=dev)
        self._randomize(torch.ones(H, dtype=torch.bool, device=dev))
        self.found = torch.full((H,), -1, dtype=torch.int32, device=dev)
        self.ycert = torch.zeros(H, N, dtype=torch.int32, device=dev)
        self.act = torch.arange(NQ, device=dev)  # puzzle of each active unit (C chains each)
        self.stopped = torch.zeros(NQ, dtype=torch.bool, device=dev)
        self.stop_t = torch.full((NQ,), -1, dtype=torch.long, device=dev)
        self.ok = torch.zeros(NQ, dtype=torch.bool, device=dev)
        self.bestP = torch.full((NQ,), float('inf'), device=dev)
        self.bestok = torch.zeros(NQ, dtype=torch.bool, device=dev)
        self.restarts = torch.zeros(NQ, dtype=torch.long, device=dev)  # restarts paid per puzzle (all its chains)
        self.t = 0
        self.next_restart = self.restart
        self.interval = self.restart
        self.elapsed = 0.0

    def _fl_move(self):
        """flop of one move of one chain (module docstring)."""
        return 4 * N + 4 * V + (2 * N if self.focused else 0) + FL_PRODUCT / 4096 + (4 * N if self.test == 'term' else 0)

    def _fl_restart(self):
        """flop of one restart of one chain: one exact field product."""
        return FL_PRODUCT

    def current_temp(self):
        """The temperature of the next `check` moves: `temp`, or the value of the cyclic schedule `anneal`."""
        if self.anneal is None:
            return self.temp
        hi, lo, period = self.anneal
        return hi * (lo / hi) ** ((self.t % period) / period)

    def _periodic_refresh(self):
        """Called every 4,096 moves: recompute the fields exactly, which removes the rounding drift of the updates."""
        self._refresh()

    def _refresh(self, rows=None):
        """Recompute G, P and Q from the boards of the chains `rows` (a bool mask over chains; None: all chains)."""
        y = self.y.long() if rows is None else self.y[rows].long()
        Y = F.one_hot(y, V).float()
        G = torch.einsum('hnd,nm->hmd', Y, self.S)
        P = 0.5 * (Y * G).sum((1, 2))
        Q = 0.5 * torch.einsum('hnd,nm,hmd->h', Y, self.Mblk, Y) if self.Mblk is not None else torch.zeros_like(P)
        if rows is None:
            self.G.copy_(G); self.P.copy_(P); self.Q.copy_(Q)
        else:
            self.G[rows] = G; self.P[rows] = P; self.Q[rows] = Q

    def _randomize(self, rows):
        """Give the chains `rows` (bool mask) a new board: uniform random digits at free cells, the clues elsewhere."""
        r = torch.randint(0, V, (int(rows.sum()), N), device=self.dev, generator=self.gen)
        self.y[rows] = torch.where(self.fr[rows], r, self.tgt[rows]).to(torch.int32)
        self._refresh(rows)

    def _moves(self, nmoves):
        """Run `nmoves` moves of every active chain in one kernel launch."""
        H = len(self.y)
        if H == 0:
            return
        assert H * 512 < 2 ** 31
        seed_t = torch.tensor([self.seed * 1000003 + self.t // self.check], dtype=torch.int64, device=self.dev)
        temp = torch.full((H,), self.current_temp(), device=self.dev)
        theta = torch.full((H,), self.theta, device=self.dev)
        M = self.Mblk if self.Mblk is not None else self.S
        _moves[(H,)](self.S, M, self.B, self.x, self.y, self.G, self.P, self.Q, self.freelist, self.fmask, self.nfree,
                     temp, theta, self.found, self.ycert, seed_t, nmoves, self.t, self.ebar, self.tau + self.tol,
                     MODE=2 if self.focused else 0, TERM=self.test == 'term', N=N, V=V, NB=128, VB=16, num_warps=1)

    PER_CHAIN = ('tgt', 'fr', 'x', 'freelist', 'fmask', 'nfree', 'y', 'G', 'P', 'Q', 'found', 'ycert')  # tensors with one row per active chain

    def _compact(self):
        """Drop the chains of stopped puzzles from the per-chain tensors."""
        keep_u = ~self.stopped[self.act]
        keep = keep_u.repeat_interleave(self.C)
        for nm in self.PER_CHAIN:
            setattr(self, nm, getattr(self, nm)[keep].contiguous())
        self.act = self.act[keep_u]

    def step_check(self):
        """check moves, then the check: stops, the lowest-P board, restarts, compaction."""
        self._moves(self.check)
        self.t += self.check
        if self.t % 4096 == 0:
            self._periodic_refresh()
        nA, C = len(self.act), self.C
        right = (self.y.long() == self.tgt).all(1).view(nA, C)
        mn, am = self.P.view(nA, C).min(1)
        better = mn < self.bestP[self.act]
        self.bestP[self.act] = torch.where(better, mn, self.bestP[self.act])
        self.bestok[self.act] = torch.where(better, right.gather(1, am[:, None]).squeeze(1), self.bestok[self.act])
        fu = self.found.view(nA, C).long()
        fu = torch.where(fu >= 0, fu, torch.full_like(fu, 1 << 60))
        first, fc = fu.min(1)
        anyc = first < (1 << 60)
        if bool(anyc.any()):
            rc = (self.ycert.long() == self.tgt).all(1).view(nA, C)
            a = self.act[anyc]
            self.ok[a] = rc[anyc].gather(1, fc[anyc][:, None]).squeeze(1)
            self.stop_t[a] = first[anyc]
            self.stopped[a] = True
        if self.restart and self.t >= self.next_restart:
            live = self.found < 0
            if bool(live.any()):
                self._randomize(live)
            live_u = self.act[~self.stopped[self.act]]  # puzzles that stopped at this check pay no restart
            self.restarts[live_u] += 1
            self.interval = int(round(self.interval * self.growth / self.check)) * self.check
            self.next_restart = self.t + self.interval
        if bool(self.stopped[self.act].float().mean() > 0.25):
            self._compact()

    def done(self):
        """True when every puzzle has stopped or the move limit is reached."""
        return self.t >= self.T_cap or len(self.act) == 0

    def run(self, on_check=None, log_every=1 << 20):
        """Run checks until `done`. on_check(search) is called after every check (for example to save the state).
        Returns `result()`."""
        t0 = time.time() - self.elapsed
        with torch.no_grad():
            while not self.done():
                self.step_check()
                self.elapsed = time.time() - t0
                if on_check is not None:
                    on_check(self)
                if self.t % log_every == 0:
                    print(f'  t {self.t}: active {len(self.act)} / {self.NQ}  {self.elapsed:.0f} s', flush=True)
        return self.result()

    def result(self):
        """Per-puzzle lists `flops` (in-kernel estimate, all chains up to the puzzle's stop), `solved` (the returned
        board equals the solution) and `stopped` (a chain passed the stopping test); and the scalars `moves` (moves per
        chain so far), `T_cap`, `fl_move`, `seconds`."""
        steps = torch.where(self.stopped, self.stop_t, torch.full_like(self.stop_t, self.t)).double()
        # a restart's exact product is paid by every chain of the puzzle while it runs
        flops = steps * self.fl_move * self.C + self.restarts.double() * self._fl_restart() * self.C
        solved = torch.where(self.stopped, self.ok, self.bestok)
        return {'flops': flops.cpu().tolist(), 'solved': solved.cpu().tolist(), 'stopped': self.stopped.cpu().tolist(),
                'moves': self.t, 'T_cap': self.T_cap, 'fl_move': self.fl_move, 'seconds': round(self.elapsed, 1)}

    STATE = PER_CHAIN + ('act', 'stopped', 'stop_t', 'ok', 'bestP', 'bestok', 'restarts')

    def state_dict(self):
        """The complete search state at a check, on the CPU, including the generator state."""
        out = {k: getattr(self, k).cpu() for k in self.STATE}
        out.update(t=self.t, next_restart=self.next_restart, interval=self.interval, elapsed=self.elapsed,
                   gen=self.gen.get_state())
        return out

    def load_state_dict(self, sd):
        """Restore a state saved by `state_dict` into a search built with the same arguments."""
        for k in self.STATE:
            setattr(self, k, sd[k].to(self.dev))
        self.t, self.next_restart, self.interval, self.elapsed = sd['t'], sd['next_restart'], sd['interval'], sd['elapsed']
        self.gen.set_state(sd['gen'])


def summarize(res):
    """Summary of a `Search.result()` as a dict: the number of puzzles, solved, stopped, wrong stops (stopped on a
    board that is not the solution), budget failures (never stopped), and the mean / median / p99 / p99.9 / max flop."""
    fl = np.asarray(res['flops'])
    sv, st = np.asarray(res['solved'], bool), np.asarray(res['stopped'], bool)
    q = lambda p: float(np.quantile(fl, p))
    return {'n': len(fl), 'solved': int(sv.sum()), 'stopped': int(st.sum()), 'wrong_stops': int((st & ~sv).sum()),
            'budget_failures': int((~st).sum()), 'unstopped_but_right': int((~st & sv).sum()),
            'mean_flop': float(fl.mean()), 'median_flop': q(0.5), 'p99_flop': q(0.99), 'p999_flop': q(0.999),
            'max_flop': float(fl.max())}
