"""The Sudoku model of the paper (table row `cram-0-fit`): the general pairwise energy and its focused Gibbs search.
Every coupled cell pair has its own 9 x 9 cost table and every cell its own input cost; the fitted weights are searched
as they are, with no processing after the fit. `experiments/sudoku.py` fits the tables by pseudolikelihood with an L1
proximal term and runs this search. The moves follow cram_sudoku_sparse.SparseSearch (same cell choice, random keys
and heat bath), and the protocol (chains, checks, stops, results) is that of cram_sudoku_search.Search.

Dimension names: h chains, n and m = 81 cells, d and c digits (9; in einsum strings a and c), k neighbour slots
(D = the largest learned degree), b boards.

Energy of a board y (one digit y_i per cell):
    E(y) = 1/2 sum_(i != j) S[i, j, y_i, y_j] + sum_i B[i, y_i],   S[j, i] = S[i, j]^T,
with S (n, m, d, c) the pair cost tables (S[i, j, a, c]: the cost of digit a at cell i with digit c at cell j) and
B (n, d) the input costs (called Bc in the code). A clue is not a term of the energy: clue cells are clamped to their
digit and never move.
Each chain keeps the fields G[i, a] = sum_j S[i, j, a, y_j]; the local energy of digit a at cell i is L[i, a] = G[i, a] + B[i, a],
and E changes by L[i, new] - L[i, old] when cell i moves. The coupled cells of i are the j with any nonzero entry in
S[i, j] (learned weights only); R[i, k] = S[nbr[i, k], i] is the block that moves the fields of the k-th neighbour when
cell i changes digit. Cells with fewer than D neighbours are padded with themselves and a zero block.

A move (one warp per chain): the cell (focused: a random free cell whose flag is set, else a random free cell), its digit
by heat bath from softmax(-L[cell] / T), the fields of its neighbours (G[nbr, :] += R[:, :, new] - R[:, :, old]), their
focus flags (own local energy L[m, y_m] > thr, thr = theta + ebar, with ebar the mean own local energy of the cells of
the training solutions), the energy and the stopping test (E <= tau, with tau the highest energy of a training
solution plus a tolerance; `solution_stats` fits tau and ebar).

Counted floating-point operations (flop) per move, as the in-kernel estimate of cram_sudoku_sparse (a multiply-add is
2 flop, index, comparison and random-number operations 0; the paper's counts are rebuilt by `cram/opcount.py`):
2 D d (each neighbour field element: one difference and one add) + 2 D (own local energy and focus test of each
neighbour) + 1 (the moved cell's flag) + 5 d (heat bath with input costs) + 4 (energy and certificate). A restart pays
one sparse product, 2 n D d d, and the exact field recompute every 4,096 moves 2 n D d d / 4096 per move.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .cram_sudoku_search import N, V, Search


# The move kernel: one program per chain runs `nmoves` moves. Arguments (flattened tensors): nbr (n, D) neighbour
# lists; R (n, D, d, c) with R[i, k] = S[nbr[i, k], i]; Bc (n, d) input costs; y (h, n) the board; G (h, n, d) fields;
# flag (h, n) focus flags; E (h,) energy; free (h, n) the free cells first; fmask (h, n) 1 at free cells; nfree (h,);
# temp (h,); found (h,) the move at which the chain first passed the stopping test, or -1; ycert (h, n) the board at
# that move; seed (1,); move0 the number of moves already made; thr the focus threshold; tau the energy threshold.
# MODE 2: focused cell choice, 0: uniform. NB, VB, DB: block sizes (powers of two that cover n, d and D).
@triton.jit
def _moves_general(nbr_ptr, R_ptr, Bc_ptr, y_ptr, G_ptr, flag_ptr, E_ptr, free_ptr, fmask_ptr, nfree_ptr,
                   temp_ptr, found_ptr, ycert_ptr, seed_ptr, nmoves, move0, thr, tau,
                   MODE: tl.constexpr, N: tl.constexpr, V: tl.constexpr, D: tl.constexpr,
                   NB: tl.constexpr, VB: tl.constexpr, DB: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, NB)
    cols = tl.arange(0, VB)
    ks = tl.arange(0, DB)
    rmask = rows < N
    cmask = cols < V
    kmask = ks < D
    km2 = kmask[:, None] & cmask[None, :]
    ybase = y_ptr + pid * N
    gbase = G_ptr + pid * N * V
    fbase = flag_ptr + pid * N
    mbase = fmask_ptr + pid * N
    E = tl.load(E_ptr + pid)
    T = tl.load(temp_ptr + pid)
    nf = tl.load(nfree_ptr + pid)
    found = tl.load(found_ptr + pid)
    sd = tl.load(seed_ptr)
    for k in range(nmoves):
        sk = sd * 65536 + k
        base = (pid * 512).to(tl.int32)
        u0 = tl.rand(sk, base)
        idx = tl.minimum((u0 * nf).to(tl.int32), nf - 1)
        cell = tl.load(free_ptr + pid * N + idx)
        if MODE == 2:
            fl = tl.load(fbase + rows, mask=rmask, other=0) != 0
            score = tl.where(fl, tl.rand(sk, base + 64 + rows), -1.0)
            cc = tl.argmax(score, 0)
            cell = tl.where(tl.max(score, 0) >= 0.0, cc, cell)
        Li = tl.load(gbase + cell * V + cols, mask=cmask, other=0.0) + tl.load(Bc_ptr + cell * V + cols, mask=cmask, other=0.0)
        uu = tl.rand(sk, base + 16 + cols)
        gum = -tl.log(-tl.log(tl.maximum(uu, 1e-20)))
        lg = tl.where(cmask, -Li / T + gum, float('-inf'))
        new = tl.argmax(lg, 0)
        old = tl.load(ybase + cell)
        new = tl.where(found >= 0, old, new)
        d = (new != old).to(tl.float32)
        nb = tl.load(nbr_ptr + cell * D + ks, mask=kmask, other=0)
        rb = R_ptr + ((cell * D + ks[:, None]) * V + cols[None, :]) * V  # R[cell, k, a, :]
        delta = (tl.load(rb + new, mask=km2, other=0.0) - tl.load(rb + old, mask=km2, other=0.0)) * d
        pg = gbase + nb[:, None] * V + cols[None, :]
        Gn = tl.load(pg, mask=km2, other=0.0) + delta
        tl.store(pg, Gn, mask=km2)
        ynb = tl.load(ybase + nb, mask=kmask, other=0)
        own = tl.sum(tl.where(cols[None, :] == ynb[:, None], Gn, 0.0), 1) + tl.load(Bc_ptr + nb * V + ynb, mask=kmask, other=0.0)
        fnb = tl.load(mbase + nb, mask=kmask, other=0) != 0
        tl.store(fbase + nb, ((own > thr) & fnb).to(tl.int32), mask=kmask)
        lnew = tl.sum(tl.where(cols == new, Li, 0.0), 0)
        E = E + (lnew - tl.sum(tl.where(cols == old, Li, 0.0), 0)) * d
        tl.debug_barrier()
        tl.store(ybase + cell, new)
        tl.store(fbase + cell, (lnew > thr).to(tl.int32))
        tl.debug_barrier()
        cert = (found < 0) & (E <= tau)
        yv = tl.load(ybase + rows, mask=rmask & cert, other=0)
        tl.store(ycert_ptr + pid * N + rows, yv, mask=rmask & cert)
        found = tl.where(cert, move0 + k + 1, found)
    tl.store(E_ptr + pid, E)
    tl.store(found_ptr + pid, found)


def neighbour_blocks(S):
    """From S (n, m, d, c): nbr (n, D) int32 (padded with the cell itself), R (n, D, d, c) with R[i, k] = S[nbr[i, k], i]
    (zero on padded slots), and D. A pair is coupled when any entry of its block is nonzero."""
    nz = (S != 0).flatten(2).any(-1) & ~torch.eye(N, dtype=torch.bool, device=S.device)
    D = int(nz.sum(1).max())
    nbr = torch.arange(N, device=S.device)[:, None].repeat(1, D)
    R = torch.zeros(N, D, V, V, device=S.device, dtype=S.dtype)
    for i in range(N):
        js = torch.nonzero(nz[i]).squeeze(1)
        nbr[i, :len(js)] = js
        R[i, :len(js)] = S[js, i]
    return nbr.to(torch.int32).contiguous(), R.contiguous(), D


def local_energies(S, Bc, y):
    """(h, n, d): L[i, a] = sum_j S[i, j, a, y_j] + B[i, a] for boards y (h, n)."""
    return torch.einsum('nmac,hmc->hna', S, F.one_hot(y.long(), V).float()) + Bc[None]


def board_energy(S, Bc, y):
    """(h,): E(y) for boards y (h, n) of digits 0-8, with S (n, m, d, c) and Bc (n, d)."""
    Y = F.one_hot(y.long(), V).float()
    G = torch.einsum('nmac,hmc->hna', S, Y)
    return 0.5 * (Y * G).sum((1, 2)) + (Y * Bc[None]).sum((1, 2))


def solution_stats(S, Bc, sol):
    """(tau, ebar) as floats, fitted from the training solutions sol (b, n) (digits 0-8): tau the highest energy of a
    training solution (the stopping threshold) and ebar the mean own local energy L[i, y_i] (the focus reference)."""
    L = local_energies(S, Bc, sol)
    own = L.gather(2, sol[..., None].long()).squeeze(-1)
    return float(board_energy(S, Bc, sol).max()), float(own.mean())


def lambda_min(S):
    """Smallest eigenvalue (a float) of the symmetric (n d) x (n d) matrix that holds the tables S (n, m, d, c)."""
    M = S.permute(0, 2, 1, 3).reshape(N * V, N * V).double()
    return float(torch.linalg.eigvalsh(M)[0])


class GeneralSearch(Search):
    """Search on the general energy (S (n, m, d, c), Bc (n, d)) with the general move kernel; the protocol, outputs and
    state handling are those of Search, and the keyword arguments are Search's (sol, free, temp, theta, C, tau, tol,
    ebar, cap, check, seed, anneal, ...). `tau` is the energy threshold: a chain stops at E <= tau + tol. The attribute
    P of the base class holds the energy E here."""

    PER_CHAIN = ('tgt', 'fr', 'x', 'freelist', 'fmask', 'nfree', 'y', 'G', 'P', 'Q', 'found', 'ycert', 'flag')
    STATE = PER_CHAIN + ('act', 'stopped', 'stop_t', 'ok', 'bestP', 'bestok', 'restarts')

    def __init__(self, S, Bc, sol, free, **kw):
        dev = torch.device(kw.get('device', 'cuda'))
        self.Sg = S.to(dev).float().contiguous()
        self.Bc = Bc.to(dev).float().contiguous()
        self.nbr, self.R, self.D = neighbour_blocks(self.Sg)
        self.focused = kw.get('focused', True)
        kw['test'] = 'pair'
        super().__init__(torch.zeros(N, N), torch.zeros(10, V), sol, free, **kw)
        self.thr = self.theta + self.ebar
        self._refresh()  # the base class filled G with its own (zero) couplings

    def _fl_move(self):
        """flop of one move of one chain (module docstring)."""
        refresh = 2 * N * self.D * V * V / 4096  # the exact field recompute every 4,096 moves
        return 2 * self.D * V + (2 * self.D + 1 if self.focused else 0) + 5 * V + 4 + refresh

    def _fl_restart(self):
        """flop of one restart of one chain: one sparse field product."""
        return 2 * N * self.D * V * V

    def _refresh(self, rows=None):
        """Recompute the fields G, the energy and the focus flags from the boards of the chains `rows` (None: all)."""
        if not hasattr(self, 'Sg') or not hasattr(self, 'ebar'):  # called by the base constructor before setup
            return
        y = self.y if rows is None else self.y[rows]
        Gs = [torch.einsum('nmac,hmc->hna', self.Sg, F.one_hot(y[lo:lo + 8192].long(), V).float()) for lo in range(0, len(y), 8192)]
        G = torch.cat(Gs) if Gs else torch.zeros(0, N, V, device=self.dev)
        Y = F.one_hot(y.long(), V).float()
        E = 0.5 * (Y * G).sum((1, 2)) + (Y * self.Bc[None]).sum((1, 2))
        own = (G + self.Bc[None]).gather(2, y.long()[..., None]).squeeze(-1)
        fr = self.fr if rows is None else self.fr[rows]
        flag = ((own > self.theta + self.ebar) & fr).to(torch.int32)
        if not hasattr(self, 'flag'):
            self.flag = torch.zeros(len(self.fr), N, dtype=torch.int32, device=self.dev)
        if rows is None:
            self.G.copy_(G); self.P.copy_(E); self.flag.copy_(flag)
        else:
            self.G[rows] = G; self.P[rows] = E; self.flag[rows] = flag

    def _periodic_refresh(self):
        self._refresh()  # removes float drift of the incremental fields every 4,096 moves (counted in _fl_move)

    def _moves(self, nmoves):
        """Run `nmoves` moves of every active chain in one kernel launch."""
        H = len(self.y)
        if H == 0:
            return
        assert H * 512 < 2 ** 31
        seed_t = torch.tensor([self.seed * 1000003 + self.t // self.check], dtype=torch.int64, device=self.dev)
        temp = torch.full((H,), self.current_temp(), device=self.dev)
        _moves_general[(H,)](self.nbr, self.R, self.Bc, self.y, self.G, self.flag, self.P, self.freelist, self.fmask,
                             self.nfree, temp, self.found, self.ycert, seed_t, nmoves, self.t, self.thr, self.tau + self.tol,
                             MODE=2 if self.focused else 0, N=N, V=V, D=self.D, NB=128, VB=16,
                             DB=triton.next_power_of_2(self.D), num_warps=1)
