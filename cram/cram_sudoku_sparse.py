"""Sparse focused Gibbs search on the same-digit pair energy of `cram_sudoku`: the same moves as
cram_sudoku_search.Search, but a move touches only the cells with a nonzero coupling to the moved cell.
The hand-set Sudoku energy of the paper (`experiments/hand_set.py`) is searched with this class.

Dimension names: h chains, n = 81 cells, d = 9 digits, k neighbour slots (D = the largest degree).

Energy: E(y) = 1/2 sum_(i != j) S_ij [y_i = y_j] - sum_i B[x_i, y_i], S = A + A^T (`cram_sudoku`).
The couplings S are stored as neighbour lists read from the nonzero entries of S (the weights only, no cell positions):
nbr[i, k] the k-th cell with S_ik != 0, w[i, k] its weight. When every nonzero S_ik has one value h (one shared level,
as in the hand-set energy), w = 1 and the fields are exact integer counts: G[j, a] = number of neighbours of j holding
a, the energy is h * count, and no periodic refresh is needed. Otherwise w = S_ik, hscale = 1, and the fields are
refreshed exactly every 4,096 moves with a sparse product.

A move (one warp per chain): the cell (focused: a random free cell whose focus flag is set, else a random free cell;
the flags are kept per cell and updated for the moved cell and its neighbours only), its digit by heat bath from
softmax(-(hscale * G[cell] - B[x_cell]) / T), the two neighbour-field updates, the focus flags of the neighbours whose
own digit is the old or the new one, the pair energy and the stopping test (hscale * P <= tau + tol). The random
numbers have the same keys and offsets as cram_sudoku_search._moves, so with w = S, hscale = 1 the two kernels make the
same moves (tests/test_cram_sudoku.py).

Counted floating-point operations (flop) per move, the in-kernel estimate (a multiply-add is 2 flop, every element of
a field-vector update 2 flop, index, digit-comparison and random-number operations 0, the Gumbel transform of the
random numbers counted as random-number generation): 4 D (two neighbour-field updates) + (D + 1) (focus: one comparison
own > thr per changed cell, thr = (theta + ebar) / hscale precomputed) + heat bath (5 d with input biases: the 4 d of
the dense kernel plus the hscale multiply; 2 d + 1 without biases: one scalar -hscale/T, then a multiply and an add
per digit) + 4 (pair energy, stopping test) [+ 2 nnz d / 4096 for the refresh with general weights]. A restart pays one
sparse product, 2 nnz d. For D = 20 and no biases this is 123 flop per move, against 551 for the dense kernel.
The paper's operation counts follow the convention of `baselines/flops/fullcount.py`.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .cram_sudoku_search import N, V, Search


# The move kernel: one program per chain runs `nmoves` moves. Arguments (flattened tensors): nbr (n, D) neighbour
# lists; w (n, D) their weights; B (t, d) input biases; x (h, n) tokens; y (h, n) the board; G (h, n, d) fields;
# flag (h, n) focus flags; P (h,) pair energy in units of hscale; free (h, n) the free cells first; fmask (h, n) 1 at
# free cells; nfree (h,); temp (h,); found (h,) the move at which the chain first passed the stopping test, or -1;
# ycert (h, n) the board at that move; seed (1,); move0 the number of moves already made; hscale the shared level
# (1 with general weights); thr the focus threshold; tau the energy threshold.
# MODE 2: focused cell choice, 0: uniform. NOBIAS: all input biases are zero. NB, VB, DB: block sizes.
@triton.jit
def _moves_sparse(nbr_ptr, w_ptr, B_ptr, x_ptr, y_ptr, G_ptr, flag_ptr, P_ptr, free_ptr, fmask_ptr, nfree_ptr,
                  temp_ptr, found_ptr, ycert_ptr, seed_ptr, nmoves, move0, hscale, thr, tau,
                  MODE: tl.constexpr, NOBIAS: tl.constexpr, N: tl.constexpr, V: tl.constexpr, D: tl.constexpr,
                  NB: tl.constexpr, VB: tl.constexpr, DB: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    rows = tl.arange(0, NB)
    cols = tl.arange(0, VB)
    ks = tl.arange(0, DB)
    rmask = rows < N
    cmask = cols < V
    kmask = ks < D
    ybase = y_ptr + pid * N
    gbase = G_ptr + pid * N * V
    fbase = flag_ptr + pid * N
    mbase = fmask_ptr + pid * N
    P = tl.load(P_ptr + pid)
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
        Gi = tl.load(gbase + cell * V + cols, mask=cmask, other=0.0)
        uu = tl.rand(sk, base + 16 + cols)
        gum = -tl.log(-tl.log(tl.maximum(uu, 1e-20)))
        if NOBIAS:  # no input cost: logits = -(hscale / T) G + Gumbel noise
            lg = tl.where(cmask, Gi * (-hscale / T) + gum, float('-inf'))
        else:
            xc = tl.load(x_ptr + pid * N + cell)
            Li = hscale * Gi - tl.load(B_ptr + xc * V + cols, mask=cmask, other=0.0)
            lg = tl.where(cmask, -Li / T + gum, float('-inf'))
        new = tl.argmax(lg, 0)
        old = tl.load(ybase + cell)
        new = tl.where(found >= 0, old, new)
        d = (new != old).to(tl.float32)
        nb = tl.load(nbr_ptr + cell * D + ks, mask=kmask, other=0)
        wv = tl.load(w_ptr + cell * D + ks, mask=kmask, other=0.0) * d
        po = gbase + nb * V + old
        go = tl.load(po, mask=kmask, other=0.0) - wv
        tl.store(po, go, mask=kmask)
        pn = gbase + nb * V + new
        gn = tl.load(pn, mask=kmask, other=0.0) + wv
        tl.store(pn, gn, mask=kmask)
        ynb = tl.load(ybase + nb, mask=kmask, other=-1)
        own = tl.where(ynb == old, go, gn)
        changed = kmask & ((ynb == old) | (ynb == new))
        fnb = tl.load(mbase + nb, mask=kmask, other=0) != 0
        tl.store(fbase + nb, ((own > thr) & fnb).to(tl.int32), mask=changed)
        gnew = tl.sum(tl.where(cols == new, Gi, 0.0), 0)
        P = P + (gnew - tl.sum(tl.where(cols == old, Gi, 0.0), 0)) * d
        tl.store(ybase + cell, new)
        tl.store(fbase + cell, (gnew > thr).to(tl.int32))
        tl.debug_barrier()
        cert = (found < 0) & (hscale * P <= tau)
        yv = tl.load(ybase + rows, mask=rmask & cert, other=0)
        tl.store(ycert_ptr + pid * N + rows, yv, mask=rmask & cert)
        found = tl.where(cert, move0 + k + 1, found)
    tl.store(P_ptr + pid, P)
    tl.store(found_ptr + pid, found)


def neighbour_lists(S):
    """From the nonzero pattern of S (n, n): nbr (n, D) int32, weights (n, D), D the largest degree; slots beyond a
    cell's degree point at the cell itself with weight 0 (never read: the kernel masks k >= D only, so every cell must
    have degree D; asserted)."""
    nz = (S != 0) & ~torch.eye(N, dtype=torch.bool, device=S.device)
    deg = nz.sum(1)
    D = int(deg.max())
    assert bool((deg == D).all()), f'learned degrees differ ({int(deg.min())}..{D}); pad support'
    nbr = torch.stack([torch.nonzero(nz[i]).squeeze(1) for i in range(N)]).to(torch.int32)
    return nbr.contiguous(), S.gather(1, nbr.long()).contiguous(), D


class SparseSearch(Search):
    """Search with the sparse move kernel. Same arguments, protocol, outputs and state handling as Search; only the
    energy-threshold test (test='pair') is implemented."""

    PER_CHAIN = Search.PER_CHAIN + ('flag',)
    STATE = PER_CHAIN + ('act', 'stopped', 'stop_t', 'ok', 'bestP', 'bestok', 'restarts')

    def __init__(self, A, B, sol, free, *, test='pair', **kw):
        assert test == 'pair', 'sparse search implements the energy-threshold test'
        A = A.float()
        S = A + A.T
        vals = S[S != 0]
        one_level = bool((vals == vals[0]).all()) and float(vals[0]) > 0
        self.hscale = float(vals[0]) if one_level else 1.0
        self.one_level = one_level
        self.nbr, w, self.D = neighbour_lists(S.to(kw.get('device', 'cuda')))
        self.w = (torch.ones_like(w) if one_level else w).contiguous()
        self.nnz = N * self.D
        self.nobias = bool((B == 0).all())
        super().__init__(A, B, sol, free, test='pair', **kw)
        self.thr = (self.theta + self.ebar) / self.hscale  # focus: hscale * own - ebar > theta <=> own > thr

    def _fl_move(self):
        """flop of one move of one chain (module docstring)."""
        refresh = 0.0 if self.one_level else 2 * self.nnz * V / 4096
        heat = 2 * V + 1 if self.nobias else 5 * V
        return 4 * self.D + ((self.D + 1) if self.focused else 0) + heat + 4 + refresh

    def _fl_restart(self):
        """flop of one restart of one chain: one sparse field product."""
        return 2 * self.nnz * V

    def _refresh(self, rows=None):
        """Recompute the fields G, the pair energy and the focus flags from the boards of the chains `rows` (None: all)."""
        y = self.y.long() if rows is None else self.y[rows].long()
        Gs = []
        for lo in range(0, len(y), 65536):  # (h, n, D, d) gathered in chunks
            Y = F.one_hot(y[lo:lo + 65536], V).float()
            Gs.append(torch.einsum('hnkd,nk->hnd', Y[:, self.nbr.long()], self.w))
        G = torch.cat(Gs) if Gs else torch.zeros(0, N, V, device=self.dev)
        Y = F.one_hot(y, V).float()
        P = 0.5 * (Y * G).sum((1, 2))
        own = G.gather(2, y[..., None]).squeeze(-1)
        fr = self.fr if rows is None else self.fr[rows]
        flag = ((own > (self.theta + self.ebar) / self.hscale) & fr).to(torch.int32)
        if not hasattr(self, 'flag'):
            self.flag = torch.zeros(len(self.fr), N, dtype=torch.int32, device=self.dev)
        if rows is None:
            self.G.copy_(G); self.P.copy_(P); self.flag.copy_(flag)
        else:
            self.G[rows] = G; self.P[rows] = P; self.flag[rows] = flag

    def _periodic_refresh(self):
        """Called every 4,096 moves. With one shared level the fields are integer counts and need no refresh."""
        if not self.one_level:
            self._refresh()

    def _moves(self, nmoves):
        """Run `nmoves` moves of every active chain in one kernel launch."""
        H = len(self.y)
        if H == 0:
            return
        assert H * 512 < 2 ** 31
        seed_t = torch.tensor([self.seed * 1000003 + self.t // self.check], dtype=torch.int64, device=self.dev)
        temp = torch.full((H,), self.current_temp(), device=self.dev)
        _moves_sparse[(H,)](self.nbr, self.w, self.B, self.x, self.y, self.G, self.flag, self.P, self.freelist, self.fmask,
                            self.nfree, temp, self.found, self.ycert, seed_t, nmoves, self.t, self.hscale,
                            self.thr, self.tau + self.tol, MODE=2 if self.focused else 0, NOBIAS=self.nobias, N=N, V=V, D=self.D,
                            NB=128, VB=16, DB=triton.next_power_of_2(self.D), num_warps=1)
