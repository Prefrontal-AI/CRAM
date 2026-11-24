import triton
import triton.language as tl


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
