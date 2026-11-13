"""Fused Triton kernels for the Maze-Hard hidden (latent) layer: the forward relaxation (`settle`) and the adjoint
relaxation with its parameter gradient (`adjoint_mu_grad`). The fits of the Maze-Hard model use them on a GPU; the
evaluation uses the PyTorch loop. Both compute the quantities of the PyTorch code in another way.

The same update as `cram_maze_hard_simple.GatedLatentEnergy.settle` (temperature 1), computed by one Triton program per
maze that keeps the maze's state in an L2-resident scratch buffer and runs every Jacobi step in a single launch:

    psi_i <- la_i - logsumexp( lg_(k,i) - psi_(i+d_k) for the four side offsets d_k, lc_i ),

from psi = +inf, checked every `check` steps with the same test (|new - psi| <= tol (1 + |new|), or equal), at most
`max_steps`. Differences from the PyTorch loop: each maze stops at its own convergence check (the loop runs every maze
of the batch until the last one converges; both stop within the tolerance of the same fixed point), and exp / log come
from Triton's libdevice. Results agree to the tolerance, not bitwise.

Dimension names: b mazes, k offsets (4 side offsets, in `zoffs` order), h w grid rows and columns.
"""

import torch
import triton
import triton.language as tl


# One program per maze. Arguments: lg (b, 4, h, w) log edge weight per side offset, la (b, h, w) log diagonal, lc
# (b, h, w) log source, flattened; psi_a and psi_b (b, h, w) two buffers that the Jacobi steps alternate between (the
# result ends in psi_a); steps (b,) the step of the passing convergence check, or max_steps; d0i..d3j the four side
# offsets; BH and BW the block sizes (powers of two that cover h and w).
@triton.jit
def _settle_kernel(lg_ptr, la_ptr, lc_ptr, psi_a, psi_b, steps_ptr, H, W, tol, max_steps, check,
                   d0i, d0j, d1i, d1j, d2i, d2j, d3i, d3j, BH: tl.constexpr, BW: tl.constexpr):
    b = tl.program_id(0)
    ri = tl.arange(0, BH)[:, None]
    cj = tl.arange(0, BW)[None, :]
    inside = (ri < H) & (cj < W)
    base = b * H * W
    off = base + ri * W + cj
    inf = float('inf')
    la = tl.load(la_ptr + off, mask=inside, other=0.0)
    lc = tl.load(lc_ptr + off, mask=inside, other=-inf)
    g0 = tl.load(lg_ptr + b * 4 * H * W + 0 * H * W + ri * W + cj, mask=inside, other=-inf)
    g1 = tl.load(lg_ptr + b * 4 * H * W + 1 * H * W + ri * W + cj, mask=inside, other=-inf)
    g2 = tl.load(lg_ptr + b * 4 * H * W + 2 * H * W + ri * W + cj, mask=inside, other=-inf)
    g3 = tl.load(lg_ptr + b * 4 * H * W + 3 * H * W + ri * W + cj, mask=inside, other=-inf)
    tl.store(psi_a + off, tl.full((BH, BW), inf, tl.float32), mask=inside)
    tl.debug_barrier()
    src = psi_a
    dst = psi_b
    steps = max_steps
    t = 0
    done = 0
    while (t < max_steps) & (done == 0):
        t += 1
        old = tl.load(src + off, mask=inside, other=inf)
        i0 = ri + d0i
        j0 = cj + d0j
        m0 = inside & (i0 >= 0) & (i0 < H) & (j0 >= 0) & (j0 < W)
        n0 = tl.load(src + base + i0 * W + j0, mask=m0, other=inf)
        i1 = ri + d1i
        j1 = cj + d1j
        m1 = inside & (i1 >= 0) & (i1 < H) & (j1 >= 0) & (j1 < W)
        n1 = tl.load(src + base + i1 * W + j1, mask=m1, other=inf)
        i2 = ri + d2i
        j2 = cj + d2j
        m2 = inside & (i2 >= 0) & (i2 < H) & (j2 >= 0) & (j2 < W)
        n2 = tl.load(src + base + i2 * W + j2, mask=m2, other=inf)
        i3 = ri + d3i
        j3 = cj + d3j
        m3 = inside & (i3 >= 0) & (i3 < H) & (j3 >= 0) & (j3 < W)
        n3 = tl.load(src + base + i3 * W + j3, mask=m3, other=inf)
        x0 = g0 - n0
        x1 = g1 - n1
        x2 = g2 - n2
        x3 = g3 - n3
        mx = tl.maximum(tl.maximum(tl.maximum(x0, x1), tl.maximum(x2, x3)), lc)
        fin = mx > -inf
        mxs = tl.where(fin, mx, 0.0)
        ssum = tl.exp(x0 - mxs) + tl.exp(x1 - mxs) + tl.exp(x2 - mxs) + tl.exp(x3 - mxs) + tl.exp(lc - mxs)
        lse = tl.where(fin, mxs + tl.log(ssum), -inf)
        new = la - lse
        tl.store(dst + off, new, mask=inside)
        tl.debug_barrier()
        if t % check == 0:
            same = (new == old) | (tl.abs(new - old) <= tol * (1.0 + tl.abs(new)))
            bad = tl.where(inside & ~same, 1, 0)
            if tl.max(tl.max(bad, 1), 0) == 0:
                done = 1
                steps = t
        tmp = src
        src = dst
        dst = tmp
    tl.store(steps_ptr + b, steps)
    # the result is in `src` (the last buffer written); copy it to psi_a so the caller reads one buffer
    res = tl.load(src + off, mask=inside, other=inf)
    tl.debug_barrier()
    tl.store(psi_a + off, res, mask=inside)


def settle(lg, la, lc, zoffs, tol=1e-6, max_steps=4000, check=8):
    """(psi (b, h, w), steps (b,)) from the parts (lg (b, 4, h, w), la (b, h, w), lc (b, h, w)), float32 on CUDA.
    zoffs: the four side offsets in the order of lg's second axis; tol, max_steps, check as GatedLatentEnergy.settle."""
    assert lg.is_cuda and lg.dtype == torch.float32 and len(zoffs) == 4
    B, _, H, W = lg.shape
    lg, la, lc = lg.contiguous(), la.contiguous(), lc.contiguous()
    psi_a = torch.empty((B, H, W), device=lg.device, dtype=torch.float32)
    psi_b = torch.empty_like(psi_a)
    steps = torch.empty((B,), device=lg.device, dtype=torch.int64)
    (a0, b0), (a1, b1), (a2, b2), (a3, b3) = zoffs
    _settle_kernel[(B,)](lg, la, lc, psi_a, psi_b, steps, H, W, float(tol), int(max_steps), int(check),
                         a0, b0, a1, b1, a2, b2, a3, b3, BH=triton.next_power_of_2(H), BW=triton.next_power_of_2(W),
                         num_warps=4)
    return psi_a, steps


@triton.jit
def _adjoint_kernel(psi_ptr, r_ptr, lg_ptr, la_ptr, mu, v_a, v_b, part_ptr, steps_ptr, H, W, tol, max_steps, check,
                    d0i, d0j, d1i, d1j, d2i, d2j, d3i, d3j, BH: tl.constexpr, BW: tl.constexpr):
    """One maze: shares P at psi, the adjoint relaxation v <- r + sum_j P_j(i - d_j) v(i - d_j), and the partial
    gradient sum_i v_i dla_i/dmu (la_i = log(deg_i + e^mu), so dla_i/dmu = e^(mu - la_i)) over cells with finite psi."""
    b = tl.program_id(0)
    ri = tl.arange(0, BH)[:, None]
    cj = tl.arange(0, BW)[None, :]
    inside = (ri < H) & (cj < W)
    base = b * H * W
    off = base + ri * W + cj
    inf = float('inf')
    psi = tl.load(psi_ptr + off, mask=inside, other=inf)
    fin = psi < inf
    r = tl.where(fin, tl.load(r_ptr + off, mask=inside, other=0.0), 0.0)
    la = tl.load(la_ptr + off, mask=inside, other=0.0)
    # incoming weights Q_j(i) = P_j(i - d_j), with P_j(c) = exp(lg_j(c) - psi(c + d_j) - la(c) + psi(c)) where all finite
    q0 = tl.zeros((BH, BW), tl.float32)
    q1 = tl.zeros((BH, BW), tl.float32)
    q2 = tl.zeros((BH, BW), tl.float32)
    q3 = tl.zeros((BH, BW), tl.float32)
    for j in tl.static_range(4):
        if j == 0:
            di, dj = d0i, d0j
        elif j == 1:
            di, dj = d1i, d1j
        elif j == 2:
            di, dj = d2i, d2j
        else:
            di, dj = d3i, d3j
        ci = ri - di
        cj2 = cj - dj
        mc = inside & (ci >= 0) & (ci < H) & (cj2 >= 0) & (cj2 < W)
        psic = tl.load(psi_ptr + base + ci * W + cj2, mask=mc, other=inf)
        lgc = tl.load(lg_ptr + b * 4 * H * W + j * H * W + ci * W + cj2, mask=mc, other=-inf)
        lac = tl.load(la_ptr + base + ci * W + cj2, mask=mc, other=0.0)
        ok = mc & (psic < inf) & fin & (lgc > -inf)
        q = tl.where(ok, tl.exp(tl.where(ok, lgc - psi - lac + psic, 0.0)), 0.0)
        if j == 0:
            q0 = q
        elif j == 1:
            q1 = q
        elif j == 2:
            q2 = q
        else:
            q3 = q
    tl.store(v_a + off, r, mask=inside)
    tl.debug_barrier()
    src = v_a
    dst = v_b
    steps = max_steps
    t = 0
    done = 0
    while (t < max_steps) & (done == 0):
        t += 1
        old = tl.load(src + off, mask=inside, other=0.0)
        i0 = ri - d0i
        j0 = cj - d0j
        n0 = tl.load(src + base + i0 * W + j0, mask=inside & (i0 >= 0) & (i0 < H) & (j0 >= 0) & (j0 < W), other=0.0)
        i1 = ri - d1i
        j1 = cj - d1j
        n1 = tl.load(src + base + i1 * W + j1, mask=inside & (i1 >= 0) & (i1 < H) & (j1 >= 0) & (j1 < W), other=0.0)
        i2 = ri - d2i
        j2 = cj - d2j
        n2 = tl.load(src + base + i2 * W + j2, mask=inside & (i2 >= 0) & (i2 < H) & (j2 >= 0) & (j2 < W), other=0.0)
        i3 = ri - d3i
        j3 = cj - d3j
        n3 = tl.load(src + base + i3 * W + j3, mask=inside & (i3 >= 0) & (i3 < H) & (j3 >= 0) & (j3 < W), other=0.0)
        new = r + q0 * n0 + q1 * n1 + q2 * n2 + q3 * n3
        tl.store(dst + off, new, mask=inside)
        tl.debug_barrier()
        if t % check == 0:
            dmax = tl.max(tl.max(tl.where(inside, tl.abs(new - old), 0.0), 1), 0)
            nmax = tl.max(tl.max(tl.where(inside, tl.abs(new), 0.0), 1), 0)
            if dmax <= tol * (1.0 + nmax):
                done = 1
                steps = t
        tmp = src
        src = dst
        dst = tmp
    v = tl.load(src + off, mask=inside, other=0.0)
    dla = tl.where(fin & inside, tl.exp(mu - la), 0.0)
    tl.store(part_ptr + b, tl.sum(tl.sum(v * dla, 1), 0))
    tl.store(steps_ptr + b, steps)


def adjoint_mu_grad(psi, r, lg, la, mu, zoffs, tol=1e-6, max_steps=4000, check=8):
    """(dl/dmu (scalar tensor), adjoint steps (b,)) for the latent with one shared mass m = e^mu and fixed edges and
    source (cram_maze_hard_graph, mass='shared', source='goal'): the adjoint relaxation of GatedLatentEnergy.adjoint and
    its local term, in one launch per maze. The convergence test is per maze (the PyTorch loop tests the batch).
    psi (b, h, w): the settled state; r (b, h, w): dl/dpsi; lg (b, 4, h, w), la (b, h, w): the parts; mu: the log mass."""
    B, H, W = psi.shape
    psi, r, lg, la = (x.contiguous().float() for x in (psi, r, lg, la))
    v_a = torch.empty_like(psi)
    v_b = torch.empty_like(psi)
    part = torch.empty((B,), device=psi.device, dtype=torch.float32)
    steps = torch.empty((B,), device=psi.device, dtype=torch.int64)
    (a0, b0), (a1, b1), (a2, b2), (a3, b3) = zoffs
    _adjoint_kernel[(B,)](psi, r, lg, la, float(mu), v_a, v_b, part, steps, H, W, float(tol), int(max_steps), int(check),
                          a0, b0, a1, b1, a2, b2, a3, b3, BH=triton.next_power_of_2(H), BW=triton.next_power_of_2(W),
                          num_warps=4)
    return part.sum(), steps
