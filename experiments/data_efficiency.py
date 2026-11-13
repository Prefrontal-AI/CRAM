"""Data efficiency of CRAM on Sudoku-Extreme: the Table 2 recipe fitted on k training puzzles, each stored with symmetry
variants. The rows are the ones that TRM's dataset builder stores, so CRAM, TRM, EBR and FRM train on the same rows.

The paper's figure (each puzzle with 99 stored variants), for k in KS and the training seeds 0-5:

  export DE_VARIANTS=99 CRAM_E_ROOT=DE2-v99
  python data_efficiency.py fit --k 30 --seed 0                      # one cold fit in this process, then 7 warm fits
  python data_efficiency.py search --k 30 --seed 0 --data test10k    # the first 10,000 test puzzles
  python data_efficiency.py export                                   # ../results/sudoku-data-eff-shared-v99.json

Fit: the recipe of sudoku.py (pseudolikelihood with proximal L1, SGD with momentum 0.5 and cosine decay from 540, 32
full-batch updates, zero start), with one change: the per-update digit relabeling is off (FullConfig.aug = False),
because the stored variants are the augmentation. A stored variant is the puzzle after a random digit relabeling, a
transposition with probability 1/2, and permutations of the bands, the stacks, the rows within a band and the columns
within a stack (TRM's builder).

Data (TRM_DATA, default runs/trm/data): seed s, k < 1,000: sudoku-smalldata/s<s>-n<k>/train (baselines/trm/trm_smalldata.py:
the first k groups of default_rng(s).permutation(1000) over TRM's 1,000-puzzle set, each the original and its 1,000
stored copies); k = 1,000: TRM's set sudoku-extreme-1k-aug-1000/train for every seed; k = 3,000 and 10,000:
sudoku-smalldata/s<s>-n<k> (the 1,000-puzzle set plus new puzzles, same builder).
DE_VARIANTS=V keeps the original and the first V of the 1,000 stored copies of each group (rows 0..V of every
1,001-row group), so the training set is a row subset of the same files. The default, 1000, keeps all rows.
TRM tokens: 0 pad, 1 blank, digit d (1-9) as d + 1; so CRAM's digit (0-8) is label - 2 and a free cell is input == 1.

Everything after the fit (the stopping threshold tau, the search, its checkpointing) is data_efficiency_common's.
Outputs go to runs/CRAM-E/<CRAM_E_ROOT>; the default folder name is DE2-v<V> (DE1 for V = 1000). The names are only labels.
Above BIG rows the full-batch gradient and prepare()'s training-set statistics are summed over chunks of CHUNK rows: the
same updates and the same tau (a maximum) up to the floating-point summation order, in less GPU memory (one full batch
of 1,001,000 rows peaks at 11.9 GiB).
Dimension names: q puzzles, n = 81 cells.
"""
import os

VARIANTS = int(os.environ.get('DE_VARIANTS', '1000'))  # stored copies kept per original (the paper's figure: 99)
assert 0 <= VARIANTS <= 1000
os.environ.setdefault('CRAM_E_ROOT', 'DE1' if VARIANTS == 1000 else f'DE2-v{VARIANTS}')  # before data_efficiency_common reads it

import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402

import data_efficiency_common as E1  # noqa: E402
import sudoku as CF  # noqa: E402
from cram import cram_sudoku_general as GS  # noqa: E402

HERE = Path(__file__).resolve().parent
assert E1.ROOT == HERE / 'runs/CRAM-E' / os.environ['CRAM_E_ROOT']
TRM_DATA = Path(os.environ.get('TRM_DATA', HERE / 'runs' / 'trm' / 'data'))
KS = (1, 3, 10, 12, 14, 17, 19, 21, 23, 25, 27, 30, 100, 300, 1000, 3000, 10000)  # numbers of training puzzles that `export` collects (12-27 resolve the range where the solve rate rises)
BIG = 4_000_000  # rows above which the fit and prepare() run in chunks of CHUNK rows (k = 10,000: 10,010,000 rows)
CHUNK = 2_000_000
SEEDS = (0, 1, 2, 3, 4, 5) if VARIANTS < 1000 else (0, 1, 2)  # training seeds that `export` collects
NUM_AUG = 1000  # stored copies per original in TRM's files


def data_dir(k, seed):
    """The folder of TRM's stored training rows for k puzzles and seed `seed`."""
    return TRM_DATA / ('sudoku-extreme-1k-aug-1000' if k == 1000 else f'sudoku-smalldata/s{seed}-n{k}') / 'train'


def load_rows(k, seed):
    """TRM's stored rows -> inputs, labels (q, n) uint8 tokens, read into host memory; checks the group layout."""
    d = data_dir(k, seed)
    x = np.load(d / 'all__inputs.npy')
    y = np.load(d / 'all__labels.npy')
    g = np.load(d / 'all__group_indices.npy')
    assert len(g) == k + 1 and (np.diff(g) == 1 + NUM_AUG).all() and g[-1] == len(x) == len(y), f'{d}: not k groups of 1,001 rows'
    if VARIANTS < NUM_AUG:
        keep = np.arange(len(x)) % (1 + NUM_AUG) <= VARIANTS  # the original (row 0 of its group) and the first VARIANTS copies
        x, y = x[keep], y[keep]
        assert len(x) == k * (1 + VARIANTS)
    return x, y


def train_any(cfg, sol, free):
    """sudoku.train (relabeling off) on at most BIG rows; above it the same 32 full-batch updates, with the gradient of
    the pseudolikelihood loss summed over chunks of CHUNK rows.
    sol: (q, n) int64 digits 0-8. free: (q, n) bool. Returns (W, B, seconds of the update loop), as sudoku.train."""
    assert not cfg.aug
    if len(sol) <= BIG:
        return CF.train(cfg, sol, free)
    import math
    dev = sol.device
    iu = CF.pair_index(dev)
    W = torch.zeros(iu.shape[1], CF.V, CF.V, device=dev, requires_grad=True)
    B = torch.zeros(CF.N, CF.V, device=dev, requires_grad=True)
    bufs = [torch.zeros_like(W), torch.zeros_like(B)]
    torch.manual_seed(cfg.seed)
    denom = free.sum()
    t0 = time.perf_counter()
    for k in range(cfg.updates):
        gW, gB = torch.zeros_like(W), torch.zeros_like(B)
        for lo in range(0, len(sol), CHUNK):
            s, f = sol[lo:lo + CHUNK], free[lo:lo + CHUNK]
            nll = -torch.log_softmax(-CF.digit_energies(CF.full_S(W, iu), B, s), -1).gather(-1, s[..., None]).squeeze(-1)
            g = torch.autograd.grad((nll * f).sum() / denom, (W, B))
            gW += g[0]; gB += g[1]
        lr = cfg.lr * 0.5 * (1 + math.cos(math.pi * k / cfg.updates))
        with torch.no_grad():
            for p, g, buf in zip((W, B), (gW, gB), bufs):
                buf.mul_(cfg.momentum).add_(g)
                p.sub_(lr * buf)
                p.copy_(p.sign() * (p.abs() - lr * cfg.lam).clamp_min(0))
    torch.cuda.synchronize()
    return W.detach(), B.detach(), time.perf_counter() - t0


_prepare = CF.prepare


def prepare_any(W, B, sol):
    """sudoku.prepare on at most BIG rows; above it tau (a maximum) and ebar (a mean) are accumulated over chunks of
    the training solutions sol (q, n). Returns the same dict as sudoku.prepare."""
    if len(sol) <= BIG:
        return _prepare(W, B, sol)
    S, g = CF.ready(W, B)
    _, _, D = GS.neighbour_blocks(S)
    tau, own_sum = -math.inf, 0.0
    for lo in range(0, len(sol), CHUNK):
        c = sol[lo:lo + CHUNK]
        tau = max(tau, float(GS.board_energy(S, B, c).max()))
        own_sum += float(GS.local_energies(S, B, c).gather(2, c[..., None].long()).squeeze(-1).double().sum())
    return {'S': S, 'B': B, 'g': g, 'ebar': own_sum / (len(sol) * CF.N), 'tau_train': tau, 'hbar': float(W[W > 0].mean()), 'D': D}


CF.prepare = prepare_any  # data_efficiency_common.search calls CF.prepare


def to_cram(x, y):
    """TRM tokens -> (q, n) int64 digits 0-8 and (q, n) bool free mask (blank input cells)."""
    sol = y.astype(np.int64) - 2
    free = x == 1
    assert sol.min() >= 0 and sol.max() <= 8 and ((x == 1) | (x.astype(np.int64) == y)).all(), 'unexpected tokens'
    return sol, free


def fit(k: int, seed: int, repeats: int = 7):
    """Cold fit (first in this fresh process) and `repeats` warm fits. The clock runs from TRM's rows in host memory to the
    inference-ready model: token conversion, transfer, the 32 updates and prepare() (S, neighbour blocks, g, ebar, tau,
    hbar). Reading the files, saving and diagnostics are outside it. CUDA context timed apart."""
    out = E1.run_dir(k, seed)
    if (out / 'fit.json').exists():
        print(out / 'fit.json', 'exists', flush=True)
        return
    dev = torch.device('cuda')
    t0 = time.perf_counter(); torch.zeros(1, device=dev); torch.cuda.synchronize(); ctx = time.perf_counter() - t0
    x, y = load_rows(k, seed)
    cfg = CF.FullConfig(aug=False, split=seed, seed=seed)

    def one():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s, f = to_cram(x, y)
        st, ft = torch.as_tensor(s, device=dev), torch.as_tensor(f, device=dev)
        W, B, loop_s = train_any(cfg, st, ft)
        hbar_ok = bool((W > 0).any())
        m = prepare_any(W, B, st) if hbar_ok else None
        torch.cuda.synchronize()
        return time.perf_counter() - t0, loop_s, W, B, m, s, f

    torch.cuda.reset_peak_memory_stats()
    cold, cold_loop, W, B, m, s, f = one()
    peak = torch.cuda.max_memory_allocated() / 2**30
    warm = [one() for _ in range(repeats)]
    same = all(torch.equal(w[2], W) and torch.equal(w[3], B) for w in warm)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / 'model.tmp.pt'
    torch.save({'W': W.cpu(), 'B': B.cpu()}, tmp)
    os.replace(tmp, out / 'model.pt')
    np.savez_compressed(out / 'train.npz', solutions=s.astype(np.int8), free=f)
    Wc = W.cpu()
    eye = torch.eye(9, dtype=torch.bool)
    originals = s[::1 + VARIANTS]
    rec = {'k': k, 'seed': seed, 'data': str(data_dir(k, seed)), 'variants': VARIANTS, 'per_update_relabeling': False,
           'examples': int(len(s)), 'unique_examples': len({r.tobytes() for r in s.astype(np.uint8)}),
           'config': dataclasses.asdict(cfg), 'fit_gpu_cold_s': cold, 'loop_cold_s': cold_loop,
           'fit_gpu_warm_s': [w[0] for w in warm], 'fit_gpu_warm_median_s': statistics.median(w[0] for w in warm),
           'warm_bitwise_equal': same, 'cuda_context_s': ctx, 'peak_memory_gib': peak,
           'nonzero_params': int((W != 0).sum() + (B != 0).sum()), 'params': int(W.numel() + B.numel()),
           'nonzero_offdiag': int((Wc[:, ~eye] != 0).sum()), 'nonzero_B': int((B != 0).sum()),
           'positive_W': int((W > 0).sum()), 'negative_W': int((W < 0).sum()),
           'min_W': float(W.min()), 'max_W': float(W.max()),
           'model_sha256': hashlib.sha256((out / 'model.pt').read_bytes()).hexdigest(),
           'gpu': torch.cuda.get_device_name(), 'node': os.uname().nodename}
    if m is not None:
        orig_E = GS.board_energy(m['S'], B, torch.as_tensor(originals, device=dev))
        rec.update(tau=m['tau_train'], tau_originals=float(orig_E.max()), ebar=m['ebar'], hbar=m['hbar'], g=m['g'], D=m['D'])
    else:
        rec.update(tau=None, degenerate='no positive entry of W: hbar undefined, no search')
    E1.dump(rec, out / 'fit.json')
    print(json.dumps({z: rec.get(z) for z in ('k', 'seed', 'examples', 'fit_gpu_cold_s', 'fit_gpu_warm_median_s', 'peak_memory_gib',
                                              'nonzero_params', 'nonzero_offdiag', 'negative_W', 'tau', 'hbar', 'D')}), flush=True)


def search(k: int, seed: int, data: str = 'screen'):
    """data_efficiency_common.search on this study's run folder. data 'test10k': the first 10,000 test puzzles (the
    paper's figure); any other value: all 412,786 test puzzles, written to <data>.json."""
    E1.search(k, seed, data)


def export(out: str = ''):
    """Per k x seed: test solve rate, wrong stops, budget failures, nonzero parameters, mean flop and fit times."""
    rows = []
    for k in KS:
        for s in SEEDS:
            d = E1.run_dir(k, s)
            if not (d / 'fit.json').exists():
                continue
            f = json.loads((d / 'fit.json').read_text())
            r = {'k': k, 'seed': s, 'training_examples': f['examples'], 'nonzero_params': f['nonzero_params'], 'params': f['params'],
                 'fit_gpu_cold_s': f['fit_gpu_cold_s'], 'fit_gpu_warm_median_s': f['fit_gpu_warm_median_s'], 'tau': f.get('tau')}
            for dd in ('screen', 'test10k', 'test'):
                p = d / f'{dd}.json'
                if p.exists():
                    sm = json.loads(p.read_text()).get('summary', {})
                    r[dd] = {z: sm.get(z) for z in ('n', 'solved', 'wrong_stops', 'budget_failures', 'mean_flop', 'median_flop')}
            rows.append(r)
    out = out or ('../results/sudoku-data-eff-shared.json' if VARIANTS == 1000 else f'../results/sudoku-data-eff-shared-v{VARIANTS}.json')
    path = (HERE / out).resolve()
    E1.dump({'source': f"experiments/data_efficiency.py; experiments/runs/CRAM-E/{os.environ['CRAM_E_ROOT']}/k<k>-s<seed>/{{fit,screen,test10k}}.json",
             'variants': VARIANTS,
             'rows': rows}, path)
    print(path)


if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict({'fit': fit, 'search': search, 'export': export})
