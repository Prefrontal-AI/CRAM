"""Scores and inference operation count (flop) per puzzle of EBR (Energy-Based Reasoner), from the per-sample records of
EBR's own evaluate.py (github.com/wwillchen/ebr at 8c5271c; written by ebr/score.sbatch). CPU only; no model runs.
The `gram` mode imports EBR's evaluators.gram from $THIRD_PARTY/ebr for its N-Queens and graph-coloring metrics.

EBR's evaluators record, for every sample (one descent trajectory from a fresh random state) and every segment end
(`cycles` = 21 descent microsteps), whether the decoded board is exact, the energy E and the tangential gradient norm
|grad_M E|. A run at depth D and breadth B therefore also holds every smaller setting: segment d of a sample does not
depend on the depth asked for (the Sudoku and maze runners keep the noise at its trained final value past the trained
depth). The settings scored here:
- fixed depth d, N samples: each sample answers at segment d; the answer is the lowest-energy sample of the first N
  (EBR's pass@N rule; N = 1 is sample 0);
- stopping at |grad_M E| < tau (EBR's adaptive stop, as its N-Queens / Graph Coloring protocol gtan_stop_index): each
  sample answers at its first segment end with |grad_M E| < tau, else at the cap d; then the same pick.
Flop per puzzle (ebr/flops.py components, one sample): fixed depth: N x (d x segment) + (N > 1) x N x energy;
adaptive: sum over the N samples of s_i x (segment + measure) + (N > 1) x N x energy, where s_i is the sample's
segment count (the stop test needs E and grad E at every segment end: one measure, forward + backward).

  python ebr/report.py sudoku <score dir> <flops.json> <out.json>     # dir of shard_<i>_of_16.{npz,json}
  python ebr/report.py maze <score npz> <flops.json> <out.json>
  python ebr/report.py gram <score npz> <flops.json> <out.json>         # N-Queens (nq.npz), Graph Coloring (gc.npz)

Dimension names: D = segments (depth), P = puzzles, B = samples per puzzle; in the gram mode S = samples, U = unique inputs.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'trm'))

DEPTHS = (1, 2, 4, 8, 16, 32, 64)
NS = (1, 2, 5, 8)
TAUS = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)


def digest(path):
    """SHA-256 of a file, as a hexadecimal string."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pick(exact, energy, n):
    """exact, energy [P, B] -> solved flag [P] of the lowest-energy sample among the first n (sample 0 if n = 1)."""
    if n == 1:
        return exact[:, 0]
    j = energy[:, :n].argmin(1)
    return exact[np.arange(len(exact)), j]


def settings(exact, energy, gtan, f, depths=DEPTHS, ns=NS, taus=TAUS):
    """Score every inference setting.

    exact [D, P, B] bool, energy [D, P, B], gtan [D, P, B] (|grad_M E|) at each segment end; f = the components of ebr/flops.py.
    Returns {name: (solved [P] bool, flop [P], settings dict)}; names are D<depth>-N<samples> and gtan<tau>-cap<depth>-N<samples>."""
    D, P, B = exact.shape
    out = {}
    seg, mea, ene = f['segment_flop'], f['measure_flop'], f['energy_flop']
    for d in depths:
        if d > D:
            continue
        for n in ns:
            if n > B:
                continue
            ok = pick(exact[d - 1], energy[d - 1], n)
            flop = np.full(P, n * d * seg + (n > 1) * n * ene, float)
            out[f'D{d}-N{n}'] = (ok, flop, dict(depth=d, samples=n, stop='fixed depth'))
        for tau in taus:
            hit = gtan[:d] < tau
            stop = np.where(hit.any(0), hit.argmax(0), d - 1)  # [P, B], 0-based segment
            pi, bi = np.meshgrid(np.arange(P), np.arange(B), indexing='ij')
            ex, en = exact[stop, pi, bi], energy[stop, pi, bi]
            for n in ns:
                if n > B:
                    continue
                ok = pick(ex, en, n)
                flop = (stop[:, :n] + 1).sum(1) * (seg + mea) + (n > 1) * n * ene
                out[f'gtan{tau}-cap{d}-N{n}'] = (ok, flop.astype(float), dict(depth=d, samples=n, stop=f'|grad_M E| < {tau}', tau=tau))
    return out


def load_sudoku(folder):
    """The 16 shards of one checkpoint -> exact, energy, gtan [D, 422786, B] in official test order, and the metas."""
    shards = sorted(Path(folder).glob('shard_*_of_*.npz'), key=lambda p: int(p.stem.split('_')[1]))
    metas = [json.loads(p.with_suffix('.json').read_text())['meta'] for p in shards]
    assert len(shards) == metas[0]['num_shards'] and sorted(m['shard'] for m in metas) == list(range(len(shards)))
    assert len({m['checkpoint'] for m in metas}) == 1 and len({m['breadth'] for m in metas}) == 1
    B = metas[0]['breadth']
    z = [np.load(p) for p in shards]
    board = np.concatenate([a['board'] for a in z])
    P = len(board) // B
    by = board.reshape(P, B)
    assert (by == by[:, :1]).all() and np.array_equal(by[:, 0], np.arange(P)), 'boards out of order'
    arr = {k: np.concatenate([a[k] for a in z], 1).reshape(-1, P, B) for k in ('exact', 'energy', 'gtan')}
    return arr['exact'].astype(bool), arr['energy'].astype(np.float64), arr['gtan'].astype(np.float64), metas, shards


def summarize(sets, mask):
    """Per setting of `sets`, the solved count, rate and flop statistics over the puzzles selected by mask [P] bool."""
    return {name: dict(solved=int(ok[mask].sum()), n=int(mask.sum()), rate=float(ok[mask].mean()),
                       mean_flop=float(flop[mask].mean()), p99_flop=float(np.percentile(flop[mask], 99)), knobs=k)
            for name, (ok, flop, k) in sets.items()}


def sudoku(folder, flops, out):
    """Sudoku-Extreme: score the 16 shards in `folder`; write `out` (JSON) and the per-puzzle flags beside it (.flags.npz)."""
    from unseen import unseen_mask  # baselines/trm/unseen.py: official test rows outside dev10k
    exact, energy, gtan, metas, shards = load_sudoku(folder)
    f = json.loads(Path(flops).read_text())
    sets = settings(exact, energy, gtan, f)
    mask, dev_idx = unseen_mask()
    assert mask.size == exact.shape[1]
    sweep_rows = sweep_official_rows()
    sub = np.zeros_like(mask)
    sub[sweep_rows] = True
    np.savez_compressed(Path(out).with_suffix('.flags.npz'), **{k: ok for k, (ok, _, _) in sets.items()},
                        **{f'flop_{k}': fl.astype(np.float32) for k, (_, fl, _) in sets.items()})
    rec = dict(method='EBR', checkpoint=metas[0]['checkpoint'], checkpoint_sha256=digest(metas[0]['checkpoint']),
               breadth=metas[0]['breadth'], depth=metas[0]['depth'], flops=flops, flops_sha256=digest(flops),
               shards={str(p): digest(p) for p in shards}, script=str(Path(__file__)), script_sha256=digest(__file__),
               unseen=summarize(sets, mask), sweep10k=summarize(sets, sub), all=summarize(sets, np.ones_like(mask)))
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    for key in ('D16-N1', 'D64-N1', 'D64-N5', 'D64-N8'):
        u = rec['unseen'][key]
        print(f"{key}: unseen {u['rate'] * 100:.3f}% ({u['solved']}/{u['n']}), {u['mean_flop']:.3g} flop; sweep10k {rec['sweep10k'][key]['rate'] * 100:.3f}%")


def sweep_official_rows():
    """Official test rows of full-all[10000:20000], the compute sweep's puzzles (as baselines/sweep/build.py sudoku_rows)."""
    data = HERE.parents[1] / 'data'
    idx = json.loads((data / 'full-test/index.json').read_text())
    rows = np.concatenate([np.array(idx['shards']['full-frmtest']['official_rows']), np.array(idx['shards']['full-00']['official_rows'])[:9000]])
    assert idx['full_all_order']['full-frmtest'] == [10000, 1000] and idx['full_all_order']['full-00'][0] == 11000
    return rows


def maze(npz, flops, out):
    """EBR's maze runner: arrays exact (canonical path), shortest, valid, energy [segments, mazes x n] (n samples per maze,
    contiguous). Maze-Unique has one path, so exact = shortest. EBR's published Maze-Hard numbers are exact (the labeled path):
    the released checkpoint gives 88.2 exact and 95.3 shortest against the published 88.3."""
    a = np.load(npz)
    meta = json.loads(Path(npz).with_suffix('.json').read_text())['meta']
    n = meta['n']
    D, R = a['exact'].shape
    M = R // n
    f = json.loads(Path(flops).read_text())
    rec = dict(method='EBR', npz=str(npz), npz_sha256=digest(npz), meta=meta, flops=flops, script=str(Path(__file__)), script_sha256=digest(__file__))
    for key in ('exact', 'shortest'):
        ex = a[key].reshape(D, M, n).astype(bool)
        en = a['energy'].reshape(D, M, n).astype(np.float64)
        gt = a['gtan'].reshape(D, M, n).astype(np.float64) if 'gtan' in a.files else np.full_like(en, np.inf)
        sets = settings(ex, en, gt, f, depths=tuple(d for d in (1, 2, 4, 6, 8, 16, 32) if d <= D), ns=tuple(k for k in (1, 2, 5) if k <= n),
                        taus=TAUS if 'gtan' in a.files else ())
        rec[key] = summarize(sets, np.ones(M, bool))
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    for key in (f'D{D}-N1', f'D{D}-N{n}'):
        print(key, {k: round(rec[k][key]['rate'] * 100, 2) for k in ('exact', 'shortest')}, f"{rec['exact'][key]['mean_flop']:.3g} flop")


def nqueens_gram_coverage(a, stop):
    """N-Queens Cov@20 in GRAM's per-row form, from EBR's samples read out at ``stop`` [S, U].

    Per test row (an input with s solutions occurs s times): the distinct valid samples (of 20) that equal one of the
    input's first 20 completions in find_all_solutions order, over their number. cov20_row_mean averages over the rows;
    cov20_group_mean averages within groups of rows with equal completion count min(s, 20), then over the groups without
    weights, the form GRAM's evaluator prints and the paper's N-Queens appendix uses for CRAM (experiments/nqueens_export.py).
    EBR's own cov20 instead counts any valid completion, capped at 20 per input, pooled over inputs."""
    from evaluators.gram import nqueens_solutions
    S, U = stop.shape
    rows = np.diff(a['puzzle_indices']).astype(int)
    si, ui = np.meshgrid(np.arange(S), np.arange(U), indexing='ij')
    P = a['decode'][stop, si, ui].astype(np.int64)
    n = int(round(P.shape[-1] ** 0.5))
    cov = np.zeros(U)
    for u in range(U):
        first20 = set(nqueens_solutions(a['inputs'][u].reshape(n, n), n))
        cov[u] = len({p.tobytes() for p in P[:, u]} & first20) / len(first20)
    groups = np.minimum(rows, 20)
    group_means = [np.average(cov[groups == k], weights=rows[groups == k]) for k in np.unique(groups)]
    return dict(cov20_row_mean=100.0 * float(np.average(cov, weights=rows)), cov20_group_mean=100.0 * float(np.mean(group_means)))


def gram(npz, flops, out):
    """N-Queens / Graph Coloring from EBR's gram runner (20 samples per unique test input, every segment recorded).

    Scores at EBR's two stopping rules (|grad_M E| < 0.2 at a segment end, cap at the trained depth; and the trained
    depth): EBR's own cells (evaluators.gram.gram_cells: N = 1 is sample 0, N = 20 the lowest raw energy; Cov@20 with
    the denominator capped at 20 solutions per input), and for Graph Coloring also Cov@20 by GRAM's definition (no cap:
    distinct proper canonical colorings found over all canonical proper colorings, summed over graphs), which the
    project page reports. Flop per input: N = 1 is sample 0's segments x (segment + measure); N = 20 sums the 20
    samples and adds 20 energy forwards for the pick (the runner's energy_raw is E(z), one forward)."""
    from evaluators.gram import all_colorings, canonical, coloring_adjacency, gram_cells, gtan_stop_index
    a = np.load(npz)
    task, nodes, cap = str(a['task']), int(a['nodes']), int(a['train_halt'])
    f = json.loads(Path(flops).read_text())
    seg, mea, ene = f['segment_flop'], f['measure_flop'], f['energy_flop']
    S, U = a['gtan'].shape[1:]
    kw = {} if task == 'gc' else {'rows_per_input': np.diff(a['puzzle_indices'])}
    rec = dict(method='EBR', task=task, npz=str(npz), npz_sha256=digest(npz), flops=flops, script=str(Path(__file__)), script_sha256=digest(__file__))
    for name, stop in ((f'gtan<0.2', gtan_stop_index(a['gtan'], 0.2, cap)), (f'fixed D{cap}', np.full((S, U), cap - 1))):
        cells = gram_cells(task, a['decode'], a['energy_raw'], stop, a['inputs'], nodes, **kw)
        per = stop + 1
        cells['flop_n1'] = float((per[0] * (seg + (mea if name.startswith('gtan') else 0))).mean())
        cells['flop_n20'] = float((per.sum(0) * (seg + (mea if name.startswith('gtan') else 0)) + S * ene).mean())
        if task == 'gc':
            si, ui = np.meshgrid(np.arange(S), np.arange(U), indexing='ij')
            P = a['decode'][stop, si, ui].astype(int)
            found = total = 0
            for u in range(U):
                adj = coloring_adjacency(a['inputs'][u], nodes)
                edges = list(zip(*np.nonzero(np.triu(adj, 1))))
                sols = all_colorings(adj)
                got = {tuple(int(v) - 3 for v in P[s, u, :nodes]) for s in range(S)}
                found += len({c for c in got if all(c[i] != c[j] for i, j in edges) and canonical(c) == c} & sols)
                total += len(sols)
            cells['cov20_gram_uncapped'] = 100.0 * found / total
        else:
            cells.update(nqueens_gram_coverage(a, stop))
        rec[name] = cells
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k.startswith(('gtan', 'fixed'))}))


if __name__ == '__main__':
    if sys.argv[1] == 'gram':
        sys.path.insert(0, str(Path(os.environ.get('THIRD_PARTY', HERE.parents[1] / 'third_party')) / 'ebr'))
    {'sudoku': sudoku, 'maze': maze, 'gram': gram}[sys.argv[1]](*sys.argv[2:5])
