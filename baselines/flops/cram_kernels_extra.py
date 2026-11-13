"""CRAM's remaining rows counted under the arithmetic-only convention of fullcount.py, as cram_kernels.py counts the
CRAM Sudoku-Extreme row of Table 2. It calls no baseline code: it runs CRAM's own search code under FullCounter.

Each row's search is counted once on real inputs at batch 1 with the row's real model, and the per-puzzle totals of the
existing test records are rebuilt from their saved per-puzzle move or step counts, exactly as the old formula used
them, with the old per-move / per-step constants replaced by the counted ones (no rescoring). Only the prediction
pipeline is counted: model-level preparation done once per model (S = A + A^T, g from the smallest eigenvalue, the
fitted thresholds) is not, and comparisons against the labels are class compare, outside the total.

- hand-sudoku: experiments/hand_set.py sudoku (cram_sudoku_sparse.SparseSearch, one level, no input costs, D = 20, focused).
  The Triton kernel _moves_sparse is counted through cram_kernels.move_sparse (its line-by-line transcription), run on
  one chain of the real hand-set search; the work outside the kernel (the constructor: first boards, their fields,
  energies and focus flags; each check of 4,096 moves: bookkeeping and stops, no refresh at one level) is counted on
  the real SparseSearch with the kernel switched off, as cram_kernels.count_general_overheads does.
- hand-maze-unique: experiments/hand_set.py maze (maze_unique.gibbs, T 0, random focus: focused Gibbs search), counted
  directly per maze; total = setup + per_move x moves, solved from two mazes and checked on the others.
- cram-sgd-fg (the Maze-Unique row of Table 2): the free 3x3 kernel trained with SGD (models in
  experiments/runs/CU13/final/fg-s*; the folder name is only a label), searched by maze_unique.gibbs at T 0 with random
  focus, counted directly per maze on each seed's model; total = setup + per_move x moves.
- cram-nqueens-8, cram-nqueens-10: cram.cram_nqueens.solve (PyTorch) with the selected methods' models (8 mpl,
  10 cd), counted directly per chain; per chain = start + per_move x moves; N=1 = chain 0, N=20 = the sum of 20 chains.
- cram-gc<n>-<arm>: cram.cram_graph_coloring.solve is NumPy, which FullCounter does not see; gc_solve_torch
  transcribes it line by line into PyTorch, drawing the same random numbers from the same NumPy generator in the same
  order (checked: the same moves and boards as solve), and the transcription is counted. Per graph = setup (coupling
  assembly, thresholds) + chains x start + per_move x (sum of the chains' moves).
- cram-latent-zmu (Maze-Hard; experiments/maze_hard_latent.py, models in experiments/runs/CRAM-HE/final-triton/CHE8-mu;
  the folder name is only a label): the inference of maze_hard_latent.evaluate with descent, counted per maze on
  training mazes (the test file keeps its single scoring): the latent's settle (setup + per x psi steps), the
  attention, maze_h3.descend (per x descent steps). The test records keep only each maze's old flop; (psi steps,
  descent steps) are decoded from it (unique for every maze, asserted).

Each group writes one record <out_dir>/cram-<row>.json; baselines/flops/fc2_export.py reads the records.

  ../../.venv/bin/python cram_kernels_extra.py <group> <out_dir>      (from baselines/flops, on a compute node)
  groups: hand-sudoku hand-maze sgd-maze nqueens gc maze-hard-zmu

Dimension names: n cells (vertices), d digits (colors), k neighbour slots (D), c chains, h chains in a batch.
"""
import hashlib
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CWD = os.getcwd()  # output paths are relative to the caller's folder (cram_kernels changes to experiments/ on import)
sys.path.insert(0, str(HERE))
import cram_kernels as CK  # noqa: E402  (puts experiments/ on sys.path and changes to experiments/)
from fullcount import CONVENTION  # noqa: E402

ROOT = CK.ROOT
EXP = ROOT / 'experiments'
CPU = torch.device('cpu')
_count = CK._count


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------
def sha(path):
    """SHA-256 hex digest of the file at `path`."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    """`path` relative to the repository root."""
    return str(Path(path).resolve().relative_to(ROOT))


def brief(s):
    """A FullCounter summary without the per-operator table (kept only where a record shows one component in full)."""
    return {k: v for k, v in s.items() if k not in ('by_op', 'convention')}


def scaled(s, k):
    """A summary of k identical repeats, per repeat."""
    assert s['total'] % k == 0 and s['total_with_compare'] % k == 0, (s['total'], k)
    return dict(total=s['total'] // k, total_with_compare=s['total_with_compare'] // k,
                classes={c: v / k for c, v in s['classes'].items()},
                by_op={op: {c: v / k for c, v in d.items()} for op, d in s['by_op'].items()})


def linear(rows, key='m'):
    """total = a + b * m from the rows with the smallest and largest m; residuals on every row (0 when exact)."""
    lo, hi = min(rows, key=lambda r: r[key]), max(rows, key=lambda r: r[key])
    assert hi[key] > lo[key], 'need two different move counts'
    out = {}
    for t in ('total', 'total_with_compare'):
        b = (hi['count'][t] - lo['count'][t]) / (hi[key] - lo[key])
        a = lo['count'][t] - b * lo[key]
        res = [r['count'][t] - (a + b * r[key]) for r in rows]
        out[t] = dict(setup=a, per=b, max_abs_residual=max(abs(x) for x in res))
    return out


def stats(per_seed_arrays):
    """Per-seed means, their mean (the table's statistic) and pooled quantiles."""
    pooled = np.concatenate(per_seed_arrays)
    q = lambda p: float(np.quantile(pooled, p))  # noqa: E731
    means = [float(a.mean()) for a in per_seed_arrays]
    return dict(per_seed_mean=means, mean_over_seeds=float(np.mean(means)), pooled_median=q(0.5), pooled_p99=q(0.99),
                pooled_max=float(pooled.max()))


def meta(extra_sources=()):
    """Provenance fields of a record: this script, the counter, `extra_sources` (each with its SHA-256), host and time."""
    srcs = [HERE / 'cram_kernels_extra.py', HERE / 'cram_kernels.py', HERE / 'fullcount.py', *extra_sources]
    return dict(script='baselines/flops/cram_kernels_extra.py', script_sha256=sha(HERE / 'cram_kernels_extra.py'),
                source_sha256={rel(p): sha(p) for p in srcs}, slurm_job=os.environ.get('SLURM_JOB_ID'),
                host=platform.node(), torch=torch.__version__, written=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                convention=CONVENTION)


def dump(out_dir, name, rec):
    """Write `rec` to <out_dir>/cram-<name>.json through a temporary file."""
    p = Path(out_dir) / (name if name.startswith('cram-') else f'cram-{name}')
    p = p.with_name(p.name + '.json')
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(rec, indent=1) + '\n')
    os.replace(tmp, p)
    print(p, flush=True)


# ---------------------------------------------------------------------------------------------------------------
# Hand-set Sudoku row (experiments/hand_set.py sudoku; results/hand-energy.json; records experiments/runs/HAND-E1/sudoku-test-s*)
# ---------------------------------------------------------------------------------------------------------------
def hand_sudoku(out_dir, seeds=(0, 1, 2)):
    """Count the hand-set Sudoku energy's search and rebuild the per-puzzle totals of its test records; writes cram-hand-sudoku.json."""
    import hand_set as HE
    from cram import cram_sudoku_search as SS
    from cram.cram_sudoku_sparse import SparseSearch
    A, B = HE.sudoku_hand()
    s = HE.SUDOKU_SEARCH
    g = 9 / float(torch.linalg.eigvalsh((A + A.T).double())[0].abs())  # model-level (temperature unit), not counted
    kw = dict(temp=s['temp'] / g, theta=s['theta'] / g, focused=True, C=s['C'], test='pair', tau=0.0, tol=s['tol'], ebar=0.0,
              cap=s['cap'], check=s['check'], restart=0, growth=1.0, seed=0,
              anneal=(s['anneal'][0] / g, s['anneal'][1] / g, s['anneal'][2]), device='cpu')
    sol, free, _ = SS.load_puzzles('full-all', HE.TEST_LO, 2)
    holder = {}
    init1 = _count(lambda: holder.setdefault('s1', SparseSearch(A, B, sol[:1], free[:1], **kw)))
    init2 = _count(lambda: holder.setdefault('s2', SparseSearch(A, B, sol[:2], free[:2], **kw)))
    ss = holder['s1']
    assert ss.one_level and ss.nobias and ss.D == 20 and ss.hscale == HE.SUDOKU_H and ss.focused and ss.restart == 0
    assert ss.fl_move == 124.0

    # the kernel: move_sparse (MODE 2 = focused, NOBIAS) on chain 0 of the real search
    st = dict(G=ss.G[0].clone(), y=ss.y[0].clone(), flag=ss.flag[0].clone(), fmask=ss.fmask[0], free=ss.freelist[0], nbr=ss.nbr,
              w=ss.w, B=ss.B, x=ss.x[0], P=ss.P[0].clone(), T=torch.tensor(ss.current_temp()), hscale=torch.tensor(ss.hscale),
              nf=ss.nfree[0], found=ss.found[0].clone(), thr=torch.tensor(ss.thr), tau=torch.tensor(ss.tau + ss.tol),
              move0=torch.tensor(0, dtype=torch.int32), k=torch.tensor(0, dtype=torch.int32))
    reps = 64
    kern = scaled(_count(lambda: [CK.move_sparse(st, 81, 9, ss.D, focused=True, nobias=True) for _ in range(reps)]), reps)
    synth = _count(lambda: CK.move_sparse(CK._sparse_state(D=20), 81, 9, 20, focused=True, nobias=True))
    assert kern['total'] == synth['total'], (kern['total'], synth['total'])

    # the work outside the kernel: one check of 4,096 moves (t % 4096 == 0: _periodic_refresh is a no-op at one level)
    ss._moves = lambda nmoves: None
    chk = _count(ss.step_check)
    s3 = SparseSearch(A, B, sol[:1], free[:1], **kw)  # a check at which a chain has passed (the stop path)
    s3._moves = lambda nmoves: None
    s3.found[1] = 17
    chk_stop = _count(s3.step_check)
    assert bool(s3.stopped[0])
    check = max(chk, chk_stop, key=lambda x: x['total'])
    start = init1
    per_puzzle_start = {t: init2[t] - init1[t] for t in ('total', 'total_with_compare')}

    recs, old, new, new_wc, sources = [], [], [], [], {}
    for seed in seeds:
        p = EXP / f'runs/HAND-E1/sudoku-test-s{seed}.json'
        d = json.loads(p.read_text())
        sources[rel(p)] = sha(p)
        sig, pp = d['signature'], d['per_puzzle']
        C, flm = sig['C'], sig['fl_move']
        assert sig['restart'] == 0 and sig['check'] == 4096 and flm == 124.0 and sig['focused'] and sig['sparse']
        fl = np.asarray(pp['flops'], np.float64)
        steps = fl / (flm * C)
        assert np.array_equal(steps, np.round(steps))
        nchk = np.ceil(steps / 4096)
        nw = steps * C * kern['total'] + start['total'] + nchk * check['total']
        nwc = steps * C * kern['total_with_compare'] + start['total_with_compare'] + nchk * check['total_with_compare']
        assert abs(fl.mean() - d['summary']['mean_flop']) <= 1e-6 * fl.mean()
        old.append(fl); new.append(nw); new_wc.append(nwc)
        recs.append(dict(seed=seed, puzzles=len(fl), old_mean=float(fl.mean()), new_mean=float(nw.mean()),
                         new_with_compare_mean=float(nwc.mean()), mean_steps=float(steps.mean()), mean_checks=float(nchk.mean())))
    exp = json.loads((ROOT / 'results/hand-energy.json').read_text())
    paper_old = next(r for r in exp['rows'] if r['benchmark'] == 'sudoku-extreme')['flop_per_puzzle']
    rec = dict(row='sudoku-extreme hand-energy (Hand-set energy, focused Gibbs)', paper_value_old=paper_old,
               search='cram_sudoku_sparse.SparseSearch: one level (h = 1, w = 1, hscale = 1), no input costs (NOBIAS), focused '
                      '(MODE 2), D = 20, C = 4, check 4,096, no restarts; kernel _moves_sparse',
               counted=dict(move_per_chain=kern, move_check_synthetic_state_total=synth['total'],
                            start_per_puzzle=start, start_per_puzzle_slope=per_puzzle_start,
                            start_note='start = the SparseSearch constructor on one puzzle (as cram_kernels counts the main '
                                       'row): S = A + A^T twice (SparseSearch and Search, model-level), the first random '
                                       'boards of the 4 chains, their fields (einsum over the 20 neighbours), energies and '
                                       'focus flags. start_per_puzzle_slope = constructor(2 puzzles) - constructor(1): the '
                                       'part that scales with the puzzles of a batch',
                            check_per_puzzle=check, check_no_stop=brief(chk), check_with_stop=brief(chk_stop),
                            check_note='step_check with the kernel switched off at t = 4,096: one level, so no field '
                                       'refresh; bookkeeping (min, where, stop) only'),
               old_constants=dict(fl_move=124.0, formula='flops = steps x fl_move x C (Search.result; no restarts)',
                                  fl_move_formula='4 D + (D + 1) + (2 d + 1) + 4 = 124 at D 20, d 9'),
               formula='per puzzle = steps x C x move + start + ceil(steps / 4096) x check; steps = old flops / (124 x 4) '
                       '(Search.result: stop_t of the first passing chain, or the moves run at the cap)',
               per_seed=recs, old=stats(old), new=stats(new), new_with_compare=stats(new_wc),
               ratio_new_over_old=float(np.mean([r['new_mean'] for r in recs]) / np.mean([r['old_mean'] for r in recs])),
               source_records=sources, notes=[
                   'The kernel count per move equals the old fl_move (124): the transcription counts the neighbour-field '
                   'updates (3 D: weight multiply, subtract, add), the Gumbel transform (4 d), the heat-bath logits (2 d + 2) '
                   'and the energy and certificate (4) plus 2 integer scalar operations; the old count had 4 D + (D + 1) + '
                   '(2 d + 1) + 4, whose focus comparisons are not counted now.',
                   'Steps are stop_t, the first passing move; the chains run on to the next check (up to 4,095 more moves '
                   'each). The old formula and this rebuild both charge stop_t only.'])
    rec.update(meta([EXP / 'hand_set.py', EXP.parent / 'cram/cram_sudoku_sparse.py', EXP.parent / 'cram/cram_sudoku_search.py']))
    dump(out_dir, 'hand-sudoku', rec)


# ---------------------------------------------------------------------------------------------------------------
# Maze-Unique focused Gibbs (maze_unique.gibbs), counted per maze
# ---------------------------------------------------------------------------------------------------------------
def count_per_maze(model, run_one, mazes, key):
    """Count run_one(model, x, y, free) on each test maze of `mazes` at batch 1 on the CPU.

    Returns (fit, rows): fit is the linear law total = setup + per x m of `linear`, with m = result[key] (moves); rows
    holds each maze's m and FullCounter summary."""
    from cram import maze_unique as M
    data = M.load('test')
    rows = []
    for i in mazes:
        x, y, fr = M.to_device(data, [i], CPU)
        holder = {}
        with torch.no_grad():
            s = _count(lambda: holder.setdefault('r', run_one(model, x, y, fr)))
        rows.append(dict(maze=int(i), m=int(holder['r'][key][0]), count=s))
    fit = linear(rows)
    for r in rows[1:]:
        r['count'] = brief(r['count'])
    return fit, rows


def gibbs_one(model, x, y, fr):
    """The focused Gibbs search of the Maze-Unique rows on one maze: T 0, random focus, at most 20,000 moves, seed 0."""
    from cram import maze_unique as M
    return M.gibbs(model, x, y, fr, T=0.0, max_moves=20000, focus='random', gen=torch.Generator().manual_seed(0))


def rebuild_maze(paths, fit, old_fn):
    """Per-maze totals of the test records `paths` (one per seed) from their saved step counts and the counted `fit`.

    old_fn(steps) is the old formula; it must reproduce the flop that each record saved."""
    recs, old, new, new_wc, sources = [], [], [], [], {}
    for seed, p in enumerate(paths):
        d = json.loads(Path(p).read_text())
        sources[rel(p)] = sha(p)
        m = np.asarray(d['per_maze']['steps'], np.float64)
        fo = np.asarray(d['per_maze']['flop'], np.float64)
        assert np.allclose(fo, old_fn(m)), 'old formula does not reproduce the record'
        nw = fit['total']['setup'] + fit['total']['per'] * m
        nwc = fit['total_with_compare']['setup'] + fit['total_with_compare']['per'] * m
        old.append(fo); new.append(nw); new_wc.append(nwc)
        recs.append(dict(seed=seed, mazes=len(m), mean_steps=float(m.mean()), old_mean=float(fo.mean()), new_mean=float(nw.mean()),
                         new_with_compare_mean=float(nwc.mean())))
    return dict(per_seed=recs, old=stats(old), new=stats(new), new_with_compare=stats(new_wc),
                ratio_new_over_old=float(np.mean([r['new_mean'] for r in recs]) / np.mean([r['old_mean'] for r in recs])),
                source_records=sources)


def hand_maze(out_dir, mazes=(0, 1, 2, 3, 4, 5)):
    """Count the hand-set Maze-Unique energy's search and rebuild the per-maze totals; writes cram-hand-maze-unique.json."""
    import hand_set as HE
    from cram import maze_unique as M
    model = HE.maze_hand()
    assert (model.R, model.de, model.w) == (1, 2, 1)
    sc = json.loads((EXP / 'runs/HAND-E1/maze-test-s0.json').read_text())['search']
    assert sc == dict(search='gibbs', T=0.0, focus='random', max_moves=20000, eta=0.5, tol=1e-3, margin=3.0, max_steps=6000, seed=0)
    fit, rows = count_per_maze(model, gibbs_one, mazes, 'moves')
    k = 2 * model.w + 1
    old_fn = lambda m: M.flop_per_step(model) + m * (4 * (k * k - 1) + 2 * 900)  # noqa: E731  (gibbs_flop)
    rb = rebuild_maze([EXP / f'runs/HAND-E1/maze-test-s{s}.json' for s in range(3)], fit, old_fn)
    exp = json.loads((ROOT / 'results/hand-energy.json').read_text())
    rec = dict(row='maze-unique hand-energy (Hand-set energy, focused Gibbs T=0)',
               paper_value_old=next(r for r in exp['rows'] if r['benchmark'] == 'maze-unique')['flop_per_puzzle'],
               search='maze_unique.gibbs, T 0, focus random, max 20,000 moves (maze_unique.Search("gibbs")), '
                      'CRAMMazeEnergy R 1, de 2, window 1 set by hand (hand_set.maze_hand)',
               counted=dict(fit=fit, mazes=rows,
                            note='batch 1, CPU; total = setup + per x moves; setup includes the kernel W, the first full '
                                 'current and the final fixed-point check; per = one move (focus over 900 cells, the field '
                                 'change scattered to the 3 x 3 window, the board update)'),
               old_constants=dict(setup=float(M.flop_per_step(model)), per_move=float(4 * (k * k - 1) + 2 * 900),
                                  formula='maze_unique.gibbs_flop: 64,800 + 1,832 x moves'),
               formula='per maze = setup + per_move x moves (moves from the record\'s per_maze steps)', **rb)
    rec.update(meta([EXP / 'hand_set.py', EXP.parent / 'cram/maze_unique.py', EXP.parent / 'cram/cram_maze.py']))
    dump(out_dir, 'hand-maze-unique', rec)


# ---------------------------------------------------------------------------------------------------------------
# N-Queens (cram.cram_nqueens.solve, PyTorch; results/nqueens-results.json training_methods, sizes 8 mpl, 10 cd)
# ---------------------------------------------------------------------------------------------------------------
def nqueens(out_dir, rows_counted=8):
    """Count cram_nqueens.solve with the selected models of sizes 8 and 10 and rebuild the per-input totals; writes cram-nqueens-<n>.json."""
    from cram import cram_nqueens as Q
    res = json.loads((ROOT / 'results/nqueens-results.json').read_text())
    for n in (8, 10):
        tm = res['training_methods']['sizes'][str(n)]
        method = tm['selected_method']
        blk = tm['rows'][method]
        c = n * n
        te_in, _, _ = Q.load_split(n, 'test')
        fits, recs, old1, old20, new1, new20, nw1c, nw20c, sources = [], [], [], [], [], [], [], [], {}
        for s in range(3):
            d = EXP / f'runs/NQ/alt/n{n}/final/{method}-s{s}'
            fin = json.loads((d / 'final.json').read_text())
            sel = fin['selection']
            model = torch.load(d / 'model.pt', weights_only=False)
            sources[rel(d / 'final.json')] = sha(d / 'final.json')
            sources[rel(d / 'records.npz')] = sha(d / 'records.npz')
            rows = []
            for i in range(rows_counted):
                holder = {}
                cnt = _count(lambda: holder.setdefault('r', Q.solve(model, te_in[i:i + 1], chains=1, temperature=sel['temperature'],
                                                                     cap=sel['cap'], seed=i, device='cpu', stop=sel['stop'])))
                rows.append(dict(row=i, m=int(holder['r'][3][0, 0]), count=cnt))
            fit = linear(rows)
            assert all(fit[t]['max_abs_residual'] == 0 for t in fit), fit
            # two chains of one row: per-chain start and moves add up (no per-row term)
            holder = {}
            cnt2 = _count(lambda: holder.setdefault('r', Q.solve(model, te_in[:1], chains=2, temperature=sel['temperature'],
                                                                  cap=sel['cap'], seed=0, device='cpu', stop=sel['stop'])))
            m2 = holder['r'][3][0]
            assert cnt2['total'] == 2 * fit['total']['setup'] + fit['total']['per'] * m2.sum(), 'chains do not add'
            for r in rows[1:]:
                r['count'] = brief(r['count'])
            fits.append(dict(seed=s, stop=sel['stop'], temperature=sel['temperature'], cap=sel['cap'], fit=fit, rows=rows))
            with np.load(d / 'records.npz') as f:
                mv, fo = f['moves'].astype(np.float64), f['flop'].astype(np.float64)
            assert np.allclose(fo, Q.init_flop(c) + mv * Q.move_flop(c))
            per = lambda t: fit[t]['setup'] + fit[t]['per'] * mv  # noqa: E731
            pn, pw = per('total'), per('total_with_compare')
            old1.append(fo[:, 0]); old20.append(fo.sum(1)); new1.append(pn[:, 0]); new20.append(pn.sum(1))
            nw1c.append(pw[:, 0]); nw20c.append(pw.sum(1))
            assert abs(fo[:, 0].mean() - fin['test']['flop_n1']) < 1e-6 * fin['test']['flop_n1']
            assert abs(fo.sum(1).mean() - fin['test']['flop_all_chains']) < 1e-6 * fin['test']['flop_all_chains']
            recs.append(dict(seed=s, rows=len(mv), mean_moves=float(mv.mean()), old_n1=float(fo[:, 0].mean()), new_n1=float(pn[:, 0].mean()),
                             old_n20=float(fo.sum(1).mean()), new_n20=float(pn.sum(1).mean()),
                             new_with_compare_n1=float(pw[:, 0].mean()), new_with_compare_n20=float(pw.sum(1).mean())))
        for f in fits[1:]:
            for t in ('total', 'total_with_compare'):
                assert f['fit'][t]['per'] == fits[0]['fit'][t]['per'] and f['fit'][t]['setup'] == fits[0]['fit'][t]['setup']
        rec = dict(row=f'nqueens-{n} training_methods {method} (selected; paper caption flop per input for 20 samples)',
                   paper_value_old=dict(flop_n1=blk['flop_n1']['mean'], flop_n20=blk['flop_n20']['mean']),
                   search=f'cram_nqueens.solve: focused heat-bath Gibbs, stop "{fits[0]["stop"]}", T {fits[0]["temperature"]} x unit, '
                          f'cap {fits[0]["cap"]}, 20 chains per input (chain 0 = N=1)',
                   counted=dict(per_seed=fits, note='batch 1 row, 1 chain, CPU; per chain = setup + per x moves (exact on '
                                                    'every counted row and on a 2-chain run); setup = the first fields (a dense '
                                                    'einsum over every cell pair, 8 c^2), the energy and the first stopping test'),
                   old_constants=dict(init_flop=Q.init_flop(c), move_flop=Q.move_flop(c),
                                      formula='per chain init_flop(c) + moves x move_flop(c) = 2c^2 + 2c + 1 + moves x (5c + 14)'),
                   formula='per input: N=1 = setup + per x moves[chain 0]; N=20 = sum over the 20 chains of (setup + per x moves)',
                   per_seed=recs,
                   old=dict(n1=stats(old1), n20=stats(old20)), new=dict(n1=stats(new1), n20=stats(new20)),
                   new_with_compare=dict(n1=stats(nw1c), n20=stats(nw20c)),
                   ratio_new_over_old=dict(n1=float(np.mean([r['new_n1'] for r in recs]) / np.mean([r['old_n1'] for r in recs])),
                                           n20=float(np.mean([r['new_n20'] for r in recs]) / np.mean([r['old_n20'] for r in recs]))),
                   source_records=sources, notes=[
                       'The N=20 prediction of the majority metric (gram_metrics, tiebreak = -energy) negates 20 energies per '
                       'input: 20 flop, not added (below 1e-5 of the total).'])
        rec.update(meta([EXP.parent / 'cram/cram_nqueens.py', EXP / 'nqueens_free.py']))
        dump(out_dir, f'nqueens-{n}', rec)


# ---------------------------------------------------------------------------------------------------------------
# Graph coloring (cram.cram_graph_coloring.solve, NumPy float64), transcribed line by line into PyTorch
# ---------------------------------------------------------------------------------------------------------------
def gc_solve_torch(model, a, *, samples, temperature, cap, seed, stop='tau'):
    """graph_coloring.solve, statement by statement, in PyTorch float64 (np.einsum -> torch.einsum,
    np.take_along_axis -> gather, np.flatnonzero -> nonzero, fancy indexing as written). The random numbers are drawn
    from the same NumPy generator with the same shapes in the same order (random-number generation, 0 flop), so the
    chains make the same moves as solve. The scalar thresholds T, focus_at and thr are Python floats in the source
    (3 flop per graph, not seen by FullCounter, left out)."""
    from cram import cram_graph_coloring as G
    C = G.C
    n = int(round((1 + math.sqrt(1 + 8 * a.shape[1])) / 2))
    rng = np.random.default_rng(seed)
    W0 = torch.as_tensor(model['W0'].astype(np.float64))
    D = torch.as_tensor(model['D'].astype(np.float64))
    at = torch.as_tensor(a).to(torch.float64)
    K = W0 + torch.einsum('bt,ptcd->bpcd', at, D) if D.dim() == 4 else W0 + at[..., None, None] * D  # assemble_np
    I, J = (torch.as_tensor(v) for v in G.pairs(n))
    S = torch.zeros(K.shape[0], n, n, C, C, dtype=torch.float64)  # pair_tensor_np
    S[:, I, J] = K
    S[:, J, I] = K.transpose(-1, -2)
    B = torch.as_tensor(np.ascontiguousarray(G.bias_np(model, n)))
    nb = len(a)
    inst = torch.as_tensor(np.repeat(np.arange(nb), samples))
    R = len(inst)
    T = temperature * model['hbar']
    focus_at = model['ebar'] + model['theta'] * model['hbar']
    thr = model['tau'] + 1e-3
    y = torch.as_tensor(rng.integers(C, size=(R, n)))
    ar = torch.arange(n)
    g = B[None] + S[inst[:, None, None], ar[None, :, None], ar[None, None, :], :, y[:, None, :]].sum(2)
    E = .5 * (torch.gather(g, 2, y[..., None])[..., 0] + B[ar, y]).sum(1)
    best, bestE = y.clone(), E.clone()
    if stop == 'terms':
        thr_p = torch.as_tensor(model['tpair'])[torch.arange(len(I))[None], torch.as_tensor((a > 0).astype(np.int64))]
        thr_v = torch.as_tensor(model['tvert']) + 1e-6

        def passes(rows):
            si = inst[rows]
            pt = S[si[:, None], I[None], J[None], y[rows][:, I], y[rows][:, J]]
            return (pt <= thr_p[si] + 1e-6).all(1) & (B[ar[None], y[rows]] <= thr_v[None]).all(1)
    else:
        def passes(rows):
            return E[rows] <= thr
    stopped = passes(torch.arange(R))
    moves = torch.zeros(R, dtype=torch.int64)
    cc = torch.arange(C)
    for _ in range(cap):
        act = torch.nonzero(~stopped).squeeze(1)
        if len(act) == 0:
            break
        e = torch.gather(g[act], 2, y[act, :, None])[..., 0]
        hot = e > focus_at
        score = torch.as_tensor(rng.random(tuple(hot.shape)))
        score = torch.where(hot.any(1)[:, None], torch.where(hot, score, -1.), score)
        v = score.argmax(1)
        local = g[act, v]
        u = torch.as_tensor(rng.uniform(np.finfo(float).tiny, 1, size=tuple(local.shape)))
        new = (-local / T - torch.log(-torch.log(u))).argmax(1)
        old = y[act, v]
        k = torch.arange(len(act))
        E[act] += local[k, new] - local[k, old]
        si = inst[act]
        g[act] += (S[si[:, None, None], ar[None, :, None], v[:, None, None], cc[None, None, :], new[:, None, None]]
                   - S[si[:, None, None], ar[None, :, None], v[:, None, None], cc[None, None, :], old[:, None, None]])
        y[act, v] = new
        moves[act] += 1
        better = act[E[act] < bestE[act]]
        best[better], bestE[better] = y[better], E[better]
        stopped[act] = passes(act)
    out = torch.where(stopped[:, None], y, best)
    return dict(boards=out.reshape(nb, samples, n).numpy(), moves=moves.reshape(nb, samples).numpy(),
                stopped=stopped.reshape(nb, samples).numpy())


def graph_coloring(out_dir, graphs=6):
    """Count the PyTorch transcription of cram_graph_coloring.solve for every selected model; writes cram-gc<n>-<arm>.json."""
    sys.path.insert(0, str(EXP))
    sys.path.insert(0, str(EXP.parent))  # the cram package
    import graph_coloring as GR  # the run script: recipe folders
    from cram import cram_graph_coloring as G
    selected = json.loads((EXP / 'runs/CRAM-GC/selected.json').read_text())
    export = json.loads((ROOT / 'results/graph-coloring-results.json').read_text())
    for key, sel in selected.items():
        if key.startswith('_'):
            continue
        n, arm = sel['n'], sel['arm']
        stem = EXP / 'runs/CRAM-GC/final' / GR.tag_of(arm, n, sel['updates'], sel['lr'], sel['lam'])
        with np.load(stem / 'cpu' / 'model.npz') as f:
            model = {k: f[k] for k in f.files}
        model |= json.loads((stem / 'cpu' / 'model.json').read_text())
        x, yl = G.load_split(n, 'test')
        at, _ = G.convert(x, yl, n)
        kw = dict(temperature=sel['temperature'], cap=sel['cap'], stop=sel['stop'])
        # the transcription makes the same moves as solve (a few graphs, 20 chains, the test seed 0)
        ref = G.solve(model, at[:4], samples=20, seed=0, **kw)
        tr = gc_solve_torch(model, at[:4], samples=20, seed=0, **kw)
        assert np.array_equal(ref['moves'], tr['moves']) and np.array_equal(ref['boards'], tr['boards'])
        rows = []
        for i in range(graphs):
            holder = {}
            cnt = _count(lambda: holder.setdefault('r', gc_solve_torch(model, at[i:i + 1], samples=1, seed=i, **kw)))
            rows.append(dict(graph=i, m=int(holder['r']['moves'][0, 0]), count=cnt))
        # (graph setup + chain start) + per x moves, fitted on chains that stop before the cap: a chain that reaches
        # the cap leaves the loop by its range, without the last emptiness test (2 comparisons fewer, no arithmetic)
        one = linear([r for r in rows if r['m'] < sel['cap']])
        for t in ('total', 'total_with_compare'):
            one[t]['residuals_all_rows'] = [r['count'][t] - (one[t]['setup'] + one[t]['per'] * r['m']) for r in rows]
        assert all(x == 0 for x in one['total']['residuals_all_rows']), one
        holder = {}
        cnt2 = _count(lambda: holder.setdefault('r', gc_solve_torch(model, at[:1], samples=2, seed=0, **kw)))
        m2 = holder['r']['moves'][0]
        fit = {}
        for t in ('total', 'total_with_compare'):
            per = one[t]['per']
            start = cnt2[t] - per * m2.sum() - one[t]['setup']  # second chain's start
            fit[t] = dict(graph_setup=one[t]['setup'] - start, chain_start=start, per_move=per)
        for r in rows[1:]:
            r['count'] = brief(r['count'])
        recs, o1, o20, n1, n20, w1, w20, sources = [], [], [], [], [], [], [], {}
        for s in range(3):
            d = stem / f"test-{sel['stop']}-T{sel['temperature']:g}-S{sel['samples']}-cap{sel['cap']}-s{s}"
            sources[rel(d / 'outputs.npz')] = sha(d / 'outputs.npz')
            sources[rel(d / 'final.json')] = sha(d / 'final.json')
            fin = json.loads((d / 'final.json').read_text())
            with np.load(d / 'outputs.npz') as f:
                mv = f['moves'].astype(np.float64)
                cf = f['chain_flop'].astype(np.float64)
            asm = G.assembly_flop(model, n)
            assert np.allclose(cf, G.init_flop(n) + mv * G.move_flop(n))
            fo1, fo20 = asm + cf[:, 0], asm + cf.sum(1)
            assert abs(fo1.mean() - fin['flop_1']) < 1e-6 * fin['flop_1'] and abs(fo20.mean() - fin['flop_20']) < 1e-6 * fin['flop_20']
            pg = lambda t, k: fit[t]['graph_setup'] + k * fit[t]['chain_start'] + fit[t]['per_move'] * mv[:, :k].sum(1)  # noqa: E731
            o1.append(fo1); o20.append(fo20)
            n1.append(pg('total', 1)); n20.append(pg('total', 20)); w1.append(pg('total_with_compare', 1)); w20.append(pg('total_with_compare', 20))
            recs.append(dict(seed=s, graphs=len(mv), mean_moves=float(mv.mean()), old_flop_1=float(fo1.mean()), new_flop_1=float(n1[-1].mean()),
                             old_flop_20=float(fo20.mean()), new_flop_20=float(n20[-1].mean()),
                             new_with_compare_flop_1=float(w1[-1].mean()), new_with_compare_flop_20=float(w20[-1].mean())))
        er = next(r for r in export['rows'] if r['n'] == n and r['arm'] == arm)
        rec = dict(row=f'graph-coloring-{n} cram-{arm}' + (' (the paper\'s row: no supplied structure)' if arm == 'full' else ''),
                   paper_value_old=dict(flop_1=er['metrics']['flop_1']['mean'], flop_20=er['metrics']['flop_20']['mean']),
                   search=f'graph_coloring.solve (NumPy): focused heat-bath search, stop "{sel["stop"]}", T {sel["temperature"]} '
                          f'x hbar, cap {sel["cap"]}, 20 chains (chain 0 = the single output)',
                   counted=dict(fit=fit, rows_one_chain=rows, two_chain_count=brief(cnt2), two_chain_moves=m2.tolist(),
                                transcription='gc_solve_torch (same moves and boards as solve on 4 test graphs x 20 chains, seed 0)',
                                note='graph_setup = coupling assembly K(a) (dense, zeros included, multiply-add 2) and the '
                                     'thresholds; chain_start = the first random coloring\'s fields (a sum over every vertex '
                                     'pair), energy and stopping test; per_move = the heat-bath logits with the Gumbel '
                                     'transform, the energy and field updates of every vertex, the move counter and the '
                                     'stopping test (terms: one add per pair threshold, as written)'),
                   old_constants=dict(assembly_flop=G.assembly_flop(model, n), init_flop=G.init_flop(n), move_flop=G.move_flop(n)),
                   formula='per graph, k chains: graph_setup + k x chain_start + per_move x (sum of the k chains\' moves); '
                           'flop_1: k = 1 (chain 0), flop_20: k = 20',
                   per_seed=recs, old=dict(flop_1=stats(o1), flop_20=stats(o20)), new=dict(flop_1=stats(n1), flop_20=stats(n20)),
                   new_with_compare=dict(flop_1=stats(w1), flop_20=stats(w20)),
                   ratio_new_over_old=dict(flop_1=float(np.mean([r['new_flop_1'] for r in recs]) / np.mean([r['old_flop_1'] for r in recs])),
                                           flop_20=float(np.mean([r['new_flop_20'] for r in recs]) / np.mean([r['old_flop_20'] for r in recs]))),
                   source_records=sources)
        rec.update(meta([EXP.parent / 'cram/cram_graph_coloring.py', EXP / 'graph_coloring.py']))
        dump(out_dir, f'gc{n}-{arm}', rec)


# ---------------------------------------------------------------------------------------------------------------
# Maze-Hard with a latent (maze_hard_latent.py, row cram-latent-zmu): the inference of maze_hard_latent.evaluate with
# descent (eta 0.2): settle the latent, the attention, damped descent of the answers.
# ---------------------------------------------------------------------------------------------------------------
def _decode_maze_hard(fl, Fp, Fa, Ff):
    """(psi steps, descent steps) of each maze from evaluate's float flop: fl = ps Fp + Fa + used Ff (1 + 1/16), ps a
    multiple of the settle check (8), used a multiple of the certificate interval (16) or the 2,048 budget. Unique for
    every maze of both rows (asserted)."""
    out = []
    useds = np.array(sorted(set(range(16, 2049, 16)) | {2048}), np.float64)
    for f in fl:
        ps = (f - Fa - useds * Ff * (1 + 1 / 16)) / Fp
        ok = (ps > 0) & (np.abs(ps - np.round(ps)) < 1e-3) & (np.round(ps) % 8 == 0)
        assert ok.sum() == 1, (f, ps[ok], useds[ok])
        out.append((int(round(ps[ok][0])), int(useds[ok][0])))
    return np.asarray(out, np.float64)


def maze_hard(out_dir, mazes=8):
    """Count the Maze-Hard latent model's inference on `mazes` training mazes per seed and rebuild the per-maze test totals; writes cram-latent-zmu.json."""
    from cram import maze_h3 as m3
    tok, _ = m3.load('train')  # counts depend only on the step counts; the test file stays with its single scoring
    for row, script, fin in (('cram-latent-zmu', 'maze_hard_latent', EXP / 'runs/CRAM-HE/final-triton/CHE8-mu'),):  # the folder name is only a label
        R = __import__(script)
        ts = json.loads((fin / 'test-scores.json').read_text())
        sources = {rel(fin / 'test-scores.json'): sha(fin / 'test-scores.json')}
        comps, recs, old, new, new_wc = {}, [], [], [], []
        for seed in (0, 1, 2):
            rec = json.loads((fin / f'final-s{seed}.json').read_text())
            model = R.build(R.Recipe(**rec['recipe']))  # the latent on the open cells (cram_maze_hard_graph), attention couplings
            model.load_state_dict(torch.load(fin / f'final-s{seed}.pt', map_location='cpu'))
            model.eval()
            sources[rel(fin / f'final-s{seed}.pt')] = sha(fin / f'final-s{seed}.pt')
            zrows, drows, att = [], [], []
            for i in range(mazes):
                T = torch.as_tensor(tok[i:i + 1])
                h = {}
                with torch.no_grad():
                    cz = _count(lambda: h.setdefault('z', model.settle(T)))
                    psi, ps = h['z']
                    ca = _count(lambda: h.setdefault('a', torch.softmax(model.att_scores(T, psi), 1)))
                    cd = _count(lambda: h.setdefault('d', m3.descend(model, T, h['a'], steps=2048, eta=0.2)))
                zrows.append(dict(maze=i, m=int(ps[0]), count=cz))
                drows.append(dict(maze=i, m=int(h['d'][1][0]), count=cd))
                att.append(ca)
            fz, fd = linear(zrows), linear(drows)
            assert all(f[t]['max_abs_residual'] == 0 for f in (fz, fd) for t in ('total',)), (fz, fd)
            assert len({a['total'] for a in att}) == 1
            Fp, Fa, Ff = m3.flop_psi_step(model), m3.flop_attention(model), m3.flop_field(model)
            sd = ts['seeds'][str(seed)]
            fl = np.asarray((sd['per_maze'] if 'per_maze' in sd else sd['result']['per_maze'])['flop'], np.float64)
            st = _decode_maze_hard(fl, Fp, Fa, Ff)
            ps, used = st[:, 0], st[:, 1]
            per = lambda t: (fz[t]['setup'] + fz[t]['per'] * ps + att[0][t] + fd[t]['setup'] + fd[t]['per'] * used)  # noqa: E731
            nw, nwc = per('total'), per('total_with_compare')
            assert abs(fl.mean() - ts['seeds'][str(seed)]['result']['flop_mean']) < 1e-6 * fl.mean()
            old.append(fl); new.append(nw); new_wc.append(nwc)
            comps[str(seed)] = dict(latent=fz, latent_rows=[dict(maze=r['maze'], m=r['m'], total=r['count']['total']) for r in zrows],
                                    latent_count_maze0=zrows[0]['count'], attention=att[0],
                                    descent=fd, descent_rows=[dict(maze=r['maze'], m=r['m'], total=r['count']['total']) for r in drows],
                                    descent_count_maze0=drows[0]['count'], old_constants=dict(flop_psi_step=Fp, flop_attention=Fa, flop_field=Ff))
            recs.append(dict(seed=seed, mazes=len(fl), mean_psi_steps=float(ps.mean()), mean_descent_steps=float(used.mean()),
                             old_mean=float(fl.mean()), new_mean=float(nw.mean()), new_with_compare_mean=float(nwc.mean())))
        rec = dict(row=f'maze-hard {row}', paper_value_old=float(np.mean([r['old_mean'] for r in recs])),
                   search='maze_hard_latent.evaluate with descent (eta 0.2): settle the latent (Jacobi steps, a check every 8), '
                          'the attention softmax, maze_h3.descend (2,048-step budget, certificate every 16 steps)',
                   counted=dict(per_seed=comps, note='batch 1, CPU, the first training mazes (counts depend only on the step '
                                'counts); latent = setup + per x psi steps, descent = setup + per x descent steps (per = one '
                                'step plus 1/16 of a certificate check), exact on every counted maze'),
                   old_constants=dict(formula='ps x flop_psi_step + flop_attention + used x flop_field x (1 + 1/16)'),
                   formula='per maze = latent setup + latent per x ps + attention + descent setup + descent per x used; (ps, '
                           'used) decoded from each maze\'s old flop (unique for every maze)',
                   per_seed=recs, old=stats(old), new=stats(new), new_with_compare=stats(new_wc),
                   ratio_new_over_old=float(np.mean([r['new_mean'] for r in recs]) / np.mean([r['old_mean'] for r in recs])),
                   source_records=sources)
        rec.update(meta([EXP / f'{script}.py', EXP.parent / 'cram/maze_h3.py', EXP.parent / 'cram/cram_maze_hard_simple.py',
                         EXP / 'maze_hard_support.py', EXP.parent / 'cram/cram_maze_hard_graph.py']))
        dump(out_dir, row, rec)


def free_maze(out_dir, mazes=(0, 1, 2, 3, 4, 5), runs_dir='runs/CU13/final', row='cram-sgd-fg'):
    """The free-kernel Maze-Unique model (CRAMMazeFreeEnergy; runs/CU13/final/fg-s*, the folder name is only a label), row
    cram-sgd-fg: focused Gibbs search at T 0. Counted directly per maze on each seed's model, as hand_maze; writes
    cram-sgd-fg.json."""
    import maze_unique as U
    from cram import maze_unique as M
    runs = [f'{runs_dir}/fg-s{s}' for s in range(3)]
    for row, one, key, paths, src, old_fn, label in (
            (row, gibbs_one, 'moves', [EXP / r / 'score-test-gibbs.json' for r in runs], EXP / 'maze_unique.py',
             lambda model: (lambda m: M.flop_per_step(model) + m * (4 * 8 + 2 * 900)), 'CRAM PL, free kernel / focused Gibbs T 0'),):
        fits = {}
        for s in range(3):
            model = U.load_run(runs[s]).eval()
            assert type(model).__name__ == 'CRAMMazeFreeEnergy'
            fits[s] = count_per_maze(model, one, mazes, key)
        fit, rows = fits[0]
        for s in (1, 2):
            for t in ('total', 'total_with_compare'):
                assert fits[s][0][t]['per'] == fit[t]['per'] and fits[s][0][t]['setup'] == fit[t]['setup']
        model = U.load_run(runs[0])
        rb = rebuild_maze(paths, fit, old_fn(model))
        rec = dict(row=f'maze-unique {row} ({label})', models=f'{runs_dir}/fg-s{{0,1,2}}',
                   counted=dict(fit=fit, mazes=rows, fits_seeds_1_2_equal=True,
                                note=f'batch 1, CPU; total = setup + per x {key}; setup includes building the kernel from the 4 '
                                     'free tables'),
                   formula=f"per maze = setup + per x {key} (from the record's per_maze steps)", **rb)
        rec.update(meta([src, EXP.parent / 'cram/maze_unique.py', EXP.parent / 'cram/cram_maze.py']))
        dump(out_dir, row, rec)


GROUPS = {'hand-sudoku': hand_sudoku, 'hand-maze': hand_maze, 'nqueens': nqueens, 'gc': graph_coloring,
          'maze-hard-zmu': maze_hard, 'sgd-maze': free_maze}

if __name__ == '__main__':
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
    GROUPS[sys.argv[1]](os.path.join(CWD, sys.argv[2]))
