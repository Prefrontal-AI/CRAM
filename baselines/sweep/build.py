"""Build results/compute-sweep/sudoku-extreme.json from the raw outputs of the compute sweep (baselines/sweep/).

One row per (method, knob, setting, training seed): solved, n, solve rate, mean and 99th-percentile flop per puzzle, the knob
values, the checkpoint, the scoring script and its SHA-256, the source files. Rows marked table1 reproduce the setting
of the paper's main table and are read from the Table-2 score files. Each method keeps its own answer rule, exact-solve
criterion and flop convention (flop per network call or step x calls; a multiply-add is 2 flop):
- HRM, TRM: k ACT steps (adaptive computation time: one step is one forward call of the wrapper); flop = k x one eval
  ACT step (results/op-counts.json).
- EqR: depth D, breadth B; answer = lowest convergence score (top1_conv) among the first B trajectories of the B64 pass
  (path noise as the Table-2 B128 point), or the single-trajectory pass (B = 1, noise 0.01);
  random initial latents (std 1) in both; flop = D x B x one network evaluation.
- FRM: N + 1 network calls (the held_t sampler with N iterations); flop = calls x one forward.
- GRAM, knob "act_steps x samples": majority vote over N sampled trajectories of k ACT steps (ties by mean
  sigmoid(q_halt)); flop = k x N x one eval ACT step.
- TRMP, knob "rollouts": best of the first K rollouts by terminal confidence; flop per puzzle = K x (embedding +
  readout) + realized steps of those K rollouts x (step + halting head).
  Knob "step_cap": 64 rollouts, each stopped at step c at the latest; a rollout still running at c answers there
  with confidence q at c (as best_of with steps = c); flop = 64 x (embedding + readout) + sum of min(steps, c) x (step +
  halting head).
- EBR, knob "depth x samples" or "tau x samples": the lowest energy among the first N sampled trajectories, each run D
  segments, or stopped at the first segment end with tangential gradient norm |grad_M E| < tau (at most 64 segments);
  flop = segments x (one segment, + one gradient measure under the tau rule) per trajectory, + N
  energy evaluations when N > 1 (baselines/runs/ebr/flops/sudoku.json, baselines/ebr/report.py).
- TRMP-fast (the TRMP lattice model, baselines/trmp/lattice.py), knobs "budget" (steps per puzzle at 64 chains) and
  "chains" (at budget 1,000): each setting is its own run of the chain search (sweep/trmp_lattice_sweep.py); answer = the
  first chain on a solved board, a puzzle unsolved within the budget is wrong; flop per puzzle = a_K + b_K x steps run on
  the puzzle, a_K and b_K counted by FullCounter over the whole search with K chains (count-k<K>.json).
  Added only when every setting of the three seeds exists.
Operation counts: every per-unit cost above is the arithmetic-only count of results/op-counts.json
(baselines/flops/fc2_export.py; convention in baselines/flops/fullcount.py). The older matmul-only counts of
FlopCounterMode (results/flop-counts.json and files next to the checkpoints) are only compared with the `*_old` fields of op-counts.json to check
that both files describe the same model. TRMP adds the counted noise arithmetic per continuing step and the input
check; EqR the top1_conv selection over B; EBR scales its precomputed per-setting flop by the segment's count ratio
(measure and energy ratios agree to 3e-4).
Puzzles: Sudoku-Extreme full-all rows 10,000-19,999 (official rows from data/full-test/index.json).
Dimensions: p = puzzles, k = steps or rollouts, d = depths, b = trajectories.

This script calls no model code; the passes it reads come from sweep/submit.sh and sweep/trmp_lattice.sbatch.
  .venv/bin/python baselines/sweep/build.py      (CPU; from the repository root)
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[2]
T = CODE / 'baselines'
R = T / 'runs'  # outputs of the baseline fits and scorers
S = R / 'compute-sweep'
OUT = CODE / 'results/compute-sweep'
DATA = CODE / 'data'
SEEDS = (0, 1, 2)
ACT_POINTS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
POW2 = (1, 2, 4, 8, 16, 32, 64)
OPC = CODE / 'results/op-counts.json'
UNITS = json.loads(OPC.read_text())['components']['baselines']  # per-unit operation counts (convention of baselines/flops/fullcount.py)


def sha(path):
    """SHA-256 of a file, as hex."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    """Path relative to the repository root, as a string."""
    return str(Path(path).resolve().relative_to(CODE))


def jload(path):
    """Read a JSON file."""
    return json.loads(Path(path).read_text())


def row(method, setting, seed, solved, flop, knobs, checkpoint, scripts, sources, table1=False, note=None, knob=None):
    """One output row. solved: (p,) bool per puzzle; flop: scalar or (p,) flop per puzzle; knobs: the setting's values."""
    solved = np.asarray(solved, bool)
    flop = np.broadcast_to(np.asarray(flop, float), solved.shape)
    r = dict(method=method, knob=knob, setting=setting, seed=seed, solved=int(solved.sum()), n=int(solved.size), solve_rate=float(solved.mean()),
             mean_flop=float(flop.mean()), p99_flop=float(np.percentile(flop, 99)), knobs=knobs, table1=table1,
             checkpoint=rel(checkpoint) if checkpoint else None, checkpoint_sha256=sha(checkpoint) if checkpoint and Path(checkpoint).is_file() else None,
             scoring_scripts={rel(p): sha(p) for p in scripts}, sources=[rel(p) for p in sources])
    if note:
        r['note'] = note
    return r


def sudoku_rows():
    """Official rows of full-all[10000:20000] and the check that EqR's first10k holds the same puzzles."""
    idx = jload(DATA / 'full-test/index.json')
    rows = np.concatenate([np.array(idx['shards']['full-frmtest']['official_rows']), np.array(idx['shards']['full-00']['official_rows'])[:9000]])
    assert idx['full_all_order']['full-frmtest'] == [10000, 1000] and idx['full_all_order']['full-00'][0] == 11000
    off = np.load(DATA / 'frm/official/test.npz')
    full = np.load(DATA / 'full-test/full-all.npz')
    assert np.array_equal(full['solutions'][10000:20000].reshape(10000, -1), off['solutions'][rows].reshape(10000, -1))
    eq = np.load(CODE / 'experiments/runs/eqr/data/sudoku-eval/first10k/all__labels.npy')
    assert np.array_equal(eq.reshape(10000, -1).astype(np.int64) - 1, off['solutions'][rows].reshape(10000, -1).astype(np.int64))
    return rows


def act_rows(method, task, rows):
    """HRM / TRM from the 32-step pass; the Table-2 16-step point from score.right.npy."""
    flops = jload(CODE / 'results/flop-counts.json')['cells']
    if method == 'HRM':
        assert next(c for c in flops if c['method'] == 'HRM' and c['benchmark'] == 'Sudoku-Extreme')['components']['flop_per_act_step'] == UNITS['HRM'][task]['per_step_old']
        per_step = UNITS['HRM'][task]['per_step']
        t1 = 'hrm/sudoku-e40000-s{k}'
        scripts = [T / 'sweep/act_steps.py', T / 'hrm/score.py']
        fsrc = OPC
    else:
        fsrc = CODE / 'experiments/runs/trm/T1/sudoku-flops.json'  # T1 is only the name of the run folder
        assert jload(fsrc)['flop_per_act_step'] == UNITS['TRM'][task]['per_step_old']
        per_step, fsrc = UNITS['TRM'][task]['per_step'], OPC
        t1 = 'trm/sudoku-s{k}'
        scripts = [T / 'sweep/act_steps.py', CODE / 'baselines/trm/trm_bench.py']
    out = []
    for k in SEEDS:
        rec = S / f'{method.lower()}/{task}-s{k}/act32.json'
        right = np.load(rec.with_suffix('.npz'))['right']  # (p, k)
        meta = jload(rec)
        assert meta['check']['mismatches'] == 0, (rec, meta['check'])
        right = right[rows]
        ckpt = meta['checkpoint']
        for s in ACT_POINTS:
            if s == 16:
                continue  # the Table-2 row below (the sweep's step 16 equals it puzzle by puzzle: check above)
            out.append(row(method, f'act{s}', k, right[:, s - 1], s * per_step, dict(act_steps=s), ckpt, scripts, [rec, fsrc], knob='act_steps'))
        ref = R / t1.format(k=k) / 'score.right.npy'
        r1 = np.load(ref)
        r1 = r1[rows]
        out.append(row(method, 'act16', k, r1, 16 * per_step, dict(act_steps=16), ckpt, scripts[1:], [ref, fsrc], table1=True, knob='act_steps'))
    return out


def eqr_rows(task):
    """EqR: the single-trajectory pass, the first B of 64 trajectories at each depth, and the Table-2 D64 B128 point."""
    bench = jload(CODE / 'experiments/runs/eqr/E1/bench.json')  # E1 is only the name of the run folder
    assert bench['sudoku (released checkpoint)']['flop_per_nfe'] == UNITS['EqR'][task]['per_nfe_old']
    per, sel = UNITS['EqR'][task]['per_nfe'], UNITS['EqR'][task]['select_over_B'] / 128  # top1_conv selection: per trajectory considered
    scripts = [T / 'sweep/eqr_sweep.py', CODE / 'baselines/eqr/eqr_score.py']
    fsrc = OPC
    out = []
    for k in SEEDS:
        d = S / f'eqr/{task}-s{k}'
        run = R / ('eqr/sudoku-s%d' % k)
        ckpt = run / 'model.pt'
        # one trajectory per puzzle, noise 0.01 (the single-trajectory pass d64-b1)
        b1 = np.load(d / 'd64-b1.npz')
        m1 = jload(d / 'd64-b1.json')
        n = 10000
        ex1 = b1['exact'][:n, :, 0]
        for j, D in enumerate(b1['depths'].tolist()):
            out.append(row('EqR', f"D{D}-B1-n{m1['noise']}-single", k, ex1[:, j], D * per, dict(depth=D, breadth=1, noise=m1['noise'], random_init=True, answer='first', trajectories='single-trajectory pass'),
                           ckpt, scripts, [d / 'd64-b1.json', fsrc], knob='depth (B = 1, single-trajectory pass)'))
        # random initial latents, B trajectories (first B of the B64 pass), top1_conv
        parts = sorted(d.glob('d64-b64-g*.npz'))
        exact = np.concatenate([np.load(p)['exact'] for p in parts])  # (p, d, b)
        score = np.concatenate([np.load(p)['score'] for p in parts])
        puzzles = np.concatenate([np.load(p)['puzzles'] for p in parts])
        assert np.array_equal(puzzles, np.arange(n)), 'B64 shards do not cover the puzzles in order'
        m64 = jload(parts[0].with_suffix('.json'))
        depths = np.load(parts[0])['depths'].tolist()
        for j, D in enumerate(depths):
            for B in POW2:
                if B > exact.shape[2]:
                    continue
                win = score[:, j, :B].argmin(1)
                out.append(row('EqR', f"D{D}-B{B}-n{m64['noise']}", k, exact[np.arange(n), j, win], D * B * per + (B * sel if B > 1 else 0),
                               dict(depth=D, breadth=B, noise=m64['noise'], random_init=True, answer='top1_conv', trajectories='first B of 64'),
                               ckpt, scripts, [p.with_suffix('.json') for p in parts] + [fsrc], knob='depth x breadth'))
        # Table 2: D64 B128 from the score file of the fit
        ref = run / 'score-first10k-D64-B128.npz'
        rj = jload(ref.with_suffix('.json'))
        assert rj['n'] == n
        out.append(row('EqR', f"D64-B128-n{rj['noise']}", k, np.load(ref)['top1_conv'], 64 * 128 * per + 128 * sel,
                       dict(depth=64, breadth=128, noise=rj['noise'], random_init=True, answer='top1_conv'), ckpt, scripts[1:], [ref.with_suffix('.json'), fsrc], table1=True, knob='depth x breadth'))
    return out


def trmp_rows(task):
    """TRMP: best of the first K rollouts (knob "rollouts"), and 64 rollouts with a per-rollout step cap (knob "step_cap")."""
    cell = next(c for c in jload(CODE / 'results/flop-counts.json')['cells'] if c['method'] == 'TRMP' and c['benchmark'] == 'Sudoku-Extreme')
    c = cell['components']
    assert c['per_step'] > 0
    u = UNITS['TRMP'][task]
    per_step, per_rollout, noise, check = u['per_step'], u['per_rollout'], u['noise_per_continuing_step'], u['input_check']
    assert u['cap_correction'] == 0 and u['select_per_rollout_considered'] == 0
    scripts = [T / 'sweep/trmp_sweep.py', T / 'trmp/score.py', T / 'flops/trmp_count.py']
    n = 10000
    out = []
    for k in SEEDS:
        d = S / f'trmp/{task}-s{k}'
        parts = sorted(d.glob('rollouts-g*.npz'))
        metas = [jload(p.with_suffix('.json')) for p in parts]
        for m in metas:
            assert m['check']['answer_mismatch_puzzles'] == 0 and m['check']['steps_mismatch_puzzles'] == 0, m['check']
        z = [np.load(p) for p in parts]
        ti = np.concatenate([f['test_index'] for f in z])
        assert np.array_equal(ti[:n], np.arange(n))
        exact = np.concatenate([f['exact'] for f in z])[:n]  # (p, k)
        conf = np.concatenate([f['confidence'] for f in z])[:n]
        steps = np.concatenate([f['steps'] for f in z])[:n].astype(np.int64)
        assert np.array_equal(np.concatenate([f['dataset_index'] for f in z])[:n], np.arange(10000, 20000))
        run = R / f'trmp/{task}-s{k}'
        for K in POW2:
            win = conf[:, :K].argmax(1)  # ties to the lowest index, as infer.best_of
            flop = check + K * per_rollout + steps[:, :K].sum(1) * per_step + (steps[:, :K].sum(1) - K) * noise
            out.append(row('TRMP', f'K{K}', k, exact[np.arange(n), win], flop,
                           dict(rollouts=K, step_cap=metas[0]['step_cap'], sigma=metas[0]['sigma'], rollout_subset='first K of 64',
                                mean_rollout_steps=float(steps[:, :K].mean())),
                           run / 'model.pt', scripts, [p.with_suffix('.json') for p in parts] + [OPC], table1=(K == 64),
                           note='K = 64 is the Table-2 setting; its answers and steps equal the Table-2 shards score/test/*.npz (checked)' if K == 64 else None,
                           knob='rollouts'))
        # per-rollout step cap c at 64 rollouts (sweep/trmp_caps.py)
        cparts = sorted(d.glob('caps-g*.npz'))
        cmetas = [jload(p.with_suffix('.json')) for p in cparts]
        for m in cmetas:
            assert m['check']['answer_mismatch_puzzles'] == 0 and m['check']['steps_mismatch_puzzles'] == 0, m['check']
        cz = [np.load(p) for p in cparts]
        assert np.array_equal(np.concatenate([f['test_index'] for f in cz])[:n], np.arange(n))
        cex = np.concatenate([f['exact_at'] for f in cz])[:n]  # (p, c, r)
        ccf = np.concatenate([f['confidence_at'] for f in cz])[:n]
        csteps = np.concatenate([f['steps'] for f in cz])[:n].astype(np.int64)
        assert np.array_equal(csteps, steps), 'steps of the cap pass differ from the rollout pass'
        caps = cz[0]['caps'].tolist()
        for j, cap in enumerate(caps):
            win = ccf[:, j].argmax(1)  # ties to the lowest index, as infer.best_of
            used = np.minimum(csteps, cap)
            flop = check + 64 * per_rollout + used.sum(1) * per_step + (used.sum(1) - 64) * noise
            out.append(row('TRMP', f'cap{cap}', k, cex[np.arange(n), j, win], flop,
                           dict(rollouts=64, step_cap=cap, sigma=cmetas[0]['sigma'], mean_rollout_steps=float(used.mean())),
                           run / 'model.pt', [T / 'sweep/trmp_caps.py', T / 'trmp/score.py', T / 'flops/trmp_count.py'],
                           [p.with_suffix('.json') for p in cparts] + [OPC], table1=(cap == caps[-1]),
                           note='a rollout still running at the cap answers there with confidence q at the cap, as best_of with steps = cap' if cap < caps[-1] else
                           'the Table-2 cap; its answers and steps equal the Table-2 shards score/test/*.npz (checked)', knob='step_cap'))
    return out


def frm_rows(task, rows):
    """FRM: the held_t sampler at each iteration count N of the sweep, and the Table-2 point (N = 4,096) from score-unseen."""
    out = []
    for k in SEEDS:
        base = R / ('frm-compiled' if k == 0 else f'frm-compiled-s{k}')
        d = S / f'frm/{task}-s{k}'
        root = base / 'sudoku'
        sel = jload(root / f'runs/s{k}/selection.json')['selected']
        assert sel['flops_per_forward'] == UNITS['FRM'][task]['per_call_old'], (sel['flops_per_forward'], UNITS['FRM'][task]['per_call_old'])
        per = UNITS['FRM'][task]['per_call']
        scripts = [T / 'sweep/frm_sweep.py', T / 'frm/score_unseen.py']
        for p in sorted(d.glob('held*.json')):
            if 'check' in p.name:
                continue
            m = jload(p)
            z = np.load(p.with_suffix('.npz'))
            assert np.array_equal(z['official_rows'], rows)
            out.append(row('FRM', f"iters{m['held_iters']}", k, z['solved'], m['nfe'] * per, dict(held_iters=m['held_iters'], nfe=m['nfe'], sampler='held_t'),
                           sel['checkpoint'], scripts, [p, root / f'runs/s{k}/selection.json'], knob='iterations'))
        ref = np.load(root / 'score-unseen.solved.npz')
        pos = np.searchsorted(ref['official_rows'], rows)
        assert np.array_equal(ref['official_rows'][pos], rows)
        out.append(row('FRM', 'iters4096', k, ref['solved'][pos], 4097 * per, dict(held_iters=4096, nfe=4097, sampler='held_t'),
                       sel['checkpoint'], scripts[1:], [root / 'score-unseen.json'], table1=True, knob='iterations'))
    return out


def gram_rows(rows):
    """GRAM: majority vote over N sampled trajectories of k ACT steps, from the scoring of baselines/gram/score.py.

    The per-puzzle flags runs/gram/score/sudoku-s<k>/score.right.npz (official test order) are checked against a vote
    recomputed here from the raw shards (per sample and step: exact flag, answer hash, sigmoid(q_halt)): the most frequent
    answer among the first N samples, ties by the larger mean q_halt, then the lowest sample index (np.argmax).
    """
    assert jload(R / 'gram/flops/sudoku.json')['flop_per_act_step'] == UNITS['GRAM']['sudoku-extreme']['per_step_old']
    per_step, vote20 = UNITS['GRAM']['sudoku-extreme']['per_step'], UNITS['GRAM']['sudoku-extreme']['vote']
    out = []
    for k in SEEDS:
        d = R / f'gram/score/sudoku-s{k}'
        merged = jload(d / 'score.json')
        flags = np.load(d / 'score.right.npz')
        shards = sorted(d.glob('shard-*.npz'), key=lambda p: int(p.stem.split('-')[1]))
        metas = [jload(p.with_suffix('.json')) for p in shards]
        assert {m['checkpoint_sha256'] for m in metas} == {metas[0]['checkpoint_sha256']}
        z = [np.load(p) for p in shards]
        assert np.array_equal(np.concatenate([f['samples'] for f in z]), np.arange(merged['samples']))
        steps = z[0]['steps'].tolist()
        right = np.concatenate([f['right'][rows] for f in z], 1)  # (p, samples, steps)
        hashes = np.concatenate([f['hash'][rows] for f in z], 1)
        q = np.concatenate([f['q'][rows] for f in z], 1).astype(np.float64)
        ckpt = metas[0]['checkpoint']
        assert sha(ckpt) == metas[0]['checkpoint_sha256']
        scripts = [Path(merged['script']), Path(metas[0]['script'])]
        for a, st in enumerate(steps):
            for n in (1, 2, 5, 10, 20):
                if n == 1:
                    ok = right[:, 0, a]
                else:
                    h, qq = hashes[:, :n, a], q[:, :n, a]
                    win = np.empty(len(h), np.int64)
                    for i in range(len(h)):  # explicit vote, independent of score.py's vectorized one
                        best = None
                        for j in range(n):
                            same = h[i] == h[i, j]
                            key = (int(same.sum()), float(qq[i][same].mean()))
                            if best is None or key > best[0]:
                                best = (key, j)
                        win[i] = best[1]
                    ok = right[np.arange(len(h)), win, a]
                ref = flags[f'steps{st}_n{n}'][rows]
                assert np.array_equal(ok, ref), (k, st, n, int((ok != ref).sum()))
                out.append(row('GRAM', f'steps{st}-n{n}', k, ok, st * n * per_step + (vote20 * n / 20 if n > 1 else 0),
                               dict(act_steps=st, samples=n, vote='majority over decoded boards, ties by mean sigmoid(q_halt)'),
                               ckpt, scripts, [d / 'score.json', d / 'score.right.npz', R / 'gram/flops/sudoku.json'],
                               table1=(st == 16 and n == 20), knob='act_steps x samples',
                               note='published setting (16 steps, majority@20); the paper table shows the published 96.1% with a dagger' if (st == 16 and n == 20) else None))
    return out


def ebr_rows(task, rows):
    """EBR: fixed depth and the gradient-norm stop, from the per-puzzle flags of ebr/report.py over the official test order."""
    name = 'sudoku'
    flops = R / f'ebr/flops/{name}.json'
    e = UNITS['EBR'][name]
    assert jload(flops)['segment_flop'] == e['segment']['old']
    seg_ratio = e['segment']['new'] / e['segment']['old']
    scripts = [T / 'ebr/report.py']
    out = []
    for k in SEEDS:
        run = R / f'ebr/{name}-s{k}'
        rep = jload(run / 'report.json')
        z = np.load(run / 'report.flags.npz')
        sets = {key: (z[key][rows], z[f'flop_{key}'][rows].astype(float) * seg_ratio, rep['all'][key]['knobs']) for key in rep['all']}
        ckpt, sources, table_key = rep['checkpoint'], [run / 'report.json', run / 'report.flags.npz', flops], 'D64-N5'
        for key, (ok, flop, knobs) in sets.items():
            out.append(row('EBR', key, k, ok, flop, knobs, ckpt, scripts, sources, table1=(key == table_key),
                           knob='tau x samples' if key.startswith('gtan') else 'depth x samples',
                           note='published setting (lowest energy of 5 samples)' if key == table_key else None))
    return out


def trmp_lattice_rows():
    """TRMP-fast (the TRMP lattice model): one row per (chains, budget) setting and seed, or none until all 51 settings exist."""
    import sys
    sys.path.insert(0, str(T / 'sweep'))
    from trmp_lattice_sweep import GRID, N
    runs = {0: R / 'trmp-lattice/warm-s0/cold', 1: R / 'trmp-lattice/sudoku-extreme-s1', 2: R / 'trmp-lattice/sudoku-extreme-s2'}
    outs = {k: S / f'trmp-lattice/sudoku-extreme-s{k}' for k in SEEDS}
    if not all((outs[k] / f'k{K}-b{B}.json').exists() for k in SEEDS for K, B in GRID):
        print('TRMP-fast: sweep incomplete, no rows')
        return []
    d = np.load(DATA / 'full-test/full-all.npz')['solutions'][10000:10000 + N].reshape(N, 81).astype(np.int64) - 1
    scripts = [T / 'sweep/trmp_lattice_sweep.py', T / 'trmp/lattice.py']
    out = []
    for k in SEEDS:
        tpl1 = jload(runs[k] / 'score/flops.json')  # the count of the Table-2 search (trmp/lattice_score.py)
        for K, B in GRID:
            j = jload(outs[k] / f'k{K}-b{B}.json')
            cnt = jload(outs[k] / f'count-k{K}.json')
            z = np.load(outs[k] / f'k{K}-b{B}.npz')
            assert j['model_sha256'] == cnt['model_sha256'] == sha(runs[k] / 'model.pt') and j['training_seed'] == k and j['count'] == N
            assert cnt['linear_exact'] and cnt['convention'] == tpl1['convention']
            if K == 64:  # check (2): the count of the Table-2 setting equals the count stored with the Table-2 score
                assert (cnt['per_puzzle_constant'], cnt['per_step']) == (tpl1['per_puzzle_constant'], tpl1['per_step']), (cnt, tpl1)
            if (K, B) == (64, 1000):  # check (1): the first 10,000 puzzles of the Table-2 shard 0
                c = j.get('check_table1_shard0') or {}
                if not c:
                    ref = np.load(runs[k] / 'score/test/0000000.npz')
                    c = dict(prediction_mismatch_puzzles=int((ref['prediction'][:N] != z['prediction']).any(-1).sum()),
                             steps_mismatch_puzzles=int((ref['steps'][:N] != z['steps']).sum()))
                # The rerun is not bitwise equal (68% of step counts differ; the compiled forward was re-traced after other
                # chain counts ran in the process, which changes bf16 rounding and so the stochastic search path), so
                # check (1) compares solved counts on the same puzzles and records the mismatches.
                ref = np.load(runs[k] / 'score/test/0000000.npz')
                ref_solved = int((ref['prediction'][:N].astype(np.int64) == d).all(-1).sum())
                assert abs(ref_solved - j['solved']) <= 10, (ref_solved, j['solved'], c)
                check1 = dict(c, table1_solved=ref_solved, sweep_solved=j['solved'])
            steps = z['steps'].astype(np.int64)
            solved = (z['prediction'].astype(np.int64) == d).all(-1)
            assert int(solved.sum()) == j['solved']
            flop = cnt['per_puzzle_constant'] + cnt['per_step'] * steps
            out.append(row('TRMP-fast', f'K{K}-B{B}', k, solved, flop, dict(chains=K, budget=B, check_every=1),
                           runs[k] / 'model.pt', scripts, [outs[k] / f'k{K}-b{B}.json', outs[k] / f'count-k{K}.json'], table1=(K, B) == (64, 1000),
                           note=(f"the Table-2 setting, rerun: {check1['sweep_solved']} solved against {check1['table1_solved']} in the first 10,000 puzzles of the Table-2 shard "
                                 f"score/test/0000000.npz; not bitwise equal ({check1['prediction_mismatch_puzzles']} answers and {check1['steps_mismatch_puzzles']} step counts differ)") if (K, B) == (64, 1000) else None,
                           knob='chains' if B == 1000 and K < 64 else 'budget'))
            out[-1].update(mean_steps=float(steps.mean()), timeouts=j['timeouts'], wrong=j['wrong'])
    return out


def summarize(rows):
    """Mean over training seeds of each (method, knob, setting): solve rate (mean, sample sd) and mean and p99 flop."""
    groups = {}
    for r in rows:
        groups.setdefault((r['method'], r['knob'], r['setting'], r['table1']), []).append(r)
    out = []
    for (m, kn, s, t1), g in groups.items():
        rate = np.array([r['solve_rate'] for r in g])
        out.append(dict(method=m, knob=kn, setting=s, table1=t1, knobs={k: v for k, v in g[0]['knobs'].items() if k not in ('mean_rollout_steps', 'sigma')}, seeds=[r['seed'] for r in g],
                        solve_rate_mean=float(rate.mean()), solve_rate_sd=float(rate.std(ddof=1)) if len(g) > 1 else None,
                        mean_flop=float(np.mean([r['mean_flop'] for r in g])), p99_flop=float(np.mean([r['p99_flop'] for r in g]))))
    return sorted(out, key=lambda r: (r['method'], r['knob'], r['mean_flop']))


def main():
    rows = sudoku_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    task, bench = 'sudoku-extreme', 'Sudoku-Extreme'
    puzzles = 'full-all rows 10,000-19,999 (FRM test 1,000 + first 9,000 of full-00; EqR first10k), official rows from data/full-test/index.json'
    rs = act_rows('HRM', task, rows) + act_rows('TRM', task, rows) + eqr_rows(task) + frm_rows(task, rows) + trmp_rows(task)
    rs += gram_rows(rows)
    rs += trmp_lattice_rows()
    if all((R / f"ebr/sudoku-s{k}/report.json").exists() for k in SEEDS):
        rs += ebr_rows(task, rows)
    doc = dict(description=f'{bench}: solve rate against inference flop per puzzle for the looped baselines of the main table, rescored from the '
                           'Table-2 checkpoints at fixed grids of their inference settings (the compute sweep, baselines/sweep/). No retraining; '
                           'no setting chosen on test.',
               created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), puzzles=puzzles, build_script=rel(__file__), build_script_sha256=sha(__file__),
               conventions=__doc__.split('\n\n')[1], rows=rs, summary=summarize(rs))
    (OUT / f'{task}.json').write_text(json.dumps(doc, indent=1) + '\n')
    print(task, len(rs), 'rows')


if __name__ == '__main__':
    main()
