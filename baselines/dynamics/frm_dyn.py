"""Dynamics of FRM's clean estimate run past the Table-2 number of network calls.

FRM (flow-map reasoning) has no trained recurrence depth: training never unrolls the network. Its Table-2 inference is
a sampler, so the reference depth here is the Table-2 number of network calls (NFE), and the state is the model's clean
estimate x0_hat (the softmax of the capped logits, clue cells pinned), which every sampler step feeds forward.
The loops below repeat flow_map_reasoning.methods.flm.conditional_generate_samples line by line (same forward,
capping, self-conditioning, clue re-pinning and RNG draws), with instrumentation added:

- sudoku: the held-tau fixed-point iteration (t_hold 0.4, self-conditioning). Variant `table` is the Table-2 sampler
  (renoise 1.0: z = t * x0_hat + (1 - t) * fresh noise every iteration); variant `det` is the deterministic held-tau map
  z = x0_hat (no noise). At logged iteration t it also runs the sampler's readout forward (no RNG), whose argmax is the
  answer of a t-iteration run; the Table-2 run is t = 4,096 (4,097 calls). Seeding and batch as baselines/sweep/frm_sweep.py.
- maze: the Table-2 sde_selfcond sampler (descending time, NFE 257 = 256 Euler/restart-SDE steps + readout, sampler
  seed 0, batch 250, as frm_maze.evaluate), then the readout map at tau = 1 iterated: z = x0_hat, x0_hat = f(z, 1).
  Network call c <= 256 is an integration step; call 257 is the Table-2 readout; calls > 257 continue the readout map.

Per puzzle and logged t: relative change of x0_hat on free cells since t - 1, whether the argmax answer changed since
t - 1, exactness of the argmax of x0_hat, and (sudoku) exactness of the readout answer.

Upstream code: the FRM clone in <root>/upstream (package flow_map_reasoning) and the adapters frm_reproduction.py and
frm_maze.py in the campaign root <root> (the folder of one FRM fit, baselines/frm/setup.sh).
Run with FRM's Python, cwd = the campaign root, PYTHONPATH = <root>/upstream (dynamics/run.sbatch does this):
  python frm_dyn.py sudoku <root> --seed K --variant table --iters 32768 --out <prefix>
  python frm_dyn.py maze <root> --seed K --calls 2056 --out <prefix>
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V1_DATA = Path(os.environ.get('V1_DATA', str(Path(__file__).resolve().parents[2] / 'data')))
FULL = V1_DATA / 'full-test'


def digest(path):
    """SHA-256 of a file, as hex."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid(n, extra=()):
    """Sorted call counts to log for a run of n calls: 72 log-spaced values, the powers of two, `extra` and n."""
    g = set(np.unique(np.round(np.geomspace(1, n, 72)).astype(int)).tolist())
    g |= {2 ** i for i in range(31) if 2 ** i <= n} | {d for d in extra if d <= n} | {n}
    return sorted(g)


class Recorder:
    """Per-puzzle statistics at logged t, kept on the GPU until the end of a batch."""

    def __init__(self, torch, logged, sol, free):
        self.torch, self.logged, self.sol, self.free = torch, set(logged), sol, free
        self.rows = {k: [] for k in ('t', 'rel', 'changed', 'exact')}
        self.prev = None

    def step(self, t, probs, pinned):
        """Record call t. probs: (b, cells, v) clean estimate; pinned: probs with clue cells set to the clue one-hot."""
        torch = self.torch
        ids = pinned.argmax(-1)
        if t in self.logged:
            if self.prev is None:
                rel = torch.full((len(ids),), float('nan'), device=ids.device)
                changed = torch.zeros(len(ids), dtype=torch.bool, device=ids.device)
            else:
                f = self.free.unsqueeze(-1)
                d = ((probs - self.prev[0]) * f).square().sum((1, 2)).sqrt()
                rel = d / (self.prev[0] * f).square().sum((1, 2)).sqrt().clamp_min(1e-12)
                changed = ((ids != self.prev[1]) & self.free.bool()).any(-1)
            self.rows['t'].append(t)
            self.rows['rel'].append(rel)
            self.rows['changed'].append(changed)
            self.rows['exact'].append((ids == self.sol).all(-1))
        self.prev = (probs, ids)

    def result(self):
        """The logged calls t and the arrays rel, changed, exact, each (b, len(t)), on the CPU."""
        out = {'t': np.array(self.rows['t'])}
        for k in ('rel', 'changed', 'exact'):
            out[k] = self.torch.stack(self.rows[k], 1).cpu().numpy()
        return out


def sudoku(a):
    """Sudoku-Extreme: iterate the held-tau map for a.iters calls on FRM's 1,000 test puzzles and write the records."""
    import torch
    root = a.root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(HERE.parent / 'frm'))
    from frm_reproduction import arrays, load_model
    from score_unseen import selection

    out_json = a.out.with_suffix('.json')
    if out_json.exists():
        print(f'{out_json} exists', flush=True)
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    sel = selection(root, a.seed)
    ckpt = Path(sel['checkpoint'])
    sha = digest(ckpt)
    assert sha == sel['checkpoint_sha256']
    model, meta = load_model(root, ckpt)
    model.eval()
    lut = meta['lut_a2g']
    s = sel['sampler']
    assert s['kind'] == 'held_t' and s['self_cond'] and s['held_iters'] == 4096
    sol_np, clue_np = arrays(FULL / 'full-frmtest.npz')
    index = json.loads((FULL / 'index.json').read_text())
    rows = np.array(index['shards']['full-frmtest']['official_rows'])
    V, cap = 10, 30.0
    logged = grid(a.iters, extra=(4096,))
    readout_at = set(logged)
    t0 = time.perf_counter()
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        dev = 'cuda'
        sol = torch.tensor(sol_np, dtype=torch.long, device=dev)
        clue = torch.tensor(clue_np, dtype=torch.long, device=dev)
        B, L = sol.shape
        assert B <= 1024  # frm_sweep's batch: the 1,000 puzzles are one batch
        mask3 = clue.bool().unsqueeze(-1)
        clue_onehot = torch.nn.functional.one_hot(sol, V).float()
        free = (~clue.bool()).float()

        def f(z, tau, sc):
            raw = model(z, tau, self_cond=sc, puzzle_identifiers=None, input_cond=None)
            capped = cap * torch.tanh(raw / cap)
            return capped - capped.logsumexp(dim=-1, keepdim=True), capped

        g = torch.linspace(0.0, 1.0, 4097, device=dev)
        phys = lut(g)
        j = int((phys - float(s['t_hold'])).abs().argmin().item())
        tau_h = torch.full((B,), float(g[j].item()), device=dev, dtype=torch.float32)
        t_h = float(phys[j].item())
        z = torch.randn((B, L, V), device=dev, dtype=torch.float32)
        z = torch.where(mask3, clue_onehot, z)
        sc = None
        rec = Recorder(torch, logged, sol, free)
        readout_exact, readout_ids_4096 = [], None
        for it in range(1, a.iters + 1):
            logprob, raw = f(z, tau_h, sc)
            sc = raw
            probs = logprob.exp()
            rec.step(it, probs, torch.where(mask3, clue_onehot, probs))
            if a.variant == 'table':
                eps = torch.randn_like(z)
                z_sde = t_h * probs + (1.0 - t_h) * eps
                z = (1.0 - s['renoise']) * z + s['renoise'] * z_sde
            else:
                z = probs
            z = torch.where(mask3, clue_onehot, z)
            if it in readout_at:  # the answer of an `it`-iteration run (one more forward, no RNG)
                r = torch.where(mask3, clue_onehot, f(z, tau_h, sc)[0].exp()).argmax(-1)
                readout_exact.append((r == sol).all(-1))
                if it == 4096:
                    readout_ids_4096 = r.cpu().numpy()
            if it in (64, 1024, 4096) or it % 8192 == 0:
                print(f'iter {it}  {time.perf_counter() - t0:.0f} s', flush=True)
        torch.cuda.synchronize()
        res = rec.result()
        res['readout_exact'] = torch.stack(readout_exact, 1).cpu().numpy()
    seconds = time.perf_counter() - t0
    summary = dict(method='FRM', task='sudoku-extreme', seed=a.seed, variant=a.variant, n=int(B), steps=a.iters, logged=logged,
                   trained_depth=4096, depth_unit='held-tau iteration (one network call; Table-2 run = 4,096 iterations + 1 readout)',
                   state='x0_hat on free cells (softmax of capped logits)', t_hold=s['t_hold'], tau_hold=float(tau_h[0]), t_phys=t_h,
                   renoise=s['renoise'] if a.variant == 'table' else 0.0, puzzles='full-frmtest (FRM 1,000 test puzzles = full-all rows 10,000-10,999)',
                   readout_solved=dict(zip(map(str, logged), res['readout_exact'].sum(0).tolist())),
                   argmax_solved=dict(zip(map(str, logged), res['exact'].sum(0).tolist())),
                   checkpoint=str(ckpt), checkpoint_sha256=sha, step=sel['step'], seconds=seconds, gpu=torch.cuda.get_device_name(),
                   slurm_job=os.environ.get('SLURM_JOB_ID'), script_sha256=digest(__file__))
    if a.variant == 'table' and readout_ids_4096 is not None:  # check: the 4,096-iteration flags of the same puzzles from the compute sweep (sweep/frm_sweep.py --check4096)
        ref = np.load(a.cs1 / f'sudoku-extreme-s{a.seed}/held4096-check-frmtest.npz')
        assert np.array_equal(ref['official_rows'], rows)
        mine = (readout_ids_4096 == sol_np).all(1)
        summary['check'] = dict(ref='compute-sweep held4096-check-frmtest.npz', ref_solved=int(ref['solved'].sum()), solved=int(mine.sum()),
                                mismatches=int((ref['solved'] != mine).sum()))
    np.savez_compressed(a.out.with_suffix('.npz'), official_rows=rows, **res)
    tmp = out_json.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(summary, indent=1) + '\n')
    os.replace(tmp, out_json)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('logged', 'readout_solved', 'argmax_solved')}, indent=1), flush=True)


def maze(a):
    """Maze-Unique: run the Table-2 sampler, then iterate the readout map up to a.calls network calls, and write the records."""
    import torch
    root = a.root.resolve()
    sys.path.insert(0, str(root))
    import frm_maze as F
    from flow_map_reasoning import build_sampler_config

    out_json = a.out.with_suffix('.json')
    if out_json.exists():
        print(f'{out_json} exists', flush=True)
        return
    a.out.parent.mkdir(parents=True, exist_ok=True)
    lane = root / f'runs/paper1000/s{a.seed}'
    sel = json.loads((lane / 'selection.json').read_text())['selected']
    assert sel['sampler'] == 'sde_selfcond' and sel['nfe'] == 257
    model, meta = F.load_model(root, sel['checkpoint'])
    model.eval()
    lut = meta['lut_a2g']
    sha = digest(sel['checkpoint'])
    solutions, clues, path = F.load_split(root, 'paper1000', 'test', lane)
    recipe = root / 'upstream/experiments/sudoku_extreme_train/samplers' / 'sde_selfcond.yaml'
    cfg, _ = build_sampler_config([str(recipe)], [f'sampler.num_samples={len(solutions)}', 'sampler.batch_size=250'])
    assert cfg.self_cond and cfg.time_schedule in (None, 'linear') and cfg.noise_scale > 0
    steps = sel['nfe'] - 1
    V, cap = 6, 30.0
    logged = grid(a.calls, extra=(steps, steps + 1, steps + 2))
    t0 = time.perf_counter()
    parts, ids257 = [], []
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        torch.manual_seed(sel['sampler_seed'])
        np.random.seed(sel['sampler_seed'])
        random.seed(sel['sampler_seed'])
        dev = 'cuda'

        def f(z, tau, sc):
            raw = model(z, tau, self_cond=sc, puzzle_identifiers=None, input_cond=None)
            capped = cap * torch.tanh(raw / cap)
            return capped - capped.logsumexp(dim=-1, keepdim=True), capped

        for lo in range(0, len(solutions), cfg.batch_size):
            sol = torch.tensor(solutions[lo:lo + cfg.batch_size], dtype=torch.long, device=dev)
            clue = torch.tensor(clues[lo:lo + cfg.batch_size], dtype=torch.long, device=dev)
            B, L = sol.shape
            mask3 = clue.bool().unsqueeze(-1)
            clue_onehot = torch.nn.functional.one_hot(sol, V).float()
            rec = Recorder(torch, logged, sol, (~clue.bool()).float())
            tau_vals = torch.linspace(0.0, 1.0, steps + 1, device=dev)
            z = torch.randn((B, L, V), device=dev, dtype=torch.float32)
            z = torch.where(mask3, clue_onehot, z)
            sc = None
            call = 0
            for i in range(steps):
                tau_in = tau_vals[i].expand(B)
                t_in = lut(tau_in)
                t_next = lut(tau_vals[i + 1].expand(B))
                dt = t_next - t_in
                logprob, raw = f(z, tau_in, sc)
                sc = raw
                probs = logprob.exp()
                call += 1
                rec.step(call, probs, torch.where(mask3, clue_onehot, probs))
                v = (probs - z) / (1.0 - t_in.view(-1, 1, 1) + 1e-5)
                z_ode = z + dt.view(-1, 1, 1) * v
                tn = t_next.view(-1, 1, 1)
                eps = torch.randn_like(z)
                z_sde = tn * probs + (1.0 - tn) * eps
                z = (1.0 - cfg.noise_scale) * z_ode + cfg.noise_scale * z_sde
                z = torch.where(mask3, clue_onehot, z)
            tau1 = tau_vals[steps].expand(B)
            while call < a.calls:  # call 257: the Table-2 readout; later calls iterate the readout map at tau = 1
                logprob, raw = f(z, tau1, sc)
                sc = raw
                probs = logprob.exp()
                call += 1
                pinned = torch.where(mask3, clue_onehot, probs)
                rec.step(call, probs, pinned)
                if call == steps + 1:
                    ids257.append(pinned.argmax(-1).cpu().numpy())
                z = pinned
            parts.append(rec.result())
            print(f'{lo + B}/{len(solutions)}  {time.perf_counter() - t0:.0f} s', flush=True)
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    res = {'t': parts[0]['t']}
    for k in ('rel', 'changed', 'exact'):
        res[k] = np.concatenate([p[k] for p in parts])
    ids257 = np.concatenate(ids257)
    ref = np.load(lane / 'scored/selected/test-seed0/predictions.npz')
    check = dict(ref='scored/selected/test-seed0/predictions.npz (Table 2)', ref_solved=int(ref['solved'].sum()),
                 solved=int((ids257 == solutions).all(1).sum()), mismatched_predictions=int((ref['predictions'] != ids257).any(1).sum()))
    summary = dict(method='FRM', task='maze-unique', seed=a.seed, variant='table+readout-map', n=int(len(solutions)), steps=a.calls,
                   logged=res['t'].tolist(), trained_depth=sel['nfe'],
                   depth_unit='network call (1-256 sde_selfcond integration, 257 Table-2 readout, >257 readout map at tau = 1)',
                   state='x0_hat on free cells (softmax of capped logits)', noise_scale=cfg.noise_scale,
                   argmax_solved=dict(zip(map(str, res['t'].tolist()), res['exact'].sum(0).tolist())), check=check,
                   checkpoint=sel['checkpoint'], checkpoint_sha256=sha, step=sel['step'], arm=sel['arm'], seconds=seconds,
                   gpu=torch.cuda.get_device_name(), slurm_job=os.environ.get('SLURM_JOB_ID'), script_sha256=digest(__file__))
    np.savez_compressed(a.out.with_suffix('.npz'), **res)
    tmp = out_json.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(summary, indent=1) + '\n')
    os.replace(tmp, out_json)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('logged', 'argmax_solved')}, indent=1), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('task', choices=['sudoku', 'maze'])
    p.add_argument('root', type=Path)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--variant', choices=['table', 'det'], default='table')
    p.add_argument('--iters', type=int, default=32768)
    p.add_argument('--calls', type=int, default=2056)
    p.add_argument('--cs1', type=Path, default=Path(__file__).resolve().parents[1] / 'runs/compute-sweep/frm')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    sudoku(a) if a.task == 'sudoku' else maze(a)
