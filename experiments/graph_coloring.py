"""Graph-coloring CRAM on GRAM's graph-coloring sets (the graph-coloring rows of Table 3): recipe screen, timed fit
and test. Run from experiments/ (on a Slurm cluster: slurm/gpu.sbatch or slurm/cpu.sbatch).

  python graph_coloring.py --mode screen --n 8 --arm full       # with the grid in --updates, --lr, --lam, --temperature, --stop
  python graph_coloring.py --mode summary                        # the selection rule on a screen folder (prints the ranking)
  python graph_coloring.py --mode fit --n 8 --arm full --device cuda
  python graph_coloring.py --mode test --n 8 --arm full --stop terms --temperature 2

  screen   fit each (updates, lr, lam) on the training rows outside the validation graphs, search the validation graphs
           at each temperature with inference seeds 0, 1, 2; one JSON per fit under runs/CRAM-GC/screen/
  summary  per size and arm, the best screen candidate by the selection rule (see `summary`)
  fit      the selected recipe on all training rows: the cold fit (the first fit in this fresh process) and 7 warm
           fits (the fit times), model saved under runs/CRAM-GC/final/
  test     the selected recipe's CPU model on GRAM's test split, once per inference seed (refuses without selected.json)

The energy, the training and the search are in cram/cram_graph_coloring.py:
    E(y) = sum_i B_i[y_i] + sum_(p = (i, j)) K_p(a)[y_i, y_j],
y the coloring and K_p(a) a 3 x 3 table that is an affine function of the adjacency tokens a. The arm sets what K_p
may read; the paper reports the arm `full` (every pair reads every token). The default of --arm is `pos`, so pass
--arm full. The selected recipes are in runs/CRAM-GC/selected.json (a copy: recipes/graph-coloring.json).
The folder name CRAM-GC is only a label.

Completed outputs are skipped, so a requeued job resumes at its first unfinished stage. Each fit takes seconds.
Dimension names (as cram/cram_graph_coloring.py): b graphs, n vertices, p vertex pairs i < j, c colors (3).
"""
import dataclasses
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

from cram import cram_graph_coloring as G

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs' / 'CRAM-GC'


@dataclasses.dataclass
class Args:
    mode: Literal['screen', 'summary', 'fit', 'test']  # stage to run
    n: int = 8  # vertices: GRAM's graph-coloring-8 or graph-coloring-10
    arm: Literal['full', 'pos', 'shared'] = 'pos'  # what each pair table may read (cram/cram_graph_coloring.py)
    updates: tuple[int, ...] = (64,)  # full-batch SGD updates (screen: every value)
    lr: tuple[float, ...] = (10.,)  # peak learning rate, cosine decay to 0 (screen: every value)
    lam: tuple[float, ...] = (0.,)  # L1 weight, proximal soft-threshold lr_k * lam (screen: every value)
    temperature: tuple[float, ...] = (.35,)  # heat-bath temperature in units of hbar (screen: every value)
    stop: tuple[str, ...] = ('tau',)  # stopping test: tau (whole energy) or terms (every term at most its threshold)
    samples: int = 20  # independent chains per graph; chain 0 is the single-output prediction
    cap: int = 10000  # moves per chain before it returns its lowest-energy coloring
    seeds: tuple[int, ...] = (0, 1, 2)  # inference seeds (training is full batch and deterministic)
    device: str = 'cuda'  # fit device (screen and fit); test searches on the CPU with the CPU-fitted model
    tag: str = ''  # screen output subfolder


def tag_of(arm, n, u, lr, lam):
    """The folder or file name of a recipe: arm, vertices n, updates u, learning rate lr and L1 weight lam."""
    return f'n{n}-{arm}-u{u}-lr{lr:g}-lam{lam:g}'


def hardware(device):
    """The name of the GPU for a CUDA device, else the processor's model name."""
    if str(device).startswith('cuda'):
        return torch.cuda.get_device_name()
    return next((l.split(':', 1)[1].strip() for l in Path('/proc/cpuinfo').read_text().splitlines()
                 if l.startswith('model name')), 'unknown')


def screen(a: Args):
    """Fit every (updates, lr, lam) of the arguments on the training rows outside the validation graphs and search the
    validation graphs at every (stopping test, temperature, seed). Writes one JSON per fit under
    runs/CRAM-GC/screen/<tag>/ with the fit summary and the score of every search."""
    x, y = G.load_split(a.n, 'train')
    av, _ = G.convert(x, y, a.n)
    train_rows, val_rows = G.validation_graphs(av)
    A_val = G.adjacency(av[val_rows], a.n)
    out_dir = ROOT / 'screen' / (a.tag or 'default')
    for u in a.updates:
        for lr in a.lr:
            for lam in a.lam:
                path = out_dir / f'{tag_of(a.arm, a.n, u, lr, lam)}.json'
                if path.exists():
                    continue
                model, timing = G.fit(x, y, a.n, arm=a.arm, updates=u, lr=lr, lam=lam, device=a.device, rows=train_rows)
                rec = dict(arm=a.arm, n=a.n, updates=u, lr=lr, lam=lam, timing=timing, summary=G.summarize(model, a.n),
                           val_graphs=int(len(val_rows)), train_rows=int(train_rows.sum()), searches=[])
                if np.isfinite(model['hbar']):
                    for stop in a.stop:
                        for T in a.temperature:
                            for s in a.seeds:
                                t0 = time.perf_counter()
                                res = G.solve(model, av[val_rows], samples=a.samples, temperature=T, cap=a.cap, seed=s,
                                              stop=stop)
                                sc = G.score(A_val, res)
                                rec['searches'].append(dict(stop=stop, temperature=T, seed=s,
                                                            seconds=time.perf_counter() - t0, **sc))
                G.save_json(path, rec)
                best = max(rec['searches'], key=lambda r: r['proper_1'], default={})
                print(path.name, f"pl {timing['final_pl']:.4f} nz {rec['summary']['params_nonzero']}/{rec['summary']['params']}",
                      {k: best.get(k) for k in ('temperature', 'proper_1', 'conflicts_1', 'coverage_20', 'flop_1')}, flush=True)


def summary(a: Args):
    """The selection rule on the screen folders named in --tag (joined by '+'): per (n, arm), the seed means of every
    (fit, stopping test, temperature). Among the candidates within one validation graph of the best single-output
    proper rate (proper_1): the highest Cov@20, then the lower 20-chain flop, the larger L1 weight, fewer updates and
    the lower learning rate. Prints the ranking and writes runs/CRAM-GC/screen/summary-<tag>.json."""
    out = {}
    for n in (8, 10):
        for arm in G.ARMS:
            cands = []
            paths = [p for t in (a.tag or 'default').split('+') for p in sorted((ROOT / 'screen' / t).glob(f'n{n}-{arm}-*.json'))]
            for path in paths:
                r = json.loads(path.read_text())
                for stop, T in sorted({(s.get('stop', 'tau'), s['temperature']) for s in r['searches']}):
                    ss = [s for s in r['searches'] if s['temperature'] == T and s.get('stop', 'tau') == stop]
                    m = {k: float(np.mean([s[k] for s in ss])) for k in ('proper_1', 'proper_canonical_1', 'conflicts_1',
                         'conflicts_majority_20', 'coverage_20', 'coverage_canonicalized_20', 'flop_1', 'flop_20',
                         'wrong_stops_all', 'budget_failures_all')}
                    cands.append(dict(n=n, arm=arm, updates=r['updates'], lr=r['lr'], lam=r['lam'], stop=stop, temperature=T,
                                      graphs=r['val_graphs'], nonzero=r['summary']['params_nonzero'],
                                      params=r['summary']['params'], **m))
            if not cands:
                continue
            best = max(c['proper_1'] for c in cands)
            near = [c for c in cands if c['proper_1'] >= best - 1 / c['graphs'] - 1e-12]
            near.sort(key=lambda c: (-c['coverage_20'], c['flop_20'], -c['lam'], c['updates'], c['lr']))
            out[f'n{n}-{arm}'] = near[0]
            print(f'n{n} {arm}: best proper_1 {best:.4f}; {len(near)} within one graph; top by the rule:')
            for c in near[:5]:
                print('   ', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
    G.save_json(ROOT / 'screen' / f"summary-{a.tag or 'default'}.json", out)


def recipe(a: Args):
    """The selected recipe of (a.n, a.arm) from runs/CRAM-GC/selected.json."""
    sel = json.loads((ROOT / 'selected.json').read_text())[f'n{a.n}-{a.arm}']
    return sel


def final_dir(a: Args, sel, device):
    """The folder of the final fit of recipe sel on a device: runs/CRAM-GC/final/<recipe>/{cuda,cpu}."""
    return ROOT / 'final' / f"{tag_of(a.arm, a.n, sel['updates'], sel['lr'], sel['lam'])}" / ('cuda' if str(device).startswith('cuda') else 'cpu')


def fit_stage(a: Args, device):
    """Cold fit and 7 warm fits in this process, on all training rows. Saves the cold fit's model and the timing
    record (cold and warm seconds, CUDA context seconds, hardware, whether each warm fit gave the same model).
    Returns (the cold fit's model, the timing record); a finished fit is loaded from its folder."""
    sel = recipe(a)
    dest = final_dir(a, sel, device)
    if (dest / 'final.json').exists():
        with np.load(dest / 'model.npz') as f:
            m = {k: f[k] for k in ('W0', 'D', 'B', 'tpair', 'tvert')}
        return m | json.loads((dest / 'model.json').read_text()), json.loads((dest / 'final.json').read_text())
    x, y = G.load_split(a.n, 'train')  # reading files: before the clock
    context = 0.
    if str(device).startswith('cuda'):
        t = time.perf_counter()
        torch.cuda.init()
        torch.zeros(1, device=device)
        torch.cuda.synchronize()
        context = time.perf_counter() - t
    kw = dict(arm=a.arm, updates=sel['updates'], lr=sel['lr'], lam=sel['lam'], device=device)
    model, timing = G.fit(x, y, a.n, **kw)
    warm = []
    for _ in range(7):
        m2, t2 = G.fit(x, y, a.n, **kw)
        warm.append(t2['fit_seconds'])
        same = all(np.array_equal(m2[k], model[k]) for k in ('W0', 'D', 'B', 'tpair', 'tvert'))
        timing.setdefault('warm_same_model', []).append(same)
    timing.update(cold_seconds=timing['fit_seconds'], warm_seconds=statistics.median(warm), warm_repeats=warm,
                  cuda_context_seconds=context, hostname=platform.node(), hardware=hardware(device),
                  torch_threads=torch.get_num_threads(), summary=G.summarize(model, a.n), recipe=sel)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / 'model.npz', **{k: model[k] for k in ('W0', 'D', 'B', 'tpair', 'tvert')})
    G.save_json(dest / 'model.json', {k: model[k] for k in ('arm', 'n', 'theta', 'tau', 'ebar', 'hbar', 'final_pl')})
    (dest / 'log.jsonl').write_text(json.dumps({k: v for k, v in timing.items() if k != 'summary'}) + '\n')
    G.save_json(dest / 'final.json', timing)
    print(dest, json.dumps({k: timing[k] for k in ('cold_seconds', 'warm_seconds', 'cuda_context_seconds', 'hardware')}), flush=True)
    return model, timing


def test(a: Args):
    """The selected recipe's CPU-fitted model on GRAM's test split: `samples` chains per graph, once per inference
    seed. The stopping test, temperature, samples and cap must equal the selected ones. Writes outputs.npz (boards,
    energies, moves, flop per chain) and final.json (the scores) per seed."""
    sel = recipe(a)
    for key in ('samples', 'cap'):
        if getattr(a, key) != sel[key]:
            raise RuntimeError(f'test settings differ from selected.json: {key}')
    if a.temperature != (sel['temperature'],) or a.stop != (sel['stop'],):
        raise RuntimeError('test temperature or stopping test differs from selected.json')
    model, timing = fit_stage(a, 'cpu')
    x, y = G.load_split(a.n, 'test')
    at, yt = G.convert(x, y, a.n)
    A = G.adjacency(at, a.n)
    for s in a.seeds:
        dest = final_dir(a, sel, 'cpu').parent / f"test-{sel['stop']}-T{sel['temperature']:g}-S{sel['samples']}-cap{sel['cap']}-s{s}"
        if (dest / 'final.json').exists():
            continue
        t0 = time.perf_counter()
        res = G.solve(model, at, samples=sel['samples'], temperature=sel['temperature'], cap=sel['cap'], seed=s,
                      stop=sel['stop'])
        sc = G.score(A, res)
        sc.update(seconds=time.perf_counter() - t0, seed=s, label_match_1=int((res['boards'][:, 0] == yt).all(1).sum()),
                  recipe=sel, n_test=int(len(at)))
        dest.mkdir(parents=True, exist_ok=True)
        np.savez(dest / 'outputs.npz', boards=res['boards'], energy=res['energy'], stopped=res['stopped'],
                 moves=res['moves'], chain_flop=res['chain_flop'])
        G.save_json(dest / 'final.json', sc)
        print(dest.name, json.dumps({k: sc[k] for k in ('proper_1', 'proper_canonical_1', 'conflicts_1',
                                                        'conflicts_majority_20', 'coverage_20', 'flop_1', 'flop_20')}),
              flush=True)


def main():
    """Record the hashes of the data files (runs/CRAM-GC/data.json), then run the stage. One CPU thread."""
    a = tyro.cli(Args)
    torch.set_num_threads(1)
    G.save_json(ROOT / 'data.json', G.SOURCE | {f'graph-coloring-{n}/{s}/all__{k}.npy': G.sha256(
        G.DATA / f'graph-coloring-{n}' / s / f'all__{k}.npy') for n in (8, 10) for s in ('train', 'test')
        for k in ('inputs', 'labels')})
    if a.mode == 'screen':
        screen(a)
    elif a.mode == 'summary':
        summary(a)
    elif a.mode == 'fit':
        fit_stage(a, a.device)
    else:
        test(a)


if __name__ == '__main__':
    main()
