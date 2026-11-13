"""Hand-set graph-coloring CRAM energy: a reference row of Table 3, like the hand-set N-Queens energy. Run from experiments/.

  python graph_coloring_hand_set.py --mode dev --n 8              # temperature on the validation graphs (seed 0)
  python graph_coloring_hand_set.py --mode test --n 8 --seed 0    # the test graphs once, 20 chains

Energy: cost 1 for every edge whose two vertices share a color, nothing else (the `shared` arm of
cram/cram_graph_coloring.py with W0 = 0, D = the identity, B = 0). A proper coloring has energy 0 and any other
coloring at least 1, so the energy test with tau = 0 (E <= 1e-3) is exact. It uses the rules, which no trained row gets.
Search: the focused heat-bath search of the learned rows (cram_graph_coloring.solve), cap 10,000 moves, focus ebar = 0, theta 0.45 and
temperature unit hbar = 1 (the training statistics of this energy: own local energy 0 at every proper coloring, one
positive coupling value). The energy does not prefer a color order, so each output is relabeled to canonical color
order (colors by first appearance, GRAM's normalize_coloring), the output format of GRAM's labels; this relabeling is
part of the rules the reference is given. Conflicts do not depend on it.
Outputs: runs/CRAM-GC/hand/n<n>/ (dev.json; test-s<seed>/final.json and outputs.npz in the format of graph_coloring.py test).
Dimension names: b graphs, n vertices, c colors (3), k chains per graph.
"""
import dataclasses
import json
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from cram import cram_graph_coloring as G

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs' / 'CRAM-GC' / 'hand'
TEMPS = (0.3, 1.0, 3.0)
CAP = 10_000


@dataclasses.dataclass
class Args:
    mode: Literal['dev', 'test']  # validation temperature choice, or the test graphs once
    n: int = 8  # vertices (8 or 10)
    seed: int = 0  # test: search seed


def model():
    """The hand-set model in the format of cram_graph_coloring.solve: non-edge table W0 = 0 (c, c), edge table
    D = identity (cost 1 for equal colors), bias B = 0, stopping threshold tau = 0, temperature unit hbar = 1."""
    C = G.C
    return dict(W0=np.zeros((C, C)), D=np.eye(C), B=np.zeros(C), tau=0.0, hbar=1.0, ebar=0.0, theta=0.45)


def relabel(res):
    """The search result with every board (b, k, n) relabeled to canonical color order (colors by first appearance)."""
    b = res['boards']
    return dict(res, boards=G.canonical(b.reshape(-1, b.shape[-1])).reshape(b.shape))


def run(a_edges, samples, T, seed):
    """Search `samples` chains per graph at temperature T; a_edges: (b, p) edge indicators. Returns the relabeled result."""
    return relabel(G.solve(model(), a_edges, samples=samples, temperature=T, cap=CAP, seed=seed, stop='tau'))


def main(a: Args):
    """--mode dev: choose the temperature on the validation graphs (one chain, seed 0; highest proper rate, ties to
    the lower flop). --mode test: the test graphs once with 20 chains at that temperature."""
    out = ROOT / f'n{a.n}'
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'dev':
        x, y = G.load_split(a.n, 'train')
        edges, _ = G.convert(x, y, a.n)
        _, val = G.validation_graphs(edges)
        A = G.adjacency(edges[val], a.n)
        rows = []
        for T in TEMPS:
            res = run(edges[val], 1, T, 0)
            s = G.score(A, res, ks=(1,))
            rows.append(dict(temperature=T, proper_1=s['proper_1'], conflicts_1=s['conflicts_1'], flop_1=s['flop_1']))
            print(rows[-1], flush=True)
        best = max(rows, key=lambda r: (r['proper_1'], -r['flop_1']))
        (out / 'dev.json').write_text(json.dumps(dict(rule='highest validation proper N = 1, ties to the lower flop', graphs=int(len(val)),
                                                      searches=rows, temperature=best['temperature']), indent=1) + '\n')
        return
    T = json.loads((out / 'dev.json').read_text())['temperature']
    x, y = G.load_split(a.n, 'test')
    edges, _ = G.convert(x, y, a.n)
    A = G.adjacency(edges, a.n)
    res = run(edges, 20, T, a.seed)
    s = G.score(A, res)
    d = out / f'test-s{a.seed}'
    d.mkdir(exist_ok=True)
    np.savez_compressed(d / 'outputs.npz', boards=res['boards'], energy=res['energy'], stopped=res['stopped'],
                        moves=res['moves'], chain_flop=res['chain_flop'])
    (d / 'final.json').write_text(json.dumps(dict(s, temperature=T, seed=a.seed, relabeled=True), indent=1) + '\n')
    print(json.dumps({k: s[k] for k in ('proper_1', 'conflicts_1', 'coverage_20', 'conflicts_majority_20', 'flop_20')}))


if __name__ == '__main__':
    main(tyro.cli(Args))
