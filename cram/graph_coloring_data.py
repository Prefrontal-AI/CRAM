"""The graph-coloring data of GRAM (a baseline model of the paper) and the rule checker, shared by the CRAM runs and
the scoring of the baselines (NumPy only).

Data source: the Hugging Face dataset `CUN-bjy/graph-coloring` (revision 08d5f1f8), encoded by GRAM's own builder
(https://github.com/ahn-ml/gram-release at commit f2e32b9, `dataset/build_graph_coloring_dataset.py`, unmodified).
`scripts/prepare_data.py multi-solution` runs the builder and writes
`<data>/gram/graph-coloring-{8,10}/{train,test}/all__{inputs,labels}.npy` (`<data>` is `data/` or `CRAM_DATA`;
`GRAM_GC_DATA` overrides the folder `<data>/gram`). The folder path must not contain "sudoku" when GRAM is trained on
it: GRAM's loss head switches to its Sudoku checker when the data path contains that word.
The task: color the n vertices of a graph with 3 colors so that no edge joins two vertices of one color. A graph with
several proper colorings occurs in several training rows, one row per coloring as its label.
Sizes (rows / distinct graphs): 8 vertices train 7,002 / 2,059, test 255 / 255; 10 vertices train 13,465 / 1,542,
test 192 / 192.

Encoding: inputs are the upper triangle of the adjacency matrix with its diagonal, n(n+1)/2 tokens in row-major order
(0 diagonal, 1 no edge, 2 edge); a label or a prediction has the n vertex colors 3, 4, 5 first, then padding (0).
`convert` returns the edge indicators of the p = n(n-1)/2 vertex pairs and the colors as 0, 1, 2.
Splits: 'train' and 'test' as GRAM built them; `validation_graphs` holds out a seeded 10% of the distinct training
graphs as the validation set on which settings are chosen. No augmentation is applied (the labels are in canonical
color order, which a relabeling of the colors would break).

Checker (GRAM's evaluators/graph_coloring_utils.py definitions): conflicts = edges whose two vertices share a color;
proper = no conflict; canonical = colors in order of first appearance; GRAM's validity = proper and canonical;
coverage counts distinct valid outputs over all canonical proper colorings of the graph.

Dimension names: b graphs, n vertices, p vertex pairs i < j in row-major order (GRAM's token order), s samples.

Usage from another environment (no package import needed):
  spec = importlib.util.spec_from_file_location('gcd', '<repo>/cram/graph_coloring_data.py')
  x, y = gcd.load_split(8, 'test'); m = gcd.check_tokens(x, preds)   # preds (b, n(n+1)/2) or (b, s, n(n+1)/2)
"""
import os
from pathlib import Path

import numpy as np

C = 3  # colors
DATA = Path(os.environ.get('GRAM_GC_DATA', Path(os.environ.get('CRAM_DATA', Path(__file__).resolve().parents[1] / 'data')) / 'gram'))
SOURCE = dict(repo='CUN-bjy/graph-coloring', revision='08d5f1f8791a532cfb48fac88660e285fc72d857',
              builder='GRAM f2e32b9 dataset/build_graph_coloring_dataset.py (unmodified)', folder=str(DATA))


def pairs(n):
    """(I, J): the vertex indices i < j of the p pairs, in row-major order."""
    return np.triu_indices(n, 1)


def offdiag_positions(n):
    """Positions of the p off-diagonal tokens in GRAM's n(n+1)/2 upper-triangle sequence."""
    tri = [(i, j) for i in range(n) for j in range(i, n)]
    return np.array([k for k, (i, j) in enumerate(tri) if i < j])


def load_split(n, split):
    """GRAM's encoded arrays (inputs, labels) for graph-coloring-n, split 'train' or 'test'."""
    d = DATA / f'graph-coloring-{n}' / split
    x, y = np.load(d / 'all__inputs.npy'), np.load(d / 'all__labels.npy')
    if x.shape[1] != n * (n + 1) // 2:
        raise ValueError('sequence length is not n(n+1)/2')
    return x, y


def convert(x, y, n):
    """GRAM tokens -> edge indicators a (b, p) float32 and colors (b, n) in 0..2."""
    tok = x[:, offdiag_positions(n)]
    if not np.isin(tok, (1, 2)).all():
        raise ValueError('off-diagonal tokens must be 1 (no edge) or 2 (edge)')
    col = y[:, :n].astype(np.int64) - 3
    if ((col < 0) | (col >= C)).any():
        raise ValueError('colors outside 3..5')
    return (tok == 2).astype(np.float32), col


def adjacency(a, n):
    """Edge indicators a (b, p) -> symmetric adjacency matrices (b, n, n) bool."""
    I, J = pairs(n)
    A = np.zeros((len(a), n, n), bool)
    A[:, I, J] = a > 0
    A[:, J, I] = a > 0
    return A


def validation_graphs(a, frac=0.1, seed=20261106):
    """CRAM's validation split of the training rows: a seeded 10% of the distinct graphs.
    Returns (training-row mask without those graphs, one row index per held-out graph)."""
    keys = [r.tobytes() for r in (a > 0)]
    uniq = sorted(set(keys))
    rng = np.random.default_rng(seed)
    held = set(uniq[k] for k in rng.permutation(len(uniq))[:round(frac * len(uniq))])
    val_rows = np.array([k in held for k in keys])
    first = {}
    for r, k in enumerate(keys):
        if k in held and k not in first:
            first[k] = r
    return ~val_rows, np.array(sorted(first.values()))


def conflicts(A, boards):
    """Edges whose two vertices share a color. A (b, n, n); boards (b, ..., n) colors -> (b, ...)."""
    n = A.shape[1]
    I, J = pairs(n)
    e = A[:, I, J].reshape((A.shape[0],) + (1,) * (boards.ndim - 2) + (len(I),))
    return ((boards[..., I] == boards[..., J]) & e).sum(-1)


def canonical(boards):
    """Relabel colors by first appearance (GRAM's normalize_coloring)."""
    flat = boards.reshape(-1, boards.shape[-1])
    out = np.empty_like(flat)
    for r, row in enumerate(flat):
        m = {}
        out[r] = [m.setdefault(v, len(m)) for v in row]
    return out.reshape(boards.shape)


def is_canonical(boards):
    """(...,) bool: the colors of boards (..., n) appear in the order 0, 1, 2 of first appearance."""
    return (canonical(boards) == boards).all(-1)


def all_colorings(n):
    """(3^n, n): every assignment of the 3 colors to n vertices."""
    return np.array(np.meshgrid(*[np.arange(C)] * n, indexing='ij')).reshape(n, -1).T


def proper_canonical_sets(A):
    """Per graph, the set of proper colorings in canonical form (GRAM's generate_all_valid_colorings)."""
    n = A.shape[1]
    allc = all_colorings(n)
    allc = allc[is_canonical(allc)]
    I, J = pairs(n)
    out = []
    for adj in A:
        ok = ~((allc[:, I] == allc[:, J]) & adj[I, J]).any(1)
        out.append({tuple(r) for r in allc[ok]})
    return out


def check_tokens(x, preds):
    """Rule check of GRAM-format predictions. x (b, L) input tokens; preds (b, L) or (b, s, L) tokens.

    Returns per graph (and sample): conflicts, proper, canonical, valid (= proper and canonical, GRAM's
    verify_coloring); a prediction with a non-color token among its first n tokens has conflicts -1 and is invalid.
    Also returns coverage: distinct valid samples summed over graphs / canonical proper colorings summed over graphs.
    """
    L = x.shape[1]
    n = int(round((-1 + np.sqrt(1 + 8 * L)) / 2))
    single = preds.ndim == 2
    p = preds[:, None] if single else preds
    col = p[..., :n].astype(np.int64) - 3
    ok_tok = ((col >= 0) & (col < C)).all(-1)
    col = np.where(ok_tok[..., None], col, 0)
    A = adjacency((x[:, offdiag_positions(n)] == 2).astype(np.float32), n)
    conf = np.where(ok_tok, conflicts(A, col), -1)
    proper = conf == 0
    canon = is_canonical(col) & ok_tok
    valid = proper & canon
    sols = proper_canonical_sets(A)
    found = sum(len({tuple(r) for r, v in zip(col[g], valid[g]) if v}) for g in range(len(x)))
    out = dict(conflicts=conf, proper=proper, canonical=canon, valid=valid,
               coverage=found / sum(len(s) for s in sols), solutions_total=sum(len(s) for s in sols))
    if single:
        out.update({k: out[k][:, 0] for k in ('conflicts', 'proper', 'canonical', 'valid')})
    return out
