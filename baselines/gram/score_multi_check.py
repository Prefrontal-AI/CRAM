"""Check of gram/score_multi.py merge: GRAM's own evaluators and the shared checkers must
agree on synthetic outputs that exercise every branch, and a planted bug must break the agreement.

Synthetic shard per test set (N-Queens 8x8, graph coloring 8): 20 samples per row at one step count (16). A sample is a
valid completion of the input (N-Queens) or a proper coloring in any color order (graph coloring), a corrupted copy
of one, a copy of an earlier sample (so votes are not all 1) with the same halting logit (so ties occur), or for graph
coloring the same coloring with different tokens in the padding positions (equal colorings, different sequences).

Run from the GRAM source tree on a compute node, with the GRAM environment:
  python gram/score_multi_check.py <scratch dir> [planted]
`planted` runs the merge of a copy of score_multi.py whose vote takes sample 0 always; the check then must report
disagreement for N = 20.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[2] / 'data/gram'


def synth_nqueens(rng, inputs, n, ns=20):
    sys.path.insert(0, str(HERE.parents[1] / 'cram'))
    import nqueens_data as nq
    rows, L = inputs.shape
    P = np.empty((rows, ns, 1, L), np.uint8)
    cache = {}
    for i in range(rows):
        key = inputs[i].tobytes()
        if key not in cache:
            cache[key] = nq.completions(inputs[i] == 2, n)
        comp = cache[key]
        for j in range(ns):
            u = rng.random()
            if j > 0 and u < 0.3:
                P[i, j, 0] = P[i, rng.integers(j), 0]
                continue
            b = comp[rng.integers(len(comp))].astype(np.uint8) + 1
            if u > 0.8:  # corrupt: move or add a queen, or write a padding token
                c = rng.integers(L)
                b[c] = rng.choice([0, 1, 2])
            P[i, j, 0] = b
    return P


def synth_gc(rng, inputs, n, ns=20):
    sys.path.insert(0, str(HERE.parents[1] / 'cram'))
    import graph_coloring_data as gcd
    rows, L = inputs.shape
    A = gcd.adjacency((inputs[:, gcd.offdiag_positions(n)] == 2).astype(np.float32), n)
    allc = gcd.all_colorings(n)
    I, J = gcd.pairs(n)
    P = np.zeros((rows, ns, 1, L), np.uint8)
    for i in range(rows):
        ok = allc[~((allc[:, I] == allc[:, J]) & A[i][I, J]).any(1)]
        for j in range(ns):
            u = rng.random()
            if j > 0 and u < 0.25:
                P[i, j, 0] = P[i, rng.integers(j), 0]
                continue
            if j > 0 and u < 0.35:  # same coloring, different padding tokens
                P[i, j, 0] = P[i, rng.integers(j), 0]
                P[i, j, 0, n + rng.integers(L - n)] = rng.integers(6)
                continue
            col = ok[rng.integers(len(ok))].copy()
            if rng.random() < 0.4:
                col = gcd.canonical(col[None])[0]
            if u > 0.8:  # corrupt: a random color, or a non-color token (0, 1, 2) at a vertex position
                col[rng.integers(n)] = rng.integers(3) if rng.random() < 0.5 else rng.integers(-3, 0)
            P[i, j, 0, :n] = col + 3
    return P


def main(scratch, planted=False):
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    script = HERE / 'score_multi.py'
    if planted:
        src = script.read_text()
        bug = "out[a:a + 2048] = np.argmax(2 * votes + mean_q, 1)"
        assert bug in src
        (scratch / 'gram').mkdir(exist_ok=True)
        (scratch / 'gram' / 'score.py').write_text((HERE / 'score.py').read_text())
        script = scratch / 'gram' / 'score_multi.py'
        script.write_text(src.replace(bug, "out[a:a + 2048] = 0  # planted: always sample 0"))
    rng = np.random.default_rng(0)
    results = {}
    for task, n, d, synth in (('nqueens', 8, 'nqueens-8', synth_nqueens), ('gc', 8, 'graph-coloring-8', synth_gc)):
        inputs = np.load(DATA / d / 'test' / 'all__inputs.npy')
        P = synth(rng, inputs, n)
        logit = rng.choice([-1.0, 0.0, 1.0], size=P.shape[:3]).astype(np.float32)
        for i in range(len(P)):  # a copy keeps the logit of the first equal sample, so equal means (ties) are common
            for j in range(20):
                logit[i, j, 0] = logit[i, (P[i, :, 0] == P[i, j, 0]).all(-1).argmax(), 0]
        shard = scratch / f'{task}.npz'
        np.savez(shard, preds=P, q_logit=logit, inputs=inputs.astype(np.uint8), samples=np.arange(20), steps=np.array([16]))
        out = scratch / f'{task}-score.json'
        out.unlink(missing_ok=True)
        subprocess.run([sys.executable, str(script), 'merge', task, str(n), str(DATA / d), str(out), str(shard)], check=True,
                       env=dict(__import__('os').environ, CHECKERS=str(HERE.parents[1] / 'cram')))
        r = json.loads(out.read_text())
        results[task] = {m: r['gram'][m]['agree'] for m in r['gram']}
        print(task, results[task], json.dumps(r['table']['steps16-n20']), flush=True)
    ok = all(all(v.values()) for v in results.values())
    expect = not planted
    print('PLANTED' if planted else 'CLEAN', 'agree' if ok else 'disagree', '-> check', 'PASSES' if ok == expect else 'FAILS')
    sys.exit(0 if ok == expect else 1)


if __name__ == '__main__':
    main(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == 'planted')
