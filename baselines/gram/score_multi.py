"""GRAM on N-Queens and graph coloring: sampled trajectories and GRAM's own metrics.

`run` is gram/score.py's sampling (one trajectory per sample, latent from the learned prior, halting off, greedy decoding
at temperature 0, sample s seeded with SEED_BASE + s) but it records, for each test row, sample and step count in
--steps, the whole decoded token sequence and the halting logit q_halt, because these tasks have many valid answers
and GRAM scores them with rule checkers, not with the label.

`merge` computes, from the recorded outputs:
- GRAM's metrics exactly as eval_pass_at_k.py computes them: GRAM's own evaluator classes (evaluators/nqueens.py
  NQueens, evaluators/graph_coloring.py GraphColoring) and its correctness function
  (evaluators/sampling_strategies.py compute_seq_correctness) are run on the recorded outputs at --headline steps
  (16, the paper's setting) with the first N samples, N = 1 and 20 (the paper's Table 1).
- The same metrics for every recorded step count and N in {1, 2, 5, 10, 20} with this repository's shared checkers
  (cram/nqueens_data.py, cram/graph_coloring_data.py, also used by the CRAM runs); at the headline
  setting they must agree with GRAM's evaluators (recorded as `agree`).
- The rule-checked single-sample solve rate: the fraction of test rows whose sample-0 output is a valid N-Queens board
  that holds every given queen (`completes`), or a proper 3-coloring of the graph (no conflict, any color order).
  GRAM's own N-Queens accuracy does not check the given queens; its graph-coloring validity also requires the
  canonical color order. Both are reported beside it.
Majority vote (GRAM's rule): the most frequent full token sequence among the first N samples, ties by the higher mean
sigmoid(q_halt), then by the earlier sample.

Run from the GRAM source tree with experiments/runs/gram/pydeps on PYTHONPATH (gram/score_multi.sbatch does this):
  python gram/score_multi.py run <out.npz> <checkpoint.pt> <first> <last> <steps,comma> <hydra overrides...>
  python gram/score_multi.py merge <nqueens|gc> <n> <data dir> <out.json> <shard.npz>...
The checker modules are read from $CHECKERS (default: beside this file), loaded by file path.
"""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path.cwd()
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import SEED_BASE, digest, load  # noqa: E402

NS = (1, 2, 5, 10, 20)


def run(out, checkpoint, first, last, steps, overrides):
    from models.losses import IGNORE_LABEL_ID
    steps = sorted(steps)
    model, loader, ck_step = load(overrides, checkpoint, max(steps))
    preds = {s: [] for s in range(first, last)}
    qs = {s: [] for s in range(first, last)}
    inputs = []
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.device('cuda'):
        for s in range(first, last):
            torch.manual_seed(SEED_BASE + s)
            for b, (_, batch, _) in enumerate(loader):
                if b == int(os.environ.get('GRAM_SCORE_BATCHES', 10**9)):  # smoke tests only
                    break
                batch = {k: v.cuda() for k, v in batch.items()}
                real = (batch['labels'] != IGNORE_LABEL_ID).any(-1)  # padding rows of the last batch have no label
                if s == first:
                    inputs.append(batch['inputs'][real].cpu().numpy().astype(np.uint8))
                carry = model.initial_carry(batch)
                p, q = [], []
                for k in range(1, max(steps) + 1):
                    carry, _, _, out_k, done = model(carry=carry, batch=batch, return_keys=['preds', 'q_halt_logits'])
                    if k in steps:
                        p.append(out_k['preds'][real].to(torch.uint8))
                        q.append(out_k['q_halt_logits'][real].float())
                    assert bool(done) == (k == max(steps)), (k, bool(done))
                preds[s].append(torch.stack(p, 1).cpu().numpy())  # (rows, steps, L)
                qs[s].append(torch.stack(q, 1).cpu().numpy())
            print(f'sample {s} done, {time.perf_counter() - t0:.0f} s', flush=True)
    order = range(first, last)
    P = np.stack([np.concatenate(preds[s]) for s in order], 1)  # (rows, samples, steps, L)
    Q = np.stack([np.concatenate(qs[s]) for s in order], 1)  # (rows, samples, steps)
    np.savez(out, preds=P, q_logit=Q, inputs=np.concatenate(inputs), samples=np.arange(first, last), steps=np.array(steps))
    meta = dict(checkpoint=str(checkpoint), checkpoint_step=ck_step, checkpoint_sha256=digest(checkpoint), overrides=list(overrides),
                samples=[first, last], steps=steps, seed_base=SEED_BASE, seconds=time.perf_counter() - t0, rows=int(P.shape[0]),
                gpu=torch.cuda.get_device_name(), script=str(Path(__file__).resolve()), script_sha256=digest(__file__))
    Path(out).with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta), flush=True)


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def majority(P, q):
    """GRAM's vote per row over the samples axis. P (rows, n, L) tokens, q (rows, n) sigmoid(q_halt) as float64.
    Returns the index of the earliest sample of the winning sequence: most votes, then highest mean q, then earliest."""
    out = np.empty(len(P), np.int64)
    for a in range(0, len(P), 2048):
        p = P[a:a + 2048]
        same = (p[:, :, None, :] == p[:, None, :, :]).all(-1)  # (rows, n, n)
        votes = same.sum(-1)
        mean_q = (same * q[a:a + 2048, None, :]).sum(-1) / votes
        out[a:a + 2048] = np.argmax(2 * votes + mean_q, 1)  # 0 <= mean_q <= 1, so votes decide first; argmax: earliest
    return out


def gram_evaluate(task, data_dir, inputs, labels, P, logit, n_samples):
    """GRAM's own evaluator over the recorded outputs of one step count: P (rows, samples, L), logit (rows, samples)."""
    from dataset.common import PuzzleDatasetMetadata
    from evaluators.sampling_strategies import compute_seq_correctness
    meta = PuzzleDatasetMetadata(**json.loads((Path(data_dir) / 'test' / 'dataset.json').read_text()))
    path = str(data_dir).lower()  # GRAM keys its checker on the path, as eval_pass_at_k.py does
    if task == 'nqueens':
        from evaluators.nqueens import NQueens
        ev = NQueens(data_path=path, eval_metadata=meta, pass_Ks=[k for k in NS if k <= n_samples], aggregated_voting=True)
    else:
        from evaluators.graph_coloring import GraphColoring
        ev = GraphColoring(data_path=path, eval_metadata=meta, pass_Ks=[k for k in NS if k <= n_samples],
                           aggregated_voting=True, max_colors=meta.vocab_size - 3)
    ev.begin_eval()
    for a in range(0, len(P), 768):
        x = torch.from_numpy(inputs[a:a + 768].astype(np.int64))
        y = torch.from_numpy(labels[a:a + 768].astype(np.int64))
        batch = dict(inputs=x, labels=y)
        samples = []
        for j in range(n_samples):
            pr = torch.from_numpy(P[a:a + 768, j].astype(np.int64))  # argmax output dtype: int64
            ok, tok, cnt = compute_seq_correctness(x, y, pr, 'cpu', data_path=path)
            samples.append(dict(preds=pr, q_halt_logits=torch.from_numpy(logit[a:a + 768, j]), exact_correct=ok,
                                token_accuracy=tok, valid=cnt > 0, h_states=None))
        ev.update_batch_samples(batch, samples, n_samples, None)
    return ev.result(save_path=None, rank=0, world_size=1)


def merge(task, n, data_dir, out, shards):
    from models.losses import IGNORE_LABEL_ID
    here = Path(os.environ.get('CHECKERS', Path(__file__).resolve().parent))
    parts = sorted((np.load(p) for p in shards), key=lambda z: int(z['samples'][0]))
    samples = np.concatenate([z['samples'] for z in parts])
    assert np.array_equal(samples, np.arange(samples.size)), samples
    steps = parts[0]['steps'].tolist()
    assert all(z['steps'].tolist() == steps for z in parts)
    P = np.concatenate([z['preds'] for z in parts], 1)  # (rows, samples, steps, L)
    logit = np.concatenate([z['q_logit'] for z in parts], 1)
    q = 1 / (1 + np.exp(-logit.astype(np.float64)))  # GRAM: sigmoid of the logit in float64
    d = Path(data_dir) / 'test'
    inputs, labels = np.load(d / 'all__inputs.npy'), np.load(d / 'all__labels.npy').astype(np.int64)
    meta = json.loads((d / 'dataset.json').read_text())
    labels[labels == meta['ignore_label_id']] = IGNORE_LABEL_ID  # as GRAM's loader does
    assert all(np.array_equal(z['inputs'], inputs) for z in parts), 'test rows out of order'
    rows, ns = P.shape[0], P.shape[1]
    res = dict(task=task, n=n, rows=rows, samples=ns, steps=steps, data=str(data_dir),
               test_sha256={f: digest(d / f) for f in ('all__inputs.npy', 'all__labels.npy')},
               shards=[str(p) for p in shards], shard_sha256=[digest(p) for p in shards],
               script=str(Path(__file__).resolve()), script_sha256=digest(__file__), table={}, gram={})
    per = {}
    if task == 'nqueens':
        nq = module('nqueens_data', here / 'nqueens_data.py')
        res['checker'] = dict(file=str(here / 'nqueens_data.py'), sha256=digest(here / 'nqueens_data.py'))
        given = inputs == 2
        boards = P == 2  # (rows, samples, steps, c); GRAM's verify_nqueens reads token 2 as a queen
        gram_ok = nq.valid(boards, n)  # (rows, samples, steps)
        comp = nq.completes(boards.reshape(rows, ns * len(steps), -1), given, n).reshape(rows, ns, len(steps))
        per.update(gram_valid=gram_ok, completes=comp)
        # GRAM's coverage compares the whole token sequence with its enumerated boards (1 empty, 2 queen), so a sample
        # with another token (0, pad) on an empty cell is valid but covers nothing; such samples enter as empty boards
        clean = np.isin(P, (1, 2)).all(-1)  # (rows, samples, steps)
        for a, k in enumerate(steps):
            cov = nq.gram_metrics(given, boards[:, :20, a] & clean[:, :20, a, None], n, tiebreak=q[:, :20, a],
                                  k_values=(20,)) if ns >= 20 else None
            for m in NS:
                if m > ns:
                    continue
                c = majority(P[:, :m, a], q[:, :m, a]) if m > 1 else np.zeros(rows, np.int64)
                e = dict(steps=k, n=m, acc=float(gram_ok[np.arange(rows), c, a].mean()),
                         acc_completion=float(comp[np.arange(rows), c, a].mean()))
                if m == 20 and cov is not None:
                    e.update(cov20=float(cov['coverage20'].mean()), cov_all=float(cov['coverage_all'].mean()))
                res['table'][f'steps{k}-n{m}'] = e
            res['table'][f'steps{k}-single'] = dict(steps=k, completes_sample0=float(comp[:, 0, a].mean()),
                                                     completes_mean=float(comp[:, :, a].mean()),
                                                     valid_sample0=float(gram_ok[:, 0, a].mean()))
    else:
        gc = module('graph_coloring_data', here / 'graph_coloring_data.py')
        res['checker'] = dict(file=str(here / 'graph_coloring_data.py'), sha256=digest(here / 'graph_coloring_data.py'))
        chk = gc.check_tokens(inputs, P.transpose(0, 2, 1, 3).reshape(rows, len(steps) * ns, -1))
        shape = lambda v: np.asarray(v).reshape(rows, len(steps), ns).transpose(0, 2, 1)  # (rows, samples, steps)
        proper, canon, valid = (shape(chk[k]) for k in ('proper', 'canonical', 'valid'))
        # GRAM's count_conflicts: edges whose end vertices carry the same token (raw tokens, as GRAM compares them)
        A = gc.adjacency((inputs[:, gc.offdiag_positions(n)] == 2).astype(np.float32), n)
        I, J = gc.pairs(n)
        conf = ((P[..., I] == P[..., J]) & A[:, I, J][:, None, None, :]).sum(-1)  # (rows, samples, steps)
        # GRAM's verify_coloring on raw tokens: no conflict, and the color tokens (3, 4, 5; others are skipped) appear
        # in canonical order. It accepts outputs with non-color tokens at vertex positions; the shared checker does not
        # (its `valid`); `proper` (any color order, color tokens only) is the rule-checked solve rate.
        nxt, seen, canon_g = np.zeros(conf.shape, np.int64), np.zeros(conf.shape + (3,), bool), np.ones(conf.shape, bool)
        for v in range(n):
            cv = P[..., v].astype(np.int64) - 3
            col = cv >= 0
            cc = np.clip(cv, 0, 2)
            new = col & ~np.take_along_axis(seen, cc[..., None], -1)[..., 0]
            canon_g &= ~new | (cc == nxt)
            np.put_along_axis(seen, cc[..., None], (np.take_along_axis(seen, cc[..., None], -1)[..., 0] | col)[..., None], -1)
            nxt += new
        gram_valid = (conf == 0) & canon_g
        totals = sum(len(x) for x in gc.proper_canonical_sets(A))  # GRAM's coverage denominator

        def gram_coverage(sel_samples, a):  # GRAM's compute_coverage_for_puzzle: distinct first-n token tuples of valid samples
            return sum(len({P[g, j, a, :n].tobytes() for j in sel_samples if gram_valid[g, j, a]}) for g in range(rows)) / totals

        per.update(conflicts=conf, proper=proper, canonical=canon, valid=valid, gram_valid=gram_valid)
        for a, k in enumerate(steps):
            for m in NS:
                if m > ns:
                    continue
                c = majority(P[:, :m, a], q[:, :m, a]) if m > 1 else np.zeros(rows, np.int64)
                sel = np.arange(rows), c, a
                e = dict(steps=k, n=m, conflicts=int(conf[sel].sum()), valid=float(gram_valid[sel].mean()),
                         valid_strict=float(valid[sel].mean()), proper=float(proper[sel].mean()),
                         coverage=gram_coverage(range(m), a), coverage_strict=gc.check_tokens(inputs, P[:, :m, a])['coverage'])
                res['table'][f'steps{k}-n{m}'] = e
            res['table'][f'steps{k}-single'] = dict(steps=k, proper_sample0=float(proper[:, 0, a].mean()),
                                                     proper_mean=float(proper[:, :, a].mean()),
                                                     valid_sample0=float(gram_valid[:, 0, a].mean()),
                                                     valid_strict_sample0=float(valid[:, 0, a].mean()))
    # GRAM's own evaluators at the headline step count
    a = steps.index(int(os.environ.get('GRAM_HEADLINE_STEPS', 16)))
    for m in (1, 20):
        if m > ns:
            continue
        r = gram_evaluate(task, data_dir, inputs, labels, P[:, :m, a], logit[:, :m, a], m)
        res['gram'][f'n{m}'] = {k: float(v) for k, v in r.items()}
        mine = res['table'][f'steps{steps[a]}-n{m}']
        if task == 'nqueens':
            cnt = {int(key.split('_')[-1][:-3]): v for key, v in r.items() if key.startswith('NQueens/count_')}
            cov = sum(r[f'NQueens/answer_coverage_{s}sol'] * c for s, c in cnt.items()) / sum(cnt.values())
            res['gram'][f'n{m}']['cov_row_mean'] = cov
            ok = abs(r['NQueens/majority_exact'] - mine['acc']) < 1e-6 and (m < 20 or abs(cov - mine['cov20']) < 1e-6)  # GRAM sums in float32
        else:
            ok = int(r['GraphColoring/majority_conflicts']) == mine['conflicts'] and \
                abs(r['GraphColoring/majority_constraint'] - mine['valid']) < 1e-6 and \
                abs(r[f'GraphColoring/coverage@{m}'] - mine['coverage']) < 1e-6
        res['gram'][f'n{m}']['agree'] = bool(ok)
        print(f'GRAM evaluator, N={m}, steps {steps[a]}: agrees with the shared checker: {ok}', flush=True)
    np.savez_compressed(Path(out).with_suffix('.per-sample.npz'), steps=np.array(steps), **per)
    Path(out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: v for k, v in res['table'].items() if k.startswith(f'steps{steps[a]}')}, indent=1), flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'run':
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), [int(x) for x in sys.argv[6].split(',')], sys.argv[7:])
    elif mode == 'merge':
        merge(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6:])
    else:
        raise SystemExit(__doc__)
