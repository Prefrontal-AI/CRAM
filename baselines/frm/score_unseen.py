"""Score the selected checkpoint of a timed FRM (Flow Reasoning Model) Sudoku campaign on the 412,786 unseen puzzles.

The unseen puzzles are the official 422,786-puzzle test split minus dev10k (data/frm/dev10k/indices.json).
They are the shards of data/full-test outside full-dev10k (full-frmtest and full-00..15); each shard's index
entry names its official rows. The script uses the campaign's own loader (EMA weights; frm_reproduction.py in the
campaign root), its copy of the authors' code (flow_map_reasoning: SamplerConfig, run_sampler) and its sampler, with the sampler settings of the campaign's selection (runs/s<k>/selection.json). Only the
batch size may differ from the selection's 128 (it changes which random numbers each puzzle draws, not the sampler).
Run with FRM's python and environment (frm/score_unseen.sbatch).

  python score_unseen.py shard <root> --seed K --shard full-03 [--batch 1024] [--n N] [--out PATH] [--check-campaign-test]
  python score_unseen.py aggregate <root> --seed K   # -> <root>/score-unseen.json and score-unseen.solved.npz

shard writes <root>/runs/s<K>/unseen/<shard>.json and the per-puzzle solved flags (.npz, with the official rows).
--check-campaign-test (full-frmtest only) also requires the predictions to equal the campaign's selected test
predictions puzzle by puzzle (meaningful at --batch 128, the selection's batch).
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V1_DATA = Path(os.environ.get('V1_DATA', HERE.parents[1] / 'data'))
FULL = V1_DATA / 'full-test'
OFFICIAL = V1_DATA / 'frm/official/test.npz'
DEV10K = V1_DATA / 'frm/dev10k/indices.json'
UNSEEN_SHARDS = ['full-frmtest'] + [f'full-{k:02d}' for k in range(16)]


def digest(path):
    """SHA-256 of a file, as a hexadecimal string."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selection(root, seed):
    """The campaign's selected checkpoint and sampler settings for one seed (runs/s<seed>/selection.json)."""
    return json.loads((root / f'runs/s{seed}/selection.json').read_text())['selected']


def shard(args):
    """Sample one shard with the selected checkpoint; write the record and the per-puzzle solved flags."""
    import torch
    root = args.root.resolve()
    sys.path.insert(0, str(root))
    from frm_reproduction import arrays, load_model, profile
    from flow_map_reasoning import SamplerConfig, run_sampler

    sel = selection(root, args.seed)
    ckpt = Path(sel['checkpoint'])
    sha = digest(ckpt)
    assert sha == sel['checkpoint_sha256'], f'{ckpt} is not the selected checkpoint'
    model, meta = load_model(root, ckpt)
    cost = profile(model)
    assert cost['params'] == sel['params']
    path = FULL / f'{args.shard}.npz'
    solutions, clues = arrays(path)
    rows = np.array(json.loads((FULL / 'index.json').read_text())['shards'][args.shard]['official_rows'])
    assert len(rows) == len(solutions)
    if args.n:
        solutions, clues, rows = solutions[:args.n], clues[:args.n], rows[:args.n]
    settings = {**sel['sampler'], 'batch_size': args.batch}
    sampler = SamplerConfig(**settings)
    it = sampler.held_iters
    assert sampler.kind == 'held_t' and it == sampler.steps
    calls = [0]
    model.register_forward_pre_hook(lambda _m, a: calls.__setitem__(0, calls[0] + len(a[0])))
    model.eval()
    with torch.no_grad(), torch.random.fork_rng(devices=[0]):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = run_sampler(model, solutions, clues, sampler, lut_a2g=meta['lut_a2g'], vocab_size=10, device='cuda')
        torch.cuda.synchronize()
        seconds = time.perf_counter() - t0
    assert np.array_equal(pred[clues.astype(bool)], solutions[clues.astype(bool)]), 'a clue changed'
    assert calls[0] == len(solutions) * (it + 1), (calls[0], len(solutions))
    solved = np.all(pred == solutions, axis=1)
    res = dict(method='FRM', seed=args.seed, shard=args.shard, n=int(len(solved)), solved=int(solved.sum()),
               accuracy=float(solved.mean()), seconds=round(seconds, 1), seconds_per_puzzle=seconds / len(solved),
               nfe=calls[0] / len(solved), flops_per_puzzle=calls[0] / len(solved) * cost['flops_per_forward'],
               checkpoint=str(ckpt), checkpoint_sha256=sha, step=sel['step'], stage=sel['stage'],
               sampler=asdict(sampler), data=str(path), data_sha256=digest(path),
               peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9, gpu=torch.cuda.get_device_name(),
               slurm_job=os.environ.get('SLURM_JOB_ID'))
    if args.check_campaign_test:
        assert args.shard == 'full-frmtest' and not args.n
        ref_sol, ref_clue = arrays(root / 'data/test.npz')
        assert np.array_equal(ref_sol, solutions) and np.array_equal(ref_clue, clues), 'full-frmtest != campaign test.npz'
        ref = np.load(root / f'runs/s{args.seed}/selected/test/predictions.npz')
        res['campaign_test_check'] = dict(batch=args.batch, predictions_equal=bool(np.array_equal(ref['predictions'], pred)),
                                          solved_equal=bool(np.array_equal(ref['solved'], solved)),
                                          campaign_solved=int(ref['solved'].sum()))
    out = Path(args.out) if args.out else root / f'runs/s{args.seed}/unseen/{args.shard}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'), solved=solved, official_rows=rows)
    out.write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: res[k] for k in ('seed', 'shard', 'n', 'solved', 'accuracy', 'seconds', 'seconds_per_puzzle', 'nfe', 'peak_memory_gb')}
                     | ({'campaign_test_check': res['campaign_test_check']} if 'campaign_test_check' in res else {})), flush=True)


def aggregate(args):
    """Check that the 17 shard records cover exactly the unseen rows; write <root>/score-unseen.json and the flags."""
    root = args.root.resolve()
    sel = selection(root, args.seed)
    index = json.loads((FULL / 'index.json').read_text())
    official = np.load(OFFICIAL)
    assert digest(OFFICIAL) == index['official_sha256']
    o_sol, o_clue = official['solutions'], official['clue_mask']
    dev = np.array(json.loads(DEV10K.read_text()))
    assert len(o_sol) == index['n_official'] == 422786 and len(set(dev.tolist())) == 10000
    assert sorted(index['shards']['full-dev10k']['official_rows']) == sorted(dev.tolist())
    recs, flags, rows = [], [], []
    for name in UNSEEN_SHARDS:
        r = json.loads((root / f'runs/s{args.seed}/unseen/{name}.json').read_text())
        f = np.load(root / f'runs/s{args.seed}/unseen/{name}.npz')
        assert r['checkpoint_sha256'] == sel['checkpoint_sha256'] and r['n'] == index['shards'][name]['n'] == len(f['solved'])
        assert r['data_sha256'] == digest(FULL / f'{name}.npz')
        g = np.load(FULL / f'{name}.npz')
        assert np.array_equal(g['solutions'], o_sol[f['official_rows']]) and np.array_equal(g['clue_mask'], o_clue[f['official_rows']]), name
        recs.append(r)
        flags.append(f['solved'])
        rows.append(f['official_rows'])
    rows, flags = np.concatenate(rows), np.concatenate(flags)
    unseen = np.setdiff1d(np.arange(len(o_sol)), dev)
    assert len(rows) == len(set(rows.tolist())) == 412786 and np.array_equal(np.sort(rows), unseen), 'not the unseen rows'
    samplers = {json.dumps({k: v for k, v in r['sampler'].items()}, sort_keys=True) for r in recs}
    assert len(samplers) == 1, samplers
    sampler = recs[0]['sampler']
    for k, v in sel['sampler'].items():
        assert k == 'batch_size' or sampler[k] == v, (k, sampler[k], v)
    order = np.argsort(rows)
    np.savez_compressed(root / 'score-unseen.solved.npz', official_rows=rows[order], solved=flags[order])
    res = dict(method='FRM', seed=args.seed, solved=int(flags.sum()), n=int(len(flags)), exact=float(flags.mean()),
               split='official Sudoku-Extreme test rows outside dev10k (412,786 unseen)', sampler=sampler,
               selection_batch_size=sel['sampler']['batch_size'], checkpoint=sel['checkpoint'],
               checkpoint_sha256=sel['checkpoint_sha256'], step=sel['step'], stage=sel['stage'], nfe=recs[0]['nfe'],
               flops_per_puzzle=recs[0]['flops_per_puzzle'], gpu_seconds=round(sum(r['seconds'] for r in recs), 1),
               shards={r['shard']: dict(n=r['n'], solved=r['solved'], seconds=r['seconds'], slurm_job=r['slurm_job']) for r in recs},
               frm_test_1000=dict(solved=recs[0]['solved'], n=recs[0]['n']),
               flags='score-unseen.solved.npz (official row indices ascending)', script=str(Path(__file__).resolve()),
               script_sha256=digest(__file__))
    (root / 'score-unseen.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: res[k] for k in ('seed', 'solved', 'n', 'exact', 'step', 'gpu_seconds')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['shard', 'aggregate'])
    p.add_argument('root', type=Path)  # the campaign folder, e.g. baselines/runs/frm-compiled-s1/sudoku
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--shard', default='')
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--n', type=int, default=0)  # > 0: only the first n puzzles of the shard (benchmarks)
    p.add_argument('--out', default='')
    p.add_argument('--check-campaign-test', action='store_true')
    a = p.parse_args()
    shard(a) if a.mode == 'shard' else aggregate(a)
