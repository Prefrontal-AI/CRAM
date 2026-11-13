"""GRAM on a test split: sampled trajectories, recorded after several step counts, for majority voting.

GRAM's evaluation (paper Sec. 4.1; evaluators/sudoku.py): each sample is one trajectory of the trained model in eval
mode, which draws the latent from the learned prior at every high-level cycle and never halts early (the Q-halt test
applies only in training), so the answer after step k of a K-step run is the answer of a k-step run. N samples are N
independent trajectories; the N decoded answers are aggregated by majority vote, ties broken by the mean sigmoid of
the halting logit (q_halt), as GRAM's evaluator does. No verifier and no value head select answers.

One run records, for each test puzzle, each sample in [first, last) and each step count in --steps: whether the
decoded answer is exactly right (every labelled cell, as trm_bench.py score), a 64-bit hash of the decoded answer (all
cells; equal answers have equal hashes) and q_halt. Samples are split over jobs: sample s uses the torch seed
seed_base + s, so the shards of one checkpoint are independent and reproducible. `merge` builds the tables.

Run from the GRAM source tree (ahn-ml/gram-release f2e32b9) with experiments/runs/gram/pydeps on PYTHONPATH, the model compiled as pretrain.py compiles it:
  python gram/score.py run <out.npz> <checkpoint.pt> <first> <last> <steps,comma> <hydra overrides...>
  python gram/score.py merge <out.json> <shard.npz>...
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path.cwd()
sys.path.insert(0, str(SRC))
SEED_BASE = 1_000_000  # sample s of a checkpoint uses seed SEED_BASE + s; disjoint from the training seeds 0-2


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(overrides, checkpoint, max_steps):
    import pretrain
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(SRC / 'config'), version_base=None):
        hcfg = compose(config_name='cfg_pretrain', overrides=list(overrides) + [f'arch.halt_max_steps={max_steps}'])
    cfg = pretrain.load_synced_config(hcfg, rank=0, world_size=1)
    cfg.checkpoint_path = None
    cfg.load_checkpoint = None
    loader, meta = pretrain.create_dataloader(cfg, 'test', test_set_mode=True, epochs_per_iter=1,
                                              global_batch_size=cfg.global_batch_size, rank=0, world_size=1)
    os.environ.pop('DISABLE_COMPILE', None)  # compiled, as pretrain.py compiles it
    model, _, _ = pretrain.create_model(cfg, meta, rank=0, world_size=1)
    ck = torch.load(checkpoint, map_location='cuda', weights_only=False)
    sd = {k.removeprefix('_orig_mod.'): v for k, v in ck['model'].items()}
    getattr(model, '_orig_mod', model).load_state_dict(sd, assign=True)
    model.eval()
    return model, loader, ck.get('step')


def run(out, checkpoint, first, last, steps, overrides):
    from models.losses import IGNORE_LABEL_ID
    steps = sorted(steps)
    model, loader, ck_step = load(overrides, checkpoint, max(steps))
    weights = torch.randint(-2**62, 2**62, (4096,), generator=torch.Generator().manual_seed(12345), dtype=torch.int64).cuda()
    rows = {s: dict(right=[], hash=[], q=[]) for s in range(first, last)}
    t0 = time.perf_counter()
    with torch.inference_mode(), torch.device('cuda'):
        for s in range(first, last):
            torch.manual_seed(SEED_BASE + s)
            for b, (_, batch, _) in enumerate(loader):
                if b == int(os.environ.get('GRAM_SCORE_BATCHES', 10**9)):  # smoke tests only
                    break
                batch = {k: v.cuda() for k, v in batch.items()}
                labels = batch['labels']
                mask = labels != IGNORE_LABEL_ID
                real = mask.any(-1)  # padding rows of the last batch have no label
                carry = model.initial_carry(batch)
                r, h, q = [], [], []
                for k in range(1, max(steps) + 1):
                    carry, _, _, preds, done = model(carry=carry, batch=batch, return_keys=['preds', 'q_halt_logits'])
                    if k in steps:
                        p = preds['preds']
                        r.append(((p == labels) | ~mask).all(-1)[real])
                        h.append((p.long() * weights[:p.shape[-1]]).sum(-1)[real])
                        q.append(torch.sigmoid(preds['q_halt_logits'].float())[real])
                    assert bool(done) == (k == max(steps)), (k, bool(done))
                rows[s]['right'].append(torch.stack(r, -1).cpu().numpy())
                rows[s]['hash'].append(torch.stack(h, -1).cpu().numpy())
                rows[s]['q'].append(torch.stack(q, -1).cpu().numpy().astype(np.float16))
            print(f'sample {s} done, {time.perf_counter() - t0:.0f} s', flush=True)
    stack = lambda key: np.stack([np.concatenate(rows[s][key]) for s in range(first, last)], 1)  # (puzzles, samples, steps)
    np.savez(out, right=stack('right'), hash=stack('hash'), q=stack('q'), samples=np.arange(first, last), steps=np.array(steps))
    meta = dict(checkpoint=str(checkpoint), checkpoint_step=ck_step, checkpoint_sha256=digest(checkpoint), overrides=list(overrides),
                samples=[first, last], steps=steps, seed_base=SEED_BASE, seconds=time.perf_counter() - t0,
                gpu=torch.cuda.get_device_name(), script=str(Path(__file__).resolve()), script_sha256=digest(__file__))
    Path(out).with_suffix('.json').write_text(json.dumps(meta, indent=1) + '\n')
    print(json.dumps(meta), flush=True)


def majority(hashes, q):
    """Majority vote per puzzle over the samples axis: index of a sample of the chosen answer. hashes, q: (puzzles, n).
    The key is (votes, mean q_halt of those votes), compared in that order: 2 * votes + mean, as 0 <= mean <= 1."""
    same = hashes[:, :, None] == hashes[:, None, :]  # (puzzles, n, n)
    votes = same.sum(-1)
    mean_q = (same * q.astype(np.float64)[:, None, :]).sum(-1) / votes
    return np.argmax(2 * votes + mean_q, 1)


def merge(out, shards, ns=(1, 2, 5, 10, 20)):
    parts = [np.load(p) for p in shards]
    order = np.argsort([int(p['samples'][0]) for p in parts])
    parts = [parts[i] for i in order]
    samples = np.concatenate([p['samples'] for p in parts])
    assert np.array_equal(samples, np.arange(samples.size)), samples
    steps = parts[0]['steps']
    assert all(np.array_equal(p['steps'], steps) for p in parts)
    right = np.concatenate([p['right'] for p in parts], 1)
    hashes = np.concatenate([p['hash'] for p in parts], 1)
    q = np.concatenate([p['q'] for p in parts], 1)
    flags = {}
    res = dict(n=int(right.shape[0]), samples=int(samples.size), steps=steps.tolist(), shards=[str(p) for p in shards],
               shard_sha256=[digest(p) for p in shards], script=str(Path(__file__).resolve()), script_sha256=digest(__file__),
               rule='majority vote over the first N samples (decoded-answer hash), ties by mean sigmoid(q_halt)', table={})
    for a, k in enumerate(steps.tolist()):
        for n in ns:
            if n > samples.size:
                continue
            c = majority(hashes[:, :n, a], q[:, :n, a]) if n > 1 else np.zeros(right.shape[0], np.int64)
            ok = right[np.arange(right.shape[0]), c, a]
            res['table'][f'steps{k}-n{n}'] = dict(steps=k, n=n, solved=int(ok.sum()), exact=float(ok.mean()))
            flags[f'steps{k}_n{n}'] = ok
    np.savez_compressed(Path(out).with_suffix('.right.npz'), **flags)
    Path(out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: round(v['exact'], 5) for k, v in res['table'].items()}), flush=True)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'run':
        run(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), [int(x) for x in sys.argv[6].split(',')], sys.argv[7:])
    elif mode == 'merge':
        merge(sys.argv[2], sys.argv[3:])
    else:
        raise SystemExit(__doc__)
