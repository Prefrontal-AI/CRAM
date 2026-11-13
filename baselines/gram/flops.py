"""GRAM's inference flop per trajectory: baselines/trm/trm_bench.py `flops` (FlopCounterMode, math SDPA, batch 1,
eval mode, halt_max_steps ACT steps) on GRAM's model, unchanged except one thing during counting only: GRAM's attention
(models/layers.py) rearranges the SDPA output from [b, h, s, d] to [b, s, h, d] and calls .view, which needs the
[b, s, h, d] memory layout that the fused kernels return; the math kernel returns [b, h, s, d], so its output is copied
into that layout (a copy; no arithmetic). One trajectory; majority@N costs N trajectories (the vote is a comparison).

Run from the GRAM source tree with experiments/runs/gram/pydeps on PYTHONPATH:
  python gram/flops.py <out.json> <hydra overrides...>
  python gram/flops.py --full <out.json> <hydra overrides...>

--full (the count by the paper's convention, every arithmetic operation): trm_bench.bench_flops_full (FullCounter,
baselines/flops/fullcount.py; prediction pipeline only, loss head excluded) for one trajectory, plus the sample selection
of the paper's majority@N, counted with FullCounter on a torch port of gram/score.py's selection (per sample: the
64-bit answer hash and sigmoid(q_halt); then `majority`, ties by mean sigmoid(q_halt)). The port is checked against
score.majority on random answers with ties. Per puzzle at N samples: N x trajectory + N x (hash + sigmoid) + vote(N).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'baselines/trm'))
import models.layers as layers  # noqa: E402

_sdpa = layers.scaled_dot_product_attention
layers.scaled_dot_product_attention = lambda *a, **k: _sdpa(*a, **k).transpose(1, 2).contiguous().transpose(1, 2)
from trm_bench import bench_flops  # noqa: E402

N_SAMPLES = 20  # the paper's main-table setting, majority@20


def majority_torch(hashes, q):
    """gram/score.py `majority` in torch (same operations), so that FullCounter sees the vote. hashes, q: (puzzles, n)."""
    same = hashes[:, :, None] == hashes[:, None, :]
    votes = same.sum(-1)
    mean_q = (same * q.double()[:, None, :]).sum(-1) / votes
    return torch.argmax(2 * votes + mean_q, 1)


def selection(rec, outputs, batch, model, n=N_SAMPLES):
    """Count the majority@n selection of one puzzle and assemble the per-puzzle total."""
    import numpy as np
    from fullcount import FullCounter
    from score import majority
    from trm_bench import combine
    g = torch.Generator().manual_seed(0)
    for _ in range(200):  # the port equals score.majority, ties included
        h = torch.randint(0, 4, (64, n), generator=g)
        q = torch.rand(64, n, generator=g, dtype=torch.float64).to(torch.float16).double()
        assert np.array_equal(majority_torch(h, q).numpy(), majority(h.numpy(), q.numpy()))
    weights = torch.randint(-2**62, 2**62, (4096,), generator=torch.Generator().manual_seed(12345), dtype=torch.int64).cuda()
    preds = torch.argmax(outputs['logits'], dim=-1)
    with torch.no_grad():  # as trm_bench.bench_flops_full: composites decompose before the counter
        with FullCounter() as c:  # per sample, as score.py run: answer hash and sigmoid(q_halt)
            hsh = (preds.long() * weights[:preds.shape[-1]]).sum(-1)
            qs = torch.sigmoid(outputs['q_halt_logits'].float())
        per_sample = c.summary()
        hashes = torch.cat([hsh] * (n - 1) + [hsh + 1], 0)[None]  # (1, n): shapes as one puzzle's n samples
        qq = torch.cat([qs] * n, 0)[None].to(torch.float16)
        with FullCounter() as c:
            majority_torch(hashes, qq)
        vote = c.summary()
    traj = rec['components']['trajectory']
    total = combine([(traj['full'], n), (per_sample, n), (vote, 1)])
    rec['components']['selection_per_sample'] = dict(full=per_sample, note='gram/score.py run: 64-bit hash of the decoded answer, sigmoid(q_halt)')
    rec['components']['vote'] = dict(full=vote, samples=n, note='gram/score.py majority (numpy at merge time), counted on its torch port')
    rec['components']['puzzle'] = dict(full=total, samples=n)
    rec['samples'] = n
    rec['flop_per_trajectory'] = traj['full']['total']
    rec['flop_per_puzzle'] = total['total']
    rec['flop_per_puzzle_with_compare'] = total['total_with_compare']
    rec['flop_per_trajectory_with_compare'] = traj['full']['total_with_compare']
    rec['flop_per_puzzle_old'] = traj['old_matmul_only'] * n
    rec['flop_per_puzzle_formula'] = (f'{n} x (16 x act_step + decode) + {n} x selection_per_sample + vote({n}); '
                                      f'old: {n} x 16 x act_step (matmul only; the vote was not counted)')
    rec['ratio_per_puzzle'] = total['total'] / rec['flop_per_puzzle_old']
    import hashlib
    rec['scripts_sha256'] = {str(Path(f).resolve()): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                             for f in (__file__, rec['script'], Path(__file__).parent / 'score.py')}


if __name__ == '__main__':
    if sys.argv[1] == '--full':
        sys.path.append(str(Path(__file__).resolve().parent))  # score.py (its majority)
        sys.path.append(str(Path(__file__).resolve().parents[1] / 'flops'))
        from trm_bench import bench_flops_full  # noqa: E402
        bench_flops_full(sys.argv[2], sys.argv[3:], finish=selection)
    else:
        bench_flops(sys.argv[1], sys.argv[2:])
