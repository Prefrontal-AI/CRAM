"""Inference flop per puzzle of TRMP (the reimplementation in baselines/trmp/; no upstream code).

Per-rollout costs by FlopCounterMode (multiply-add = 2; matmuls only, as the TRM and FRM rows), batch 1, eval mode:
one supervision step (model.step: cycles x rounds of message passing and the answer GRU), the halting head (run after
every step of a continuing rollout, infer.best_of), the clue embedding and the symbol readout (once per rollout).
Per puzzle = 64 x (embedding + readout) + (sum of the realized steps of its 64 rollouts) x (step + halting head), with
the steps that score.py saved for each test rollout (<run>/score/test/*.npz, `steps`, shape (puzzles, 64)).
A run whose test scoring has not finished contributes no total.

  ../.venv/bin/python flops/trmp_count.py --out runs/flops/trmp.json      # from baselines/ (one GPU or CPU)

--full (the default above is unchanged): every component is counted with flops/fullcount.py
FullCounter (arithmetic only; comparisons and selections recorded and reported as the with_compare sensitivity total), and the counted region is the whole of infer.best_of, line by line (`full_parts`):
- input_check (per puzzle): best_of's `(x.sum(-1) > 0).all()`;
- embedding (per rollout): `model.embedding`;
- step (per realized step): `model.step`;
- halt (per realized step): `depth[active] += 1`, q = -logsumexp(-halt_logits), the finiteness test, `q > 0`,
  `done.any()`, `~done`, `remaining.any()`; at the step cap best_of sets done = ones_like (no `q > 0`), so a rollout
  that reaches the cap saves `cap_correction` = halt - halt_at_cap (0 arithmetic; 1 comparison in the with_compare total);
- readout (per rollout, once, at its halting step): symbol logits, finiteness test, the allowed-symbol mask and argmax;
- noise (per continuing rollout step, i.e. realized steps - 1 per rollout): `a + sigma * randn` (the draws count 0,
  recorded as rng);
- select (per puzzle): argmax of the 64 terminal confidences.
Per puzzle = input_check + 64 (embedding + readout) + S (step + halt) - C cap_correction + (S - 64) noise + select,
S = sum of the 64 realized steps, C = rollouts that reached the cap (steps from <run>/score/test/*.npz).
Components run on an untrained model with batch 1 (one rollout), as the default count; the counts do not depend on the
weights. `validate` checks the formula end to end: FullCounter over infer.best_of itself (trained model.pt, the run's
sigma, bf16 autocast as score.py) on dev puzzles, at the Table-2 cap and at caps 2 and 1 (rollouts stopped by the cap), must equal the formula with
the depths best_of returned. Writes one record per benchmark: <full_dir>/trmp-<task>.json.

  ../.venv/bin/python flops/trmp_count.py --full     # from baselines/, on a GPU node
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import tyro

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'trmp'))
from infer import matrix_count  # noqa: E402
from model import RECIPES, Trainable  # noqa: E402

TASKS = {'sudoku-extreme': 'Sudoku-Extreme', 'maze-unique': 'Maze-Unique'}


@dataclasses.dataclass
class Args:
    out: str = 'runs/flops/trmp.json'  # result JSON (relative to baselines/)
    runs: str = 'runs/trmp'  # folder of the fits <task>-s<seed>
    seeds: tuple[int, ...] = (0, 1, 2)  # training seeds whose test step counts are read when they exist
    full: bool = False  # count all arithmetic of infer.best_of with FullCounter (see the docstring) instead
    full_dir: str = 'runs/flops-full'  # --full: folder of the records trmp-<task>.json (relative to baselines/)
    validate_puzzles: int = 2  # --full: dev puzzles run through best_of under FullCounter for the end-to-end check


def count(fn):
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        result = fn()
    return int(fc.get_total_flops()), {str(k): int(v) for k, v in fc.get_flop_counts()['Global'].items()}, result


@torch.no_grad()
def per_rollout(task):
    """Counted flop of each part of one rollout (batch 1)."""
    r = RECIPES[task]
    model = Trainable(task).eval()
    x = torch.ones(1, r['cells'], r['symbols'])
    emb, emb_ops, clues = count(lambda: model.embedding(x))
    state = model.initial(x)
    step, step_ops, state = count(lambda: model.step(state, clues))
    halt, halt_ops, _ = count(lambda: model.halt_logits(state[1]))
    readout, readout_ops, _ = count(lambda: model.symbol(state[1]))
    formula = matrix_count(r)
    return dict(embedding=emb, step=step, halt_head=halt, readout=readout, ops=dict(embedding=emb_ops, step=step_ops,
                halt_head=halt_ops, readout=readout_ops), step_formula_infer_matrix_count=formula,
                step_matches_formula=step == formula, params=sum(p.numel() for p in model.parameters()),
                rollouts=r['rollouts'], step_cap=r['inference_steps'])


def totals(run, parts):
    """Per-puzzle flop from the saved test step counts of one run, or None if its test scoring is incomplete."""
    test_json = run / 'score' / 'test.json'
    if not test_json.exists():
        return None
    rec = json.loads(test_json.read_text())
    shards = sorted((run / 'score' / 'test').glob('*.npz'))
    steps = np.concatenate([np.load(p)['steps'] for p in shards]).astype(np.int64)
    assert steps.shape == (rec['count'], parts['rollouts']), steps.shape
    assert steps.min() >= 1 and steps.max() <= parts['step_cap']
    per_step = parts['step'] + parts['halt_head']
    flop = parts['rollouts'] * (parts['embedding'] + parts['readout']) + steps.sum(1) * per_step
    return dict(run=str(run), puzzles=int(len(flop)), solved=rec['solved'], sigma=rec['sigma'],
                model_sha256=rec['model_sha256'], mean_rollout_steps=float(steps.mean()),
                mean_steps_per_puzzle=float(steps.sum(1).mean()), capped_rollouts=int((steps == parts['step_cap']).sum()),
                mean_flop_per_puzzle=float(flop.mean()), sd_flop_per_puzzle=float(flop.std(ddof=1)),
                min_flop=int(flop.min()), max_flop=int(flop.max()),
                test_json_sha256=hashlib.sha256(test_json.read_bytes()).hexdigest(),
                shards_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in shards})


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def full_parts(task, sigma=0.1, device='cpu'):
    """FullCounter count (and FlopCounterMode's, matmuls only) of each component of infer.best_of, one rollout."""
    from torch.utils.flop_counter import FlopCounterMode
    from fullcount import FullCounter
    r = RECIPES[task]
    model = Trainable(task).to(device).eval()
    x = torch.ones(1, r['cells'], r['symbols'], device=device)
    parts = {}

    def run(name, fn):
        with torch.no_grad():
            with FlopCounterMode(display=False) as old:
                fn()
            with FullCounter() as fc:
                out = fn()
        s = fc.summary()
        parts[name] = dict(full=s['total'], full_with_compare=s['total_with_compare'], flopcountermode_matmul_only=int(old.get_total_flops()), classes=s['classes'],
                           by_op=s['by_op'], matmul_class_equals_flopcountermode=s['classes']['matmul'] == int(old.get_total_flops()))
        return out

    run('input_check', lambda: (x.sum(-1) > 0).all())
    clues = run('embedding', lambda: model.embedding(x))
    state = run('step', lambda: model.step(model.initial(x), clues))
    active = torch.arange(1, device=device)

    def halt(at_cap):
        depth = torch.zeros(1, dtype=torch.long, device=device)
        depth[active] += 1
        q = -torch.logsumexp(-model.halt_logits(state[1]).float(), dim=-1)
        if not torch.isfinite(q).all():
            raise ValueError('Nonfinite halting confidence')
        done = torch.ones_like(q, dtype=torch.bool) if at_cap else (q > 0)
        done.any()
        remaining = ~done
        remaining.any()
        return q
    q = run('halt', lambda: halt(False))
    run('halt_at_cap', lambda: halt(True))

    def readout():
        done = torch.ones(1, dtype=torch.bool, device=device)
        finished = active[done]
        logits = model.symbol(state[1][done]).float()
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite symbol logits')
        return logits.masked_fill(x[finished] == 0, -torch.inf).argmax(-1)
    run('readout', readout)
    gen = torch.Generator(device=device).manual_seed(0)

    def noise():
        a = state[1]
        return a + sigma * torch.randn(a.shape, dtype=a.dtype, device=a.device, generator=gen)
    run('noise', noise)
    conf = torch.zeros(1, r['rollouts'], device=device)
    run('select', lambda: conf.argmax(-1))
    return parts


def full_formula(c, steps, cap):
    """Per-puzzle flop (int64 array) from the component totals c and realized steps (puzzles, rollouts)."""
    k = steps.shape[1]
    S, C = steps.sum(1), (steps == cap).sum(1)
    corr = c['halt'] - c['halt_at_cap']
    return (c['input_check'] + k * (c['embedding'] + c['readout']) + S * (c['step'] + c['halt']) - C * corr
            + (S - k) * c['noise'] + c['select'])


def validate(task, run, c, cw, n, cap):
    """FullCounter over infer.best_of on n dev puzzles; equals full_formula with the depths best_of returned?"""
    from fullcount import FullCounter
    sys.path.insert(0, str(HERE.parent / 'trmp'))
    import fit as trmp_fit
    import score
    from infer import best_of
    model = Trainable(task).cuda().eval()
    model.load_state_dict(torch.load(run / 'model.pt', map_location='cuda', weights_only=True))
    sigma = json.loads((run / 'score' / 'selection.json').read_text())['sigma']
    x = score.load_split(task, 'dev')[0][:n].cuda()
    gen = torch.Generator(device='cuda').manual_seed(score.RNG_BASE)
    with FullCounter() as fc, trmp_fit.amp_context(True):
        _, work = best_of(model, x, rollouts=score.ROLLOUTS, steps=cap, sigma=sigma, generator=gen)
    steps = work['rollout_steps'].cpu().numpy().astype(np.int64)
    formula, formula_w = int(full_formula(c, steps, cap).sum()), int(full_formula(cw, steps, cap).sum())
    return dict(run=str(run), model_sha256=sha256(run / 'model.pt'), sigma=sigma, puzzles=n, cap=cap,
                rollout_steps=steps.tolist(), counted_best_of=fc.get_total_flops(), formula=formula,
                counted_best_of_with_compare=fc.total_with_compare(), formula_with_compare=formula_w,
                equal=fc.get_total_flops() == formula and fc.total_with_compare() == formula_w, classes=fc.classes())


def main_full(args: Args):
    old = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else None
    for task, name in TASKS.items():
        r = RECIPES[task]
        parts = full_parts(task, device='cuda' if torch.cuda.is_available() else 'cpu')
        c = {k: v['full'] for k, v in parts.items()}
        cw = {k: v['full_with_compare'] for k, v in parts.items()}
        old_parts = old['tasks'][task]['per_rollout'] if old else None
        checks = {k: v['matmul_class_equals_flopcountermode'] for k, v in parts.items()}
        if old_parts:
            for k, o in (('embedding', 'embedding'), ('step', 'step'), ('halt', 'halt_head'), ('readout', 'readout')):
                checks[f'{k}_matmul_class_equals_old_record_{o}'] = parts[k]['classes']['matmul'] == old_parts[o]
        assert all(checks.values()), checks
        assert c['halt'] - c['halt_at_cap'] == 0 and cw['halt'] - cw['halt_at_cap'] == 1, 'the cap step skips one comparison'
        val = [validate(task, Path(args.runs) / f'{task}-s0', c, cw, args.validate_puzzles, cap) for cap in (r['inference_steps'], 2, 1)]
        assert all(v['equal'] for v in val), [(v['counted_best_of'], v['formula']) for v in val]
        seeds = {}
        for k in args.seeds:
            run = Path(args.runs) / f'{task}-s{k}'
            if not (run / 'score' / 'test.json').exists():
                seeds[str(k)] = None
                continue
            rec = json.loads((run / 'score' / 'test.json').read_text())
            shards = sorted((run / 'score' / 'test').glob('*.npz'))
            steps = np.concatenate([np.load(p)['steps'] for p in shards]).astype(np.int64)
            assert steps.shape == (rec['count'], r['rollouts'])
            new = full_formula(c, steps, r['inference_steps'])
            new_w = full_formula(cw, steps, r['inference_steps'])
            oldf = None
            if old_parts:
                per_step = old_parts['step'] + old_parts['halt_head']
                oldf = r['rollouts'] * (old_parts['embedding'] + old_parts['readout']) + steps.sum(1) * per_step
            seeds[str(k)] = dict(run=str(run), puzzles=int(len(new)), sigma=rec['sigma'], model_sha256=rec['model_sha256'],
                                 mean_steps_per_puzzle=float(steps.sum(1).mean()), capped_rollouts=int((steps == r['inference_steps']).sum()),
                                 mean_flop_per_puzzle_full=float(new.mean()), mean_flop_per_puzzle_full_with_compare=float(new_w.mean()),
                                 mean_flop_per_puzzle_old=None if oldf is None else float(oldf.mean()),
                                 old_record_mean_flop_per_puzzle=(old['tasks'][task]['per_seed'].get(str(k)) or {}).get('mean_flop_per_puzzle') if old else None,
                                 ratio=None if oldf is None else float(new.mean() / oldf.mean()),
                                 test_json_sha256=sha256(run / 'score' / 'test.json'),
                                 shards_sha256={p.name: sha256(p) for p in shards})
        done = [v for v in seeds.values() if v]
        res = dict(method='TRMP', benchmark=name, task=task,
                   components={k: dict(v, per=per) for (k, v), per in zip(parts.items(), (
                       'puzzle', 'rollout', 'realized step', 'realized step', 'realized step at the cap (replaces halt)',
                       'rollout', 'continuing rollout step (steps - 1 per rollout)', 'puzzle'))},
                   cap_correction=c['halt'] - c['halt_at_cap'], cap_correction_with_compare=cw['halt'] - cw['halt_at_cap'],
                   formula_full='input_check + 64 (embedding + readout) + S (step + halt) - C (halt - halt_at_cap) + (S - 64) noise + select; '
                                'S = sum of the 64 realized rollout steps, C = rollouts that reached the cap; steps from <run>/score/test/*.npz',
                   formula_old='64 (embedding + readout) + S (step + halt_head), FlopCounterMode components (runs/flops/trmp.json)',
                   sweep_components=dict(per_step=c['step'] + c['halt'], per_rollout=c['embedding'] + c['readout'],
                                         noise_per_continuing_step=c['noise'], cap_correction=c['halt'] - c['halt_at_cap'],
                                         select_per_rollout_considered=c['select'] // r['rollouts'], input_check=c['input_check'],
                                         with_compare=dict(per_step=cw['step'] + cw['halt'], per_rollout=cw['embedding'] + cw['readout'],
                                                           noise_per_continuing_step=cw['noise'], cap_correction=cw['halt'] - cw['halt_at_cap'],
                                                           select_per_rollout_considered=cw['select'] // r['rollouts'], input_check=cw['input_check']),
                                         formula='K rollouts, cap c, u = min(steps, c) of those K: input_check + K per_rollout + sum(u) per_step '
                                                 '- #(steps >= c) cap_correction + (sum(u) - K) noise + K select_per_rollout_considered'),
                   checks=checks, validation=val, per_seed=seeds, seeds_done=[k for k, v in seeds.items() if v],
                   flop_per_puzzle_full=float(np.mean([v['mean_flop_per_puzzle_full'] for v in done])) if done else None,
                   flop_per_puzzle_full_with_compare=float(np.mean([v['mean_flop_per_puzzle_full_with_compare'] for v in done])) if done else None,
                   flop_per_puzzle_old=old['tasks'][task]['flop_per_puzzle'] if old else None,
                   convention='baselines/flops/fullcount.py FullCounter (arithmetic only: matmul + arith + transc; the with_compare totals add class compare), batch 1 (one rollout) per component, eval mode, no_grad; '
                              'no custom or compiled kernel runs in TRMP (plain torch.nn modules, eager)',
                   command=' '.join([Path(sys.executable).name] + sys.argv), script_sha256=sha256(__file__),
                   fullcount_sha256=sha256(HERE / 'fullcount.py'), torch=torch.__version__,
                   created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), host=os.uname().nodename,
                   slurm_job=os.environ.get('SLURM_JOB_ID'))
        res['ratio'] = res['flop_per_puzzle_full'] / res['flop_per_puzzle_old'] if done and old else None
        res['ratio_with_compare'] = res['flop_per_puzzle_full_with_compare'] / res['flop_per_puzzle_old'] if done and old else None
        out = Path(args.full_dir) / f'trmp-{task}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1) + '\n')
        print(task, c, cw, res['flop_per_puzzle_old'], res['flop_per_puzzle_full'], res['ratio'], res['ratio_with_compare'],
              [(v['counted_best_of'], v['formula'], v['counted_best_of_with_compare'], v['formula_with_compare']) for v in val], flush=True)


def main(args: Args):
    if args.full:
        return main_full(args)
    out = {}
    for task, name in TASKS.items():
        parts = per_rollout(task)
        seeds = {}
        for k in args.seeds:
            t = totals(Path(args.runs) / f'{task}-s{k}', parts)
            seeds[k] = t
        done = [t for t in seeds.values() if t is not None]
        out[task] = dict(benchmark=name, per_rollout=parts,
                         per_step_with_halt=parts['step'] + parts['halt_head'],
                         per_seed={str(k): v for k, v in seeds.items()},
                         seeds_done=[k for k, v in seeds.items() if v is not None],
                         flop_per_puzzle=float(np.mean([t['mean_flop_per_puzzle'] for t in done])) if done else None)
    res = dict(method='TRMP', tasks=out, convention='FlopCounterMode, batch 1 per rollout, eval mode; multiply-add = 2; '
               'matmuls only (activations, normalization, pooling, noise not counted, as the TRM/FRM rows)',
               formula='64 x (embedding + readout) + sum of realized rollout steps x (step + halting head)',
               script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), host=os.uname().nodename,
               slurm_job=os.environ.get('SLURM_JOB_ID'))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({t: {k: v for k, v in d.items() if k != 'per_seed'} for t, d in out.items()}, indent=1))


if __name__ == '__main__':
    main(tyro.cli(Args))
