"""TRMP: one timed cold fit on one GPU, bf16 autocast and torch.compile, Sudoku-Extreme or Maze-Unique.

TRMP has no released code; this is a clean-room reconstruction of the ACT (adaptive computation time) recipe in
Appendix B of arXiv 2610.18649, and it calls no upstream code. The recipe: per-example gradients by vmap,
per-trajectory clipping against a slowly updated reference median (kappa = 50), global clip 0.5, step rejection above a pre-clip norm of 50, AdamW, zero-start EMA with bias
correction, 32-step persistent slots with staggered phases, and validation checkpoint selection.
Choices of this implementation (trmp/check.py compares one update with a plain fp32 torch.optim.AdamW reference):
- numerics: the forward and backward run under bf16 autocast; master weights, gradients, AdamW moments and EMA are
  fp32. One update (gradients, clipping, rejection, AdamW, EMA, reference median) is one torch.compile graph.
  AdamW is written out (the torch.optim.AdamW single-tensor equations) so that a rejected step is a masked update on
  the device instead of a host branch; `trmp/check.py` checks it against torch.optim.AdamW.
- a Sudoku path: 1,000 training puzzles, a fresh digit permutation per draw (B.4), 500 validation puzzles.
- the clock is baselines/clock.py; the rolling checkpoint is written every `ckpt_every` seconds inside an exclusion.

Dimension names: b slots, v cells, s symbols.

  python trmp/fit.py --task sudoku-extreme --out runs/trmp/sudoku-extreme-s0       # from baselines/
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
import tyro
from torch.func import functional_call, grad_and_value, vmap
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from clock import FitClock, cuda_context_seconds, write_json  # noqa: E402
from model import DORMANT, RECIPES, SOURCE, Trainable  # noqa: E402

V1_DATA = Path(os.environ.get('V1_DATA', ROOT / 'data'))
MAZE_VALIDATION = Path(os.environ.get('TRMP_MAZE_VALIDATION', ROOT / 'experiments/runs/TRMP-audit/generated-validation'))
LR, BETAS, EPS, WD, EMA, CLIP, REJECT, KAPPA, PHASES = 7.5e-4, (.9, .95), 1e-8, .04821, .9995, .5, 50., 50., 32


@dataclasses.dataclass
class Config:
    task: Literal['sudoku-extreme', 'maze-unique'] = 'sudoku-extreme'  # benchmark
    out: str = 'runs/trmp/sudoku-extreme-s0'  # output folder (fit.json, model.pt, log.jsonl, ckpt.pt)
    seed: int = 0  # training seed; Sudoku also uses it as the FRM training split train-s<seed>
    steps: Optional[int] = None  # updates to run (default: the recipe's 275,000 Sudoku / 120,000 Maze)
    schedule_steps: Optional[int] = None  # learning-rate schedule length (default: the recipe's updates)
    slots: Optional[int] = None  # persistent slots (default: the recipe's 512 Sudoku / 128 Maze)
    amp: bool = True  # bf16 autocast forward/backward (fp32 master weights and optimizer)
    compile: bool = True  # torch.compile one update
    validate_every: int = 5000  # updates between validation evaluations of the EMA model
    validate_from: int = 40000  # first update evaluated for checkpoint selection
    eval_steps: int = 128  # supervision steps of the fixed-compute validation readout
    dev_count: Optional[int] = None  # validation puzzles (default: 500 Sudoku, 100 generated Maze)
    ckpt_every: float = 1800.0  # seconds of wall clock between rolling checkpoints
    log_every: int = 500  # updates between log lines (each log line synchronizes the device)
    rows: Optional[str] = None  # Sudoku training rows: a TRM-format train folder (all__inputs.npy, all__labels.npy) instead of FRM's train-s<seed>; unused by the reported fits
    permute: bool = True  # Sudoku: a fresh digit permutation per draw (B.4); --no-permute draws the stored rows unchanged


# ---------------------------------------------------------------- data

def read_raw(cfg):
    """Read the training files into host memory (before the clock)."""
    if cfg.task == 'sudoku-extreme' and cfg.rows:
        # TRM tokens: inputs 1 = blank, d + 1 = given digit d; labels d + 1. Every stored row is one training board.
        d = Path(cfg.rows)
        inputs, labels = np.load(d / 'all__inputs.npy'), np.load(d / 'all__labels.npy')
        assert inputs.shape == labels.shape and inputs.shape[1] == 81 and labels.min() >= 2 and labels.max() <= 10
        clue = inputs > 1
        assert (inputs[clue] == labels[clue]).all(), 'a given differs from its label'
        h = hashlib.sha256()
        for f in ('all__inputs.npy', 'all__labels.npy'):
            h.update((d / f).read_bytes())
        return dict(solutions=labels.astype(np.int64) - 1, clue_mask=clue, path=str(d), sha256=h.hexdigest())
    if cfg.task == 'sudoku-extreme':
        path = V1_DATA / f'frm/split/train-s{cfg.seed}.npz'
        f = np.load(path)
        return dict(solutions=f['solutions'], clue_mask=f['clue_mask'], path=str(path),
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    sys.path.insert(0, str(ROOT / 'experiments'))
    sys.path.insert(0, str(ROOT))  # the cram package
    from cram.maze_unique import SHA256, _read
    return dict(inputs=_read('train/all__inputs.npy'), labels=_read('train/all__labels.npy'),
                path='EqR maze-30x30-unique-1k train', sha256=SHA256['train/all__inputs.npy'])


def encode_sudoku(solutions, clue_mask):
    """x (b, 81, 9): allowed digits (a given is one-hot, a blank all nine); target digit 0-8."""
    target = torch.as_tensor(np.asarray(solutions, dtype=np.int64) - 1)
    clue = torch.as_tensor(np.asarray(clue_mask, dtype=bool))
    x = torch.where(clue[..., None], F.one_hot(target, 9).float(), torch.ones(*target.shape, 9))
    return x, target


def encode_maze(tokens, target_path):
    """Maze input x (b, 900, 5): one-hot tokens, a free cell two-hot on free and path; target (b, 900) symbol index."""
    target = tokens.clone()
    free = tokens == 1
    target[free] = torch.where(target_path[free] > 0, 4, 1)
    x = F.one_hot(tokens, 5).float()
    x[..., 4] = free
    return x, target


def encode_train(cfg, raw):
    if cfg.task == 'sudoku-extreme':
        return encode_sudoku(raw['solutions'], raw['clue_mask'])
    tokens = torch.as_tensor(raw['inputs'].astype(np.int64) - 1).reshape(-1, 900)
    target_path = (raw['labels'] == 5) | (raw['inputs'] == 3) | (raw['inputs'] == 4)
    return encode_maze(tokens, torch.as_tensor(target_path).reshape(-1, 900))


def read_dev(cfg, count):
    """Validation boards for checkpoint selection (read and encoded inside an exclusion)."""
    if cfg.task == 'sudoku-extreme':
        f = np.load(V1_DATA / 'frm/dev10k/dev.npz')
        return encode_sudoku(f['solutions'][:count], f['clue_mask'][:count])
    manifest = json.loads((MAZE_VALIDATION / 'final.json').read_text())
    for kind in ('inputs', 'labels'):
        assert hashlib.sha256((MAZE_VALIDATION / f'dev/all__{kind}.npy').read_bytes()).hexdigest() == manifest[f'{kind}_sha256']
    vx = torch.as_tensor(np.load(MAZE_VALIDATION / 'dev/all__inputs.npy').astype(np.int64)) - 1
    vy = torch.as_tensor(np.load(MAZE_VALIDATION / 'dev/all__labels.npy').astype(np.int64)) - 1
    x, y = encode_maze(vx.reshape(-1, 900), (vy == 4).reshape(-1, 900))
    return x[:count], y[:count]


def sampled_batch(task, x, y, count, permute=True):
    """Draw `count` training boards, uniformly with replacement, with the task's augmentation."""
    ids = torch.randint(len(x), (count,), device=x.device)
    bx, by = x[ids].clone(), y[ids].clone()
    if task == 'sudoku-extreme' and not permute:
        return bx, by
    if task == 'sudoku-extreme':
        # Fresh digit permutation per draw (B.4): old digit d becomes perm[d].
        perm = torch.rand(count, 9, device=x.device).argsort(-1)
        inverse = perm.argsort(-1)
        bx = bx.gather(-1, inverse[:, None, :].expand_as(bx))
        by = perm.gather(1, by)
        return bx, by
    swap = torch.rand(count, 1, device=x.device) < .5
    start, end = bx[..., 2].clone(), bx[..., 3].clone()
    bx[..., 2] = torch.where(swap, end, start)
    bx[..., 3] = torch.where(swap, start, end)
    by = torch.where(swap & (by == 2), 3, torch.where(swap & (by == 3), 2, by))
    return bx, by


# ---------------------------------------------------------------- one update

def amp_context(amp):
    return torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext()


def make_update(model, names, amp):
    """One update as a pure function of tensors; mutates P (parameters), M, V, E, acc, ref, rejected in place."""
    decay = [model.get_parameter(n).ndim >= 2 for n in names]

    def loss_one(params, xx, yy, ss):
        with amp_context(amp):
            return functional_call(model, params, (xx.unsqueeze(0), yy.unsqueeze(0), tuple(v.unsqueeze(0) for v in ss)))

    derivative = vmap(grad_and_value(loss_one, has_aux=True), in_dims=(None, 0, 0, 0), randomness='error')

    def update(P, M, V, E, acc, ref, rejected, bx, by, state, lr):
        params = dict(zip(names, P))
        grads, (losses, (new_state, halt, ce)) = derivative(params, bx, by, state)
        G = [grads[n] for n in names]
        norms = sum(g.flatten(1).square().sum(1) for g in G).sqrt()
        ratios = norms / losses.clamp_min(1e-12)
        median = ratios.median()
        ref0 = torch.where(acc == 0, median, ref)
        scale = (KAPPA * ref0 / ratios.clamp_min(1e-12)).clamp(max=1)
        G = [(g * scale.reshape(-1, *([1] * (g.ndim - 1)))).mean(0) for g in G]
        total = torch.stack([g.square().sum() for g in G]).sum().sqrt()  # clip_grad_norm_ (L2, pre-clip norm)
        coef = (CLIP / (total + 1e-6)).clamp(max=1.0)
        ok = torch.isfinite(total) & (total <= REJECT)
        t = (acc + 1).float()
        bc1, bc2 = 1 - BETAS[0] ** t, 1 - BETAS[1] ** t
        for p, m, v, e, g, d in zip(P, M, V, E, G, decay):
            g = g * coef
            # torch.optim.AdamW, single-tensor: decoupled decay, lerp first moment, bias-corrected step.
            p_new = p * (1 - lr * WD) if d else p
            m_new = m.lerp(g, 1 - BETAS[0])
            v_new = v * BETAS[1] + (1 - BETAS[1]) * g * g
            p_new = p_new - (lr / bc1) * m_new / (v_new.sqrt() / bc2.sqrt() + EPS)
            p.copy_(torch.where(ok, p_new, p))
            m.copy_(torch.where(ok, m_new, m))
            v.copy_(torch.where(ok, v_new, v))
            e.copy_(torch.where(ok, e * EMA + (1 - EMA) * p_new, e))
        new_ref = (.99 * ref0 + .01 * median).clamp(min=ref0 * .986, max=ref0 * 1.014)
        ref.copy_(torch.where(ok, new_ref, ref0))
        acc.add_(ok.long())
        rejected.add_((~ok).long())
        return tuple(v.squeeze(1) for v in new_state), halt.squeeze(-1), losses.mean(), ce.mean(), total

    return update


def lr_table(schedule_steps, length, device):
    """Learning rate per update: linear warm-up over 10%, then cosine to 5% of the peak."""
    warmup = max(1, round(.1 * schedule_steps))
    k = np.arange(length, dtype=np.float64)
    mult = np.where(k < warmup, (k + 1) / warmup,
                    .05 + .95 * .5 * (1 + np.cos(math.pi * (k - warmup) / (schedule_steps - warmup))))
    return torch.as_tensor(LR * mult, dtype=torch.float32, device=device)


@torch.no_grad()
def eval_dev(model, x, y, steps, amp):
    """Validation solves: fixed-compute readout, symbols outside the allowed set masked."""
    with amp_context(amp):
        state = model.initial(x)
        clues = model.embedding(x)
        for _ in range(steps):
            state = model.step(state, clues)
        raw_logits = model.symbol(state[1]).float()
    prediction = raw_logits.masked_fill(x == 0, -torch.inf).argmax(-1)
    return {'solved': int((prediction == y).all(-1).sum()), 'count': len(x),
            'boards': float((prediction == y).all(-1).float().mean()), 'cells': float((prediction == y).float().mean()),
            'cross_entropy': float(F.cross_entropy(raw_logits.flatten(0, 1), y.flatten())), 'steps': steps}


def ema_state(model, names, E, accepted):
    """Bias-corrected EMA weights (the dormant parameters keep their values)."""
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if accepted:
        for n, e in zip(names, E):
            sd[n] = e / (1 - EMA ** accepted)
    return sd


# ---------------------------------------------------------------- fit

def main(cfg: Config):
    r = RECIPES[cfg.task]
    steps = cfg.steps or r['train_steps']
    schedule = cfg.schedule_steps or r['train_steps']
    slots = cfg.slots or r['slots']
    dev_count = cfg.dev_count or (500 if cfg.task == 'sudoku-extreme' else 100)
    out = Path(cfg.out)
    if (out / 'fit.json').exists():
        print(f'{out}/fit.json exists: nothing to do', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', dict(dataclasses.asdict(cfg), steps=steps, schedule_steps=schedule, slots=slots, dev_count=dev_count))
    ctx = cuda_context_seconds()
    device = torch.device('cuda')
    raw = read_raw(cfg)  # disk read, before the clock

    clock = FitClock()
    clock.start()  # training data are in host memory
    with clock.exclude('dev_read_encode'):
        host_dx, host_dy = read_dev(cfg, dev_count)
        dx, dy = host_dx.to(device), host_dy.to(device)
    host_x, host_y = encode_train(cfg, raw)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    x, y = host_x.to(device), host_y.to(device)
    model = Trainable(cfg.task).to(device)
    names = [n for n, _ in model.named_parameters() if not n.startswith(DORMANT)]
    P = [model.get_parameter(n) for n in names]
    for p in model.parameters():
        p.requires_grad_(False)  # gradients come from torch.func, never from .backward()
    M = [torch.zeros_like(p) for p in P]
    V = [torch.zeros_like(p) for p in P]
    E = [torch.zeros_like(p) for p in P]
    acc = torch.zeros((), dtype=torch.long, device=device)
    rejected = torch.zeros((), dtype=torch.long, device=device)
    ref = torch.zeros((), device=device)
    lrs = lr_table(schedule, steps, device)
    lr = torch.zeros((), device=device)
    bx, by = sampled_batch(cfg.task, x, y, slots, cfg.permute)
    state = model.initial(bx)
    phases = torch.arange(slots, device=device) % PHASES
    step, selected = 0, None

    ckpt = out / 'ckpt.pt'
    log_path = out / 'log.jsonl'
    if ckpt.exists():
        with clock.exclude('checkpoint_load'):
            st = torch.load(ckpt, map_location=device, weights_only=False)
        assert st['config'] == dataclasses.asdict(cfg), 'cannot resume with a different configuration'
        model.load_state_dict(st['model'])
        for dst, src in ((M, st['M']), (V, st['V']), (E, st['E'])):
            for a, b in zip(dst, src):
                a.copy_(b)
        acc.copy_(st['acc']); rejected.copy_(st['rejected']); ref.copy_(st['ref'])
        bx, by, state, phases, step, selected = st['bx'], st['by'], st['state'], st['phases'], st['step'], st['selected']
        torch.set_rng_state(st['rng_cpu'].cpu())
        torch.cuda.set_rng_state(st['rng_cuda'].cpu())
        clock.resumed, clock.prior_fit_seconds = True, st['fit_seconds']
        lines = log_path.read_text().splitlines() if log_path.exists() else []
        kept = [l for l in lines if json.loads(l).get('step', -1) <= step]
        log_path.write_text('\n'.join(kept + [json.dumps({'resume': step})]) + '\n')
        print(f'RESUME step {step}: not a cold fit', flush=True)

    update = make_update(model, names, cfg.amp)
    if cfg.compile:
        update = torch.compile(update, fullgraph=False)

    def save():
        fit_so_far = clock.elapsed()
        with clock.exclude('checkpoint_save'):
            st = dict(config=dataclasses.asdict(cfg), model=model.state_dict(), M=M, V=V, E=E, acc=acc, rejected=rejected,
                      ref=ref, bx=bx, by=by, state=state, phases=phases, step=step, selected=selected,
                      rng_cpu=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state(), fit_seconds=fit_so_far)
            tmp = ckpt.with_suffix('.pt.tmp')
            torch.save(st, tmp)
            os.replace(tmp, ckpt)

    last_save = time.perf_counter()
    stop_after = int(os.environ.get('TRMP_STOP_AFTER', -1))
    t_log, s_log = time.perf_counter(), step
    while step < steps:
        lr.copy_(lrs[step])
        new_state, halt, loss, ce, norm = update(P, M, V, E, acc, ref, rejected, bx, by, state, lr)
        # Persistent slots: refill a slot that halts or reaches 32 supervision steps.
        phases = phases + 1
        replace = (halt > 0) | (phases >= PHASES)
        fresh_x, fresh_y = sampled_batch(cfg.task, x, y, slots, cfg.permute)
        bx = torch.where(replace[:, None, None], fresh_x, bx)
        by = torch.where(replace[:, None], fresh_y, by)
        # .contiguous(): Inductor pads output strides; a carried state with other strides would recompile the update.
        state = tuple(torch.where(replace.reshape(-1, *([1] * (v.ndim - 1))), 0., v).contiguous() for v in new_state)
        phases = torch.where(replace, 0, phases)
        step += 1
        if step % cfg.log_every == 0 or step == steps or step <= 3:
            fit_s = clock.elapsed()
            now = time.perf_counter()
            rec = dict(step=step, loss=float(loss), ce=float(ce), gradient_norm=float(norm), accepted=int(acc),
                       rejected=int(rejected), fit_s=round(fit_s, 2),
                       s_per_update=round((now - t_log) / max(1, step - s_log), 5),
                       peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            t_log, s_log = now, step
            with log_path.open('a') as f:
                f.write(json.dumps(rec) + '\n')
            print('STEP ' + json.dumps(rec), flush=True)
        if cfg.validate_every and step >= cfg.validate_from and step % cfg.validate_every == 0:
            with clock.exclude('evaluation'):
                candidate = copy.deepcopy(model)
                candidate.load_state_dict(ema_state(model, names, E, int(acc)))
                score = eval_dev(candidate, dx, dy, cfg.eval_steps, cfg.amp)
                if selected is None or score['solved'] >= selected['score']['solved']:  # later ties win
                    selected = dict(step=step, score=score, model={k: v.clone() for k, v in candidate.state_dict().items()})
                with (out / 'validation.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=step, score=score, selected_step=selected['step'])) + '\n')
                print('VALID ' + json.dumps(dict(step=step, **score, selected_step=selected['step'])), flush=True)
                del candidate
        if time.perf_counter() - last_save > cfg.ckpt_every and step < steps:
            save()
            last_save = time.perf_counter()
        if step == stop_after:  # resume test only: a simulated preemption right after a checkpoint
            save()
            print(f'STOP after {step} (TRMP_STOP_AFTER)', flush=True)
            return

    final = copy.deepcopy(model)
    final.load_state_dict(selected['model'] if selected is not None else ema_state(model, names, E, int(acc)))
    fit_seconds = clock.stop()  # the inference-ready model exists

    t = time.perf_counter()
    torch.save(final.state_dict(), out / 'model.pt')
    final_save_seconds = time.perf_counter() - t
    t = time.perf_counter()
    dev = eval_dev(final, dx, dy, cfg.eval_steps, cfg.amp)
    final_dev_seconds = time.perf_counter() - t
    rec = clock.record(
        method='TRMP', task=cfg.task, source=SOURCE, config=dataclasses.asdict(cfg),
        recipe=dict(r, steps=steps, schedule_steps=schedule, slots=slots, lr=LR, betas=BETAS, weight_decay=WD, ema=EMA,
                    global_clip=CLIP, reject_above=REJECT, kappa=KAPPA, slot_phases=PHASES,
                    validate_every=cfg.validate_every, validate_from=cfg.validate_from, eval_steps=cfg.eval_steps,
                    dev_count=dev_count),
        numerics=('bf16 autocast forward and backward, fp32 master weights, AdamW moments and EMA' if cfg.amp else 'fp32')
                 + (', torch.compile of one update' if cfg.compile else ', eager'),
        params=sum(p.numel() for p in final.parameters()), trained_params=sum(p.numel() for p in P),
        steps=step, accepted=int(acc), rejected=int(rejected), selected_step=selected['step'] if selected else step,
        selected_dev=selected['score'] if selected else None, final_dev=dev, cuda_context_seconds=ctx,
        final_save_seconds=final_save_seconds, final_dev_seconds=final_dev_seconds,
        training_data=dict(path=raw['path'], sha256=raw['sha256'], count=len(host_x)),
        model_sha256=hashlib.sha256((out / 'model.pt').read_bytes()).hexdigest(),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE / 'fit.py', HERE / 'model.py', HERE.parent / 'clock.py')},
        deviations=[
            'clean-room reconstruction of the ACT recipe, not the authors\' code or fused kernels',
            'PyTorch default initialization except the documented halting head',
            'bf16 autocast for every task (the paper trains Sudoku ACT in single precision; Maze uses bf16 kernels)',
        ] + (['Sudoku: validation is the first 500 of the dev10k puzzles held out of the official test split, '
              'not the paper\'s pool from the training split; checkpoint evaluation every 5,000 updates from 40,000 '
              '(the paper does not state the Sudoku interval; this is the Maze rule)'] if cfg.task == 'sudoku-extreme' else
             ['Maze: validation is 100 generated mazes (exact identity to the authors\' file unverified)'])
        + ([f'Sudoku training rows: {cfg.rows} (every stored row a board, drawn uniformly with replacement), not FRM\'s train-s{cfg.seed}'] if cfg.rows else [])
        + ([] if cfg.permute else ['no per-draw digit permutation: the stored rows only (their stored symmetry copies per puzzle)']))
    write_json(out / 'fit.json', rec)
    ckpt.unlink(missing_ok=True)
    print(f'FIT {fit_seconds:.1f} s, {step} updates, accepted {int(acc)}, selected {rec["selected_step"]}, dev {dev}', flush=True)


if __name__ == '__main__':
    main(tyro.cli(Config))
