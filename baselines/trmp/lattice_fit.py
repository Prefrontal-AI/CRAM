"""TRMP lattice model (the paper's "ten-minute" Sudoku-Extreme model): one timed cold fit on one GPU.

Recipe: arXiv 2610.18649 Table 4, "Full gradient, factor graph" (lattice.RECIPE): widths (9, 24, 48, 12) = 16,543
parameters, deep-supervision steps of 14 cycles x 2 rounds with no detached cycle, a persistent pool of 512 boards that
is also the batch, AdamW (0.9, 0.95) with peak lr 6.8742e-3 and weight decay 0.04821 on matrices only, linear warm-up
over 10.09% then cosine decay to 5%, gradient-norm clip 1.0, bias-corrected EMA 0.9995, 20,000 updates, bf16 autocast
with TF32 matrix products. Every update runs one deep-supervision step on every pool board, takes the loss of Eq. 12
averaged over the 14 read-out cycles, then projects each board's lattice; a board leaves the pool when it is solved and
correct, when a detected conflict is real, or at age 100 (B.2.1). Checkpoint selection (B.4): from update 12,000, every
2,000 updates, the EMA weights search the 500 validation puzzles with 64 chains and a budget of 300 steps, chains
checked every 4 steps; the best (later ties win) is the fitted model.

Clock: baselines/clock.py (the CRAM fit-time protocol); validation is excluded. Data as our TRMP ACT row: FRM's
train-s<seed> (1,000 puzzles), validation dev10k[:500].

Dimension names: b pool boards, v cells, s symbols.

  python trmp/lattice_fit.py --seed 0 --out runs/trmp-lattice/sudoku-extreme-s0        # from baselines/
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
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import tyro
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from clock import FitClock, cuda_context_seconds, write_json  # noqa: E402
import lattice as L  # noqa: E402

V1_DATA = Path(os.environ.get('V1_DATA', ROOT / 'data'))
R = L.RECIPE


@dataclasses.dataclass
class Config:
    out: str = 'runs/trmp-lattice/sudoku-extreme-s0'  # output folder (fit.json, model.pt, log.jsonl, ckpt.pt)
    seed: int = 0  # training seed; also selects FRM's training split train-s<seed>
    steps: Optional[int] = None  # updates to run (default: the recipe's 20,000)
    schedule_steps: Optional[int] = None  # learning-rate schedule length (default: the recipe's updates)
    amp: bool = True  # bf16 autocast forward/backward with TF32 matrix products (fp32 weights, optimizer and EMA)
    compile: bool = True  # torch.compile the forward and loss (the backward through AOTAutograd)
    validate: bool = True  # checkpoint selection on validation (B.4); off: the final EMA weights
    validate_from: int = R['validate_from']  # first update evaluated for checkpoint selection
    validate_every: int = R['validate_every']  # updates between validation searches
    dev_count: int = 500  # validation puzzles (dev10k[:dev_count])
    ckpt_every: float = 600.0  # seconds of wall clock between rolling checkpoints
    log_every: int = 250  # updates between log lines (each log line synchronizes the device)


def read_raw(cfg):
    path = V1_DATA / f'frm/split/train-s{cfg.seed}.npz'
    f = np.load(path)
    return dict(solutions=f['solutions'], clue_mask=f['clue_mask'], path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def lattice_of(solutions, clue_mask):
    """x0 (b, 81, 9): a given is one-hot, a blank holds all nine digits; digits 0-8 (b, 81)."""
    d = torch.as_tensor(np.asarray(solutions, dtype=np.int64) - 1)
    c = torch.as_tensor(np.asarray(clue_mask, dtype=bool))
    return torch.where(c[..., None], F.one_hot(d, 9).float(), torch.ones(*d.shape, 9)), d


def read_dev(count):
    f = np.load(V1_DATA / 'frm/dev10k/dev.npz')
    return lattice_of(f['solutions'][:count], f['clue_mask'][:count])


def lr_multiplier(k, schedule_steps):
    """Linear warm-up over 10.09% of the schedule, then cosine decay to 5% of the peak."""
    warmup = max(1, round(R['warmup'] * schedule_steps))
    if k < warmup:
        return (k + 1) / warmup
    f = R['final_lr_fraction']
    return f + (1 - f) * 0.5 * (1 + math.cos(math.pi * (k - warmup) / max(1, schedule_steps - warmup)))


def ema_weights(net, E, updates):
    sd = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if updates:
        for (name, _), e in zip(net.named_parameters(), E):
            sd[name] = e / (1 - R['ema'] ** updates)
    return sd


def evaluate(net, x0, digits, chains, budget, check_every, amp, seed, compile_eval):
    """Chain search on validation puzzles; returns solved count and step statistics."""
    gen = torch.Generator(device=x0.device).manual_seed(seed)
    fn = torch.compile(net) if compile_eval else net
    t = time.perf_counter()
    pred, steps, resets = L.chain_search(fn, x0, chains=chains, budget=budget, check_every=check_every,
                                         slots=len(x0), amp=amp, generator=gen)
    torch.cuda.synchronize()
    solved = (pred == digits).all(-1)
    return dict(solved=int(solved.sum()), count=len(x0), timeouts=int((pred < 0).any(-1).sum()),
                wrong=int(((pred >= 0).all(-1) & ~solved).sum()), mean_steps=float(steps.float().mean()),
                resets=int(resets.sum()), seconds=round(time.perf_counter() - t, 2), chains=chains, budget=budget,
                check_every=check_every)


def main(cfg: Config):
    steps = cfg.steps or R['steps']
    schedule = cfg.schedule_steps or steps
    out = Path(cfg.out)
    if (out / 'fit.json').exists():
        print(f'{out}/fit.json exists: nothing to do', flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', dict(dataclasses.asdict(cfg), steps=steps, schedule_steps=schedule, recipe=R))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ctx = cuda_context_seconds()
    device = torch.device('cuda')
    raw = read_raw(cfg)  # disk read, before the clock

    clock = FitClock()
    clock.start()  # training data are in host memory
    if cfg.validate:
        with clock.exclude('dev_read_encode'):
            dx0, dd = read_dev(cfg.dev_count)
            dx0, dd = dx0.to(device), dd.to(device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    sols = torch.as_tensor(raw['solutions'].astype(np.int64) - 1).to(device)
    clues = torch.as_tensor(raw['clue_mask'].astype(bool)).to(device)
    net = L.LatticeNet().to(device)
    params = [p for _, p in net.named_parameters()]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=R['weight_decay']), dict(params=no_decay, weight_decay=0.0)],
                            lr=R['lr'], betas=R['betas'], fused=True)
    E = [torch.zeros_like(p) for p in params]
    pool = R['pool']
    x0, y = L.draw_starts(sols, clues, pool)
    x = x0.clone()
    age = torch.zeros(pool, dtype=torch.long, device=device)
    step, selected = 0, None

    ckpt = out / 'ckpt.pt'
    log_path = out / 'log.jsonl'
    if ckpt.exists():
        with clock.exclude('checkpoint_load'):
            st = torch.load(ckpt, map_location=device, weights_only=False)
        assert st['config'] == dataclasses.asdict(cfg), 'cannot resume with a different configuration'
        net.load_state_dict(st['net'])
        opt.load_state_dict(st['opt'])
        for a, b in zip(E, st['E']):
            a.copy_(b)
        x, x0, y, age, step, selected = st['x'], st['x0'], st['y'], st['age'], st['step'], st['selected']
        torch.set_rng_state(st['rng_cpu'].cpu())
        torch.cuda.set_rng_state(st['rng_cuda'].cpu())
        clock.resumed, clock.prior_fit_seconds = True, st['fit_seconds']
        lines = log_path.read_text().splitlines() if log_path.exists() else []
        kept = [l for l in lines if json.loads(l).get('step', -1) <= step]
        log_path.write_text('\n'.join(kept + [json.dumps({'resume': step})]) + '\n')
        print(f'RESUME step {step}: not a cold fit', flush=True)

    def forward_loss(x, y, given):
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=cfg.amp):
            heads = net(x)
        loss, parts = L.lattice_loss(heads, x, y, given)
        return loss, parts, heads.keep[-1].detach(), heads.sym[-1].detach(), heads.conf[-1].detach()

    fwd = torch.compile(forward_loss) if cfg.compile else forward_loss

    def save():
        fit_so_far = clock.elapsed()
        with clock.exclude('checkpoint_save'):
            st = dict(config=dataclasses.asdict(cfg), net=net.state_dict(), opt=opt.state_dict(), E=E, x=x, x0=x0, y=y,
                      age=age, step=step, selected=selected, rng_cpu=torch.get_rng_state(),
                      rng_cuda=torch.cuda.get_rng_state(), fit_seconds=fit_so_far)
            tmp = ckpt.with_suffix('.pt.tmp')
            torch.save(st, tmp)
            os.replace(tmp, ckpt)

    last_save = time.perf_counter()
    stop_after = int(os.environ.get('TRMP_STOP_AFTER', -1))
    t_log, s_log = time.perf_counter(), step
    counts = torch.zeros(4, dtype=torch.long, device=device)  # solved-correct, true conflict, aged, false conflict
    while step < steps:
        for g in opt.param_groups:
            g['lr'] = R['lr'] * lr_multiplier(step, schedule)
        given = x0.sum(-1) == 1
        loss, parts, keep, sym, conf = fwd(x, y, given)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, R['clip'], foreach=True)
        opt.step()
        with torch.no_grad():
            torch._foreach_mul_(E, R['ema'])
            torch._foreach_add_(E, params, alpha=1 - R['ema'])
        # Projection and pool (B.2.1).
        nx, conflict, solved, _, _ = L.project(x, given, keep, sym, conf, L.THETA_CLS_TRAIN)
        age = age + 1
        correct = solved & (nx == y).all(-1).all(-1)
        real = conflict & L.conflicts(nx, y)[1]
        aged = age >= R['max_age']
        discard = correct | real | aged
        counts += torch.stack((correct.sum(), real.sum(), aged.sum(), (conflict & ~real).sum()))
        fx0, fy = L.draw_starts(sols, clues, pool)
        d3 = discard[:, None, None]
        x = torch.where(d3, fx0, nx)
        x0 = torch.where(d3, fx0, x0)
        y = torch.where(d3, fy, y)
        age = torch.where(discard, 0, age)
        step += 1
        if step % cfg.log_every == 0 or step == steps or step <= 3:
            fit_s = clock.elapsed()
            now = time.perf_counter()
            c = counts.tolist()
            rec = dict(step=step, loss=float(loss), bce=float(parts[0]), ce=float(parts[1]), conf=float(parts[2]),
                       gradient_norm=float(norm), lr=opt.param_groups[0]['lr'], solved_correct=c[0], true_conflict=c[1],
                       aged=c[2], false_conflict=c[3], fit_s=round(fit_s, 2),
                       s_per_update=round((now - t_log) / max(1, step - s_log), 5),
                       peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            counts.zero_()
            t_log, s_log = now, step
            with log_path.open('a') as f:
                f.write(json.dumps(rec) + '\n')
            print('STEP ' + json.dumps(rec), flush=True)
        if cfg.validate and step >= cfg.validate_from and (step - cfg.validate_from) % cfg.validate_every == 0:
            with clock.exclude('evaluation'):
                cand = copy.deepcopy(net)
                cand.load_state_dict(ema_weights(net, E, step))
                cand.eval()
                score = evaluate(cand, dx0, dd, R['dev_chains'], R['dev_budget'], R['dev_check_every'], cfg.amp,
                                 seed=20261112 + step, compile_eval=cfg.compile)
                if selected is None or score['solved'] >= selected['score']['solved']:  # later ties win
                    selected = dict(step=step, score=score, model={k: v.clone() for k, v in cand.state_dict().items()})
                with (out / 'validation.jsonl').open('a') as f:
                    f.write(json.dumps(dict(step=step, score=score, selected_step=selected['step'])) + '\n')
                print('VALID ' + json.dumps(dict(step=step, **score, selected_step=selected['step'])), flush=True)
                del cand
        if time.perf_counter() - last_save > cfg.ckpt_every and step < steps:
            save()
            last_save = time.perf_counter()
        if step == stop_after:  # resume test only: a simulated preemption right after a checkpoint
            save()
            print(f'STOP after {step} (TRMP_STOP_AFTER)', flush=True)
            return

    final = copy.deepcopy(net)
    final.load_state_dict(selected['model'] if selected is not None else ema_weights(net, E, step))
    fit_seconds = clock.stop()  # the inference-ready model exists

    t = time.perf_counter()
    torch.save(final.state_dict(), out / 'model.pt')
    final_save_seconds = time.perf_counter() - t
    rec = clock.record(
        method='TRMP lattice (factor graph, 16k)', task='sudoku-extreme', source=L.SOURCE, config=dataclasses.asdict(cfg),
        recipe=dict(R, steps=steps, schedule_steps=schedule, theta=L.THETA, theta_cls_train=L.THETA_CLS_TRAIN,
                    theta_cls_eval=L.THETA_CLS_EVAL, temperature=L.TEMP, loss_weights=[L.W_POS, L.W_NEG, L.W_SYM, L.W_CONF]),
        numerics=('bf16 autocast forward and backward with TF32 matrix products, fp32 weights, AdamW moments and EMA'
                  if cfg.amp else 'fp32') + (', torch.compile of the forward and loss' if cfg.compile else ', eager'),
        params=sum(p.numel() for p in final.parameters()), steps=step,
        selected_step=selected['step'] if selected else step, selected_dev=selected['score'] if selected else None,
        cuda_context_seconds=ctx, final_save_seconds=final_save_seconds,
        training_data=dict(path=raw['path'], sha256=raw['sha256'], count=len(raw['solutions'])),
        model_sha256=hashlib.sha256((out / 'model.pt').read_bytes()).hexdigest(),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (HERE / 'lattice_fit.py', HERE / 'lattice.py', HERE / 'model.py', HERE.parent / 'clock.py')},
        deviations=[
            "clean-room reimplementation from the paper and the authors' precursor tests and Maze trainer; the "
            "authors' Sudoku lattice code and Triton kernels are not available",
            'PyTorch default initialization',
            'data as our TRMP ACT row: FRM train-s<seed> (1,000 puzzles), validation dev10k[:500]; the paper draws '
            '1,000 training and 500 validation puzzles from the official training split',
            'fresh pool draws uniform with replacement (the precursor cycles a permutation); a corrupted cell takes a '
            'digit drawn uniformly from the other eight',
            'validation: on a step that is not a check step a chain keeps its lattice and its status is ignored',
        ])
    write_json(out / 'fit.json', rec)
    ckpt.unlink(missing_ok=True)
    print(f'FIT {fit_seconds:.1f} s, {step} updates, selected {rec["selected_step"]}, dev {rec["selected_dev"]}', flush=True)


if __name__ == '__main__':
    main(tyro.cli(Config))
