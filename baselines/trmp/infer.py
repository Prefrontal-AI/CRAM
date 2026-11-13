"""Adaptive TRMP best-of-k inference (arXiv 2610.18649, Appendix B.2.2). Imported by trmp/score.py; no command line.

Halting and symbol logits are cast to fp32 (a no-op in fp32) so the selection is the same under bf16 autocast.
`matrix_count` is the closed-form matrix-product operation count of one supervision step. Each rollout halts independently at q > 0 or at the step
cap; noise is added between steps only to continuing rollouts; the largest terminal q selects the answer, ties to the
lowest rollout index. No labels are accepted.
"""
from __future__ import annotations

import torch


def matrix_count(r):
    """Dense matrix flop of one supervision step of one rollout (2 flop per multiply-add)."""
    n, m, h, a = r['widths']
    v, degree = r['cells'], r['degree']
    per_round = (2*v*degree*((m+n)*h+h*h) + 2*v*3*m*(h+n+a+m) + 2*v*m*n
                 + 2*(3*v if r['relations'] else 0)*(3*n*h+h*h))
    return r['cycles']*(r['rounds']*per_round + 2*v*3*a*(m+a))


@torch.no_grad()
def best_of(model, x, *, rollouts, steps, sigma, adaptive=True, generator=None):
    """x (puzzles, cells, symbols) allowed-symbol multihot; returns answers and work counters."""
    if rollouts < 1 or steps < 1 or sigma < 0:
        raise ValueError('Positive rollout/depth budgets and nonnegative noise required')
    if x.ndim != 3 or not (x.sum(-1) > 0).all():
        raise ValueError('Every cell needs at least one allowed symbol')
    puzzles, cells, _ = x.shape
    expanded = x.repeat_interleave(rollouts, dim=0)
    state = model.initial(expanded)
    clues = model.embedding(expanded)
    active = torch.arange(len(expanded), device=x.device)
    prediction = torch.empty((len(expanded), cells), dtype=torch.long, device=x.device)
    confidence = torch.empty(len(expanded), device=x.device)
    depth = torch.zeros(len(expanded), dtype=torch.long, device=x.device)
    for index in range(1, steps+1):
        state = model.step(state, clues)
        depth[active] += 1
        if adaptive or index == steps:
            q = -torch.logsumexp(-model.halt_logits(state[1]).float(), dim=-1)
            if not torch.isfinite(q).all():
                raise ValueError('Nonfinite halting confidence')
            done = (q > 0) if adaptive and index < steps else torch.ones_like(q, dtype=torch.bool)
            if done.any():
                finished = active[done]
                logits = model.symbol(state[1][done]).float()
                if not torch.isfinite(logits).all():
                    raise ValueError('Nonfinite symbol logits')
                prediction[finished] = logits.masked_fill(expanded[finished] == 0, -torch.inf).argmax(-1)
                confidence[finished] = q[done]
        else:
            done = torch.zeros(len(active), dtype=torch.bool, device=x.device)
        remaining = ~done
        if not remaining.any():
            break
        active = active[remaining]
        clues = clues[remaining]
        state = tuple(value[remaining] for value in state)
        if sigma:
            h, a, messages, units = state
            noise = torch.randn(a.shape, dtype=a.dtype, device=a.device, generator=generator)
            state = h, a+sigma*noise, messages, units
    confidence = confidence.reshape(puzzles, rollouts)
    depth = depth.reshape(puzzles, rollouts)
    winner = confidence.argmax(-1)
    answers = prediction.reshape(puzzles, rollouts, cells)[torch.arange(puzzles, device=x.device), winner]
    return answers, dict(rollout_steps=depth, confidence=confidence, selected_rollout=winner)
