"""The training method `spl` for the N-Queens energies: pseudolikelihood (PL) at boards with some queens removed.

The search starts from the board that holds only the given queens (every free cell empty). Every training board has
exactly n queens, so plain PL never evaluates the energy near the board where the search starts. `spl` evaluates the PL
of the label at a board between the start board and the label: each free cell takes its start state (empty) with a
probability r, and r is drawn per board from the uniform distribution on (0, rate):
    loss = mean over free cells i of -log softmax(-L_i)[y_i],   L_i (2,) the local energies of cell i at the emptied board.
It is a local objective at the labeled boards: no model samples, no solver.

Importing this module installs loss_fn as training_methods.loss_fn, so that every caller of training_methods.loss_fn
accepts the method name 'spl'; every other method name passes through to the original function.
nqueens_offset.py imports this module for its method `spl`.
Dimension names as nqueens_free.py: b rows (boards), i cells (n * n), a cell states (0 empty, 1 queen).
"""
import torch

import training_methods as AT
import nqueens_free as N

METHODS = ('spl',)
N.CLASS.update({m: 'S' for m in METHODS})
_base_loss = AT.loss_fn


def spl_loss(rate, task, ctx, gen):
    """The spl loss (a scalar). rate: upper end of the per-board emptying probability. task: an object with
    field(v, ctx) -> (b, i, a). ctx: {'y': (b, i) int labels, 'free': (b, i) bool}. gen: the generator of the draws."""
    y, free = ctx['y'], ctx['free']
    start = torch.where(free, torch.zeros_like(y), y)  # the search's start board: given queens, every free cell empty
    r = rate * torch.rand(len(y), 1, device=y.device, generator=gen)  # one rate per board, U(0, rate)
    hit = free & (torch.rand(free.shape, device=y.device, generator=gen) < r)
    L = task.field(AT.onehot(torch.where(hit, start, y), 2), ctx)
    return AT.free_mean(-torch.log_softmax(-L, -1).gather(-1, y[..., None]).squeeze(-1), free)


def loss_fn(method, knob, task, ctx, gen, mstate):
    """training_methods.loss_fn with the method 'spl' added. For 'spl', knob is {'rate': float}; the other arguments
    are those of training_methods.loss_fn."""
    if method not in METHODS:
        return _base_loss(method, knob, task, ctx, gen, mstate)
    return spl_loss(knob['rate'], task, ctx, gen)


AT.loss_fn = loss_fn  # nqueens_offset.fit and nqueens_free.fit call AT.loss_fn
