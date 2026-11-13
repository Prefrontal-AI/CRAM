"""TRMP (Tiny Recursive Message Passing, arXiv 2610.18649) ACT model, clean-room reconstruction from the paper.

ACT = adaptive computation time: the model takes supervision steps until its halting head fires. The authors released
no code for this model, so no upstream code is called. `Reference` is the model as the paper states it (nn.GRUCell);
`Trainable` is the same arithmetic with the GRU equations written out (so torch.func.vmap can take per-example
gradients), for either task. RECIPES holds the paper's settings per task and `parameter_count` its Equation 11.
Imported by trmp/fit.py, score.py, check.py and the lattice files; it has no command line.

Dimension names: b slots or rollouts, v cells (81 Sudoku, 900 Maze), k incoming messages per cell, s symbols,
n message width, m cell memory width, a answer width.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

SOURCE = 'https://arxiv.org/html/2610.18649v1'
RECIPES = {
    'sudoku-extreme': dict(widths=[15, 39, 76, 20], symbols=9, cells=81, degree=20, relations=0, cycles=8, rounds=12,
                           detached_cycles=6, slots=512, train_steps=275000, inference_steps=1024, rollouts=64),
    'maze-unique': dict(widths=[16, 42, 84, 21], symbols=5, cells=900, degree=7, relations=2, cycles=12, rounds=12,
                        detached_cycles=10, slots=128, train_steps=120000, inference_steps=64, rollouts=64),
}
DORMANT = ('keep.', 'conflict.', 'initial_answer')  # lattice heads and a0: in the parameter count, never trained


def parameter_count(r):
    """Equation 11, including dormant lattice heads and ACT halt head."""
    n, m, h, a = r['widths']
    q = r['symbols']
    parts = {
        'cell': (m+n)*h + h*h + 2*h + 3*m*(h+n+a+m) + 8*m + n*m+n,
        'relation_each': 3*n*h + h*h + 2*h+n,
        'answer': 3*a*(m+a) + 8*a,
        'input_and_lattice_heads': (q+1)*n + 2*(q+1)*a + 2*q+1,
        'halting_head': a+1,
    }
    parts['total'] = (parts['cell'] + r['relations']*parts['relation_each'] + parts['answer']
                      + parts['input_and_lattice_heads'] + parts['halting_head'])
    return parts


class Relation(nn.Module):
    def __init__(self, n, width):
        super().__init__()
        self.embed = nn.Linear(n, width)
        self.decode = nn.Sequential(nn.Linear(n+width, width), nn.ReLU(), nn.Linear(width, n))

    def forward(self, x):
        summary = self.embed(x).relu().mean(-2, keepdim=True)
        return self.decode(torch.cat((x, summary.expand(*x.shape[:-1], -1)), -1))


class ExplicitGRU(nn.GRUCell):
    """Same PyTorch GRU equations, expressed with vmappable primitives."""
    def forward(self, x, hidden):
        ir, iz, inn = F.linear(x, self.weight_ih, self.bias_ih).chunk(3, -1)
        hr, hz, hn = F.linear(hidden, self.weight_hh, self.bias_hh).chunk(3, -1)
        reset, update = (ir+hr).sigmoid(), (iz+hz).sigmoid()
        candidate = (inn+reset*hn).tanh()
        return candidate+update*(hidden-candidate)


class Reference(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.task = task
        r = self.recipe = RECIPES[task]
        n, m, width, a = r['widths']
        self.embedding = nn.Linear(r['symbols'], n)
        self.message = nn.Sequential(nn.Linear(m+n, width), nn.ReLU(), nn.Linear(width, width))
        self.gru = nn.GRUCell(width+n+a, m)
        self.norm = nn.LayerNorm(m)
        self.emit = nn.Linear(m, n)
        self.answer = nn.GRUCell(m, a)
        self.answer_norm = nn.LayerNorm(a)
        self.initial_answer = nn.Parameter(torch.zeros(a))
        self.keep = nn.Linear(a, r['symbols'])
        self.symbol = nn.Linear(a, r['symbols'])
        self.conflict = nn.Linear(a, 1)
        self.halt = nn.Linear(a, 1)
        self.relations = nn.ModuleList(Relation(n, width) for _ in range(r['relations']))
        if task == 'sudoku-extreme':
            peers = [[j for j in range(81) if j != i and
                      (i//9 == j//9 or i % 9 == j % 9 or (i//27, i % 9//3) == (j//27, j % 9//3))]
                     for i in range(81)]
            assert all(len(p) == 20 for p in peers)
            self.register_buffer('peers', torch.tensor(peers))
        assert sum(p.numel() for p in self.parameters()) == parameter_count(r)['total']

    def initial(self, x):
        b, v = x.shape[:2]
        n, m, _, a = self.recipe['widths']
        return (x.new_zeros(b, v, m), x.new_zeros(b, v, a), x.new_zeros(b, v, n), x.new_zeros(b, v, 3, n))

    def round(self, state, clues):
        h, a, outgoing, unit_replies = state
        b, v, n = outgoing.shape
        if self.recipe['relations']:
            grid = outgoing.reshape(b, 30, 30, n)
            padded = F.pad(grid, (0, 0, 1, 1, 1, 1))
            neighbors = torch.stack((padded[:, :-2, 1:-1], padded[:, 2:, 1:-1], padded[:, 1:-1, :-2], padded[:, 1:-1, 2:]), -2)
            incoming = torch.cat((neighbors.reshape(b, v, 4, n), unit_replies), -2)
            # Unit replies use OLD cell messages (synchronous, two-hop exchange).
            row = self.relations[0](grid)
            col = self.relations[0](grid.transpose(1, 2)).transpose(1, 2)
            whole = self.relations[1](outgoing)
            next_units = torch.stack((row.reshape(b, v, n), col.reshape(b, v, n), whole), -2)
        else:
            incoming = outgoing[:, self.peers]
            next_units = unit_replies
        memory = h.unsqueeze(-2).expand(*incoming.shape[:-1], h.shape[-1])
        pooled = self.message(torch.cat((memory, incoming), -1)).mean(-2)
        z = torch.cat((pooled, clues, a), -1)
        new_h = self.norm(self.gru(z.reshape(b*v, -1), h.reshape(b*v, -1)))
        new_h = new_h.reshape(b, v, -1)
        return new_h, a, self.emit(new_h), next_units

    def step(self, state, clues):
        for _ in range(self.recipe['cycles']):
            state = self.cycle(state, clues)
        return state

    def cycle(self, state, clues):
        b, v = clues.shape[:2]
        for _ in range(self.recipe['rounds']):
            state = self.round(state, clues)
        h, a, outgoing, units = state
        a = self.answer_norm(self.answer(h.reshape(b*v, -1), a.reshape(b*v, -1)))
        return h, a.reshape(b, v, -1), outgoing, units


class Trainable(Reference):
    """The trained model for either task: explicit GRUs, halt head zero weight and bias -5."""
    def __init__(self, task):
        super().__init__(task)
        for name in ('gru', 'answer'):
            old = getattr(self, name)
            new = ExplicitGRU(old.input_size, old.hidden_size)
            new.load_state_dict(old.state_dict())
            setattr(self, name, new)
        nn.init.zeros_(self.halt.weight)
        nn.init.constant_(self.halt.bias, -5)

    def halt_logits(self, answer):
        return self.halt(answer.detach()).squeeze(-1)

    def forward(self, x, target, state):
        """Per-board loss; leading batch dimension is one inside vmap."""
        with torch.no_grad():
            clues = self.embedding(x)
            for _ in range(self.recipe['detached_cycles']):
                state = self.cycle(state, clues)
        state = tuple(t.detach() for t in state)
        clues = self.embedding(x)
        ce = x.new_zeros(())
        retained = self.recipe['cycles']-self.recipe['detached_cycles']
        for _ in range(retained):
            state = self.cycle(state, clues)
            logits = self.symbol(state[1])
            ce = ce+F.cross_entropy(logits.float().flatten(0, 1), target.flatten())/retained
        q_cell = self.halt_logits(state[1]).float()
        q_board = -torch.logsumexp(-q_cell, -1)
        correct = (logits.detach().argmax(-1) == target)
        loss = ce+F.binary_cross_entropy_with_logits(q_cell, correct.float())
        loss = loss+F.binary_cross_entropy_with_logits(q_board, correct.all(-1).float())
        # .float(): under bf16 autocast the carried messages are bf16; keep the carried state fp32 (a no-op in fp32).
        return loss, (tuple(t.detach().float() for t in state), q_board.detach(), ce.detach())
