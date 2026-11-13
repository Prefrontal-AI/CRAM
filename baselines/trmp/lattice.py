"""TRMP lattice model on the Sudoku factor graph: clean-room reimplementation of arXiv 2610.18649, Appendix B.1-B.2.1.

This is the model of the paper's "ten-minute" claim (Section 5.1: "a factor graph model with 16k parameters trains on
Sudoku-extreme in about ten minutes on a single H100"), with the "Full gradient, factor graph" recipe of Table 4.
The authors' Sudoku lattice code is not public. Details that the paper leaves open follow the unit tests of their
Sudoku lattice and their Maze lattice trainer (github.com/salvatomm/discopy, branch triton_kernel_branch, commit
1695740: test/neural/test_sudoku_lattice.py, docs/neural/examples/maze/lattice.py and lattice_train.py).

Dimension names: b boards or chains, v cells (81), u units (27: rows, columns, boxes), k members of a unit (9),
p ports of a cell (3, one per unit), s symbols (9 digits), c cycles, n message width, m cell memory width,
h hidden width, a answer width.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import ExplicitGRU  # noqa: E402  (nn.GRUCell equations written with plain operators)

SOURCE = 'https://arxiv.org/html/2610.18649v1'
# Table 4, column "Full gradient, factor graph"; widths and parameter count from Table 3.
RECIPE = dict(widths=(9, 24, 48, 12), cycles=14, rounds=2, detached=0, pool=512, steps=20000, lr=6.8742e-3,
              weight_decay=0.04821, warmup=0.1009, final_lr_fraction=0.05, clip=1.0, ema=0.9995, betas=(0.9, 0.95),
              max_age=100, start_probs=(0.20, 0.55, 0.25), parameters=16543,
              validate_from=12000, validate_every=2000, dev_chains=64, dev_budget=300, dev_check_every=4,
              test_chains=64, test_budget=1000, test_check_every=1)
THETA = 0.1  # a non-given candidate whose keep probability is below this is removed
THETA_CLS_TRAIN, THETA_CLS_EVAL = 0.5, 0.8  # board conflict threshold (B.2.1)
TEMP = 1.5  # temperature of the committed symbol
W_POS, W_NEG, W_SYM, W_CONF = 4.0, 0.5, 0.2, 0.1  # Eq. 12


def sudoku_units():
    """(27, 9) cell indices: nine rows, nine columns, nine 3 x 3 boxes."""
    rows = [[9 * r + c for c in range(9)] for r in range(9)]
    cols = [[9 * r + c for r in range(9)] for c in range(9)]
    boxes = [[9 * (3 * br + i) + 3 * bc + j for i in range(3) for j in range(3)] for br in range(3) for bc in range(3)]
    return torch.tensor(rows + cols + boxes)


def cell_ports(units):
    """(81, 3) flat index u * 9 + k of each cell's place in its three units."""
    ports = [[] for _ in range(81)]
    for u, members in enumerate(units.tolist()):
        for k, cell in enumerate(members):
            ports[cell].append(9 * u + k)
    assert all(len(p) == 3 for p in ports)
    return torch.tensor(ports)


def parameter_count(widths, symbols=9, relations=1):
    """Equation 11 (P_cell + n_rel P_rel + P_ans + P_io)."""
    n, m, h, a = widths
    cell = (m + n) * h + h * h + 2 * h + 3 * m * (h + n + a + m) + 8 * m + n * m + n
    rel = 3 * n * h + h * h + 2 * h + n
    ans = 3 * a * (m + a) + 8 * a
    io = (symbols + 1) * n + 2 * (symbols + 1) * a + 2 * symbols + 1
    return cell + relations * rel + ans + io


class Unit(nn.Module):
    """Deep Sets unit, Eq. 9: phi_w = ReLU(W_phi m_w); reply_w = MLP([m_w; sum_w' phi_w']) (sum for Sudoku)."""
    def __init__(self, n, h):
        super().__init__()
        self.embed = nn.Linear(n, h)
        self.decode = nn.Sequential(nn.Linear(n + h, h), nn.ReLU(), nn.Linear(h, n))

    def forward(self, members):  # (b, u, k, n)
        pooled = self.embed(members).relu().sum(-2, keepdim=True)
        return self.decode(torch.cat((members, pooled.expand(*members.shape[:-1], -1)), -1))


@dataclass
class Heads:
    keep: torch.Tensor  # (..., b, v, s) keep logits
    sym: torch.Tensor  # (..., b, v, s) symbol logits
    conf: torch.Tensor  # (..., b, v) per-cell conflict logits; their logsumexp over cells is the board logit


class LatticeNet(nn.Module):
    """The recurrent message-passing network of B.1 on the Sudoku factor graph, with the three lattice heads of B.2.1."""
    def __init__(self, widths=RECIPE['widths'], cycles=RECIPE['cycles'], rounds=RECIPE['rounds']):
        super().__init__()
        n, m, h, a = widths
        self.widths, self.cycles, self.rounds = widths, cycles, rounds
        self.encoder = nn.Linear(9, n)  # u_v = E x_v on the multi-hot lattice row
        self.message = nn.Sequential(nn.Linear(m + n, h), nn.ReLU(), nn.Linear(h, h))  # Eq. 6
        self.gru = ExplicitGRU(h + n + a, m)  # Eq. 7
        self.norm = nn.LayerNorm(m)
        self.emit = nn.Linear(m, n)  # Eq. 8
        self.unit = Unit(n, h)  # one module shared by all 27 units
        self.answer = ExplicitGRU(m, a)  # Eq. 10
        self.answer_norm = nn.LayerNorm(a)
        self.initial_answer = nn.Parameter(torch.zeros(a))  # a_0; built outside the graph, so it stays zero
        self.keep = nn.Linear(a, 9)
        self.symbol = nn.Linear(a, 9)
        self.conflict = nn.Linear(a, 1)
        units = sudoku_units()
        self.register_buffer('units', units, persistent=False)
        self.register_buffer('ports', cell_ports(units), persistent=False)
        assert sum(p.numel() for p in self.parameters()) == parameter_count(widths) == 16543 or widths != RECIPE['widths']

    def heads(self, answer):
        return self.keep(answer), self.symbol(answer), self.conflict(answer).squeeze(-1)

    def forward(self, x, readouts='all'):
        """One deep-supervision step from the initial state on lattice x (b, v, s).

        Returns Heads stacked over the read-out cycles (c, b, v, ...): every cycle (D = 0, all differentiated) when
        readouts == 'all', else only the last cycle (inference)."""
        b, v, _ = x.shape
        n, m, h, a = self.widths
        clues = self.encoder(x)
        mem = x.new_zeros(b, v, m, dtype=clues.dtype)
        out = x.new_zeros(b, v, n, dtype=clues.dtype)  # message each cell broadcasts to its units
        incoming = x.new_zeros(b, v, 3, n, dtype=clues.dtype)  # unit replies each cell reads
        ans = self.initial_answer.detach().to(clues.dtype).expand(b, v, a)
        found = []
        for cycle in range(self.cycles):
            for _ in range(self.rounds):
                # Synchronous round: units reply to the cell messages of the previous round, cells read the replies of
                # the previous round, so a cell-to-cell exchange takes two rounds.
                replies = self.unit(out[:, self.units]).reshape(b, 27 * 9, n)[:, self.ports]  # (b, v, p, n)
                e = self.message(torch.cat((mem.unsqueeze(-2).expand(b, v, 3, m), incoming), -1)).mean(-2)
                z = torch.cat((e, clues, ans), -1)
                mem = self.norm(self.gru(z.reshape(b * v, -1), mem.reshape(b * v, m))).reshape(b, v, m)
                out, incoming = self.emit(mem), replies
            ans = self.answer_norm(self.answer(mem.reshape(b * v, m), ans.reshape(b * v, a))).reshape(b, v, a)
            if readouts == 'all' or cycle == self.cycles - 1:
                found.append(self.heads(ans))
        keep, sym, conf = (torch.stack(t) for t in zip(*found))
        return Heads(keep, sym, conf)


# ---------------------------------------------------------------- loss (Eq. 12)

def conflicts(x, y):
    """Per cell: no surviving candidate agrees with the solution; per board: any such cell."""
    cell = ~((x > 0.5) & (y > 0.5)).any(-1)
    return cell, cell.any(-1)


def lattice_loss(heads, x, y, given):
    """Eq. 12 for Sudoku, averaged over the read-out cycles.

    keep: weighted BCE (positives 4, negatives 0.5) against x * y, mean over all cells and symbols; symbol CE (0.2)
    over cells that are neither given nor on a board whose lattice has lost the solution; board conflict BCE (0.1)
    on the logsumexp of the cell conflict logits. No per-cell conflict term (Maze only)."""
    keep, sym, conf = heads.keep.float(), heads.sym.float(), heads.conf.float()
    c = keep.shape[0]
    t = x * y
    bce = -(W_POS * t * F.logsigmoid(keep) + W_NEG * (1 - t) * F.logsigmoid(-keep)).mean((1, 2, 3))
    _, board = conflicts(x, y)
    mask = (~given & ~board[:, None]).float()
    target = y.argmax(-1).expand(c, -1, -1)
    ce_cell = F.cross_entropy(sym.flatten(0, 2), target.flatten(), reduction='none').view_as(target)
    ce = (ce_cell * mask).sum((1, 2)) / mask.sum().clamp_min(1)
    conf_bce = F.binary_cross_entropy_with_logits(torch.logsumexp(conf, -1), board.float().expand(c, -1),
                                                  reduction='none').mean(-1)
    total = bce + W_SYM * ce + W_CONF * conf_bce
    return total.mean(), torch.stack((bce.mean(), ce.mean(), conf_bce.mean()))


# ---------------------------------------------------------------- projection (B.2.1, Eq. 4)

@torch.no_grad()
def project(x, given, keep, sym, conf, theta_cls, generator=None, theta=THETA):
    """Pi: remove non-given candidates with keep probability < 0.1; conflict if a cell is empty or the board conflict
    probability exceeds theta_cls; solved if every cell is a singleton and there is no conflict; otherwise commit one
    uniformly chosen undecided cell to a symbol sampled at temperature 1.5 from its surviving candidates.

    x (b, v, s) float 0/1; given (b, v); keep, sym (b, v, s) and conf (b, v) logits of the last cycle.
    theta: the removal threshold (default 0.1, B.2.1). Returns (new x, conflict, solved, kill, decided)."""
    b, v, s = x.shape
    alive = x > 0.5
    kill = alive & (torch.sigmoid(keep.float()) < theta) & ~given[..., None]
    alive = alive & ~kill
    count = alive.sum(-1)
    conflict = (count == 0).any(-1) | (torch.sigmoid(torch.logsumexp(conf.float(), -1)) > theta_cls)
    solved = (count == 1).all(-1) & ~conflict
    decided = ~conflict & ~solved
    pick = torch.where(count > 1, torch.rand(b, v, device=x.device, generator=generator), -1.0).argmax(-1)  # (b,)
    rows = torch.arange(b, device=x.device)
    row_alive = alive[rows, pick]  # (b, s)
    logits = (sym.float()[rows, pick] / TEMP).masked_fill(~row_alive, -torch.inf)
    u = torch.rand(b, s, device=x.device, generator=generator).clamp_min(1e-20)
    digit = (logits - torch.log(-torch.log(u))).argmax(-1)  # Gumbel-max sample
    new_row = torch.where(decided[:, None], F.one_hot(digit, s).bool(), row_alive)
    alive = alive.clone()
    alive[rows, pick] = new_row
    return alive.to(x.dtype), conflict, solved, kill, decided


# ---------------------------------------------------------------- training starts (B.2.1; precursor Stream.hint)

def draw_starts(solutions, clue_mask, count, probs=RECIPE['start_probs'], generator=None):
    """`count` fresh pool entries from the training puzzles on the device: a uniformly drawn puzzle, a fresh digit
    relabelling (B.4), and a start that is the raw puzzle (p 0.20), a partial correct fill (0.55: U(0,1) of the blanks)
    or a partial fill with wrong digits (0.25: U(0.1,1) of the blanks, at least one; max(1, U(0.01,0.30)) of the
    filled cells get a digit drawn uniformly from the other eight). Filled cells are givens of the entry.

    solutions (N, 81) long 0-8, clue_mask (N, 81) bool. Returns x0 (b, v, s), y (b, v, s) float."""
    dev = solutions.device
    g = dict(device=dev, generator=generator)
    ids = torch.randint(len(solutions), (count,), **g)
    sol, clue = solutions[ids], clue_mask[ids]
    perm = torch.rand(count, 9, **g).argsort(-1)
    sol = perm.gather(1, sol)
    kind = torch.multinomial(torch.tensor(probs, device=dev).expand(count, 3), 1, generator=generator).squeeze(-1)
    blanks = ~clue
    nb = blanks.sum(-1)
    frac = torch.where(kind == 1, torch.rand(count, **g), 0.1 + 0.9 * torch.rand(count, **g))
    nfill = (frac * nb).floor().long()
    nfill = torch.where(kind == 2, nfill.clamp_min(1), nfill)
    nfill = torch.where(kind == 0, 0, torch.minimum(nfill, nb))
    rank = torch.where(blanks, torch.rand(count, 81, **g), 2.0).argsort(-1).argsort(-1)
    fill = blanks & (rank < nfill[:, None])
    nbad = torch.where(kind == 2, ((0.01 + 0.29 * torch.rand(count, **g)) * nfill).floor().long().clamp_min(1), 0)
    rank_bad = torch.where(fill, torch.rand(count, 81, **g), 2.0).argsort(-1).argsort(-1)
    bad = fill & (rank_bad < nbad[:, None])
    wrong = (sol + torch.randint(1, 9, (count, 81), **g)) % 9
    digit = torch.where(bad, wrong, sol)
    given = clue | fill
    y = F.one_hot(sol, 9).float()
    x0 = torch.where(given[..., None], F.one_hot(digit, 9).float(), torch.ones_like(y))
    return x0, y


# ---------------------------------------------------------------- chain search (B.2.1, Algorithm 2)

@torch.no_grad()
def chain_search(net, x0, chains, budget, check_every, theta_cls=THETA_CLS_EVAL, slots=512, amp=True,
                 generator=None, log=None, theta=THETA):
    """Solve each puzzle with `chains` independent stochastic chains run in parallel.

    Every step runs one deep-supervision step and one projection on every chain. On a check step (every
    `check_every` steps of the puzzle) a chain in conflict restarts from the puzzle and the first chain on a solved
    board gives the answer (graded afterwards; no oracle). The budget is the number of steps per puzzle; a puzzle not
    solved within it is a timeout. Puzzles stream through `slots` puzzle slots.

    x0 (P, v, s), any number of cells and symbols. Returns digits (P, v) long (-1 for a timeout), steps (P,) long (steps run on the puzzle), resets
    (P,) long (chain restarts)."""
    dev = x0.device
    P, V, Q = x0.shape  # puzzles, cells, symbols
    S = min(slots, P)
    K = chains
    digits = torch.full((P, V), -1, dtype=torch.long, device=dev)
    steps_used = torch.zeros(P, dtype=torch.long, device=dev)
    resets = torch.zeros(P, dtype=torch.long, device=dev)
    slot_puzzle = torch.arange(S, device=dev)  # puzzle index of each slot, -1 when empty
    nxt = S
    slot_steps = torch.zeros(S, dtype=torch.long, device=dev)
    x = x0[slot_puzzle].repeat_interleave(K, 0)  # (S K, v, s)
    start = x.clone()
    ctx = torch.autocast('cuda', dtype=torch.bfloat16) if amp else torch.autocast('cuda', enabled=False)
    done_count = 0
    while True:
        active = slot_puzzle >= 0
        if not bool(active.any()):
            break
        given = start.sum(-1) == 1
        with ctx:
            hd = net(x, readouts='last')
        x, conflict, solved, _, _ = project(x, given, hd.keep[-1], hd.sym[-1], hd.conf[-1], theta_cls, generator, theta)
        slot_steps = slot_steps + active.long()
        check = (slot_steps % check_every == 0) & active  # (S,)
        check_c = check.repeat_interleave(K)
        restart = conflict & check_c
        x = torch.where(restart[:, None, None], start, x)
        solved_s = (solved & check_c).view(S, K)
        found = solved_s.any(-1) & active
        first = solved_s.float().argmax(-1)  # first solved chain of each slot
        answer = x.view(S, K, V, Q)[torch.arange(S, device=dev), first].argmax(-1)  # (S, v)
        timeout = active & ~found & (slot_steps >= budget)
        finished = found | timeout
        idx = slot_puzzle.clamp_min(0)
        resets.index_add_(0, idx, (restart.view(S, K).sum(-1) * active.long()))
        if bool(finished.any()):
            fin = finished.nonzero().squeeze(-1)
            pz = slot_puzzle[fin]
            digits[pz] = torch.where(found[fin, None], answer[fin], -1)
            steps_used[pz] = slot_steps[fin]
            done_count += len(fin)
            # refill the finished slots with the next puzzles, or empty them
            k = len(fin)
            new = torch.arange(nxt, nxt + k, device=dev)
            new = torch.where(new < P, new, -1)
            nxt += k
            slot_puzzle[fin] = new
            slot_steps[fin] = 0
            fill = torch.zeros(S, dtype=torch.bool, device=dev)
            fill[fin] = True
            fill_c = fill.repeat_interleave(K)
            fresh = x0[slot_puzzle.clamp_min(0)].repeat_interleave(K, 0)
            start = torch.where(fill_c[:, None, None], fresh, start)
            x = torch.where(fill_c[:, None, None], fresh, x)
            if log is not None:
                log(done_count, P)
    return digits, steps_used, resets
