"""CRAM on Sudoku-Extreme: the general pairwise energy, fitted on 1,000 solved puzzles and searched by focused Gibbs.

The Table 2 row (record runs/table/sudoku-extreme__cram-0-fit.json), for the training splits 0, 1 and 2:

  python sudoku.py diag                      # fit the energy on train-s0, s1, s2; writes runs/CRAM-S/full/full-aug-s<k>/model.pt
  S="--scale hbar --hi 0.40 --lo-t 0.31 --theta 0.45 --stop fit"
  python sudoku.py search --split 0 $S --data dev10k --out runs/CRAM-S/full/dev/cram0-hb-fit-s0.json   # 10,000 development puzzles
  touch runs/CRAM-S/full/test-approved       # states that the search settings are final
  python sudoku.py final --split 0 $S        # the 412,786 test puzzles, once, after the split's development run passes
  python sudoku.py time --split 0 --device cuda --stop fit   # cold and warm fit time, in a fresh process
  python sudoku.py time --split 0 --device cpu --stop fit    # the same on one CPU thread
  python sudoku.py rows --stop fit           # the table record

Dimension names: b boards, n = 81 cells, m = 81 cells, p = 3,240 unordered cell pairs, a and c digits (9).

Energy. A board assigns a digit y_i to each cell i. On a board (one-hot answers),
    E(y) = 1/2 sum_(i != j) S[i, j, y_i, y_j] + sum_i B[i, y_i].
The parameters are one free 9 x 9 cost matrix W_p per unordered cell pair p = (i < j), with S[i, j] = W_p and
S[j, i] = W_p^T, and one input cost B[i, a] per cell and digit: 3,240 x 81 + 81 x 9 = 263,169 parameters, all zero at
the start. The energy holds no rule of Sudoku: the fit must find which pairs of cells interact.
Clue cells are clamped to their digits. `S` and `B` are costs: a lower energy is a better board.

Fit (`train`). The loss is the pseudolikelihood (PL) of the labeled digit of each free (non-clue) cell, given the
labeled digits of all other cells:
    loss = -mean_(free cells i) log softmax_a( -(sum_m S[i, m, a, y_m] + B[i, a]) )[y_i].
It is evaluated at the labeled board, so the fit runs no search. There are 32 full-batch updates of SGD with
momentum 0.5 (heavy ball) and a cosine learning rate, lr_k = 540 x 1/2 (1 + cos(pi k / 32)). After each step a
proximal L1 step shrinks every entry toward 0 by lr_k x lam (soft threshold); it sets most entries to exactly 0.
Augmentation: at every update, every puzzle gets a fresh random relabeling of its digits (a permutation of 1-9
applied to its solution). Cell positions are never permuted.

Inference-ready model (`prepare`). From W, B and the training solutions: the full table S; g = 9 / |lambda_min(S)|;
D, the largest number of cells that one cell is coupled to; tau, the highest energy of a training solution; ebar, the
mean over cells of a training solution's own-digit local energy (the search's focus statistic); and hbar, the mean
positive entry of W, which is the unit of the search temperatures. The fit time includes all of these.

Search (`search`, cram/cram_sudoku_general.py GeneralSearch). Focused Gibbs sampling with 4 chains per puzzle from
random boards. One move picks a free cell (a random flagged cell if any; a cell is flagged when its own local energy
exceeds theta + ebar) and resamples its digit from softmax(-local energy / T). The temperature T falls geometrically
from hi to lo over every 524,288 moves, then restarts at hi. With `--scale hbar`, hi, lo and theta are multiples of
hbar (the paper: 0.40, 0.31 and 0.45). A puzzle stops when a chain reaches E <= tau + 1e-3 (`--stop fit`: the
stopping threshold fitted on the training solutions) and returns that board. The budget is 4e11 operations per puzzle.
`--stop zero` accepts E <= 1e-3 instead, and `--scale g` uses the fixed schedule FROZEN in units of 1/g.
`difficulty_export.py` reads test records of `--scale hbar --stop zero` (cram0-hb-s<k>.json).

Data: `train-s<k>` are the 1,000-puzzle training splits, `dev10k` the 10,000 development puzzles, and `full-all`
rows 10,000 and up the 412,786 test puzzles (scripts/prepare_data.py). The training seed equals the split index.
The folder and file names under runs/CRAM-S/full (full-aug, cram0-hb-fit, timing) are only labels.
"""
import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cram import cram_sudoku as CS
from cram import cram_sudoku_search as SS

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs/CRAM-S/full'
N, V = 81, 9


@dataclasses.dataclass
class FullConfig:
    """The settings of one fit (`train`). The defaults are the paper's recipe."""
    aug: bool = True  # fresh digit relabeling of every puzzle at every update
    lam: float = 3e-3 / 9  # L1 weight per entry of W and B, applied as a proximal soft-threshold
    lr: float = 540.0  # initial SGD learning rate on W and B (cosine decay to 0)
    momentum: float = 0.5  # heavy-ball momentum
    updates: int = 32  # full-batch updates
    split: int = 0  # training split (train-s<split>)
    seed: int = 0  # seed of the relabeling draws


def dump(obj, path):
    """Write `obj` as JSON through a temporary file, so that an interrupted write leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def pair_index(device):
    """(2, p) int64: the cells (i, j), i < j, of every unordered cell pair, in the order of the rows of W."""
    return torch.triu_indices(N, N, 1, device=device)


def full_S(W, iu):
    """(n, m, a, c) from W (p, a, c): S[i, j] = W_p and S[j, i] = W_p^T; zero on the diagonal."""
    S = torch.zeros(N, N, V, V, device=W.device, dtype=W.dtype)
    S = S.index_put((iu[0], iu[1]), W)
    return S.index_put((iu[1], iu[0]), W.transpose(1, 2))


def digit_energies(S, B, sol):
    """(b, n, a): energy of the board with cell i set to a, less a constant: sum_m S[i, m, a, y_m] + B[i, a]."""
    y = F.one_hot(sol, V).float()
    return torch.einsum('nmac,bmc->bna', S, y) + B[None]


def pl_loss(S, B, sol, free):
    """Pseudolikelihood loss (a scalar): the mean over the free cells of -log p(y_i | all other cells), where
    p(. | others) = softmax over the digits a of -(sum_m S[i, m, a, y_m] + B[i, a]).
    S: (n, m, a, c). B: (n, a). sol: (b, n) int64 digits 0-8. free: (b, n) bool, True at non-clue cells."""
    nll = -torch.log_softmax(-digit_energies(S, B, sol), -1).gather(-1, sol[..., None]).squeeze(-1)
    return (nll * free).sum() / free.sum()


def board_energy(S, B, sol):
    """(b,): E(y) on one-hot boards."""
    y = F.one_hot(sol, V).float()
    return 0.5 * torch.einsum('bna,nmac,bmc->b', y, S, y) + (y * B[None]).sum((1, 2))


def train(cfg, sol, free, draws=None):
    """Fit W and B on the solved puzzles `sol` (b, n) int64 digits 0-8 with free-cell masks `free` (b, n) bool.

    Returns (W (p, a, c), B (n, a), seconds of the update loop). W and B start at zero.
    Each update: relabel the digits of every puzzle (cfg.aug), take the gradient of the pseudolikelihood loss over the
    whole training set, take one SGD step with momentum and the cosine learning rate, then apply the proximal L1 step.
    draws: the cfg.updates uniform tensors (b, 9) of the digit relabelings, to replay a GPU fit on the CPU. None draws
    them from the default generator of sol's device, seeded with cfg.seed (cram_sudoku.augment_digit: the relabeling
    of a puzzle is the argsort of 9 uniform numbers)."""
    dev = sol.device
    iu = pair_index(dev)
    W = torch.zeros(iu.shape[1], V, V, device=dev, requires_grad=True)
    B = torch.zeros(N, V, device=dev, requires_grad=True)
    bufs = [torch.zeros_like(W), torch.zeros_like(B)]
    torch.manual_seed(cfg.seed)
    t0 = time.perf_counter()
    for k in range(cfg.updates):
        if not cfg.aug:
            s, f = sol, free
        elif draws is None:
            s, f = CS.augment_digit(sol, free)  # a fresh digit permutation per puzzle
        else:
            s, f = draws[k].to(dev).argsort(1).gather(1, sol), free  # the same permutations, from recorded draws
        gW, gB = torch.autograd.grad(pl_loss(full_S(W, iu), B, s, f), (W, B))
        lr = cfg.lr * 0.5 * (1 + math.cos(math.pi * k / cfg.updates))  # cosine decay from cfg.lr toward 0
        with torch.no_grad():
            for p, g, buf in zip((W, B), (gW, gB), bufs):
                buf.mul_(cfg.momentum).add_(g)  # heavy-ball momentum
                p.sub_(lr * buf)
                p.copy_(p.sign() * (p.abs() - lr * cfg.lam).clamp_min(0))  # proximal L1: soft threshold at lr x lam
    if dev.type == 'cuda':
        torch.cuda.synchronize()
    return W.detach(), B.detach(), time.perf_counter() - t0


def compress(W, B):
    """A diagnostic that the reported model does not use: every value of W and B replaced by the nearer of two levels
    (Lloyd k-means with 2 levels). Returns (W quantized, B quantized, levels)."""
    flat = torch.cat([W.flatten(), B.flatten()])
    q, levels = CS.kmeans_levels_values(flat, 2)
    return q[:W.numel()].view_as(W), q[W.numel():].view_as(B), levels


@torch.no_grad()
def measure(W, B, iu, peer, dev_sol, dev_free):
    """Diagnostics of a fitted energy; nothing here enters the fit or the search.

    peer: (n, m) bool, True where two cells share a row, a column or a box (the rules of Sudoku, used only to group
    the learned entries). dev_sol, dev_free: (b, n) development solutions and free-cell masks.
    Returns a dict: counts, signs and ranges of the entries of W in four groups (pairs that the rules couple or not,
    equal or different digits) and of B; whether the energy has the form "one positive cost on equal digits of coupled
    pairs"; the range of the energy of the development solutions; and how many single-cell changes of the first 1,000
    development solutions have energy <= 1e-3."""
    conflict = peer[iu[0], iu[1]]  # (p,)
    eye = torch.eye(V, dtype=torch.bool, device=W.device)
    groups = {'conflict_diag': W[conflict][:, eye], 'conflict_off': W[conflict][:, ~eye],
              'other_diag': W[~conflict][:, eye], 'other_off': W[~conflict][:, ~eye], 'B': B}
    row = {g: {'n': v.numel(), 'nonzero': int((v != 0).sum()), 'pos': int((v > 0).sum()), 'neg': int((v < 0).sum()),
               'min': float(v.min()), 'max': float(v.max())} for g, v in groups.items()}
    cd = groups['conflict_diag']
    h = float(cd[0, 0])
    row['equality_form'] = bool((cd == h).all() and h > 0 and all(row[g]['nonzero'] == 0 for g in groups if g != 'conflict_diag'))
    S = full_S(W, iu)
    E = board_energy(S, B, dev_sol)
    row['dev_E'] = {'min': float(E.min()), 'max': float(E.max())}
    row['dev_zero_accept'] = float((E <= 1e-3).float().mean())
    # every single-cell change of a free cell to another digit, first 1,000 dev solutions
    acc = tot = 0
    for lo in range(0, 1000, 100):
        s, fr = dev_sol[lo:lo + 100], dev_free[lo:lo + 100]
        G = digit_energies(S, B, s)  # E(y with i <- a) - E(y) = G[i, a] - G[i, y_i]
        dE = G - G.gather(2, s[..., None])
        mask = fr[..., None] & ~F.one_hot(s, V).bool()
        Ec = E[lo:lo + 100][:, None, None] + dE
        acc += int((Ec[mask] <= 1e-3).sum())
        tot += int(mask.sum())
    row['single_changes'] = tot
    row['single_change_zero_accept'] = acc
    return row


ARMS = {  # mode: a list of (run folder prefix, FullConfig overrides, training splits)
    'diag': [('full-aug', {}, (0, 1, 2)), ('full-noaug', {'aug': False}, (0, 1, 2))],  # the paper's recipe (full-aug) and the same fit without relabeling
    'sweep': [(f'full-aug-u{k}', {'updates': k}, (0, 1, 2)) for k in (4, 8, 16)],  # fewer updates
    'screen': [(f'full-aug-l{l}', {'lam': l * 1e-3 / 9}, (0,)) for l in (2, 4, 5)] + [('full-aug-u64', {'updates': 64}, (0,))],  # other L1 weights and more updates, split 0
}


def diag(mode='diag'):
    """Fit every arm of ARMS[mode] on its training splits and save each model with its diagnostics.

    Writes runs/CRAM-S/full/<arm>-s<split>/model.pt (W, B and their two-level quantization Wq, Bq) and final.json, and
    the list of all rows in runs/CRAM-S/full/<mode>.json. A run whose final.json exists is not fitted again.
    The models of the paper are full-aug-s0, full-aug-s1 and full-aug-s2 (the tensors W and B)."""
    dev_sol, dev_free, _ = SS.load_puzzles('dev10k')
    dev_sol, dev_free = torch.as_tensor(dev_sol, device='cuda'), torch.as_tensor(dev_free, device='cuda')
    iu = pair_index('cuda')
    peer = CS.conflict_graph('cuda')  # diagnostics only
    rows = []
    for arm, kw, splits in ARMS[mode]:
        for split in splits:
            out = ROOT / f'{arm}-s{split}'
            if (out / 'final.json').exists():
                rows.append(json.loads((out / 'final.json').read_text()))
                continue
            cfg = FullConfig(split=split, seed=split, **kw)
            sol, free, _ = SS.load_puzzles(f'train-s{split}')
            sol, free = torch.as_tensor(sol, device='cuda'), torch.as_tensor(free, device='cuda')
            W, B, secs = train(cfg, sol, free)
            Wq, Bq, levels = compress(W, B)
            row = {'arm': arm, 'split': split, 'config': dataclasses.asdict(cfg), 'loop_s': secs, 'levels': levels,
                   'raw': measure(W, B, iu, peer, dev_sol, dev_free),
                   'compressed': measure(Wq, Bq, iu, peer, dev_sol, dev_free)}
            out.mkdir(parents=True, exist_ok=True)
            torch.save({'W': W.cpu(), 'B': B.cpu(), 'Wq': Wq.cpu(), 'Bq': Bq.cpu()}, out / 'model.pt')
            dump(row, out / 'final.json')
            rows.append(row)
            print(json.dumps({k: row[k] for k in ('arm', 'split', 'levels', 'loop_s')}), flush=True)
            for kind in ('raw', 'compressed'):
                r = row[kind]
                print(' ', kind, {g: (r[g]['nonzero'], round(r[g]['min'], 4), round(r[g]['max'], 4)) for g in ('conflict_diag', 'conflict_off', 'other_diag', 'other_off', 'B')},
                      'eq', r['equality_form'], 'dev0', r['dev_zero_accept'], 'wrong', r['single_change_zero_accept'], flush=True)
    dump(rows, ROOT / f'{mode}.json')


# ---------------------------------------------------------------- the search on the fitted energy
# Search settings: C chains per puzzle; cap, the operation budget per puzzle; check, the moves between two exact
# recomputations of the fields; tol, the tolerance of the stopping test; theta and anneal = (high temperature, low
# temperature, period in moves), in units of 1/g. With --scale hbar the command line gives theta and the two
# temperatures in units of hbar, and only the period of `anneal` is used.
FROZEN = dict(C=4, cap=4e11, check=4096, tol=1e-3, theta=1.0, anneal=(0.9, 0.7, 524288))
MODEL = 'full-aug'  # the run folder prefix of the paper's fits


def ready(W, B):
    """The part of the inference-ready model that needs only the learned W (p, a, c): the full table S (n, m, a, c)
    and g = 9 / |lambda_min(S)|. Returns (S, g). `prepare` adds the statistics of the training solutions."""
    from cram import cram_sudoku_general as GS
    iu = pair_index(W.device)
    S = full_S(W, iu)
    g = 9 / abs(GS.lambda_min(S))
    return S, g


def prepare(W, B, sol):
    """The inference-ready model from the learned W (p, a, c), B (n, a) and the training solutions sol (b, n).

    Returns a dict: 'S' (n, m, a, c); 'B'; 'g' = 9 / |lambda_min(S)|; 'ebar', the mean own-digit local energy of a
    training solution (the search flags a cell whose own local energy exceeds theta + ebar); 'tau_train', the
    highest energy of a training solution (the fitted stopping threshold); 'hbar', the mean positive entry of W (the
    unit of the search temperatures with --scale hbar); 'D', the largest number of cells coupled to one cell."""
    from cram import cram_sudoku_general as GS
    S, g = ready(W, B)
    nbr, R, D = GS.neighbour_blocks(S)
    tau, ebar = GS.solution_stats(S, B, sol)  # tau: the highest energy E (input costs included) of a training solution
    hbar = float(W[W > 0].mean())  # the search's temperature unit
    return {'S': S, 'B': B, 'g': g, 'ebar': ebar, 'tau_train': tau, 'hbar': hbar, 'D': D}


def load_model(split, device='cuda', model=MODEL):
    """(W, B) of the fit runs/CRAM-S/full/<model>-s<split>/model.pt."""
    sd = torch.load(ROOT / f'{model}-s{split}' / 'model.pt', map_location=device, weights_only=True)
    return sd['W'], sd['B']


def search(split, data, lo, n, out, scale='g', hi=None, lo_t=None, theta=None, model=MODEL, stop='zero'):
    """Search puzzles lo .. lo + n - 1 of the set `data` with the model of `split`, and write the record `out`.

    scale 'g': the schedule FROZEN in units of 1/g. scale 'hbar': hi, lo_t and theta in units of hbar, the mean
    positive entry of W (the paper: 0.40, 0.31, 0.45).
    stop 'zero': a puzzle stops at E <= 0 + tol. stop 'fit': it stops at E <= tau + tol, where tau is the highest energy
    of a training solution of the split (the paper).
    The record holds the settings ('signature'), the summary (solved, wrong stops, budget failures, operation counts)
    and the per-puzzle results. An existing record is returned without a new search. The search writes a checkpoint
    of its state every 10 minutes and resumes from it. The generator seed of the search is the split index."""
    from cram import cram_sudoku_general as GS
    import hashlib
    out = Path(out)
    if out.exists():
        return json.loads(out.read_text())
    W, B = load_model(split, model=model)
    tsol, tfree, _ = SS.load_puzzles(f'train-s{split}')
    m = prepare(W, B, torch.as_tensor(tsol, device='cuda'))
    sol, free, idx = SS.load_puzzles(data, lo, n)
    g = m['g']
    hbar = m['hbar']
    tau = {'zero': 0.0, 'fit': m['tau_train']}[stop]
    period = FROZEN['anneal'][2]
    if scale == 'g':
        hi, lo_t, theta = FROZEN['anneal'][0], FROZEN['anneal'][1], FROZEN['theta']
        unit = 1 / g
    else:
        unit = hbar
    ss = GS.GeneralSearch(m['S'], B, sol, free, temp=hi * unit, theta=theta * unit, focused=True, C=FROZEN['C'],
                          tau=tau, tol=FROZEN['tol'], ebar=m['ebar'], cap=FROZEN['cap'], check=FROZEN['check'],
                          seed=split, anneal=(hi * unit, lo_t * unit, period))
    sha = hashlib.sha256((ROOT / f'{model}-s{split}' / 'model.pt').read_bytes()).hexdigest()
    signature = {'model': model, 'split': split, 'model_sha256': sha, 'data': data, 'lo': lo, 'n': len(idx), 'g': g,
                 'ebar': m['ebar'], 'D': ss.D, 'fl_move': ss.fl_move, 'test': 'zero', 'seed': split,
                 **{k: (list(v) if isinstance(v, tuple) else v) for k, v in FROZEN.items()}}
    if scale != 'g':  # these keys exist only in records of --scale hbar
        signature.update(scale=scale, hbar=hbar, anneal=[hi * unit, lo_t * unit, period], theta=theta * unit,
                         rel=[hi, lo_t, theta])
    if stop != 'zero':  # the fitted threshold; records of the zero test keep test='zero' and have no tau
        signature.update(test=stop, tau=tau)
    ck = out.parent / (out.stem + '-state') / 'ckpt.pt'
    resumes = 0
    if ck.exists():
        sd = torch.load(ck, map_location='cpu', weights_only=False)
        assert sd['signature'] == signature, 'checkpoint belongs to another search'
        ss.load_state_dict(sd['state'])
        resumes = sd['resumes'] + 1
        print(f'resume at move {ss.t}, {len(ss.act)} active', flush=True)
    last = [time.monotonic()]

    def on_check(s):
        if s.t % 4096 == 0 and time.monotonic() - last[0] >= 600:
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'signature': signature, 'state': s.state_dict(), 'resumes': resumes}, ck.with_suffix('.tmp.pt'))
            os.replace(ck.with_suffix('.tmp.pt'), ck)
            last[0] = time.monotonic()

    res = ss.run(on_check)
    res['puzzle'] = idx.tolist()
    result = {'signature': signature, 'summary': SS.summarize(res), 'resumes': resumes,
              'gpu': torch.cuda.get_device_name(), 'per_puzzle': res}
    dump(result, out)
    ck.unlink(missing_ok=True)
    if ck.parent.exists():
        ck.parent.rmdir()
    print(json.dumps({'out': str(out), **{k: signature[k] for k in ('split', 'data', 'n', 'g', 'D', 'fl_move')}, **result['summary'], 'seconds': res['seconds']}), flush=True)
    return result


def tag_of(scale, stop):
    """The file-name stem of a record: cram0 or cram0-hb (scale hbar), then -fit for the fitted stopping threshold."""
    return ('cram0' if scale == 'g' else 'cram0-hb') + ('' if stop == 'zero' else f'-{stop}')


def final(split, scale='g', hi=None, lo_t=None, theta=None, stop='zero'):
    """The test run of one split: the 412,786 test puzzles (full-all rows 10,000 and up), scored once.

    It refuses to start unless (1) the file runs/CRAM-S/full/test-approved exists, which states that the search
    settings are final, and (2) the split's development record used the same settings and solved at least 9,999 of the
    10,000 development puzzles with no wrong stop."""
    assert (ROOT / 'test-approved').exists(), 'test on hold: create runs/CRAM-S/full/test-approved to open it'
    tag = tag_of(scale, stop)
    dev = json.loads((ROOT / 'dev' / f'{tag}-s{split}.json').read_text())
    sm = dev['summary']
    if scale != 'g':
        assert dev['signature']['rel'] == [hi, lo_t, theta], ('dev and test settings differ', dev['signature']['rel'])
    assert dev['signature']['test'] == stop, ('dev and test stopping rules differ', dev['signature']['test'])
    assert dev['signature']['data'] == 'dev10k' and dev['signature']['n'] == 10000
    assert sm['solved'] >= 9999 and sm['wrong_stops'] == 0, f'dev gate failed: {sm}; do not open test'
    print(f"dev gate passed for split {split}: {sm['solved']}/10000, 0 wrong stops; test starts", flush=True)
    return search(split, 'full-all', 10000, 412786, ROOT / 'test' / f'{tag}-s{split}.json', scale, hi, lo_t, theta, stop=stop)


def timing(split, device, repeats=7, stop='zero'):
    """The paper's fit-time protocol: from training data in host memory to the inference-ready model (transfer,
    initialization, augmentation, every update, S, the neighbour blocks, g, ebar, the stopping threshold tau and the
    temperature unit hbar); file reading excluded. Cold = the first fit in this fresh process; warm = the median of
    `repeats` further fits. CUDA context creation is timed separately.
    On the CPU (one thread) the GPU's relabeling draws are replayed (read from disk before the clock), so the CPU fit is
    the same computation; run the GPU timing of a split first. The record also states whether the timed fit
    reproduced the saved model bit for bit. Output: runs/CRAM-S/full/timing/cram0[-fit]-s<split>-<device>.json
    (the folder name `baselines` is only a label)."""
    out = ROOT / 'timing' / (f'cram0-s{split}-{device}.json' if stop == 'zero' else f'cram0-{stop}-s{split}-{device}.json')
    if out.exists():
        return
    dev = torch.device(device)
    if device == 'cpu':
        torch.set_num_threads(1)
    sol_h, free_h, _ = SS.load_puzzles(f'train-s{split}')
    sol_h, free_h = torch.as_tensor(sol_h), torch.as_tensor(free_h)
    cfg = FullConfig(split=split, seed=split)
    draws_path = ROOT / 'timing' / f'draws-s{split}.pt'
    ctx = 0.0
    if device == 'cuda':
        t0 = time.perf_counter(); torch.zeros(1, device=dev); torch.cuda.synchronize(); ctx = time.perf_counter() - t0
        draws = None  # the GPU fit draws its own numbers; they are recorded for the CPU replay after the timed fits
    else:
        draws = torch.load(draws_path)

    def one():
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        sol, free = sol_h.to(dev), free_h.to(dev)
        W, B, loop_s = train(cfg, sol, free, draws)
        m = prepare(W, B, sol)
        if device == 'cuda':
            torch.cuda.synchronize()
        return time.perf_counter() - t0, loop_s, W, B, m

    cold, cold_loop, W, B, m = one()
    warm = [one()[0] for _ in range(repeats)]
    if device == 'cuda':  # the relabeling draws of the fit, for the CPU replay: recorded after the clock, not before it
        torch.manual_seed(cfg.seed)
        rec_draws = [torch.rand(len(sol_h), V, device=dev).cpu() for _ in range(cfg.updates)]
        if draws_path.exists():
            assert all(torch.equal(a, b) for a, b in zip(rec_draws, torch.load(draws_path))), 'draws differ from the recorded ones'
        draws_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rec_draws, draws_path)
    W0, B0 = load_model(split, 'cpu')
    rec = {'split': split, 'device': device, 'train_s': cold, 'loop_s': cold_loop, 'warm_median_s': sorted(warm)[len(warm) // 2],
           'warm_s': warm, 'cuda_context_s': ctx, 'threads': torch.get_num_threads(),
           'bitwise_equal_to_saved': bool(torch.equal(W.cpu(), W0) and torch.equal(B.cpu(), B0)),
           'max_abs_diff': float(max((W.cpu() - W0).abs().max(), (B.cpu() - B0).abs().max())),
           'same_support_as_saved': bool(torch.equal(W.cpu() != 0, W0 != 0)),
           'tau_train': m['tau_train'], 'ebar': m['ebar'], 'hbar': m['hbar'],
           'gpu': torch.cuda.get_device_name() if device == 'cuda' else None, 'node': os.uname().nodename}
    dump(rec, out)
    print(json.dumps(rec), flush=True)


def rows(stop='zero'):
    """Write the table record from the test, development and timing records of the three splits (scale hbar).
    stop 'fit': runs/table/sudoku-extreme__cram-0-fit.json (the paper's row); 'zero': sudoku-extreme__cram-0.json."""
    import statistics
    import numpy as np
    from cram.table import write_row
    seeds, fl, gpu, cpu = [], [], [], []
    for split in range(3):
        tag, tt = tag_of('hbar', stop), ('' if stop == 'zero' else f'-{stop}')
        t = json.loads((ROOT / 'test' / f'{tag}-s{split}.json').read_text())
        d = json.loads((ROOT / 'dev' / f'{tag}-s{split}.json').read_text())
        assert t['signature']['n'] == 412786 and sorted(t['per_puzzle']['puzzle']) == list(range(10000, 422786))
        assert t['signature']['test'] == d['signature']['test'] == stop
        gpu.append(json.loads((ROOT / 'timing' / f'cram0{tt}-s{split}-cuda.json').read_text()))
        cpu.append(json.loads((ROOT / 'timing' / f'cram0{tt}-s{split}-cpu.json').read_text()))
        if stop != 'zero':  # the timed fit computed the threshold the search used
            assert gpu[-1]['bitwise_equal_to_saved'] and gpu[-1]['tau_train'] == t['signature']['tau'] == d['signature']['tau']
            assert abs(cpu[-1]['tau_train'] - t['signature']['tau']) <= 1e-5, (cpu[-1]['tau_train'], t['signature']['tau'])
        seeds.append({'split': split, **t['summary'], 'dev': d['summary'], 'g': t['signature']['g'], 'D': t['signature']['D'],
                      'fl_move': t['signature']['fl_move'], 'gpu_seconds': t['per_puzzle']['seconds'],
                      'tau': t['signature'].get('tau', 0.0)})
        fl.append(np.asarray(t['per_puzzle']['flops']))
    fl = np.concatenate(fl)
    q = lambda p: float(np.quantile(fl, p))
    nz = []
    for split in range(3):
        W, B = load_model(split, 'cpu')
        nz.append(int((W != 0).sum() + (B != 0).sum()))
    test = {'zero': 'zero energy (E <= 1e-3)',
            'fit': 'fitted threshold: E <= tau + 1e-3, tau the highest energy of a training solution (input costs included)'}[stop]
    path = write_row(
        'sudoku-extreme', 'cram-0' + ('' if stop == 'zero' else f'-{stop}'),
        label='CRAM0 (general Eq. 1 energy, PL + L1, no post-fit processing)' + ('' if stop == 'zero' else ', fitted stopping threshold'),
        per_seed=[s['solved'] / s['n'] for s in seeds], n_test=412786,
        test_set='Sudoku-Extreme official test less dev10k (412,786 puzzles), x 3 training splits',
        params=N * (N - 1) // 2 * V * V + N * V, flop_per_puzzle=statistics.fmean(s['mean_flop'] for s in seeds),
        train_gpu_s=statistics.fmean(x['train_s'] for x in gpu), train_cpu_s=statistics.fmean(x['train_s'] for x in cpu),
        provenance='sudoku.py; runs/CRAM-S/full/',
        notes='One free 9x9 cost block per unordered cell pair and a free input cost per cell; digit relabeling; no equality '
              'prior, no compression; ' + {'zero': 'zero-energy stopping test.',
                                           'fit': 'stopping threshold fitted from the training solutions (tau + 1e-3, tau the '
                                                  'highest training-solution energy, refit per split, inside the fit clock).'}[stop],
        extra={'per_seed': seeds, 'wrong_stops': int(sum(s['wrong_stops'] for s in seeds)),
               'budget_failures': int(sum(s['budget_failures'] for s in seeds)),
               'unstopped_but_right': int(sum(s['unstopped_but_right'] for s in seeds)),
               'flop_pooled': {'mean': float(fl.mean()), 'median': q(0.5), 'p99': q(0.99), 'p999': q(0.999), 'max': float(fl.max())},
               'nonzero_params': nz,
               'search': {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in FROZEN.items() if k not in ('anneal', 'theta')},
                          'scale': 'hbar (mean positive entry of W)', 'anneal_rel': [0.40, 0.31, FROZEN['anneal'][2]], 'theta_rel': 0.45,
                          'test': test, 'tau_per_split': [s['tau'] for s in seeds]},
               'fit_gpu_cold_s': [x['train_s'] for x in gpu], 'fit_gpu_warm_s': [x['warm_median_s'] for x in gpu],
               'fit_cpu_cold_s': [x['train_s'] for x in cpu], 'fit_cpu_warm_s': [x['warm_median_s'] for x in cpu],
               'gpu_timing': gpu, 'cpu_timing': cpu})
    # the record also carries the cold and warm fit times at top level (means over the three splits)
    rec = json.loads(path.read_text())
    rec.update(fit_gpu_cold_s=rec['train_gpu_s'], fit_cpu_cold_s=rec['train_cpu_s'],
               fit_gpu_warm_s=statistics.fmean(x['warm_median_s'] for x in gpu),
               fit_cpu_warm_s=statistics.fmean(x['warm_median_s'] for x in cpu),
               fit_time_kind='measured-complete-fit')
    path.write_text(json.dumps(rec, indent=1) + '\n')
    print(path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=list(ARMS) + ['search', 'final', 'time', 'rows'])
    p.add_argument('--split', type=int, default=0)  # training split 0, 1 or 2; also the seed of the fit and of the search
    p.add_argument('--data', default='dev10k')  # search: dev10k | full-all | train-s<k>
    p.add_argument('--lo', type=int, default=0)  # search: first puzzle of --data
    p.add_argument('--n', type=int, default=None)  # search: number of puzzles (default: all from --lo)
    p.add_argument('--out', default='')  # search: output record (default: runs/CRAM-S/full/dev/cram0-s<split>.json)
    p.add_argument('--device', default='cuda')  # time: cuda | cpu
    p.add_argument('--scale', default='g', choices=['g', 'hbar'])  # unit of the temperatures and of theta: 1/g (FROZEN) or hbar
    p.add_argument('--hi', type=float, default=None)  # --scale hbar: the high temperature, in units of hbar
    p.add_argument('--lo-t', type=float, default=None)  # --scale hbar: the low temperature, in units of hbar
    p.add_argument('--theta', type=float, default=0.45)  # --scale hbar: the focus threshold, in units of hbar
    p.add_argument('--model', default=MODEL)  # run folder prefix of the fit under runs/CRAM-S/full (for example full-aug-u<k> of mode sweep)
    p.add_argument('--stop', default='zero', choices=['zero', 'fit'])  # stopping test: zero energy, or the threshold tau fitted on the training solutions
    a = p.parse_args()
    if a.mode in ARMS:
        diag(a.mode)
    elif a.mode == 'search':
        search(a.split, a.data, a.lo, a.n, a.out or ROOT / 'dev' / f'cram0-s{a.split}.json', a.scale, a.hi, a.lo_t, a.theta, a.model, a.stop)
    elif a.mode == 'final':
        final(a.split, a.scale, a.hi, a.lo_t, a.theta, a.stop)
    elif a.mode == 'time':
        timing(a.split, a.device, stop=a.stop)
    else:
        rows(a.stop)
