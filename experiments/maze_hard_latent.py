"""CRAM on Maze-Hard: an energy with one hidden layer on the graph of open cells (model: cram/cram_maze_hard_graph.py).

The paper's recipe is recipes/maze-hard-latent.json. Its name, CHE8-mu, is only the label of its run folders.

  mkdir -p runs/CRAM-HE && cp recipes/maze-hard-latent.json runs/CRAM-HE/selected-CHE8-mu.json
  python maze_hard_latent.py train --name <run> --seed 0 --recipe.<field> <value> ...   # development: fit on 900 training mazes, score the other 100
  python maze_hard_latent.py summary                                          # table of every development run in runs/CRAM-HE
  python maze_hard_latent.py fit --name CHE8-mu --seed 0 --warm 7 --triton    # final fit on all 1,000 training mazes
  python maze_hard_latent.py fitcpu --name CHE8-mu --seed 0                   # one cold fit on one CPU thread
  python maze_hard_latent.py score --name CHE8-mu --final-dir final-triton --row cram-latent-zmu   # the test set, once
  python maze_hard_latent.py row --name CHE8-mu --final-dir final-triton      # rewrite the table record from test-scores.json

Model (the values of the paper's recipe are in parentheses). Tokens x_i: 0 wall, 1 open, 2 S (start), 3 G (goal).
- Hidden layer. One hidden neuron z_i >= 0 per open cell (S and G are open), with the quadratic energy
      E_z(z; x) = 1/2 z^T A z - c^T z,    A = (4 + m) I - Adj,    c_i = 1 at G and 0 elsewhere,
  where Adj is the 4-neighbour adjacency of the open cells and m = e^mu > 0 is one learned mass (`degree=False`,
  `mass='shared'`, `mass_fn='exp'`, `source='goal'`). A is symmetric and positive definite, so E_z has one minimum
  z* = A^-1 c. The code stores psi_i = -log z_i, because z* spans hundreds of orders of magnitude in one maze. The
  Jacobi update of A z = c in this coordinate is
      psi_i <- log(4 + m) - log( sum_(j ~ i) e^(-psi_j) + c_i ),
  applied to all cells at once from psi = +inf (z = 0), until a test every 8 steps finds that no cell changed by more
  than 1e-6 (1 + |psi_i|).
- Attention. Each cell i reads the hidden layer in its window (itself and its 4 side neighbours, offsets d):
      alpha_id = softmax_d( beta_d - psi_(i+d) ),    with one learned bias beta_d per offset (`coupling='attention'`).
- Answer energy. Each cell has an answer p_i in [0, 1] (1: on the path). With `answers='mu'` the energy is the
  Maze-Unique energy (cram/cram_maze.py: a free 2 x 2 cost table per offset of the 3 x 3 window and an open-cell
  bias; walls clamped off, S and G clamped on) plus the links gated by the attention,
      sum_i sum_d kappa_d alpha_id p_i p_(i+d),    with one learned coupling kappa_d per offset.
  The model has 29 parameters: 1 (mu) + 5 (beta) + 5 (kappa) + 2 (bias) + 16 (tables).

Training (`fit_recipe`): Adam on the pseudolikelihood of the labeled answers of the free cells (`loss='pl'`), 4,000
updates of 64 mazes, learning rate 0.1. The loss reads z* through the attention. Its gradient with respect to mu is
computed at the fixed point, without differentiating through the relaxation: the adjoint lambda solves
A lambda = grad_z loss by the same Jacobi relaxation with the loss gradient as its source, and
d loss / d mu = -lambda^T d_mu(grad_z E_z)(z*). `--triton` computes the settle and the adjoint with fused Triton
kernels (cram/latent_settle_triton.py); they compute the same quantities as the PyTorch code.

Inference (`evaluate`): settle the hidden layer, compute the attention once, then run damped descent on the answer
logits u: u <- u + 0.2 (field(sigmoid(u)) - u), from u = 3 at every cell. Every 16 steps the rounded board is tested;
a maze stops when no single-cell flip of its rounded board lowers the energy. The budget is 2,048 steps.

`train` and `fit` share `fit_recipe`, which runs from the training arrays in host memory to the inference-ready model.
The hidden layer uses the fixed 4-neighbour adjacency of the open cells, so no support is estimated.
Only `score` opens the Maze-Hard test file, and it refuses to score a recipe twice.
The other values of the `Recipe` fields are the alternatives that the development runs compared with the paper's recipe.
Outputs: runs/CRAM-HE/<name>/ (development runs) and runs/CRAM-HE/final/<name>/ or final-triton/<name>/ (final fits).

Dimension names: b mazes, k offsets of a window (self first), h w grid rows and columns.
"""
import dataclasses
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs' / 'CRAM-HE'

import maze_hard_support as P  # noqa: E402
from cram import cram_maze_hard_graph as G  # noqa: E402
from cram import cram_maze_hard_simple as S  # noqa: E402
from cram import maze_h3 as m3  # noqa: E402
from cram import maze_h3p as mp  # noqa: E402

def save(obj, path):
    """Write `obj` as JSON through a temporary file, so that an interrupted write leaves no partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def sync():
    """Wait for queued GPU work, so that a wall-clock reading covers it."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def sha256(path):
    """SHA-256 of a file, as a hexadecimal string."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


CKPT_EVERY_S = float(os.environ.get("CHE_CKPT_EVERY_S", 600))  # seconds between rolling checkpoints of a fit


@dataclasses.dataclass
class Recipe:
    """The settings of one fit. Every final fit record (final-s<seed>.json) stores these fields, and `build` and the
    operation counts rebuild a model from them. The paper's values are those of recipes/maze-hard-latent.json:
    mass shared, mass_fn exp, source goal, loss pl, coupling attention, answers mu, degree False."""
    mass: Literal['token', 'shared'] = 'token'  # hidden mass m: one per token (open, S, G) or one shared
    mass_fn: Literal['exp', 'softplus'] = 'exp'  # m = e^mu or softplus(mu)
    mass0: float = 4 * math.e ** 4  # initial mass, 4 e^4
    source: Literal['token', 'goal'] = 'token'  # source c: relu(sg) per token (open, S, G), or 1 at G and 0 elsewhere
    source0: float = 1.0  # initial source pre-activation (token source)
    loss: Literal['ptr+pl', 'pl'] = 'ptr+pl'  # 'pl': pseudolikelihood (PL) of the answers alone, its gradient reaches the attention and the hidden layer; 'ptr+pl': a cross-entropy of the attention at each path cell's successor, plus PL with the attention held fixed
    coupling: Literal['attention', 'gate'] = 'attention'  # the answer links read the attention, or sigmoid gates on the edges of the maze graph
    updates: int = 4000  # Adam updates
    lr: float = 0.1  # Adam learning rate (one for every parameter)
    batch: int = 64  # mazes per update
    full_batch: bool = False  # every update uses all training mazes (no sampling; `batch` ignored)
    warm_cache: bool = False  # start each maze's hidden settle from its settled state at its previous update
    answers: Literal['h3', 'mu'] = 'h3'  # answer energy: 'h3': one bias per token and the gated links; 'mu': the Maze-Unique energy (free 3x3 tables, open bias, givens clamped) plus the gated links
    degree: bool = True  # diagonal of A: True: deg_i + m (the graph Laplacian + m I); False: 4 + m for every cell, A = (4 + m) I - Adj


def build(recipe):
    """The untrained model of a recipe (a cram_maze_hard_graph energy on the CPU)."""
    cls = G.GraphLatentEnergyMU if recipe.answers == 'mu' else G.GraphLatentEnergy
    model = cls(mass=recipe.mass, mass_fn=recipe.mass_fn, source=recipe.source, mass0=recipe.mass0,
                source0=recipe.source0, coupling=recipe.coupling)
    model.degree_in_diag = recipe.degree
    return model


def losses(model, recipe, T, Y, TT, psi):
    """The training losses of one batch, as (pointer cross-entropy or None, pseudolikelihood).

    T: (b, h, w) tokens. Y: (b, h, w) float labels (1 on the path). TT: (b, h, w) successor indices, or None with loss
    'pl'. psi: (b, h, w) settled hidden state -log z*, which carries the adjoint gradient.
    With loss 'pl' (the paper's) the pointer term is None and the gradient of the pseudolikelihood reaches beta and,
    through the adjoint, the hidden layer. With 'ptr+pl' the pseudolikelihood holds the couplings fixed."""
    if recipe.loss == 'ptr+pl':
        lu, s = S.pointer_loss(model, T, Y, TT, psi)
        a = torch.softmax(s, 1) if recipe.coupling == 'attention' else model.gates(T, psi)
        return lu, m3.pl_loss(model, T, Y, a.detach())
    pl = getattr(model, 'pl_loss', None)
    a = model.couplings(T, psi)
    return None, (pl(T, Y, a) if pl is not None else m3.pl_loss(model, T, Y, a))


def latent_summary(model):
    """The learned values that a record reports: the mass, log(2 + m) (the cost per step of psi along a corridor when
    the diagonal holds the degree), beta, kappa, the answer bias and, for a token source, the source."""
    with torch.no_grad():
        m = model.masses()
        out = {'mass': m.tolist(), 'step_cost_corridor': torch.log(2 + m).tolist(), 'beta': model.beta.tolist(),
               'kappa': model.kappa.tolist(), 'b': model.b.tolist()}
        if model.source == 'token':
            out['source'] = torch.relu(model.sg).tolist()
        return out


def count_params(model, recipe):
    """Counts of the parameters that receive a training gradient and enter the inference energy, as a dict:
    'params' (all), 'params_nonzero', 'latent' (mass and source) and 'stored_floats' (every stored tensor entry).
    Wall entries of mu and sg are unused (walls are not hidden nodes); the goal-only source is fixed (its scale cancels
    from the attention). Mass parameters enter as exp(.) or softplus(.), so they are nonzero for every stored value;
    sources enter as relu(.)."""
    nz = lambda t: int((t != 0).sum())
    n_mass = 3 if recipe.mass == 'token' else 1
    n_src = 3 if recipe.source == 'token' else 0
    nz_src = nz(torch.relu(model.sg[1:])) if recipe.source == 'token' else 0
    kap = model.kappa if recipe.coupling == 'attention' else model.kappa[1:]  # gates have no self term
    beta = model.beta if recipe.coupling == 'attention' or recipe.loss == 'ptr+pl' else model.beta[1:]
    n_T = model.T.numel() if hasattr(model, 'T') else 0
    return {'params': n_mass + n_src + beta.numel() + kap.numel() + model.b.numel() + n_T,
            'params_nonzero': n_mass + nz_src + nz(beta) + nz(kap) + nz(model.b) + (nz(model.T) if n_T else 0),
            'latent': n_mass + n_src, 'stored_floats': int(sum(p_.numel() for p_ in model.parameters()))}


def fit_recipe(tok, y, seed, recipe, device='cuda', ckpt=None, log=None, triton=False):
    """One fit, from the training arrays in host memory to the inference-ready model.

    tok: (n, h, w) int tokens. y: (n, h, w) bool labels. seed: seeds the torch generator and the numpy generator of
    the batch order. triton: use the fused kernels for the settle and the adjoint (GPU only).
    Returns (model, times, curve): the model with gradients off, a dict of timings in seconds ('setup_s', 'train_s',
    'total_s'), and a list with one row of losses and step counts per 50 updates.
    With `ckpt` (a path), a rolling checkpoint is written every CKPT_EVERY_S seconds and resumed from; `log` (a path)
    receives one JSON line per 50 updates.
    Each update: draw a batch, settle the hidden layer (no gradient tracking), compute the loss at the settled state,
    and call backward. The backward pass of the settled state runs the adjoint relaxation (module docstring)."""
    times = {'support_s': 0.0}  # the 4-neighbour adjacency is fixed: nothing is estimated
    sync()
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    model = build(recipe).to(device)
    model.triton_settle = triton and device != 'cpu'  # fused Triton settle and adjoint (cram/latent_settle_triton.py)
    T = torch.as_tensor(tok, device=device)
    Y = torch.as_tensor(y, device=device, dtype=torch.float32)
    TT = torch.as_tensor(P.successors(tok, y, model.aoffs), device=device) if recipe.loss == 'ptr+pl' else None
    opt = torch.optim.Adam([p_ for p_ in model.parameters() if p_.requires_grad], lr=recipe.lr)
    g = np.random.default_rng(seed)
    start, curve, before = 1, [], 0.0
    if ckpt is not None and ckpt.exists():
        st = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(st['model']); opt.load_state_dict(st['opt'])
        g.bit_generator.state = st['np_rng']; torch.set_rng_state(st['torch_rng'].cpu())  # map_location moved it
        start, curve, before = st['step'] + 1, st['curve'], st['elapsed_s']
        times['resumed_at'] = st['step']
        if log is not None and log.exists():
            keep = [ln for ln in log.read_text().splitlines() if 'u' in json.loads(ln) and json.loads(ln)['u'] <= st['step']]
            log.write_text('\n'.join(keep + [json.dumps({'resume': st['step']})]) + '\n')
    sync()
    t1 = time.perf_counter()
    times['setup_s'] = t1 - t0
    last_ck = time.perf_counter()
    cache = None  # (n, h, w) settled states of the training mazes (warm_cache); +inf until a maze is first settled
    if recipe.warm_cache:
        cache = torch.full(T.shape, math.inf, device=device, dtype=model.k.dtype)
    for u in range(start, recipe.updates + 1):
        idx = (torch.arange(len(tok), device=device) if recipe.full_batch
               else torch.as_tensor(g.integers(0, len(tok), recipe.batch), device=device))
        psi, steps = model.settle(T[idx], grad=True, psi0=None if cache is None else cache[idx])
        if cache is not None:
            cache[idx] = psi.detach()
        lu, lb = losses(model, recipe, T[idx], Y[idx], None if TT is None else TT[idx], psi)
        opt.zero_grad()
        (lb if lu is None else lu + lb).backward()
        opt.step()
        if u % 50 == 0 or u == recipe.updates:
            row = {'u': u, 'ptr': None if lu is None else float(lu.detach()), 'pl': float(lb.detach()), 'latent_steps_max': int(steps.max()),
                   'adjoint_steps': int(getattr(model, 'adjoint_steps', 0)), 'mass': model.masses().tolist()}
            curve.append(row)
            if log is not None:
                with open(log, 'a') as f:
                    f.write(json.dumps(row) + '\n')
        if ckpt is not None and time.perf_counter() - last_ck > CKPT_EVERY_S and u < recipe.updates:
            tmp = ckpt.with_suffix('.tmp')
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'np_rng': g.bit_generator.state,
                        'torch_rng': torch.get_rng_state(), 'step': u, 'curve': curve,
                        'elapsed_s': before + time.perf_counter() - t1}, tmp)
            os.replace(tmp, ckpt)
            last_ck = time.perf_counter()
    sync()
    times['train_s'] = before + time.perf_counter() - t1
    times['total_s'] = times['support_s'] + times['setup_s'] + times['train_s']
    for p_ in model.parameters():
        p_.requires_grad_(False)
    return model, times, curve


def spearman_all(psi, dist):
    """A diagnostic: the mean over mazes of the Spearman correlation of psi* with the breadth-first-search (BFS)
    distance to G, over every cell where both are finite (maze_h3.rank_corr also drops psi >= 5,000).
    psi, dist: (b, h, w). Returns a float."""
    out = []
    for p, q in zip(psi, dist):
        m = torch.isfinite(q) & torch.isfinite(p)
        if m.sum() < 3:
            continue
        rp, rq = p[m].double().argsort().argsort().double(), q[m].double().argsort().argsort().double()
        out.append(float(torch.corrcoef(torch.stack([rp, rq]))[0, 1]))
    return float(np.mean(out))


def evaluate(model, tok, y, dev=torch.device('cuda')):
    """Solve the mazes `tok` (n, h, w) and compare with the labels `y` (n, h, w), in batches of 100.

    Per batch: settle the hidden layer, compute the couplings (`model.couplings`: the attention, or the gates), then
    run the damped descent (step 0.2 from logits 3, fixed-point test every 16 steps, at most 2,048 steps).
    A maze is solved when its rounded board equals the label and the descent stopped at a fixed point.
    Operation count per maze: hidden steps x flop_psi_step + flop_attention + descent steps x flop_field x (1 + 1/16);
    the factor 1/16 is the fixed-point test.
    Returns a dict: 'solved', 'n', 'wrong_stops' (stopped at a board that is not the label), 'budget_failures' (not
    stopped), statistics of the operation count, and diagnostics that never enter a decision: the rank correlation of
    psi with the BFS distance to G, and how often the largest attention weight of a path cell is at its successor
    ('ptr_acc_path'; 'ptr_acc_ties' on the cells that maze_h3p.tie_cells marks). 'per_maze' holds the per-maze lists."""
    t = P.successors(tok, y, model.aoffs)
    r = {'exact': [], 'stopped': [], 'flop': []}
    cg, cga, steps, n_p, hit_p, n_t, hit_t = [], [], [], 0, 0, 0, 0
    for i in range(0, len(tok), 100):
        T = torch.as_tensor(tok[i:i + 100], device=dev)
        Y = torch.as_tensor(y[i:i + 100], device=dev, dtype=torch.float32)
        Tt = torch.as_tensor(t[i:i + 100], device=dev)
        with torch.no_grad():
            psi, ps = model.settle(T)
            s = model.att_scores(T, psi)
            a = model.couplings(T, psi)
            hit = s.argmax(1) == Tt
            ties = mp.tie_cells(T, Y)
            nu, used, done = m3.descend(model, T, a, steps=2048, eta=.2)
        bfs = m3.bfs(T, 3)
        cg.append(m3.rank_corr(psi, bfs))
        cga.append(spearman_all(psi, bfs))
        steps += ps.tolist()
        on = (Y > 0.5) & (T != 3)
        hit_p += int((hit & on).sum()); n_p += int(on.sum())
        hit_t += int((hit & ties).sum()); n_t += int(ties.sum())
        fl = ps.float() * m3.flop_psi_step(model) + m3.flop_attention(model) + used.float() * m3.flop_field(model) * (1 + 1 / 16)
        r['exact'] += ((nu == Y).flatten(1).all(1) & done).tolist(); r['stopped'] += done.tolist(); r['flop'] += fl.tolist()
    ex, sp, fl = np.array(r['exact'], bool), np.array(r['stopped'], bool), np.array(r['flop'])
    return {'solved': int(ex.sum()), 'n': len(ex), 'wrong_stops': int((sp & ~ex).sum()), 'budget_failures': int((~sp).sum()),
            'flop_mean': float(fl.mean()), 'flop_p99': float(np.quantile(fl, .99)), 'flop_max': float(fl.max()),
            'corr_G_rank_corr': float(np.nanmean(cg)), 'corr_G_all_cells': float(np.mean(cga)),
            'psi_steps_median': float(np.median(steps)), 'psi_steps_max': int(max(steps)),
            'ptr_acc_path': hit_p / max(n_p, 1), 'ptr_acc_ties': hit_t / max(n_t, 1), 'per_maze': r}


@dataclasses.dataclass
class Train:
    name: str  # run folder in runs/CRAM-HE
    seed: int = 0  # training seed (fit on the 900 fitting mazes, score the 100 development mazes)
    recipe: Recipe = dataclasses.field(default_factory=Recipe)
    triton: bool = False  # fused Triton kernels for the hidden settle and adjoint (same quantities)


def train(a: Train):
    """A development run: fit on 900 training mazes (maze_learned.dev_split), score the other 100 and write
    runs/CRAM-HE/<name>/dev-s<seed>.json. The test file is not opened."""
    out = ROOT / a.name
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f'dev-s{a.seed}.json'
    if dest.exists():
        print('done already', dest)
        return
    tok, y = m3.load('train')
    tr, dv = m3.dev_split()
    t = time.perf_counter()
    torch.zeros(1, device='cuda')
    sync()
    context_s = time.perf_counter() - t
    model, times, curve = fit_recipe(tok[tr], y[tr], a.seed, a.recipe, ckpt=out / f'ckpt-s{a.seed}.pt',
                                     log=out / f'log-s{a.seed}.jsonl', triton=a.triton)
    model.triton_settle = False  # evaluation runs the PyTorch settle
    torch.save(model.state_dict(), out / f'model-s{a.seed}.pt')
    ev = evaluate(model, tok[dv], y[dv])
    model.zero_temperature = True  # a diagnostic: the hidden layer's zero-temperature limit (min in place of soft-min)
    ev0 = evaluate(model, tok[dv], y[dv])
    model.zero_temperature = False
    rec = {'name': a.name, 'seed': a.seed, 'recipe': dataclasses.asdict(a.recipe), 'times': times,
           'cuda_context_s': context_s, 'curve': curve, 'latent': latent_summary(model),
           'params': count_params(model, a.recipe),
           'dev100': {k: v for k, v in ev.items() if k != 'per_maze'}, 'dev100_per_maze': ev['per_maze'],
           'dev100_T0': {k: v for k, v in ev0.items() if k != 'per_maze'},
           'hardware': torch.cuda.get_device_name(), 'torch': torch.__version__}
    save(rec, dest)
    (out / f'ckpt-s{a.seed}.pt').unlink(missing_ok=True)
    print(json.dumps({'name': a.name, 'seed': a.seed} | rec['dev100'] | {'train_s': times['total_s'], 'params': rec['params'],
                      'T0': {f: ev0[f] for f in ('solved', 'wrong_stops', 'budget_failures')}, 'latent': rec['latent']}),
          flush=True)


@dataclasses.dataclass
class Summary:
    pass


def summary(a: Summary):
    """Print one line per development run in runs/CRAM-HE and write them to runs/CRAM-HE/summary.json."""
    rows = []
    for d in sorted(p_ for p_ in ROOT.iterdir() if p_.is_dir() and p_.name not in ('tmp', 'final')):
        for f in sorted(d.glob('dev-s*.json')):
            r = json.loads(f.read_text())
            de = r['dev100']
            rows.append({'name': r['name'], 'seed': r['seed'], 'solved': de['solved'], 'wrong_stops': de['wrong_stops'],
                         'budget_failures': de['budget_failures'], 'flop_mean': round(de['flop_mean']),
                         'T0_solved': r['dev100_T0']['solved'], 'ptr_acc_path': round(de['ptr_acc_path'], 5),
                         'ptr_acc_ties': round(de['ptr_acc_ties'], 4), 'corr_G_all': round(de['corr_G_all_cells'], 5),
                         'train_s': round(r['times']['total_s'], 1), 'params': r['params']['params'],
                         'params_nonzero': r['params']['params_nonzero'],
                         'mass': [round(x, 1) for x in r['latent']['mass']],
                         'source': [round(x, 2) for x in r['latent'].get('source', [])],
                         'beta': [round(x, 2) for x in r['latent']['beta']]})
            print(json.dumps(rows[-1]))
    save(rows, ROOT / 'summary.json')


# --- final fits (all 1,000 training mazes; the test file is never opened) -------------------------------------------------

@dataclasses.dataclass
class Fit:
    name: str  # frozen recipe: runs/CRAM-HE/selected-<name>.json holds its Recipe
    seed: int = 0  # training seed
    warm: int = 7  # further full fits in this process for the warm time (median), each checked bitwise against the first
    triton: bool = False  # fused Triton kernels for the hidden settle and adjoint (same quantities); output final-triton/<name>


def fit(a: Fit):
    """The final fit of one seed on all 1,000 training mazes, then `warm` further fits in the same process.

    The first fit of a fresh process is the cold fit; its model is saved. Each warm fit must reproduce the cold model
    bit for bit ('bitwise_equal_to_cold'); the warm time is the median of the warm fits. Creating the CUDA context is
    timed separately ('cuda_context_s'). Writes final-s<seed>.json, final-s<seed>.pt and models.json (the SHA-256 that
    `score` checks) in runs/CRAM-HE/final/<name>/ (final-triton/<name>/ with --triton)."""
    recipe = Recipe(**json.loads((ROOT / f'selected-{a.name}.json').read_text())['recipe'])
    out = ROOT / ('final-triton' if a.triton else 'final') / a.name
    out.mkdir(parents=True, exist_ok=True)
    rec_path, model_path = out / f'final-s{a.seed}.json', out / f'final-s{a.seed}.pt'
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
    if rec is not None and len(rec.get('warm', [])) >= a.warm:
        print('done already', rec_path)
        return
    tok, y = m3.load('train')  # the training file only; reading it is excluded from the timer
    t = time.perf_counter()
    torch.zeros(1, device='cuda')
    sync()
    context_s = time.perf_counter() - t
    if rec is None:
        model, times, curve = fit_recipe(tok, y, a.seed, recipe, triton=a.triton, ckpt=out / f'ckpt-s{a.seed}.pt',
                                         log=out / f'log-s{a.seed}.jsonl')
        torch.save(model.state_dict(), model_path)
        rec = {'name': a.name, 'split': 'final', 'seed': a.seed, 'recipe': dataclasses.asdict(recipe), 'n_train': len(tok),
               'cuda_context_s': context_s, 'cold': times, 'curve': curve, 'warm': [], 'sha256': sha256(model_path),
               'latent': latent_summary(model), 'params': count_params(model, recipe),
               'hardware': torch.cuda.get_device_name(), 'torch': torch.__version__}
        save(rec, rec_path)
        (out / f'ckpt-s{a.seed}.pt').unlink(missing_ok=True)
        print(json.dumps({k: v for k, v in rec.items() if k not in ('warm', 'curve')}), flush=True)
    else:  # a resumed process: one discarded fit first, so that the warm repeats stay warm
        fit_recipe(tok, y, a.seed, recipe, triton=a.triton)
        rec.setdefault('resumed', []).append(len(rec['warm']))
    ref = torch.load(model_path, map_location='cuda')
    while len(rec['warm']) < a.warm:
        model, times, _ = fit_recipe(tok, y, a.seed, recipe, triton=a.triton)
        times['bitwise_equal_to_cold'] = all(torch.equal(v, ref[k]) for k, v in model.state_dict().items())
        rec['warm'].append(times)
        rec['warm_median_total_s'] = statistics.median(w['total_s'] for w in rec['warm'])
        save(rec, rec_path)
        print(json.dumps(times), flush=True)
    models = json.loads((out / 'models.json').read_text()) if (out / 'models.json').exists() else {}
    models[str(a.seed)] = {'path': str(model_path.relative_to(HERE)), 'sha256': rec['sha256']}
    save(models, out / 'models.json')


@dataclasses.dataclass
class FitCPU:
    name: str  # frozen recipe: runs/CRAM-HE/selected-<name>.json holds its Recipe
    seed: int = 0  # training seed
    threads: int = 1  # CPU threads
    warm: int = 0  # further full fits in the same process for the warm time (median); 0: the cold fit only


def fitcpu(a: FitCPU):
    """One cold fit on all 1,000 training mazes on the CPU. Saves the times and the largest weight difference to the GPU
    model of the same seed; no model is saved. The test file is never opened."""
    torch.set_num_threads(a.threads)
    recipe = Recipe(**json.loads((ROOT / f'selected-{a.name}.json').read_text())['recipe'])
    out = ROOT / 'final' / a.name
    out.mkdir(parents=True, exist_ok=True)
    dest = out / (f'cpu-s{a.seed}-t{a.threads}.json' if a.warm == 0 else f'cpu-s{a.seed}-t{a.threads}-warm{a.warm}.json')
    if dest.exists():
        print('done already', dest)
        return
    tok, y = m3.load('train')
    model, times, curve = fit_recipe(tok, y, a.seed, recipe, device='cpu', ckpt=out / f'ckpt-cpu-s{a.seed}.pt',
                                     log=out / f'log-cpu-s{a.seed}.jsonl')
    rec = {'name': a.name, 'seed': a.seed, 'threads': a.threads, 'times': times, 'end_curve': curve[-1],
           'cpu': open('/proc/cpuinfo').read().split('model name')[1].split('\n')[0].strip(': ')}
    if a.warm:  # the warm protocol: the cold fit above, then `warm` further fits in this process (each checked against it)
        rec['warm'] = []
        for _ in range(a.warm):
            m2, t2, _c = fit_recipe(tok, y, a.seed, recipe, device='cpu')
            t2['bitwise_equal_to_cold'] = all(torch.equal(v, model.state_dict()[k]) for k, v in m2.state_dict().items())
            rec['warm'].append(t2)
            rec['warm_median_total_s'] = statistics.median(w['total_s'] for w in rec['warm'])
            save(rec, dest.with_suffix('.partial.json'))
    gpu = out / f'final-s{a.seed}.pt'
    if gpu.exists():
        ref = torch.load(gpu, map_location='cpu')
        rec['max_abs_diff_to_gpu_model'] = max(float((v - ref[k]).abs().max()) for k, v in model.state_dict().items())
    save(rec, dest)
    (out / f'ckpt-cpu-s{a.seed}.pt').unlink(missing_ok=True)
    print(json.dumps(rec), flush=True)


# --- the single test scoring ----------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Score:
    name: str  # frozen recipe with final fits in runs/CRAM-HE/<final_dir>/<name>
    seeds: tuple[int, ...] = (0, 1, 2)  # training seeds to score
    final_dir: str = 'final'  # 'final' (PyTorch kernels) or 'final-triton' (fits with the fused kernels)
    row: str = 'cram-latent-zmu'  # table row written after scoring (runs/table/maze-hard__<row>.json)
    require_warm: bool = True  # refuse models without 7 bitwise-checked warm repeats (False: accept fits made with --warm 0)


def score(a: Score):
    """Score the final models of a frozen recipe on the 1,000 test mazes, once. Refuses a second scoring, and refuses any
    model whose SHA-256 differs from final/<name>/models.json and its fit record, before the test file is opened."""
    fin = ROOT / a.final_dir / a.name
    dest = fin / 'test-scores.json'
    if dest.exists():
        raise SystemExit(f'{dest} exists: the test split is scored once')
    selected = json.loads((ROOT / f'selected-{a.name}.json').read_text())
    models = json.loads((fin / 'models.json').read_text())
    recs = {}
    for seed in a.seeds:
        rec = json.loads((fin / f'final-s{seed}.json').read_text())
        h = sha256(fin / f'final-s{seed}.pt')
        assert str(seed) in models and h == models[str(seed)]['sha256'] == rec['sha256'], f'seed {seed}: model hash mismatch'
        assert not a.require_warm or (len(rec['warm']) >= 7 and all(w['bitwise_equal_to_cold'] for w in rec['warm'])), f'seed {seed}: warm fits'
        recs[seed] = rec
    ttok, ty = m3.load('test')  # the first and only read of the test file for this recipe
    out = {'name': a.name, 'final_dir': a.final_dir, 'row': a.row, 'selected': selected,
           'scored_utc': time.strftime('%Y-%m-%d %H:%M', time.gmtime()), 'seeds': {}}
    for seed, rec in recs.items():
        model = build(Recipe(**rec['recipe'])).cuda()
        model.load_state_dict(torch.load(fin / f'final-s{seed}.pt', map_location='cuda'))
        for p_ in model.parameters():
            p_.requires_grad_(False)
        ev = evaluate(model, ttok, ty)
        out['seeds'][seed] = {'result': {k: v for k, v in ev.items() if k != 'per_maze'}, 'per_maze': ev['per_maze'],
                              'sha256': rec['sha256'], 'params': rec['params'], 'latent': rec['latent'],
                              'train_gpu_cold_s': rec['cold']['total_s'], 'train_gpu_warm_median_s': rec.get('warm_median_total_s'),
                              'cuda_context_s': rec['cuda_context_s']}
        print(seed, {k: ev[k] for k in ('solved', 'wrong_stops', 'budget_failures', 'flop_mean', 'flop_p99', 'flop_max')}, flush=True)
    cpu = fin / 'cpu-s0-t1.json'
    out['cpu_seed0'] = json.loads(cpu.read_text()) if cpu.exists() else None
    save(out, dest)
    write_test_row(out)


@dataclasses.dataclass
class Row:
    name: str  # frozen recipe whose test-scores.json exists
    final_dir: str = 'final'  # as Score


def row(a: Row):
    """Rewrite the table record from test-scores.json, adding the CPU fit times when their records exist."""
    fin = ROOT / a.final_dir / a.name
    out = json.loads((fin / 'test-scores.json').read_text())
    cpu = fin / 'cpu-s0-t1.json'
    if out.get('cpu_seed0') is None and cpu.exists():
        out['cpu_seed0'] = json.loads(cpu.read_text())
    warm = fin / 'cpu-s0-t1-warm7.json'  # the CPU warm protocol (cold + 7 warm fits in one process)
    out['cpu_seed0_warm'] = json.loads(warm.read_text()) if warm.exists() else None
    write_test_row(out)


def write_test_row(out):
    """Write runs/table/maze-hard__<row>.json from the content of test-scores.json (`out`)."""
    from cram.table import write_row
    recipe = out['selected']['recipe']
    seeds = out['seeds']
    res = [s['result'] for s in seeds.values()]
    cpu = out['cpu_seed0']
    cpu_warm = out.get('cpu_seed0_warm')
    ks = sorted(seeds, key=int)
    cold = [seeds[k]['train_gpu_cold_s'] for k in ks]
    warm = [seeds[k]['train_gpu_warm_median_s'] for k in ks]
    path = write_row(
        'maze-hard', out.get('row', 'cram-latent-zmu'),
        label='CRAM + linear latent with its input at G (pseudolikelihood alone, adjoint at the fixed point / damped descent)',
        per_seed=[r['solved'] / r['n'] for r in res], n_test=1000,
        test_set='Maze-Hard test (sapientinc/maze-30x30-hard-1k @ 549754de), all 1,000 mazes, scored once per seed',
        params=seeds[ks[0]]['params']['params'], flop_per_puzzle=statistics.fmean(r['flop_mean'] for r in res),
        train_gpu_s=warm[0] if warm[0] is not None else cold[0], train_cpu_s=(cpu_warm['warm_median_total_s'] if cpu_warm else None if cpu is None else cpu['times']['total_s']),
        provenance=(f"recipe {out['name']} (experiments/recipes/maze-hard-latent.json); maze_hard_latent.py fit/score; "
                    f"runs/CRAM-HE/selected-{out['name']}.json, runs/CRAM-HE/{out.get('final_dir', 'final')}/{out['name']}/"),
        notes=(('Hidden layer on the open cells: A z = e_G with A = Laplacian + m I (one learned mass m = e^mu). ' if recipe['degree'] else
                'Hidden layer on the open cells: A z = e_G with A = (4 + m) I - Adj (one learned mass m = e^mu). ') +
               ('Fits with the fused Triton settle and adjoint. ' if out.get('final_dir') == 'final-triton' else '') +
               'Settled by the Jacobi relaxation in log z (all cells at once); '
               'pair costs kappa (alpha_ij + alpha_ji), alpha = softmax over self + 4 sides of beta_d + log z_(i+d). ' +
               ('Answer energy: the Maze-Unique energy (free 2x2 tables on the 3x3 window, open-cell bias) plus these pair costs. ' if recipe['answers'] == 'mu' else '') +
               'Training: pseudolikelihood of the answers alone, Adam lr 0.1 constant, 4,000 updates of 64 mazes; '
               'gradient to m by the adjoint relaxation at the fixed point (no differentiation through a solver). '
               'Inference: hidden settle, then damped descent (step 0.2 from logits 3, fixed-point certificate every 16 '
               'steps, at most 2,048 steps). Trained on all 1,000 training mazes. train_gpu_s: seed-0 warm fit (median of '
               '7 repeats after the cold fit, one H200) when warm fits were made, else its cold fit; train_cpu_s: seed-0 warm fit on one CPU thread when measured, else its cold fit. '
               'flop_per_puzzle: the counter of maze_hard_latent.evaluate; results/op-counts.json holds the count under the operation-count convention of baselines/flops/fullcount.py.'),
        extra={'fit_gpu_cold_s': cold, 'fit_gpu_warm_s': warm,
               'fit_cpu_cold_s': [None if cpu is None else cpu['times']['total_s']],
               'fit_cpu_warm_s': [cpu_warm['warm_median_total_s'] if cpu_warm else None],
               'cuda_context_s': [seeds[k]['cuda_context_s'] for k in ks],
               'wrong_stops': [r['wrong_stops'] for r in res], 'budget_failures': [r['budget_failures'] for r in res],
               'flop_p99': [r['flop_p99'] for r in res], 'flop_max': [r['flop_max'] for r in res],
               'stored_params': seeds[ks[0]]['params']['stored_floats']})
    print('wrote', path)


if __name__ == '__main__':
    commands = {'train': (Train, train), 'summary': (Summary, summary), 'fit': (Fit, fit), 'fitcpu': (FitCPU, fitcpu),
                'score': (Score, score), 'row': (Row, row)}
    cfg = tyro.extras.subcommand_cli_from_dict({k: v[0] for k, v in commands.items()})
    next(fn for cls, fn in commands.values() if type(cfg) is cls)(cfg)
