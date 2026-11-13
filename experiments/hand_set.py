"""Hand-set energies on Sudoku-Extreme and Maze-Unique: energies written from the rules of each puzzle, with no learned
parameter and no training, searched by focused Gibbs. They are the hand-set rows of Table 2.

  python hand_set.py check                                   # the hand energies against the puzzle rules
  python hand_set.py sudoku --data dev --seed 0              # the 10,000 development puzzles, search seed 0
  python hand_set.py sudoku --data test --seed 0             # the 412,786 test puzzles; needs the seed's development run
  python hand_set.py maze --split dev100 --seed 0            # the 100 development mazes
  python hand_set.py maze --split test --seed 0              # the 1,000 test mazes; needs the seed's development run
  python hand_set.py rows                                    # table records and results/hand-energy.json

Sudoku. The energy counts rule violations:
    E(y) = h x (number of unordered cell pairs in one row, column or box that hold the same digit),    h = 1.
The 810 such pairs are the conflict graph. Clues are clamped and there are no input costs. A board is a solution
exactly when E = 0.
Search (cram/cram_sudoku_sparse.py SparseSearch): focused Gibbs with 4 chains per puzzle from random boards. One move
picks a free cell (a random flagged cell if any; a cell is flagged when its own local energy exceeds the focus
threshold) and resamples its digit from softmax(-local energy / T). The temperature T falls geometrically from 0.9/g
to 0.7/g over every 524,288 moves; the focus threshold is 1.0/g; g = 9 / |lambda_min(S)|, where S is the coupling
matrix. A puzzle stops when a chain reaches E <= 1e-3. The budget is 4e11 operations per puzzle. T and the focus
threshold are in units of 1/g, and g is proportional to 1/h, so the move probabilities do not depend on h.

Maze-Unique. The energy is cram.cram_maze.CRAMMazeEnergy (R 1, de 2, window 1: a 3 x 3 kernel of 2 x 2 cost tables
that is a product of a row factor, a column factor and a state factor; 16 stored values), set by hand (MAZE_VALUES) so
that the cost of "on" at an open cell is
    1.5 - (number of on side neighbours) - 0.1 x (number of on diagonal neighbours).
A move at temperature 0 from the board with every open cell on then turns off an open cell with at most one on side
neighbour and keeps a cell with two. This removes dead ends cell by cell until only the path is left.
Search (cram.maze_unique.gibbs through maze_unique.score): focused Gibbs at temperature 0 from every open cell on;
each move sets one free cell that is not at its lowest-energy answer to that answer; a maze stops at a fixed point.

`seed` is the seed of the search. A test run starts only after the development run of the same seed solved all but at
most one puzzle with no wrong stop. Outputs: runs/HAND-E1/ (the folder name is only a label).

Dimension names: n = m = 81 cells, d = 9 digits; a, c answer states (0 off, 1 on), y, x kernel rows and columns.
"""
import hashlib
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs' / 'HAND-E1'
EXPORT = HERE.parent / 'results' / 'hand-energy.json'
N, V = 81, 9

# ---------------------------------------------------------------- Sudoku
SUDOKU_H = 1.0  # the one equal-digit cost on a conflict pair
SUDOKU_SEARCH = dict(temp=0.8, theta=1.0, anneal=(0.9, 0.7, 524288), C=4, cap=4e11, check=4096, tol=1e-3)  # T, theta in units of 1/g
N_DEV, N_TEST, TEST_LO = 10000, 412786, 10000  # dev10k = full-all rows 0-9,999; test = rows 10,000-422,785


def conflict_graph():
    """(n, m) bool: cells i != j share a row, a column or a 3x3 box."""
    r, c = np.divmod(np.arange(N), 9)
    b = (r // 3) * 3 + c // 3
    same = (r[:, None] == r[None]) | (c[:, None] == c[None]) | (b[:, None] == b[None])
    np.fill_diagonal(same, False)
    return torch.as_tensor(same)


def sudoku_hand(h=SUDOKU_H):
    """(A, B) in cram_sudoku_search's form: S = A + A^T = h on the conflict pairs, 0 elsewhere; B = 0 (10 tokens x 9)."""
    A = (h / 2) * conflict_graph().float()
    return A, torch.zeros(10, V)


def dump(obj, path):
    """Write `obj` as JSON through a temporary file, so that an interrupted write leaves no partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def sudoku_out(data, seed):
    """Path of the Sudoku record of one puzzle set ('dev' or 'test') and search seed."""
    return OUT / f'sudoku-{data}-s{seed}.json'


def dev_gate(path, n, data):
    """Refuse a test run unless the development record `path` holds n puzzles, at most one unsolved, no wrong stop."""
    d = json.loads(path.read_text())
    sm = d['summary'] if 'summary' in d else d
    assert sm['n'] == n and sm['solved'] >= n - 1 and sm['wrong_stops'] == 0, f'{data} dev gate failed: {path}; do not open test'


def main_sudoku(data: Literal['dev', 'test'] = 'dev', seed: int = 0):
    """One search of the hand-set Sudoku energy over the development puzzles (dev10k) or the test puzzles (full-all
    rows 10,000 and up). Writes the settings, the summary and the per-puzzle results to runs/HAND-E1/sudoku-<data>-s<seed>.json.
    The search writes a checkpoint of its state every 10 minutes and resumes from it."""
    from cram import cram_sudoku_search as SS
    from cram.cram_sudoku_sparse import SparseSearch
    out = sudoku_out(data, seed)
    if out.exists():
        print(out, 'exists; nothing to do', flush=True)
        return
    if data == 'test':  # the test is opened only after the same seed's development run passes (as sudoku.final)
        dev_gate(sudoku_out('dev', seed), N_DEV, 'sudoku')
        sol, free, idx = SS.load_puzzles('full-all', TEST_LO, N_TEST)
        assert len(idx) == N_TEST and idx[0] == TEST_LO
    else:
        sol, free, idx = SS.load_puzzles('dev10k', 0, N_DEV)
    A, B = sudoku_hand()
    s = SUDOKU_SEARCH
    g = 9 / float(torch.linalg.eigvalsh((A + A.T).double())[0].abs())
    temp, theta = s['temp'] / g, s['theta'] / g
    anneal = (s['anneal'][0] / g, s['anneal'][1] / g, s['anneal'][2])
    # ebar: the mean own-digit field of a solution, 0 for every valid board (no same-digit conflict pair); tau = 0 (zero test)
    ss = SparseSearch(A, B, sol, free, temp=temp, theta=theta, focused=True, C=s['C'], test='pair', tau=0.0, tol=s['tol'],
                      ebar=0.0, cap=s['cap'], check=s['check'], restart=0, growth=1.0, seed=seed, anneal=anneal)
    assert ss.one_level and ss.nobias and ss.D == 20 and ss.hscale == SUDOKU_H
    signature = {'data': data, 'lo': int(idx[0]), 'n': len(idx), 'h': SUDOKU_H, 'g': g, 'temp': temp, 'theta': theta,
                 'anneal': list(anneal), 'C': s['C'], 'test': 'zero', 'tau': 0.0, 'tol': s['tol'], 'cap': s['cap'],
                 'check': s['check'], 'restart': 0, 'growth': 1.0, 'seed': seed, 'focused': True, 'sparse': True,
                 'fl_move': ss.fl_move}
    ck = out.parent / (out.stem + '-state') / 'ckpt.pt'
    resumes = 0
    if ck.exists():
        sd = torch.load(ck, map_location='cpu', weights_only=False)
        assert sd['signature'] == signature, 'checkpoint belongs to another search'
        ss.load_state_dict(sd['state'])
        resumes = sd['resumes'] + 1
        print(f'resume at move {ss.t}, {len(ss.act)} active', flush=True)
    last = [time.monotonic()]

    def on_check(st):
        if st.t % 4096 == 0 and time.monotonic() - last[0] >= 600:
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'signature': signature, 'state': st.state_dict(), 'resumes': resumes}, ck.with_suffix('.tmp.pt'))
            os.replace(ck.with_suffix('.tmp.pt'), ck)
            last[0] = time.monotonic()

    res = ss.run(on_check)
    res['puzzle'] = idx.tolist()
    summary = SS.summarize(res)
    dump({'signature': signature, 'summary': summary, 'resumes': resumes, 'gpu': torch.cuda.get_device_name(),
          'per_puzzle': res}, out)
    ck.unlink(missing_ok=True)
    if ck.parent.exists():
        ck.parent.rmdir()
    print(json.dumps({'out': str(out), **summary, 'seconds': res['seconds']}), flush=True)


# ---------------------------------------------------------------- Maze-Unique
MAZE_VALUES = {  # the 16 stored values of CRAMMazeEnergy(R=1, de=2, window=1), set by hand
    'Fp': [[0.1, 1.0, 0.1]],    # row offset factor F[dr], dr = -1, 0, 1
    'Gp': [[0.1, 1.0, 0.1]],    # column offset factor G[dc]; A(dr, dc) = F G: side 0.1, diagonal 0.01 (centre removed)
    'P': [[[0.0, 0.0], [1.0, 0.0]]],   # P[a] for a = off, on: only the on state has a factor
    'Q': [[[0.0, 0.0], [-5.0, 0.0]]],  # K = Q P^T = -5 at (on, on), 0 elsewhere: off cells and walls exert no field
    'b': [0.0, -1.5],                   # open-cell bias of (off, on): an on open cell costs 1.5
}
MAZE_JUSTIFICATION = {
    'Fp, Gp = (0.1, 1, 0.1)': 'the kernel is 2 K F[dr] G[dc]: side neighbours get 2 x 0.1 x 1 = 0.2, diagonal ones 0.02, '
                              'so diagonals weigh 1/10 of a side; a rank-1 offset product cannot make them exactly 0',
    'K(on, on) = Q P^T = -5': 'scales the side coupling to -1 per on side neighbour and the diagonal one to -0.1; '
                              'K = 0 for any off state, so walls and removed cells exert no field',
    'b = (0, -1.5)': 'an on open cell pays 1.5: it stays on iff (on sides) + 0.1 (on diagonals) > 1.5, which holds for 2 '
                     'on sides and fails for at most 1 (1 + 0.1 x 4 = 1.4 < 1.5), with no ties',
    'P, Q other entries = 0': 'no coupling involves the off state, and the second factor column is unused',
}
MAZE_SEARCH_NOTE = ('focused Gibbs at T 0 from every open cell on (maze_unique.gibbs, focus random, max 20,000 moves): '
                    'each move sets one free cell that is not at its lowest-energy answer, chosen uniformly among them, '
                    'to that answer; a maze stops at a fixed point (no such cell left)')


def maze_hand():
    """The hand-set Maze-Unique energy: CRAMMazeEnergy(R=1, de=2, window=1) with the stored values MAZE_VALUES."""
    from cram import cram_maze as C
    m = C.CRAMMazeEnergy(R=1, de=2, window=1)
    m.load_state_dict({k: torch.tensor(v) for k, v in MAZE_VALUES.items()})
    return m.eval()


def maze_out(split, seed):
    """Path of the maze record of one split ('dev100' or 'test') and search seed."""
    return OUT / f'maze-{split}-s{seed}.json'


def main_maze(split: Literal['dev100', 'test'] = 'dev100', seed: int = 0, device: str = 'cuda'):
    """Score the hand-set maze energy with maze_unique.score (focused Gibbs at temperature 0, the search of the CRAM
    row); `seed` seeds the generator of the focus."""
    import maze_unique as U
    out = maze_out(split, seed)
    if out.exists():
        print(out, 'exists; nothing to do', flush=True)
        return
    if split == 'test':
        dev_gate(maze_out('dev100', seed), 100, 'maze')
    with torch.no_grad():
        r = U.score(maze_hand(), split, U.Search('gibbs', seed=seed), device)
    r['learned_params'] = 0
    dump(r, out)
    print(split, seed, U.short(r), flush=True)


# ---------------------------------------------------------------- checks
def main_check():
    """The hand energies against the rules: the Sudoku conflict graph has 810 pairs of degree 20 and one cost level;
    the maze kernel has -1 at the side neighbours, -0.1 at the diagonal ones and 0 for the off state.
    Writes runs/HAND-E1/check.json, which `rows` copies into the records.
    Optional: where fitted equality-form energies exist in runs/CRAM-S/fits/ (fits of an earlier recipe, which this
    repository does not produce), the record also states whether each equals the hand energy up to its level
    ('cram_l1_sparse_fits'; an empty list otherwise)."""
    Cg = conflict_graph()
    A, B = sudoku_hand()
    S = A + A.T
    res = {'sudoku': {'pairs': int(Cg.triu(1).sum()), 'degree': sorted(set(Cg.sum(1).tolist())),
                      'levels': sorted(set(S.flatten().tolist())), 'bias_max': float(B.abs().max()),
                      'lambda_min_S': float(torch.linalg.eigvalsh(S.double())[0])}}
    assert res['sudoku']['pairs'] == 810 and res['sudoku']['degree'] == [20] and res['sudoku']['levels'] == [0.0, SUDOKU_H]
    fits = []
    for split in range(3):
        p = HERE / 'runs/CRAM-S/fits' / f'cid-sym-nob-l1-0.003-u32-k2-s{split}' / 'model.pt'
        if p.exists():
            sd = torch.load(p, map_location='cpu', weights_only=True)
            Sl = sd['A'] + sd['A'].T
            h = float(Sl[Sl != 0][0])
            fits.append({'split': split, 'same_support': bool(torch.equal(Sl != 0, Cg)), 'level': h,
                         'equal_to_hand_times_level': bool(torch.allclose(Sl, h * S, rtol=0, atol=1e-7)),
                         'bias_max': float(sd['B'].abs().max())})
    res['sudoku']['cram_l1_sparse_fits'] = fits
    m = maze_hand()
    W = m.kernel().detach()  # (a, c, y, x)
    res['maze'] = {'stored': m.n_params(), 'on_on': W[1, 1].tolist(), 'off_rows_zero': bool((W[0] == 0).all() and (W[:, 0] == 0).all())}
    side = W[1, 1, [0, 1, 1, 2], [1, 0, 2, 1]]
    diag = W[1, 1, [0, 0, 2, 2], [0, 2, 0, 2]]
    assert torch.allclose(side, torch.full((4,), -1.0)) and torch.allclose(diag, torch.full((4,), -0.1)) and float(W[1, 1, 1, 1]) == 0
    assert res['maze']['off_rows_zero'] and res['maze']['stored'] == 16
    dump(res, OUT / 'check.json')
    print(json.dumps(res, indent=1), flush=True)


# ---------------------------------------------------------------- records
def _commit():
    """The short hash of the repository's current commit."""
    return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=HERE).stdout.strip()


def main_rows(seeds: tuple[int, ...] = (0, 1, 2)):
    """Write the table records runs/table/<benchmark>__hand-energy.json and the export results/hand-energy.json from
    the test and development records of the search seeds `seeds` and from check.json.
    The records store the SHA-256 of this file ('script_sha256'), so any edit of this file changes that field."""
    from cram.table import write_row
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    check = json.loads((OUT / 'check.json').read_text())
    export = {'experiment': 'hand-set energies, no learned parameters, no training',
              'script': 'experiments/hand_set.py', 'script_sha256': script_sha, 'commit': _commit(),
              'written': datetime.now(timezone.utc).isoformat(timespec='seconds'), 'rows': []}

    # Sudoku
    tests = [json.loads(sudoku_out('test', s).read_text()) for s in seeds]
    devs = [json.loads(sudoku_out('dev', s).read_text())['summary'] for s in seeds]
    for t in tests:
        assert t['signature']['n'] == N_TEST and sorted(t['per_puzzle']['puzzle']) == list(range(TEST_LO, TEST_LO + N_TEST))
    sm = [t['summary'] for t in tests]
    fl = np.concatenate([np.asarray(t['per_puzzle']['flops']) for t in tests])
    q = lambda p: float(np.quantile(fl, p))
    sig = tests[0]['signature']
    sud_settings = {k: sig[k] for k in ('h', 'g', 'temp', 'theta', 'anneal', 'C', 'test', 'tau', 'tol', 'cap', 'check',
                                        'restart', 'growth', 'focused', 'sparse', 'fl_move')}
    sud_flop_conv = ('cram_sudoku_sparse.SparseSearch: per move per chain 4 D (two neighbour-field '
                     'updates) + (D + 1) (focus) + (2 d + 1) (heat bath without input costs) + 4 (energy and test) = '
                     f'{sig["fl_move"]:g} flop at D = 20, d = 9; a multiply-add is 2 flop; index, comparison and RNG '
                     'operations 0; a puzzle pays all 4 chains up to its stop')
    notes = ('Hand-set energy, no learned parameters, no training: E(y) = h x (number of row, column or box cell pairs '
             'with the same digit), h = 1, on the 810-pair conflict graph; clues clamped; no input costs. The search '
             'and stopping test: sparse focused Gibbs (cram_sudoku_sparse.SparseSearch), 4 chains from random boards, '
             'temperature falling geometrically from 0.9/g to 0.7/g over every 524,288 moves (g = 9/|lambda_min(S)| = '
             f'{sig["g"]:.4f}: T {sig["anneal"][0]:.4f} -> {sig["anneal"][1]:.4f}), focus threshold 1.0/g, cap 4e11 flop, '
             'stop at energy <= 1e-3. T and the focus threshold scale with h, so the move probabilities do not depend '
             'on h. Repeats: inference seeds 0, 1, 2.')
    path = write_row('sudoku-extreme', 'hand-energy', label='Hand-set energy (focused Gibbs)',
                     per_seed=[s['solved'] / s['n'] for s in sm], n_test=N_TEST,
                     test_set='Sudoku-Extreme official test less dev10k (412,786 puzzles), x 3 inference seeds',
                     params=0, flop_per_puzzle=statistics.fmean(s['mean_flop'] for s in sm), train_gpu_s=0.0, train_cpu_s=0.0,
                     provenance='hand_set.py; runs/HAND-E1/sudoku-{dev,test}-s{0,1,2}.json',
                     notes=notes,
                     extra={'per_seed': [{'seed': s, **x, 'dev': d, 'gpu_seconds': t['per_puzzle']['seconds']}
                                         for s, x, d, t in zip(seeds, sm, devs, tests)],
                            'runs': int(len(fl)), 'wrong_runs': int(sum(x['n'] - x['solved'] for x in sm)),
                            'wrong_stops': int(sum(x['wrong_stops'] for x in sm)),
                            'budget_failures': int(sum(x['budget_failures'] for x in sm)),
                            'flop_pooled': {'mean': float(fl.mean()), 'median': q(0.5), 'p99': q(0.99), 'p999': q(0.999), 'max': float(fl.max())},
                            'search': sud_settings, 'flop_convention': sud_flop_conv, 'params_detail':
                            'Learned: 0. Hand-set: one cost h = 1 on each of the 810 unordered conflict pairs; 0 elsewhere; no input costs.',
                            'check': check['sudoku'], 'script_sha256': script_sha})
    print(path)
    export['rows'].append({
        'benchmark': 'sudoku-extreme', 'row': 'hand-energy', 'solved': int(sum(x['solved'] for x in sm)),
        'total': int(sum(x['n'] for x in sm)), 'solve_rate': sum(x['solved'] for x in sm) / sum(x['n'] for x in sm),
        'solve_rate_per_seed': [x['solved'] / x['n'] for x in sm], 'wrong_stops': int(sum(x['wrong_stops'] for x in sm)),
        'budget_failures': int(sum(x['budget_failures'] for x in sm)),
        'flop_per_puzzle': statistics.fmean(x['mean_flop'] for x in sm), 'flop_pooled': {'median': q(0.5), 'p99': q(0.99), 'max': float(fl.max())},
        'flop_convention': sud_flop_conv, 'params': 0,
        'hand_values': {'h (equal-digit cost on each row, column or box pair)': SUDOKU_H, 'support': '810 conflict pairs',
                        'input costs': 0, 'clues': 'clamped'},
        'inference': {**sud_settings, 'seeds': list(seeds), 'start': 'random boards'},
        'test_set': 'Sudoku-Extreme official test less dev10k: full-all rows 10,000-422,785 (412,786 puzzles), x 3 inference seeds',
        'dev': [{'seed': s, 'solved': d['solved'], 'n': d['n'], 'wrong_stops': d['wrong_stops'], 'mean_flop': d['mean_flop']} for s, d in zip(seeds, devs)],
        'record': 'experiments/runs/table/sudoku-extreme__hand-energy.json'})

    # Maze-Unique
    sc = [json.loads(maze_out('test', s).read_text()) for s in seeds]
    dv = [json.loads(maze_out('dev100', s).read_text()) for s in seeds]
    mfl = np.concatenate([np.asarray(r['per_maze']['flop']) for r in sc])
    maze_flop_conv = ('maze_unique.gibbs_flop: one full current (2 x 2 x 2 x 9 x 900 = 64,800 flop) plus per '
                      'move 4 x 8 (the field change at the 8 neighbours) + 2 x 900 (focus); a multiply-add is 2 flop')
    path = write_row('maze-unique', 'hand-energy', label='Hand-set energy (focused Gibbs T=0)',
                     per_seed=[r['solved'] / r['n'] for r in sc], n_test=sc[0]['n'],
                     test_set='EqR maze-30x30-unique-1k test (1,000 mazes), x 3 inference seeds',
                     params=0, flop_per_puzzle=statistics.fmean(r['mean_flop'] for r in sc), train_gpu_s=0.0, train_cpu_s=0.0,
                     provenance='hand_set.py; runs/HAND-E1/maze-{dev100,test}-s{0,1,2}.json',
                     notes=(f'Hand-set energy, no learned parameters, no training: CRAMMazeEnergy (16 stored values) set to '
                            f'{json.dumps(MAZE_VALUES)}: an on open cell costs 1.5 - (on side neighbours) - 0.1 (on diagonal '
                            f'neighbours); walls clamped off, S and G clamped on. {MAZE_SEARCH_NOTE}. Repeats: inference seeds '
                            f'0, 1, 2 (the learned CRAM row uses focus seed 0 for each training seed).'),
                     extra={'stopped': [r['stopped'] for r in sc], 'wrong_stops': [r['wrong_stops'] for r in sc],
                            'budget_failures': [r['budget_failures'] for r in sc], 'mean_steps': [r['mean_steps'] for r in sc],
                            'max_steps': [r['max_steps'] for r in sc], 'p99_flop': float(np.quantile(mfl, 0.99)),
                            'max_flop': float(mfl.max()), 'stored_params': 16, 'learned_params': 0,
                            'dev100': [{k: r[k] for k in ('n', 'solved', 'wrong_stops', 'budget_failures', 'mean_flop')} for r in dv],
                            'flop_convention': maze_flop_conv, 'check': check['maze'], 'script_sha256': script_sha})
    print(path)
    export['rows'].append({
        'benchmark': 'maze-unique', 'row': 'hand-energy', 'solved': int(sum(r['solved'] for r in sc)),
        'total': int(sum(r['n'] for r in sc)), 'solve_rate': sum(r['solved'] for r in sc) / sum(r['n'] for r in sc),
        'solve_rate_per_seed': [r['solved'] / r['n'] for r in sc], 'wrong_stops': int(sum(r['wrong_stops'] for r in sc)),
        'budget_failures': int(sum(r['budget_failures'] for r in sc)),
        'flop_per_puzzle': statistics.fmean(r['mean_flop'] for r in sc), 'flop_max': float(mfl.max()),
        'flop_convention': maze_flop_conv, 'params': 0, 'stored_values': 16,
        'hand_values': MAZE_VALUES, 'hand_justification': MAZE_JUSTIFICATION,
        'inference': {'search': MAZE_SEARCH_NOTE, **sc[0]['search'], 'seeds': list(seeds)},
        'test_set': 'EqR maze-30x30-unique-1k test (1,000 mazes), x 3 inference seeds',
        'dev': [{'seed': s, 'solved': r['solved'], 'n': r['n'], 'wrong_stops': r['wrong_stops'], 'mean_flop': r['mean_flop']} for s, r in zip(seeds, dv)],
        'record': 'experiments/runs/table/maze-unique__hand-energy.json'})
    dump(export, EXPORT)
    EXPORT.write_text(EXPORT.read_text() + '\n')
    print(EXPORT)


if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict({'sudoku': main_sudoku, 'maze': main_maze, 'check': main_check, 'rows': main_rows},
                                         config=(tyro.conf.OmitArgPrefixes,))
