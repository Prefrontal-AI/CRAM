"""Score the trained CRAM models in checkpoints/ on their test sets, with the search code and the settings of the paper's
evaluations, and compare each score with the recorded one in manifest.json.

Each task's function builds the model from the checkpoint the way its evaluation script does, and calls the same search:

    sudoku-extreme     experiments/sudoku.py (search)               cram.cram_sudoku_general.GeneralSearch
    maze-unique        experiments/maze_unique.py (score)           cram.maze_unique.gibbs
    nqueens-8, -10     experiments/nqueens_offset.py (run_final)    nqueens_free.score -> cram.cram_nqueens.solve
    graph-coloring-*   experiments/graph_coloring.py (test)         cram.cram_graph_coloring.solve
    maze-hard          experiments/maze_hard_latent.py (score)      maze_hard_latent.evaluate

Run on one GPU, from the repository root:

    uv run python checkpoints/evaluate.py                                             # every model, seed 0
    uv run python checkpoints/evaluate.py --tasks maze-unique nqueens-8 --seeds 0 1 2

The datasets are read from data/, which scripts/prepare_data.py writes (README.md, "Data").
Sudoku-Extreme has 412,786 test puzzles (about 1.5 GPU hours on one H200). By default the script scores the 10,000
development puzzles instead (about 5 minutes), whose recorded run it must reproduce in every field.
`--sudoku-set test --sudoku-n N` scores the first N test puzzles. Every other task scores its whole test set.

Dimension names: q puzzles, i cells, a cell states.
"""
import dataclasses
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]

from checkpoints.load import MANIFEST, path  # noqa: E402


@dataclasses.dataclass
class Args:
    tasks: tuple[str, ...] = tuple(MANIFEST)  # models to score
    seeds: tuple[int, ...] = (0,)  # training seeds (Sudoku, Maze-Unique, N-Queens) or search seeds (graph coloring)
    sudoku_set: Literal['dev', 'test'] = 'dev'  # Sudoku-Extreme puzzles to score: the 10,000 development puzzles or test puzzles
    sudoku_n: int = 10000  # with --sudoku-set test: the number of test puzzles, from the start of the test set (412786: all)


def entry(name, seed):
    models = MANIFEST[name]['models']
    return models[seed if len(models) > 1 else 0]


def sudoku(seed, data, n):
    """As sudoku.search(split=seed, ..., scale='hbar', stop='fit'): the stopping threshold and the focus statistic
    come from the training solutions of the split, and the temperatures are in units of hbar.

    data 'dev': the 10,000 development puzzles. The search draws its random numbers for the whole batch, so the same
    batch gives the same result: the score must equal the record in every field. data 'test': the first n test puzzles;
    a batch that differs from the recorded run (n < 412,786) takes other random draws, so only the rate is compared."""
    import sudoku as F
    from cram import cram_sudoku_general as GS
    from cram import cram_sudoku_search as SS
    exp = entry('sudoku-extreme', seed)['expected']
    sd = torch.load(path('sudoku-extreme', seed), map_location='cuda', weights_only=True)
    W, B = sd['W'], sd['B']
    train_sol, _, _ = SS.load_puzzles(f'train-s{seed}')
    m = F.prepare(W, B, torch.as_tensor(train_sol, device='cuda'))
    if data == 'dev':
        sol, free, _ = SS.load_puzzles('dev10k', 0, 10000)
    else:
        sol, free, _ = SS.load_puzzles('full-all', 10000, n)  # the test set is full-all less its first 10,000 puzzles
    hi, lo, theta = exp['rel']
    unit, fz = m['hbar'], F.FROZEN
    search = GS.GeneralSearch(m['S'], B, sol, free, temp=hi * unit, theta=theta * unit, focused=True, C=fz['C'],
                              tau=m['tau_train'], tol=fz['tol'], ebar=m['ebar'], cap=fz['cap'], check=fz['check'],
                              seed=seed, anneal=(hi * unit, lo * unit, fz['anneal'][2]))
    s = SS.summarize(search.run(lambda _: None))
    keys = ('n', 'solved', 'wrong_stops', 'budget_failures', 'mean_flop')
    got = {'set': data, **{k: s[k] for k in keys}}
    want = {'set': data, **{k: (exp['dev'] if data == 'dev' else exp)[k] for k in keys}}
    if s['n'] == want['n']:
        same = got == want
    else:
        same = s['wrong_stops'] == 0 and s['solved'] / s['n'] >= want['solved'] / want['n'] - 5e-4
    return got, want, same


def maze(seed):
    """As maze_unique.main_score(run, split='test'): focused Gibbs at T = 0 from every open cell on."""
    import maze_unique as U
    e = entry('maze-unique', seed)
    cfg = U.Train(**json.loads((ROOT / e['folder'] / 'config.json').read_text()))
    model = U.make_model(cfg)
    model.load_state_dict(torch.load(path('maze-unique', seed), map_location='cpu'))
    r = U.score(model, 'test', U.Search(**e['expected']['search']))
    keys = ('n', 'solved', 'wrong_stops', 'budget_failures', 'mean_flop')
    got, want = {k: r[k] for k in keys}, {k: e['expected'][k] for k in keys}
    return got, want, got == want


def nqueens(n, seed):
    """As nqueens_offset.run_final: 20 chains per test row, the selected stopping test and temperature."""
    import nqueens_free as N
    from cram import cram_nqueens as Q
    e = entry(f'nqueens-{n}', seed)
    model = torch.load(path(f'nqueens-{n}', seed), weights_only=False)
    inputs, labels, _ = Q.load_split(n, 'test')
    r = N.score(model, inputs, labels, n, chains=20, seed=seed, **e['expected']['search'])
    keys = ('rows', 'acc_n1', 'acc_majority20', 'cov20', 'wrong_stops')
    got, want = {k: r[k] for k in keys}, {k: e['expected'][k] for k in keys}
    return got, want, all(abs(got[k] - want[k]) <= 1e-9 for k in keys)


def graph_coloring(n, seed):
    """As graph_coloring.py test: 20 samples per test graph. `seed` is the seed of the search (one fitted model)."""
    from cram import cram_graph_coloring as G
    e = entry(f'graph-coloring-{n}', 0)
    exp = next(x for x in e['expected'] if x['seed'] == seed)
    model = dict(np.load(path(f'graph-coloring-{n}')))
    model.update(json.loads((ROOT / e['folder'] / 'model.json').read_text()))
    x, y = G.load_split(n, 'test')
    tokens, _ = G.convert(x, y, n)
    r = G.score(G.adjacency(tokens, n), G.solve(model, tokens, seed=seed, **exp['search']))
    keys = ('graphs', 'conflicts_1', 'conflicts_majority_20', 'coverage_20', 'wrong_stops_all')
    got, want = {k: r[k] for k in keys}, {k: exp[k] for k in keys}
    return got, want, all(abs(got[k] - want[k]) <= 1e-9 for k in keys)


def maze_hard(seed):
    """As maze_hard_latent.score: settle the hidden layer, then damped descent of the answers, on the 1,000 test mazes."""
    import maze_hard_latent as H
    from cram import maze_h3 as m3
    e = entry('maze-hard', seed)
    rec = json.loads((ROOT / e['folder'] / 'final.json').read_text())
    model = H.build(H.Recipe(**rec['recipe'])).cuda()
    model.load_state_dict(torch.load(path('maze-hard', seed), map_location='cuda'))
    for p in model.parameters():
        p.requires_grad_(False)
    tok, y = m3.load('test')
    r = H.evaluate(model, tok, y)
    keys = ('n', 'solved', 'wrong_stops', 'budget_failures')
    got, want = {k: r[k] for k in keys}, {k: e['expected'][k] for k in keys}
    got['flop_mean'], want['flop_mean'] = r['flop_mean'], e['expected']['flop_mean']
    return got, want, all(got[k] == want[k] for k in keys) and abs(got['flop_mean'] - want['flop_mean']) <= 1e-6 * want['flop_mean']


def main(a: Args):
    failed = 0
    for name in a.tasks:
        for seed in a.seeds:
            if name == 'sudoku-extreme':
                got, want, same = sudoku(seed, a.sudoku_set, a.sudoku_n)
            elif name == 'maze-unique':
                got, want, same = maze(seed)
            elif name == 'maze-hard':
                got, want, same = maze_hard(seed)
            elif name.startswith('nqueens'):
                got, want, same = nqueens(int(name.split('-')[1]), seed)
            else:
                got, want, same = graph_coloring(int(name.split('-')[2]), seed)
            failed += not same
            print(f"{name} seed {seed}: {'MATCHES the record' if same else 'DIFFERS from the record'}", flush=True)
            print(f'  scored now: {json.dumps(got)}')
            print(f'  recorded:   {json.dumps(want)}', flush=True)
    print('every score matches its record' if failed == 0 else f'{failed} scores differ from their records')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main(tyro.cli(Args))
