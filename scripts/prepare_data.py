"""Download the datasets at pinned revisions and write them where the code reads them (`data/`, or `CRAM_DATA`).

  python scripts/prepare_data.py sudoku          # Sudoku-Extreme: training splits, development sets, test shards
  python scripts/prepare_data.py maze-unique     # Maze-Unique (EqR's 30x30 mazes with a unique path)
  python scripts/prepare_data.py maze-hard       # Maze-Hard (HRM's 30x30 mazes)
  python scripts/prepare_data.py multi-solution --gram-src <clone of ahn-ml/gram-release at f2e32b9>   # N-Queens, graph coloring

Sudoku-Extreme (`sapientinc/sudoku-extreme`).
`scripts/sudoku_extreme_splits.json` lists, for every split file, the rows of the official `train.csv` or `test.csv`
that it holds, in order, and a SHA-256 of its arrays; this script checks each hash.
The splits are those of the FRM reproduction that the paper compares with:
- `frm/split/train-s<k>.npz`, k = 0..19: 1,000 training puzzles, drawn from `train.csv` by FRM's loader
  (balanced by rating, subsample seed k). Table 2 uses k = 0, 1, 2.
- `frm/dev10k/dev.npz`: 10,000 development puzzles from `test.csv`. `frm/split/dev.npz` is its first 256.
- `frm/split/test.npz`: the 1,000 test puzzles that FRM's reproduction scored.
- `full-test/full-all.npz`: all 422,786 puzzles of `test.csv`, ordered as dev10k (rows 0-9,999), FRM's 1,000 test
  puzzles (10,000-10,999), then the other 411,786 in official order. The test set of the paper is rows 10,000 and up
  (412,786 puzzles that no choice saw); "the first 10,000 test puzzles" are rows 10,000-19,999.
  `full-test/full-<name>.npz` are the same puzzles in 18 shards, and `index.json` maps each shard to official rows.
- `frm/official/{train,test}.npz`: the two CSV files as arrays, in official order (the baseline scorers read them).
- `sudoku-extreme/{train,test}.csv`: the downloaded files (the TRM data builder reads them).

Every file holds `solutions` (N, 81) digits 1-9, `clue_mask` (N, 81) and `ratings` (N,).
`clue_mask` is 1 at a given cell and 0 at a blank cell; `ratings` is the dataset's difficulty rating of each puzzle.

Maze-Unique (`locuslab/EqR-data`, folder `maze-30x30-unique-1k`) and Maze-Hard (`sapientinc/maze-30x30-hard-1k`) are
downloaded unchanged; `cram/maze_unique.py` and `cram/maze_learned.py` read them.
N-Queens and graph coloring are built by GRAM's own dataset builders from a clone of GRAM's repository.

Dimension names: N puzzles of a file, 81 cells of a board in row-major order.
"""
import csv
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import tyro
from huggingface_hub import hf_hub_download, snapshot_download

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('CRAM_DATA', HERE.parent / 'data'))
MAZE_UNIQUE = dict(repo='locuslab/EqR-data', revision='bcb7d45c157c06a054b715933557d266f5bd9d7f', folder='maze-30x30-unique-1k')
MAZE_HARD = dict(repo='sapientinc/maze-30x30-hard-1k', revision='549754de3d67eebe57b6b22c7d226b9412d55b07')
GRAM_COMMIT = 'f2e32b9b11737890bfd4f39ead4cbea90835f91b'
NSHARD = 16  # shards of the test puzzles that are neither development puzzles nor FRM's test puzzles


def content_sha256(sol, clue, rat):
    """A hash of a split's arrays that does not depend on the file format or the integer width."""
    parts = (np.ascontiguousarray(sol, np.int64), np.ascontiguousarray(clue, np.uint8), np.ascontiguousarray(rat, np.int64))
    return hashlib.sha256(b''.join(p.tobytes() for p in parts)).hexdigest()


def read_csv(path):
    """Read one official CSV file (columns source, question, answer, rating; a blank cell of the question is '.').

    Returns (solutions (N, 81) uint8 digits 1-9, clue_mask (N, 81) bool, ratings (N,) int, sources (N,) str)."""
    sol, clue, rat, src = [], [], [], []
    with open(path, newline='') as f:
        rows = csv.reader(f)
        assert next(rows) == ['source', 'question', 'answer', 'rating']
        for source, question, answer, rating in rows:
            sol.append(np.frombuffer(answer.encode(), np.uint8) - ord('0'))
            clue.append(np.frombuffer(question.encode(), np.uint8) != ord('.'))
            rat.append(int(rating))
            src.append(source)
    return np.stack(sol), np.stack(clue), np.array(rat), np.array(src)


def save(path, sol, clue, rat, dtypes, **extra):
    """Write one split file: the three arrays cast to `dtypes` (a dict of numpy dtype names), plus `extra` arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, solutions=sol.astype(dtypes['solutions']), clue_mask=clue.astype(dtypes['clue_mask']), ratings=rat.astype(dtypes['ratings']), **extra)


def sudoku():
    """Download train.csv and test.csv of Sudoku-Extreme, check them against the pinned hashes, and write every split
    file that the module docstring lists. Each split is a list of official rows in `sudoku_extreme_splits.json`."""
    index = json.loads((HERE / 'sudoku_extreme_splits.json').read_text())
    official = {}
    for name in ('train', 'test'):
        path = hf_hub_download(index['dataset'], f'{name}.csv', repo_type='dataset', revision=index['revision'], local_dir=DATA / 'sudoku-extreme')
        sol, clue, rat, src = read_csv(path)
        assert len(sol) == index[f'{name}_csv_rows'] and content_sha256(sol, clue, rat) == index[f'{name}_csv_content_sha256'], f'{name}.csv is not the pinned file'
        official[name] = (sol, clue, rat)
        save(DATA / f'frm/official/{name}.npz', sol, clue, rat, dict(solutions='uint8', clue_mask='uint8', ratings='int32'), sources=src)
        print(f'{name}.csv: {len(sol):,} puzzles', flush=True)
    rows = {}
    for rel, spec in index['files'].items():
        sol, clue, rat = (a[spec['rows']] for a in official[spec['csv']])
        assert content_sha256(sol, clue, rat) == spec['content_sha256'], rel
        save(DATA / rel, sol, clue, rat, spec['dtypes'])
        rows[rel] = np.array(spec['rows'])
    dev, test = rows['frm/dev10k/dev.npz'], rows['frm/split/test.npz']
    (DATA / 'frm/dev10k/indices.json').write_text(json.dumps([int(i) for i in dev]))
    assert list(rows['frm/split/dev.npz']) == list(dev[:256]) and not set(dev) & set(test)

    # the whole official test split: development puzzles, FRM's test puzzles, then the rest in 16 shards
    sol, clue, rat = official['test']
    rest = np.setdiff1d(np.arange(len(sol)), np.concatenate([dev, test]))
    shards = {'full-dev10k': dev, 'full-frmtest': test} | {f'full-{k:02d}': part for k, part in enumerate(np.array_split(rest, NSHARD))}
    wide = dict(solutions='int64', clue_mask='int64', ratings='int64')
    order, start = {}, 0
    for name, idx in shards.items():
        save(DATA / f'full-test/{name}.npz', sol[idx], clue[idx], rat[idx], wide)
        order[name] = [start, len(idx)]
        start += len(idx)
    every = np.concatenate(list(shards.values()))
    assert start == len(sol) == len(set(every.tolist()))
    assert content_sha256(sol[every], clue[every], rat[every]) == index['full_all_content_sha256'], 'full-all.npz differs from the paper\'s test order'
    save(DATA / 'full-test/full-all.npz', sol[every], clue[every], rat[every], wide)
    (DATA / 'full-test/index.json').write_text(json.dumps(dict(
        dataset=index['dataset'], revision=index['revision'], n_official=len(sol), full_all_order=order,
        shards={name: dict(n=len(idx), official_rows=[int(i) for i in idx]) for name, idx in shards.items()})))
    for name in ('train-s0', 'train-s1', 'train-s2', 'dev', 'test'):  # the shard folder also serves as a split folder
        link = DATA / f'full-test/{name}.npz'
        if not link.exists():
            link.symlink_to(Path('..') / 'frm/split' / f'{name}.npz')
    print(f'Sudoku-Extreme written to {DATA}: 20 training splits, dev10k, {start:,} test puzzles in full-all.npz', flush=True)


def maze_unique():
    """Download the Maze-Unique folder of EqR's dataset repository to `data/eqr/`."""
    root = snapshot_download(MAZE_UNIQUE['repo'], repo_type='dataset', revision=MAZE_UNIQUE['revision'],
                             allow_patterns=f"{MAZE_UNIQUE['folder']}/*", local_dir=DATA / 'eqr')
    print(f"Maze-Unique written to {Path(root) / MAZE_UNIQUE['folder']} (cram/maze_unique.py checks the four SHA-256 values)")


def maze_hard():
    """Download train.csv and test.csv of Maze-Hard to `data/maze-hard/`."""
    for name in ('train.csv', 'test.csv'):
        hf_hub_download(MAZE_HARD['repo'], name, repo_type='dataset', revision=MAZE_HARD['revision'], local_dir=DATA / 'maze-hard')
    print(f"Maze-Hard written to {DATA / 'maze-hard'}")


@dataclasses.dataclass
class MultiSolution:
    gram_src: Path  # a clone of https://github.com/ahn-ml/gram-release at commit f2e32b9
    python: str = sys.executable  # an interpreter with GRAM's dataset dependencies (see baselines/README.md)


def multi_solution(a: MultiSolution):
    """Build GRAM's N-Queens (8x8, 10x10) and graph-coloring (8, 10 vertices) sets with GRAM's own builders, run in the
    clone `a.gram_src` (which must be at GRAM_COMMIT). Writes `data/gram/`; a set whose test labels exist is skipped."""
    head = subprocess.run(['git', '-C', str(a.gram_src), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    assert head == GRAM_COMMIT, f'{a.gram_src} is at {head}, not {GRAM_COMMIT}'
    out = DATA / 'gram'
    if 'sudoku' in str(out):
        print(f'warning: GRAM chooses its checker from the data path, and {out} contains "sudoku"; set CRAM_DATA to another '
              'folder before training GRAM, TRM or HRM on these sets', flush=True)
    for n, counts in ((8, ('5', '6', '7')), (10, ('7', '8', '9'))):
        if not (out / f'nqueens-{n}/test/all__labels.npy').exists():
            subprocess.run([a.python, 'build_nqueens_dataset.py', '--n', str(n), '--remove-counts', *counts, '--output-dir', str(out / f'nqueens-{n}')],
                           cwd=a.gram_src / 'dataset', check=True)
    for n in (8, 10):
        if not (out / f'graph-coloring-{n}/test/all__labels.npy').exists():
            subprocess.run([a.python, 'build_graph_coloring_dataset.py', '--config-name', f'graph-coloring-{n}', '--output-dir', str(out / f'graph-coloring-{n}')],
                           cwd=a.gram_src / 'dataset', check=True)
    print(f'N-Queens and graph coloring written to {out}')


if __name__ == '__main__':
    tyro.extras.subcommand_cli_from_dict({'sudoku': sudoku, 'maze-unique': maze_unique, 'maze-hard': maze_hard, 'multi-solution': multi_solution})
