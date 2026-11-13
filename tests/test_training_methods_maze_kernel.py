"""The maze arms of the training-method study (experiments/training_methods.py) with the free 3x3 kernel
(ALT_MAZE_KERNEL=free). The folder names runs/ALT and runs/ALT-free are labels of the run folders. Ways it could be wrong:
1. a free run builds the separable model somewhere (train, test, the deferred probe of training_curves, count), because the
   kernel is not passed through: Maze(kernel='free') must hold a CRAMMazeFreeEnergy at its zero start, and the separable
   default must still build the 16-parameter separable model at its seeded start;
2. the plan lines of a free maze stage lack --maze-kernel free or write under runs/ALT/ (they would overwrite or mix
   with the separable runs), or Sudoku lines move out of runs/ALT/;
3. train accepts a maze run whose output root does not match its kernel;
4. the exports read the Sudoku rows from runs/ALT-free/ (root('sudoku') must stay runs/ALT).
The environment variable is read at import, so the env-dependent checks run in a fresh interpreter.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parents[1] / 'experiments'


def _py(code, kernel):
    """Run `code` in a fresh interpreter inside experiments/ with ALT_MAZE_KERNEL set; return its output."""
    env = dict(os.environ, ALT_MAZE_KERNEL=kernel)
    r = subprocess.run([sys.executable, '-c', code], cwd=HERE, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_maze_task_builds_the_chosen_kernel():
    import training_methods as A
    from cram import cram_maze as CM
    free = A.Maze(0, torch.device('cpu'), kernel='free')
    free.setup()
    assert isinstance(free.model, CM.CRAMMazeFreeEnergy) and free.model.n_params() == 18
    assert all(float(p.abs().sum()) == 0 for p in free.model.parameters())
    sep = A.Maze(0, torch.device('cpu'))
    sep.setup()
    torch.manual_seed(0)
    ref = CM.CRAMMazeEnergy(R=1, de=2, window=1)
    assert isinstance(sep.model, CM.CRAMMazeEnergy)
    assert all(torch.equal(a, b) for a, b in zip(sep.model.state_dict().values(), ref.state_dict().values()))
    with pytest.raises(AssertionError):
        A.Maze(0, torch.device('cpu'), kernel='other')


def test_roots_and_plan_lines():
    code = ("import json, training_methods as A; print(json.dumps({'rel': A.REL, 'flags': A.flags(dict(knob=None, l1=0.003), 'maze'), "
            "'sflags': A.flags(dict(knob=None, l1=1.0), 'sudoku'), 'tune': str(A.tune_dir('maze', 'pl').relative_to(A.HERE)), "
            "'stune': str(A.tune_dir('sudoku', 'pl').relative_to(A.HERE))}))")
    free = json.loads(_py(code, 'free'))
    assert free['rel'] == {'sudoku': 'runs/ALT', 'maze': 'runs/ALT-free'}
    assert free['flags'].endswith(' --maze-kernel free') and 'maze-kernel' not in free['sflags']
    assert free['tune'] == 'runs/ALT-free/tune/maze-pl' and free['stune'] == 'runs/ALT/tune/sudoku-pl'
    sep = json.loads(_py(code, 'separable'))
    assert sep['rel'] == {'sudoku': 'runs/ALT', 'maze': 'runs/ALT'} and 'maze-kernel' not in sep['flags']
    lines = _py("import training_methods as A; A.plan('tune', ('maze',), ('mpf',))", 'free').splitlines()
    assert len(lines) == 9 and all('--maze-kernel free' in l and '--out runs/ALT-free/tune/maze-mpf/' in l for l in lines)


def test_train_refuses_a_root_that_does_not_match_the_kernel():
    import training_methods as A
    for out, kernel in (('runs/ALT/x', 'free'), ('runs/ALT-free/x', 'separable')):
        with pytest.raises(AssertionError):
            A.train(A.Run(task='maze', method='pl', lr=0.03, l1=0.0, maze_kernel=kernel, out=out + '-never-created'))
    assert not (HERE / "runs/ALT/x-never-created").exists() and not (HERE / "runs/ALT-free/x-never-created").exists()
