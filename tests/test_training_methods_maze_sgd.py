"""The maze arms of the training-method study (experiments/training_methods.py) with SGD
(ALT_MAZE_OPT=sgd-cosine), the setting of the paper. The folder names runs/ALT, runs/ALT-free and runs/ALT-sgd are
labels of the run folders. Ways it could be wrong:
1. the SGD runs land in runs/ALT/ or runs/ALT-free/ (overwriting or mixing with the Adam runs), or Sudoku moves out of
   runs/ALT/;
2. the maze grids are not the registered ones (lr 0.1, 0.3, 1; L1 0, 0.003, 0.03; pl at lr 1 with 20 updates, the
   recipe of the Maze-Unique row), or the Sudoku grids change;
3. a plan line lacks --maze-opt sgd-cosine or --maze-kernel free, so the run trains Adam or the separable model;
4. train accepts an sgd-cosine run outside runs/ALT-sgd/, an Adam run inside it, or an sgd-cosine run in a process
   without ALT_MAZE_OPT=sgd-cosine (whose pl schedule would be the Adam reference's 100 updates);
5. ALT_MAZE_OPT=sgd-cosine with the separable kernel is accepted.
The SGD update itself (momentum 0.5, cosine decay, no clip) is not checked here.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / 'experiments'


def _py(code, kernel='free', opt='sgd-cosine', ok=True):
    """Run `code` in a fresh interpreter inside experiments/ with the two environment variables set (training_methods
    reads them at import); assert that it succeeds (or fails when ok is False) and return its output."""
    env = dict(os.environ, ALT_MAZE_KERNEL=kernel, ALT_MAZE_OPT=opt)
    r = subprocess.run([sys.executable, '-c', code], cwd=HERE, env=env, capture_output=True, text=True)
    assert (r.returncode == 0) == ok, r.stderr
    return r.stdout


def test_roots_grids_and_plan_lines():
    code = ("import json, training_methods as A; print(json.dumps({'rel': A.REL, 'lrs': A.LRS, 'l1s': A.L1S, 'ref': A.REF, "
            "'flags': A.flags(dict(knob=None, l1=0.0), 'maze'), 'sflags': A.flags(dict(knob=None, l1=1.0), 'sudoku')}))")
    d = json.loads(_py(code))
    assert d['rel'] == {'sudoku': 'runs/ALT', 'maze': 'runs/ALT-sgd'}
    assert d['lrs'] == {'sudoku': [6.0, 60.0, 540.0], 'maze': [0.1, 0.3, 1.0]}
    assert d['l1s'] == {'sudoku': [1.0, 0.1, 0.01], 'maze': [0.0, 0.003, 0.03]}
    assert d['ref'] == {'sudoku': {'lr': 540.0, 'updates': 32}, 'maze': {'lr': 1.0, 'updates': 20}}  # the recipe of the Maze-Unique row: lr 1, 20 updates
    assert d['flags'].endswith(' --maze-kernel free --maze-opt sgd-cosine') and 'maze' not in d['sflags']
    lines = _py("import training_methods as A; A.plan('tune', ('maze',), ('pl', 'cd'))").splitlines()
    pl = [l for l in lines if '--method pl' in l]
    assert len(pl) == 3 and all('--lr 1.0 ' in l for l in pl) and {l.split('--l1 ')[1].split()[0] for l in pl} == {'0.0', '0.003', '0.03'}
    assert len(lines) == 3 + 3 * 2 * 3 * 2  # cd: 3 lr x 2 knobs x 3 L1 x 2 regularizer weights
    assert all('--maze-kernel free --maze-opt sgd-cosine' in l and ' --out runs/ALT-sgd/tune/' in l for l in lines)
    a18 = json.loads(_py(code, opt='adam'))
    assert a18['rel']['maze'] == 'runs/ALT-free' and a18['lrs']['maze'] == [0.01, 0.03, 0.1] and 'sgd' not in a18['flags']


def test_train_refuses_wrong_roots_and_settings():
    bad = [("runs/ALT-free/x", 'free', 'sgd-cosine'), ("runs/ALT/x", 'free', 'sgd-cosine'), ("runs/ALT-sgd/x", 'free', 'adam'),
           ("runs/ALT-sgd/x", 'separable', 'adam')]
    for out, kernel, opt in bad:
        code = (f"import training_methods as A\ntry:\n    A.train(A.Run(task='maze', method='pl', lr=0.3, l1=0.0, maze_kernel='{kernel}', "
                f"maze_opt='{opt}', out='{out}-never-created'))\nexcept AssertionError:\n    print('refused')")
        assert _py(code).strip() == 'refused', (out, kernel, opt)
    code = ("import training_methods as A\ntry:\n    A.train(A.Run(task='maze', method='pl', lr=0.3, l1=0.0, maze_kernel='free', "
            "maze_opt='sgd-cosine', out='runs/ALT-sgd/x-never-created'))\nexcept AssertionError:\n    print('refused')")
    assert _py(code, opt='adam').strip() == 'refused'  # the process lacks ALT_MAZE_OPT=sgd-cosine
    for r in ('runs/ALT-free', 'runs/ALT-sgd', 'runs/ALT'):
        assert not (HERE / r / 'x-never-created').exists()


def test_sgd_needs_the_free_kernel():
    _py('import training_methods', kernel='separable', ok=False)
