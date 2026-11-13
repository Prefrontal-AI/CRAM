"""The test search of the data-efficiency study (data_efficiency.py): the search of the Table 2 Sudoku row on a model
that was fitted on k training puzzles.

data_efficiency.py imports this module and calls `search`. It sets the environment variable CRAM_E_ROOT (the run
folder under runs/CRAM-E) before the import.

Inference-ready model (sudoku.prepare): S, the neighbour blocks, g, ebar, hbar and tau, the highest energy of the run's
own training rows. Search (as sudoku.py --scale hbar --stop fit): focused Gibbs, 4 chains, temperatures falling from
0.40 hbar to 0.31 hbar over every 524,288 moves, focus threshold 0.45 hbar, stop at E <= tau + 1e-3, budget 4e11
operations per puzzle, checks every 4,096 moves, search seed = the training seed.
Dimension names: q puzzles, n = 81 cells.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import sudoku as CF
from cram import cram_sudoku_general as GS
from cram import cram_sudoku_search as SS

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'runs/CRAM-E' / os.environ.get('CRAM_E_ROOT', 'E1')  # the run folder; data_efficiency.py sets CRAM_E_ROOT before it imports this module
REL = (0.40, 0.31, 0.45)  # the search schedule of the Table 2 row: high and low temperature and the focus threshold, in units of hbar


def run_dir(k, seed):
    """The folder of the run with k training puzzles and training seed `seed`."""
    return ROOT / f'k{k}-s{seed}'


def dump(obj, path):
    """Write `obj` as JSON through a temporary file, so that an interrupted write leaves no partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def search(k: int, seed: int, data: str = 'screen'):
    """Search test puzzles with the fitted model of run (k, seed) and write <run folder>/<data>.json.

    data 'test10k': full-all rows 10,000-19,999, the first 10,000 test puzzles (the paper's data-efficiency figure).
    Any other value, such as 'test': all 412,786 test puzzles, full-all rows 10,000-422,785; `data` then only names
    the output file.
    The model, its training rows (train.npz) and fit.json come from data_efficiency.py fit. The record holds the
    settings ('signature'), the summary and the per-puzzle results. An existing record is not computed again; a run
    whose fit has no positive entry of W is recorded as skipped. The search writes a checkpoint of its state every 10
    minutes and resumes from it."""
    out = run_dir(k, seed) / f'{data}.json'
    if out.exists():
        print(out, 'exists', flush=True)
        return
    fitrec = json.loads((run_dir(k, seed) / 'fit.json').read_text())
    if fitrec.get('tau') is None:
        dump({'skipped': fitrec['degenerate']}, out)
        return
    dev = torch.device('cuda')
    sd = torch.load(run_dir(k, seed) / 'model.pt', map_location=dev, weights_only=True)
    W, B = sd['W'], sd['B']
    tr = np.load(run_dir(k, seed) / 'train.npz')
    m = CF.prepare(W, B, torch.as_tensor(tr['solutions'].astype(np.int64), device=dev))
    assert m['tau_train'] == fitrec['tau'] and m['hbar'] == fitrec['hbar'], 'model or training set changed since the fit'
    if data == 'test10k':  # the first 10,000 test puzzles
        sol, free, idx = SS.load_puzzles('full-all', 10000, 10000)
    else:
        sol, free, idx = SS.load_puzzles('full-all', 10000, 412786)
    hi, lo, theta = REL
    unit = m['hbar']
    period = CF.FROZEN['anneal'][2]
    ss = GS.GeneralSearch(m['S'], B, sol, free, temp=hi * unit, theta=theta * unit, focused=True, C=CF.FROZEN['C'],
                          tau=m['tau_train'], tol=CF.FROZEN['tol'], ebar=m['ebar'], cap=CF.FROZEN['cap'],
                          check=CF.FROZEN['check'], seed=seed, anneal=(hi * unit, lo * unit, period))
    signature = {'k': k, 'seed': seed, 'model_sha256': fitrec['model_sha256'], 'data': data, 'n': len(idx), 'g': m['g'],
                 'ebar': m['ebar'], 'hbar': unit, 'tau': m['tau_train'], 'D': ss.D, 'fl_move': ss.fl_move, 'rel': list(REL),
                 **{x: (list(v) if isinstance(v, tuple) else v) for x, v in CF.FROZEN.items() if x not in ('anneal', 'theta')}}
    ck = run_dir(k, seed) / f'{data}-state' / 'ckpt.pt'
    resumes = 0
    if ck.exists():
        st = torch.load(ck, map_location='cpu', weights_only=False)
        assert st['signature'] == signature, 'checkpoint belongs to another search'
        ss.load_state_dict(st['state'])
        resumes = st['resumes'] + 1
        print(f'resume at move {ss.t}, {len(ss.act)} active', flush=True)
    last = [time.monotonic()]

    def on_check(s):
        if time.monotonic() - last[0] >= 600:
            ck.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'signature': signature, 'state': s.state_dict(), 'resumes': resumes}, ck.with_suffix('.tmp.pt'))
            os.replace(ck.with_suffix('.tmp.pt'), ck)
            last[0] = time.monotonic()

    res = ss.run(on_check)
    res['puzzle'] = np.asarray(idx).tolist()
    result = {'signature': signature, 'summary': SS.summarize(res), 'resumes': resumes, 'gpu': torch.cuda.get_device_name(),
              'node': os.uname().nodename, 'per_puzzle': res}
    dump(result, out)
    ck.unlink(missing_ok=True)
    if ck.parent.exists():
        ck.parent.rmdir()
    print(json.dumps({'out': str(out), 'D': ss.D, **result['summary'], 'seconds': res['seconds']}), flush=True)
