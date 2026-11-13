"""Data efficiency of the baselines on Sudoku-Extreme: the solve rates of TRM, EBR and FRM on the first 10,000 test
puzzles, per number k of unique training puzzles and per training seed. This script only collects saved scores.

  DE_VARIANTS=99 python data_efficiency_baselines.py   # the paper's figure: writes ../results/sudoku-data-eff-baselines-v99.json

test10k is full-all rows 10,000-19,999; baselines/ebr/report.py sweep_official_rows() gives their official test rows.
DE_VARIANTS=99 (each training puzzle stored with 99 symmetry variants, 100 rows per puzzle; k up to 100,000):
  TRM  experiments/runs/trm/de2-v99/n<k>-s<s>: score.right.npy holds one flag per official test row (422,786), so
       test10k is its mean over those rows.
  EBR  baselines/runs/ebr/de2-v99/n<k>-s<s>: baselines/ebr/de1_report.py on its score10k folder (EBR's evaluate.py on the
       same 10,000 puzzles), setting gtan0.2-cap64-N5 (the setting of the paper's Table 2); D64-N1 and D64-N5 are
       kept beside it.
  FRM  baselines/runs/frm/de2-v99/n<k>-s<s>/runs/s<s>/test10k.json (baselines/frm/de2_adapter.py finish).
DE_VARIANTS unset (1,000 stored variants per puzzle): TRM at k = 1,000 in baselines/runs/trm/sudoku-s<s>, other k in
experiments/runs/trm/smalldata/n<k>-s<s>; EBR in baselines/runs/ebr/de1/n<k>-s<s>; no FRM runs; writes
../results/sudoku-data-eff-baselines.json.
DE_PARTIAL=1 leaves out runs that have not finished and records them under 'missing' (a preview export).
TRM at k = 30,000 makes 65,117 updates (baselines/trm/data_efficiency.sbatch sets EP = EI = 1,667 epochs:
int(1667 * 30000 / 768); no epoch setting gives 65,100).
The folder names de1 and de2-v99 are only labels.
"""
import os
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'baselines' / 'ebr'))
from report import sweep_official_rows  # noqa: E402

VARIANTS = int(os.environ.get('DE_VARIANTS', '1000'))  # stored variants per training puzzle (the paper's figure: 99)
DE2 = VARIANTS != 1000  # True: the runs with fewer stored variants (folders de2-v<VARIANTS>)
KS = tuple(sorted((1, 3, 10, 30, 100, 300, 1000, 3000, 10000) + ((50, 70, 150, 200, 500, 700, 30000, 100000) if DE2 else ())))  # numbers of unique training puzzles
TRM_UPDATES = {30000: 65117}  # k whose TRM run does not make 65,100 updates (module docstring)
SEEDS = (0, 1, 2, 3, 4, 5) if DE2 else (0, 1, 2)  # training seeds
EBR_EXTRA_SEEDS = {}  # k: further seeds that only EBR ran (none)
PARTIAL = os.environ.get('DE_PARTIAL') == '1'
EBR_KEYS = ('gtan0.2-cap64-N5', 'D64-N1', 'D64-N5')  # EBR inference settings read from each report


def sha(p):
    """SHA-256 of a file, as a hexadecimal string."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ebr_scored(d):
    """The test10k scoring job of an EBR run has finished (its output folder holds evaluate.py's results)."""
    return (d / 'fit.json').exists() and any((d / 'score10k').glob('*.json'))


def trm_dir(k, s):
    """The folder of the TRM run with k training puzzles and seed s."""
    if DE2:
        return HERE / f'runs/trm/de2-v{VARIANTS}' / f'n{k}-s{s}'
    return ROOT / 'baselines/runs/trm' / f'sudoku-s{s}' if k == 1000 else HERE / 'runs/trm/smalldata' / f'n{k}-s{s}'


def main(out: str = ''):
    """Collect one row per (method, k, seed) and write the JSON. Each row holds the run folder, the hash of the score
    file, the number of training updates, the fit time and the test10k solve rate. The script checks that every run
    made its scheduled number of updates and, for EBR, that the report scored the final checkpoint."""
    out = out or ('../results/sudoku-data-eff-baselines-v99.json' if DE2 else '../results/sudoku-data-eff-baselines.json')
    rows10k = sweep_official_rows()
    assert len(rows10k) == 10000 and len(set(rows10k.tolist())) == 10000
    rows, missing = [], []
    for k in KS:
        for s in SEEDS + EBR_EXTRA_SEEDS.get(k, ()):
            if s in SEEDS and PARTIAL and not (trm_dir(k, s) / 'score.json').exists():
                missing.append(('TRM', k, s))
            elif s in SEEDS:  # TRM: seeds 0-2; EBR_EXTRA_SEEDS are EBR runs only
                d = trm_dir(k, s)
                right = np.load(d / 'score.right.npy')
                assert right.shape == (422786,), right.shape
                fit = json.loads((d / 'fit.json').read_text())
                want = TRM_UPDATES.get(k, 65100) if DE2 else 65100
                assert min(fit['steps'], fit['total_steps']) == want and (d / f'step_{fit["steps"]}').exists(), (d, fit['steps'], fit['total_steps'])  # total_steps is the scheduler's rounded total (65,100-65,105)
                full = json.loads((d / 'score.json').read_text())
                assert abs(full['exact'] - right.mean()) < 1e-9, d
                sub = right[rows10k]
                rows.append(dict(method='TRM', k=k, seed=s, run=str(d), score_right_sha256=sha(d / 'score.right.npy'), steps=fit['steps'], scheduled_steps=fit['total_steps'], updates=min(fit['steps'], fit['total_steps']),
                                 fit_seconds=fit['fit_seconds'], params=fit['params'],
                                 test10k=dict(n=10000, solved=int(sub.sum()), rate=float(sub.mean())), test=dict(n=len(right), solved=int(right.sum()), rate=float(right.mean()))))
            d = ROOT / 'baselines/runs/ebr' / (f'de2-v{VARIANTS}' if DE2 else 'de1') / f'n{k}-s{s}'
            if PARTIAL and not ((d / 'fit.json').exists() and (d / 'score10k.report.json').exists() or ebr_scored(d)):
                missing.append(('EBR', k, s))
                continue
            fit = json.loads((d / 'fit.json').read_text())
            assert fit['steps'] == fit['total_steps'] == 60000, (d, fit['steps'])
            rep = d / 'score10k.report.json'
            if not rep.exists():
                subprocess.run([sys.executable, str(ROOT / 'baselines/ebr/de1_report.py'), str(d / 'score10k'), str(ROOT / 'baselines/runs/ebr/flops-full/sudoku.json'), str(rep)], check=True, cwd=ROOT / 'baselines')
            r = json.loads(rep.read_text())
            assert Path(r['checkpoint']).resolve() == Path(fit['final_checkpoint']).resolve(), (r['checkpoint'], fit['final_checkpoint'])
            rows.append(dict(method='EBR', k=k, seed=s, run=str(d), report=str(rep), report_sha256=sha(rep), steps=fit['steps'],
                             fit_seconds=fit['fit_seconds'], params=fit['params'],
                             test10k={key: {z: r['test10k'][key][z] for z in ('n', 'solved', 'rate', 'mean_flop')} for key in EBR_KEYS}))
    for k in KS if DE2 else ():
        for s in SEEDS:
            f = ROOT / 'baselines/runs/frm/de2-v99' / f'n{k}-s{s}' / f'runs/s{s}' / 'test10k.json'
            if PARTIAL and not f.exists():
                missing.append(('FRM', k, s))
                continue
            t = json.loads(f.read_text())
            assert t['n'] == 10000 and t['rows'] == 'full-all 10000-19999' and not t['smoke'] and t['seed'] == s, f
            flags = np.load(t['flags'])
            assert flags.shape == (10000,) and int(flags.sum()) == t['solved'], f
            rows.append(dict(method='FRM', k=k, seed=s, run=str(f.parent), record=str(f), record_sha256=sha(f), checkpoint=t['checkpoint'], step=t['step'],
                             test10k=dict(n=t['n'], solved=t['solved'], rate=t['rate'], flops_per_puzzle=t['flops_per_puzzle'])))
    path = (HERE / out).resolve()
    path.write_text(json.dumps(dict(source='experiments/data_efficiency_baselines.py', variants=VARIANTS, partial=PARTIAL, missing=[list(m) for m in missing], script_sha256=sha(__file__), rows=rows), indent=1) + '\n')
    print(f'{len(missing)} runs missing' if missing else 'complete')
    for m in ('TRM', 'EBR', 'FRM'):
        for k in KS:
            rr = [r for r in rows if r['method'] == m and r['k'] == k]
            v = [(r['test10k']['gtan0.2-cap64-N5']['rate'] if m == 'EBR' else r['test10k']['rate']) * 100 for r in rr]
            if not v:
                continue
            print(f"{m} k={k:>5}: " + ' '.join(f'{x:7.2f}' for x in v) + f'   mean {np.mean(v):7.2f}')
    print(path)


if __name__ == '__main__':
    main(*sys.argv[1:2])
