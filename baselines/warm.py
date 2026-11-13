"""Cold and warm fit times of a baseline, as CRAM's: one cold fit, then N more complete fits in the same process.

The fit itself is the method's own timed fit function, called unchanged once per repeat, each into its own folder:
<out>/cold, <out>/warm1 ... <out>/warm<N>. The only user is trmp/lattice_fit_warm.py (the TRMP lattice model on
Sudoku-Extreme, whose fit is trmp/lattice_fit.py); this file calls no upstream code itself. Cold is the first fit in
a fresh process; warm is the median of the N further fits (CRAM: N = 7). A warm repeat is valid only if every earlier
repeat ran in the same process, so a restarted job (preemption, requeue) deletes the partial set and starts again from
a cold fit; the sbatch also clears the compile caches then.
Summary: <out>/repeats.json (per-repeat fit seconds, the cold value, the warm median, model hashes).
"""
import json
import os
import shutil
import statistics
import time
from pathlib import Path

from clock import write_json


def repeat(fit_once, out, n_warm, read, meta):
    """Run one cold fit and `n_warm` warm fits in this process and write <out>/repeats.json.

    fit_once(folder) runs one complete timed fit into folder and writes folder/fit.json; read(fit.json dict) gives the
    reported seconds; `meta` is a dict of fields copied into repeats.json.
    """
    out = Path(out)
    if (out / 'repeats.json').exists():
        print(f'{out}/repeats.json exists: nothing to do', flush=True)
        return
    names = ['cold'] + [f'warm{i}' for i in range(1, n_warm + 1)]
    for n in names:  # a partial set from an earlier process: its warm repeats would not be warm, so start again
        if (out / n).exists():
            print(f'removing {out / n} (partial set from an earlier process)', flush=True)
            shutil.rmtree(out / n)
    fits = []
    for n in names:
        fit_once(out / n)
        rec = json.loads((out / n / 'fit.json').read_text())
        assert not rec.get('resumed'), f'{out / n} resumed from a checkpoint: not a complete fit'
        fits.append(dict(name=n, seconds=read(rec), fit_seconds=rec['fit_seconds'], model_sha256=rec.get('model_sha256'),
                         host=rec.get('host'), slurm_job=rec.get('slurm_job')))
        print(f'REPEAT {n}: {fits[-1]["seconds"]:.3f} s', flush=True)
    warm = [f['seconds'] for f in fits[1:]]
    write_json(out / 'repeats.json', dict(
        **meta, cold_seconds=fits[0]['seconds'], warm_median_seconds=statistics.median(warm), warm_seconds=warm,
        n_warm=n_warm, fits=fits, pid=os.getpid(), same_process=True,
        definition='cold: the first complete fit in a fresh process; warm: median of the further complete fits in that '
                   'process (baselines/warm.py)',
        finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    print(f'COLD {fits[0]["seconds"]:.3f} s, WARM median {statistics.median(warm):.3f} s over {n_warm}', flush=True)
