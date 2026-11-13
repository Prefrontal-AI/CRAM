"""The one fit clock behind every training time in the paper's main table.

Definition (the CRAM fit-time protocol, applied to every method):
- The clock starts when the training data are in host memory and stops when the inference-ready model exists.
- Inside: device transfer, model construction, compilation and its warm-up, every update, EMA finalization.
- Excluded: reading files from disk, evaluation, checkpoint and model saving. Each excluded interval is timed and
  listed in the record, so the wall clock can be rebuilt as fit_seconds + sum(excluded).
- CUDA context creation is timed separately (`cuda_context_seconds`).
- Cold: the first fit in a fresh process. A run that resumed from a checkpoint is flagged `resumed: true`.

Every boundary synchronizes the device, so queued GPU work is counted in the interval that issued it.

This file calls no baseline code. Each wrapper `<baseline>/fit.py` creates one FitClock, calls `start()` after the data
are loaded, wraps evaluation and saving in `exclude(name)`, calls `stop()` when the model is ready, and writes
`record()` to `fit.json` with `write_json`.
"""

import contextlib
import json
import os
import socket
import time
from pathlib import Path


def _sync():
    """Wait for queued GPU work to finish, when PyTorch is installed and CUDA is initialized."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
    except ImportError:
        pass


def cuda_context_seconds():
    """Time CUDA context creation (torch.cuda.init plus one tensor), reported apart from the fit."""
    import torch
    t = time.perf_counter()
    torch.cuda.init()
    torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    return time.perf_counter() - t


class FitClock:
    """Wall clock of one fit, minus the excluded intervals; `sync` is called at every boundary."""

    def __init__(self, sync=_sync):
        self.sync = sync
        self.t0 = self.t1 = None
        self.excluded = {}
        self.resumed = False
        self.prior_fit_seconds = 0.0  # fit time carried over from before a resume

    def start(self):
        """Start the clock. Call it when the training data are in host memory."""
        self.sync()
        self.t0 = time.perf_counter()

    @contextlib.contextmanager
    def exclude(self, name):
        """Time an interval inside the fit that does not count: evaluation, saving, disk reads."""
        self.sync()
        t = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.excluded[name] = self.excluded.get(name, 0.0) + time.perf_counter() - t

    def elapsed(self):
        """Fit seconds so far (wall since start minus excluded intervals, plus time before a resume)."""
        self.sync()
        end = self.t1 if self.t1 is not None else time.perf_counter()
        return self.prior_fit_seconds + (end - self.t0) - sum(self.excluded.values())

    def stop(self):
        """Stop the clock and return the fit seconds. Call it when the inference-ready model exists."""
        self.sync()
        self.t1 = time.perf_counter()
        return self.elapsed()

    def record(self, **meta):
        """The fit record as a dict: fit seconds, wall seconds, excluded intervals, device, host, code hashes and `meta`."""
        import torch
        gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
        import hashlib
        import sys
        code = {str(Path(f).resolve()): hashlib.sha256(Path(f).read_bytes()).hexdigest()
                for f in (sys.argv[0], __file__) if f and Path(f).is_file()}  # the code that ran, hashed at run time
        return dict(fit_seconds=self.elapsed(), wall_seconds=self.t1 - self.t0, excluded_seconds=self.excluded,
                    resumed=self.resumed, prior_fit_seconds=self.prior_fit_seconds, gpu=gpu,
                    host=socket.gethostname(), slurm_job=os.environ.get('SLURM_JOB_ID'), code_sha256=code,
                    torch=torch.__version__, cuda=torch.version.cuda,
                    finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **meta)


def write_json(path, obj):
    """Write `obj` as JSON to `path` through a temporary file, so that a reader never sees a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, default=str) + '\n')
    os.replace(tmp, path)
