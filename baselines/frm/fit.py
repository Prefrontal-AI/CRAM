"""FRM (Flow Reasoning Model): one timed cold fit of one training stage on one GPU. The authors' code is
github.com/helblazer811/Flow-Reasoning-Models at 6149a01 (Python package flow_map_reasoning, "FlowMapReasoning").

The frozen campaign adapters (frm_reproduction.py for Sudoku-Extreme, frm_maze.py for Maze-Unique; from an earlier
reproduction campaign, copied into the run folder by frm/setup.sh) run unchanged. Each adapter checks its own SHA-256 and
the SHA-256 of every upstream source file before it trains. The upstream package owns the model, objective, optimizer,
augmentation and schedule.

This wrapper changes only the numerics flags and adds the clock:
- Numerics: the DiT (diffusion transformer) autocasts its blocks to bf16 (authors). The authors' per-block
  torch.compile is off (DIT_USE_COMPILE=0, set by frm/fit.sbatch): on one H200 it made training slower (Stage B 3.89
  against 5.37 updates/s, Stage A 29 against 35-38), and the rule is "bf16 and compilation unless compilation slows
  the fit". With FRM_COMPILE_MODEL=1 (the default of frm/fit.sbatch) this wrapper compiles the whole model instead.
  Flash attention stays off (FLM_DISABLE_FLASH=1): the authors report NaNs with flash attention + torch.compile on
  this DiT family, and the earlier reproduction ran without it.
- Every in-run evaluation (the adapter's val_fn) and every torch.save are timed and excluded.
- fit_seconds = the adapter's own clock around the upstream main() (completed.json "seconds") minus the excluded
  intervals. That clock starts before the upstream script builds its dataset, so it also counts the dataset read.

FRM trains in two stages, A then B, each in a fresh process: the table reports fit(A) + fit(B).
Usage (frm/fit.sbatch does this, with PYTHONPATH=<campaign_root>/upstream):
  python baselines/frm/fit.py <campaign_root> <adapter.py> <stage A|B> -- <adapter arguments...>
Writes <campaign_root>/timing/fit-<stage>.json (the fit record; `timing` is only a folder name).
"""

import json
import os
import runpy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clock import FitClock, cuda_context_seconds, write_json  # noqa: E402


def main():
    """Run the adapter's train action for one stage under the fit clock and write the fit record."""
    root, adapter, stage = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3]
    rest = sys.argv[sys.argv.index('--') + 1:]
    record_path = root / 'timing' / f'fit-{stage}.json'
    if record_path.exists():
        print(f'{record_path} exists: nothing to do', flush=True)
        return
    ctx = cuda_context_seconds()
    import torch
    import flow_map_reasoning as fmr
    clock = FitClock()

    original_save = torch.save

    def timed_save(*args, **kwargs):
        with clock.exclude('checkpoint_save'):
            return original_save(*args, **kwargs)
    torch.save = timed_save

    upstream_train = fmr.train

    def train(model, loader, cfg, **kwargs):
        if os.environ.get('FRM_COMPILE_MODEL') == '1':  # whole-model torch.compile (not in the authors' code)
            model = torch.compile(model, **json.loads(os.environ.get('FRM_COMPILE_KWARGS', '{}')))
        val_fn = kwargs.get('val_fn')
        if val_fn is not None:
            def timed_val(*a, **k):
                with clock.exclude('evaluation'):
                    return val_fn(*a, **k)
            kwargs['val_fn'] = timed_val
        return upstream_train(model, loader, cfg, **kwargs)
    fmr.train = train  # the adapter captures this as its original_train, then wraps it with its own val_fn

    # The adapter resumes from the newest step_*.pt of this stage; its clock then covers only this process.
    earlier = [p for p in root.glob(f'**/stage{stage}/step_*.pt') if 'qualification' not in p.parts]
    clock.resumed = bool(earlier)  # a resumed stage is not a cold fit; collect.py skips it
    clock.start()
    sys.argv = [str(adapter), '--root', str(root), '--action', 'train', '--stage', stage, *rest]
    runpy.run_path(str(adapter), run_name='__main__')
    clock.stop()
    completed = sorted(root.glob(f'**/stage{stage}/completed.json'), key=lambda p: p.stat().st_mtime)[-1]
    done = json.loads(completed.read_text())
    excluded = sum(clock.excluded.values())
    rec = clock.record(method='FRM', upstream='FlowMapReasoning 6149a01', stage=stage, adapter=adapter.name,
                       adapter_args=rest, completed=str(completed), steps=done['step'],
                       adapter_seconds=done['seconds'], cuda_context_seconds=ctx,
                       numerics='bf16 autocast in the DiT blocks (authors), torch.compile off (slower here), flash attention off')
    rec['fit_seconds'] = done['seconds'] - excluded
    rec['note'] = ('fit_seconds = adapter_seconds - excluded; the excluded intervals all fall inside the adapter clock. '
                   'wall_seconds is this wrapper\'s clock around the whole adapter run, including imports and setup.')
    write_json(record_path, rec)
    print(f'FIT stage {stage}: {rec["fit_seconds"]:.1f} s ({done["seconds"]:.1f} s adapter clock, {excluded:.1f} s '
          f'excluded)', flush=True)


if __name__ == '__main__':
    main()
