"""EqR: one timed cold fit on one GPU (locuslab/EqR 134c7541, unmodified, through baselines/eqr/eqr_train.py).

eqr_train.py runs EqR's own pretrain.py pieces (create_dataloader, init_train_state, train_batch, EMA, torch.compile as
pretrain.py applies it) and adds a rolling checkpoint and validation. This wrapper runs it unchanged and adds the clock:
- the clock starts when EqR builds its training state (pretrain.init_train_state), after the data loaders exist;
- every EqR evaluation (validation on dev) and every checkpoint/model save are timed and excluded;
- the clock stops when eqr_train.main returns (the final EMA model is saved, excluded).
The inductor cache lives in the fresh run folder, so compilation is cold.
Usage (from baselines/eqr, EqR's Python): python fit.py <eqr_train.py arguments...>   (eqr/fit.sbatch runs it and the scoring)
"""

import sys
from pathlib import Path

import tyro

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clock import FitClock, cuda_context_seconds, write_json  # noqa: E402

sys.path.insert(0, str(Path.cwd()))
import eqr_common as C  # noqa: E402
import eqr_train as T  # noqa: E402


def main():
    a = tyro.cli(T.Args)
    run = a.run_dir.resolve()
    if (run / 'fit.json').exists():
        print(f'{run}/fit.json exists: nothing to do', flush=True)
        return
    ctx = cuda_context_seconds()
    clock = FitClock()
    C.install_stand_ins()
    import pretrain
    import evaluators.eval_fn as ev

    init = pretrain.init_train_state

    def timed_init(*args, **kwargs):
        clock.start()
        return init(*args, **kwargs)
    pretrain.init_train_state = timed_init

    evaluate = ev.evaluate

    def timed_evaluate(*args, **kwargs):
        with clock.exclude('evaluation'):
            return evaluate(*args, **kwargs)
    ev.evaluate = timed_evaluate

    save = T.atomic_save

    def timed_save(*args, **kwargs):
        with clock.exclude('checkpoint_save'):
            return save(*args, **kwargs)
    T.atomic_save = timed_save

    T.main(a)
    fit_seconds = clock.stop()
    import json
    final = json.loads((run / 'final.json').read_text())
    clock.resumed = final.get('restarts', 0) > 0
    rec = clock.record(method='EqR', upstream=f'locuslab/EqR {C.EQR_COMMIT}', preset=a.preset, seed=a.seed,
                       steps=final['step'], total_steps=final['total_updates'], params=final['num_params'],
                       cuda_context_seconds=ctx, eqr_train_final=final,
                       numerics='EqR defaults (bf16 forward); torch.compile unless DISABLE_COMPILE is set',
                       compile='off' if 'DISABLE_COMPILE' in __import__('os').environ else 'on')
    rec['fit_seconds'] = fit_seconds
    write_json(run / 'fit.json', rec)
    print(f'FIT {fit_seconds:.1f} s, {final["step"]} updates', flush=True)


if __name__ == '__main__':
    main()
