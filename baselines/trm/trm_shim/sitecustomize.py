"""Log shim for TRM's pretrain.py: print every wandb.log record as one JSON line (wandb itself is disabled).
Python imports sitecustomize at start-up when this folder is on PYTHONPATH (the job scripts set it).
Nested evaluation metrics are flattened to 'all/...' keys.
wandb.init rebinds wandb.log to its run, so the print is installed again after each init."""
import json

try:
    import wandb

    def _print_log(data, step=None, **kw):
        rec = {'step': step}

        def put(prefix, d):  # evaluation metrics arrive nested ({'all': {...}}): flatten them as 'all/...'
            for k, v in d.items():
                if isinstance(v, dict):
                    put(f'{prefix}{k}/', v)
                    continue
                try:
                    rec[prefix + k] = float(v)
                except Exception:
                    pass

        put('', data)
        print('WANDB ' + json.dumps(rec), flush=True)

    _init = wandb.init

    def _patched_init(*a, **kw):
        run = _init(*a, **kw)
        wandb.log = _print_log
        return run

    wandb.init = _patched_init
    wandb.log = _print_log
except Exception as e:  # noqa
    print('sitecustomize: wandb patch failed', e)
