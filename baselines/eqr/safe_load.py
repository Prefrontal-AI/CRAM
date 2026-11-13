"""Load an EqR checkpoint (a downloaded file, so untrusted) with an unpickler that admits only an allowlist of globals.

torch.load(weights_only=True) refuses EqR's released checkpoints: their `config` is an OmegaConf DictConfig whose
metadata holds a collections.defaultdict, which the weights-only unpickler cannot rebuild. This module is a pickle
module for torch.load(pickle_module=...): it admits torch's tensor and storage rebuilders, the plain containers, numpy's
array and RNG-state types and OmegaConf's container classes, and refuses every other global.
"""
import pickle
from _compat_pickle import IMPORT_MAPPING, NAME_MAPPING
import sys
import types

import torch

ALLOWED_MODULES = {'torch._utils', 'torch.storage', 'torch.serialization', 'omegaconf', 'omegaconf.base', 'omegaconf.nodes',
                   'omegaconf.dictconfig', 'omegaconf.listconfig'}
ALLOWED = {('collections', 'OrderedDict'), ('collections', 'defaultdict'), ('typing', 'Any'), ('builtins', 'dict'),
           ('builtins', 'list'), ('builtins', 'int'), ('builtins', 'float'), ('builtins', 'str'), ('builtins', 'tuple'),
           ('builtins', 'set'), ('builtins', 'bool'), ('builtins', 'NoneType'),
           ('torch', 'Size'), ('torch', 'device'), ('torch', 'dtype'), ('torch', 'bfloat16'), ('torch', 'float32'),
           ('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'), ('numpy', 'ndarray'),
           ('numpy', 'dtype'), ('numpy.dtypes', 'UInt32DType'), ('numpy.dtypes', 'Float64DType'),
           ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', 'scalar')}
ALLOWED_STORAGE = {'FloatStorage', 'BFloat16Storage', 'HalfStorage', 'LongStorage', 'IntStorage', 'BoolStorage',
                   'ByteStorage', 'DoubleStorage', 'UntypedStorage'}


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in NAME_MAPPING:  # protocol-2 pickles use Python 2 names (__builtin__.long); map as pickle does
            module, name = NAME_MAPPING[(module, name)]
        module = IMPORT_MAPPING.get(module, module)
        if (module, name) in ALLOWED or module in ALLOWED_MODULES or (module == 'torch' and name in ALLOWED_STORAGE):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'refused global {module}.{name}')


safe_pickle = types.ModuleType('eqr_safe_pickle')
safe_pickle.Unpickler = Unpickler
safe_pickle.load = lambda f, **kw: Unpickler(f, **kw).load()
safe_pickle.__name__ = 'eqr_safe_pickle'


def load(path, map_location='cpu'):
    """The checkpoint at `path` as a dict; OmegaConf containers in it are converted to plain dicts and lists."""
    from omegaconf import DictConfig, ListConfig, OmegaConf
    ckpt = torch.load(path, map_location=map_location, weights_only=False, pickle_module=safe_pickle)
    return {k: OmegaConf.to_container(v, resolve=True) if isinstance(v, (DictConfig, ListConfig)) else v for k, v in ckpt.items()}


if __name__ == '__main__':  # python safe_load.py <ckpt>: print what the checkpoint holds
    import json
    c = load(sys.argv[1])
    for k, v in c.items():
        if k in ('model', 'ema', 'optimizers'):
            continue
        print(k, json.dumps(v, default=str)[:3000])
    sd = c['model']
    print('model tensors', sum(v.numel() for v in sd.values()), [(k, tuple(v.shape)) for k, v in sd.items()])
    if 'ema' in c:
        print('ema', type(c['ema']), list(c['ema'])[:4] if isinstance(c['ema'], dict) else '')
