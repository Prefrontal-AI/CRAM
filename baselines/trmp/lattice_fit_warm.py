"""TRMP lattice model, Sudoku-Extreme: cold and warm fit times.

Runs trmp/lattice_fit.py's timed fit, unchanged, 1 + n_warm times in one process (baselines/warm.py). All repeats share the
compile cache of the set (TORCHINDUCTOR_CACHE_DIR, set by trmp/lattice_fit_warm.sbatch), which is empty for the cold
fit. Dynamo's recompile limit is raised so that later fits stay compiled, as the cold one.
  python trmp/lattice_fit_warm.py <seed> <out> <n_warm>        # from baselines/
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import lattice_fit as LF  # noqa: E402
from warm import repeat  # noqa: E402


def main(seed, out, n_warm):
    import torch._dynamo
    for name in ('recompile_limit', 'cache_size_limit'):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, 64)
    repeat(lambda folder: LF.main(LF.Config(seed=seed, out=str(folder))), out, n_warm, lambda r: r['fit_seconds'],
           dict(method='TRMP lattice (factor graph, 16k)', task='sudoku-extreme', seed=seed, reported_field='fit_seconds'))


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))
