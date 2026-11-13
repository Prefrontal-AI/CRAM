"""Operation-count export: every inference operation count the paper shows, under one convention, in results/op-counts.json.

Reads the counter records in baselines/runs/flops-full/ (baselines: FullCounter on each method's own inference code,
written by hrm_count.py, trmp_count.py, eqr_count.py, frm_count.py, gram/flops.py and trm/trm_bench.py; CRAM:
cram_kernels.py and cram_kernels_extra.py) and baselines/runs/ebr/flops-full/ (EBR, ebr/flops.py), and the CRAM test
records for the per-puzzle rebuilds. Writes, per paper cell (benchmark, row): the old count (matrix products only, or
CRAM's kernel formula), the count under the convention, the count with comparisons also counted (sensitivity), and the
source record with its SHA-256; plus the counted components that the per-puzzle rebuilds (cram/opcount.py), the compute
sweeps (baselines/sweep/build.py) and the difficulty export use. This script calls no baseline code.

  .venv/bin/python baselines/flops/fc2_export.py        (from the repository root, on a CPU allocation: reads ~60 MB)

Convention (baselines/flops/fullcount.py): matrix products 2 flop per multiply-add; elementwise arithmetic (add, subtract,
multiply, divide, negate, absolute value; integer arithmetic too) and transcendental evaluations (exp, log, sqrt,
sigmoid, tanh, erf) 1 flop each; comparisons, selections, min/max/argmax and mask logic recorded, not counted; indexing,
data movement, casts and random-number generation free.
"""
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FF = ROOT / 'baselines/runs/flops-full'
OUT = ROOT / 'results/op-counts.json'


def sha(p):
    """SHA-256 hex digest of the file at `p`."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    """`p` as a path relative to the repository root."""
    return str(Path(p).resolve().relative_to(ROOT))


def jl(p):
    """The JSON file at `p`, parsed."""
    return json.loads(Path(p).read_text())


def cell(benchmark, row, method, old, new, new_cmp, source, note=None, lower_bound=False, upper_bound=None):
    """One cell of the export: a (benchmark, row) pair with its old count, its count under the convention (`new`), the
    count with comparisons (`new_cmp`) and the SHA-256 of each source record."""
    src = [source] if not isinstance(source, list) else source
    return dict(benchmark=benchmark, row=row, method=method, old=float(old), new=float(new), new_with_compare=float(new_cmp),
                ratio=float(new) / float(old) if old else None, flop_is_lower_bound=lower_bound, upper_bound=upper_bound,
                sources={rel(s): sha(s) for s in src}, note=note)


def seed_mean(rec, key):
    """Mean of `key` over the seeds of a per-seed counter record."""
    return statistics.fmean(s[key] for s in rec['per_seed'].values())


def main():
    """Read every counter record and write results/op-counts.json."""
    comps, cells = {'baselines': {}}, []
    B = comps['baselines']
    # ---------------- ACT models: HRM, TRM, GRAM ----------------
    for method, files in (('HRM', ('hrm-sudoku', 'hrm-mazeunique')), ('TRM', ('trm-sudoku', 'trm-mazeunique')),
                          ('GRAM', ('gram-sudoku', 'gram-mazeunique'))):
        for task, f in zip(('sudoku-extreme', 'maze-unique'), files):
            r = jl(FF / f'{f}.json')
            c = r['components']
            B.setdefault(method, {})[task] = dict(per_step=c['act_step']['full']['total'], per_step_with_compare=c['act_step']['full']['total_with_compare'],
                                                  per_step_old=c['act_step']['old_matmul_only'], vote=c.get('vote', {}).get('full', {}).get('total', 0))
            cells.append(cell(task, method.lower(), method, r['flop_per_puzzle_old'], r['flop_per_puzzle'], r['flop_per_puzzle_with_compare'], FF / f'{f}.json'))
    # Maze-Hard HRM and TRM (appendix table, published accuracies): the same models as the counted Maze-Unique fits (README
    # recipes, 30x30 grid, 16 ACT steps at evaluation; HRM's released Maze-Hard checkpoint has the README architecture), so
    # the per-maze count does not depend on the maze and is taken from the Maze-Unique count
    for method, f in (('HRM', 'hrm-mazeunique'), ('TRM', 'trm-mazeunique')):
        r = jl(FF / f'{f}.json')
        cells.append(cell('maze-hard', method.lower(), method, r['flop_per_puzzle_old'], r['flop_per_puzzle'], r['flop_per_puzzle_with_compare'], FF / f'{f}.json',
                          note='same model and sequence length as the counted Maze-Unique fit; the count does not depend on the maze'))
    for task, f in (('nqueens-8', 'gram-nqueens8'), ('nqueens-10', 'gram-nqueens10'), ('graph-coloring-8', 'gram-gc8'), ('graph-coloring-10', 'gram-gc10')):
        r = jl(FF / f'{f}.json')
        cells.append(cell(task, 'gram', 'GRAM', r['flop_per_puzzle_old'], r['flop_per_puzzle'], r['flop_per_puzzle_with_compare'], FF / f'{f}.json', note='20 samples, majority vote'))
    # ---------------- EqR, FRM ----------------
    for task, f in (('sudoku-extreme', 'eqr-sudoku-extreme'), ('maze-unique', 'eqr-maze-unique')):
        r = jl(FF / f'{f}.json')
        c = r['components']
        B.setdefault('EqR', {})[task] = dict(per_nfe=c['next_nfe']['total'], per_nfe_with_compare=c['next_nfe']['total_with_compare'],
                                             per_nfe_old=r['old']['flop_per_nfe_recorded'], select_over_B=c['select_over_B']['total'])
        cells.append(cell(task, 'eqr', 'EqR', r['old']['per_puzzle'], r['per_puzzle_full'], r['per_puzzle_full_with_compare'], FF / f'{f}.json', note=r['setting']))
    for task, f, row in (('sudoku-extreme', 'frm-sudoku-extreme', 'frm'), ('maze-unique', 'frm-maze-unique', 'frm')):
        r = jl(FF / f'{f}.json')
        c = r['components']
        B.setdefault('FRM', {})[task] = dict(per_call=c['call']['total'], per_call_with_compare=c['call']['total_with_compare'], per_call_old=r['old']['flop_per_call_recorded'],
                                             per_puzzle_over_calls=c['puzzle']['total'] / r['old']['calls'])
        cells.append(cell(task, row, 'FRM', r['old']['per_puzzle'], r['per_puzzle_full'], r['per_puzzle_full_with_compare'], FF / f'{f}.json'))
    # ---------------- TRMP ----------------
    for task, f, row in (('sudoku-extreme', 'trmp-sudoku-extreme', 'trmp'), ('maze-unique', 'trmp-maze-unique', 'trmp-local')):
        r = jl(FF / f'{f}.json')
        B.setdefault('TRMP', {})[task] = r['sweep_components']
        cells.append(cell(task, row, 'TRMP', seed_mean(r, 'mean_flop_per_puzzle_old'), seed_mean(r, 'mean_flop_per_puzzle_full'),
                          seed_mean(r, 'mean_flop_per_puzzle_full_with_compare'), FF / f'{f}.json', note='best of 64 rollouts, realized steps'))
    # ---------------- EBR (arXiv 2611.08977; results/ebr-comparison.json) ----------------
    E = {}
    for b in ('sudoku', 'mazeunique', 'nqueens8', 'nqueens10', 'gc8', 'gc10'):
        d, o = jl(ROOT / f'baselines/runs/ebr/flops-full/{b}.json'), jl(ROOT / f'baselines/runs/ebr/flops/{b}.json')
        E[b] = {k: dict(new=d['ops'][k]['total'], new_with_compare=d['ops'][k]['total_with_compare'], old=o[k + '_flop']) for k in ('segment', 'measure', 'energy')}
        for k in ('segment', 'measure', 'energy'):
            assert d['ops'][k]['classes']['matmul'] == o[k + '_flop']
    B['EBR'] = E
    s = E['sudoku']
    # Sudoku-Extreme: the authors' stopped setting (arXiv 2611.08977 App. C): each of 5 samples stops at the first
    # segment end with |grad_M E| < 0.2, at most 64 segments. Its flop depends on the samples' segment counts s_i
    # (ebr/report.py: sum_i s_i x (segment + measure) + 5 x energy), so the mean total of segments per puzzle, S, comes from
    # each seed's report under the convention (report-fc2.json) and the old and with-compare counts use the same S.
    # Maze-Unique: fixed 16 segments, its protocol.
    e = E['sudoku']
    reps = [jl(ROOT / f'baselines/runs/ebr/sudoku-s{k}/report-fc2.json')['unseen']['gtan0.2-cap64-N5'] for k in range(3)]
    S = statistics.fmean((r['mean_flop'] - 5 * e['energy']['new']) / (e['segment']['new'] + e['measure']['new']) for r in reps)
    f = lambda k: S * (e['segment'][k] + e['measure'][k]) + 5 * e['energy'][k]
    cells.append(cell('sudoku-extreme', 'ebr', 'EBR', f('old'), f('new'), f('new_with_compare'),
                      [ROOT / 'baselines/runs/ebr/flops-full/sudoku.json'] + [ROOT / f'baselines/runs/ebr/sudoku-s{k}/report-fc2.json' for k in range(3)],
                      note=f'lowest energy of 5 samples, each stopped at |grad_M E| < 0.2, at most 64 segments; {S:.2f} segments per puzzle over the 5 samples (three-seed mean)'))
    e = E['mazeunique']
    f = lambda k: 5 * 16 * e['segment'][k] + 5 * e['energy'][k]
    cells.append(cell('maze-unique', 'ebr', 'EBR', f('old'), f('new'), f('new_with_compare'), ROOT / 'baselines/runs/ebr/flops-full/mazeunique.json',
                      note='published setting: lowest energy of 5 samples, 16 segments'))
    # ---------------- CRAM: main rows ----------------
    k = jl(FF / 'cram-kernels.json')
    ov = k['sudoku_general_overheads_s0']
    comps['cram0_sudoku'] = dict(C=ov['C'], check_moves=ov['check_moves'],
                                 move={str(D): dict(total=k[f'sudoku_general_move_D{D}']['total'], total_with_compare=k[f'sudoku_general_move_D{D}']['total_with_compare']) for D in (20, 30, 31)},
                                 start=dict(total=ov['init_per_puzzle']['total'], total_with_compare=ov['init_per_puzzle']['total_with_compare']),
                                 check=dict(total=ov['check_per_puzzle']['total'], total_with_compare=ov['check_per_puzzle']['total_with_compare']),
                                 source='baselines/flops/cram_kernels.py (move_general transcription; GeneralSearch constructor and step_check on the seed-0 model)')
    OUT.write_text(json.dumps(dict(components=comps), indent=1) + '\n')  # opcount.py reads the components below
    sys.path.insert(0, str(ROOT / 'experiments'))
    sys.path.insert(0, str(ROOT))  # the cram package
    from cram import opcount
    opcount.components.cache_clear()
    recs = [ROOT / f'experiments/runs/CRAM-S/full/test/cram0-hb-fit-s{s}.json' for s in range(3)]
    olds, news, cmps = [], [], []
    for p in recs:
        d = jl(p)
        fl = np.asarray(d['per_puzzle']['flops'], float)
        olds.append(fl.mean()); news.append(opcount.cram0_sudoku(fl, d['signature']).mean()); cmps.append(opcount.cram0_sudoku(fl, d['signature'], True).mean())
    cells.append(cell('sudoku-extreme', 'cram-0-fit', 'CRAM', np.mean(olds), np.mean(news), np.mean(cmps), recs + [FF / 'cram-kernels.json'],
                      note='3-seed mean of per-puzzle means; per seed new ' + ', '.join(f'{x:.4g}' for x in news)))
    # ---------------- CRAM: other rows (cram_kernels_extra.py records) ----------------
    extra = (('sudoku-extreme', 'hand-energy', 'cram-hand-sudoku', None), ('maze-unique', 'hand-energy', 'cram-hand-maze-unique', None),
             ('maze-unique', 'cram-sgd-fg', 'cram-sgd-fg', None),  # the Maze-Unique row of Table 2
             ('maze-hard', 'cram-latent-zmu', 'cram-latent-zmu', None),  # the Maze-Hard row of the appendix
             ('nqueens-8', 'cram-mpl', 'cram-nqueens-8', 'n20'), ('nqueens-10', 'cram-cd', 'cram-nqueens-10', 'n20'),
             ('nqueens-8', 'cram-mpl-n1', 'cram-nqueens-8', 'n1'), ('nqueens-10', 'cram-cd-n1', 'cram-nqueens-10', 'n1'),
             ('graph-coloring-8', 'cram-full', 'cram-gc8-full', 'flop_20'), ('graph-coloring-10', 'cram-full', 'cram-gc10-full', 'flop_20'),
             ('graph-coloring-8', 'cram-full-n1', 'cram-gc8-full', 'flop_1'), ('graph-coloring-10', 'cram-full-n1', 'cram-gc10-full', 'flop_1'))
    for task, row, f, sub in extra:
        r = jl(FF / f'{f}.json')
        o, n = (r['old'][sub], r['new'][sub]) if sub else (r['old'], r['new'])
        nc = (r['new_with_compare'][sub] if sub else r['new_with_compare'])['mean_over_seeds']
        cells.append(cell(task, row, 'CRAM' if 'hand' not in f else 'Hand-set energy', o['mean_over_seeds'], n['mean_over_seeds'], nc, FF / f'{f}.json'))
    # ---------------- training arithmetic per update (training-methods table; experiments/training_methods.py count --full) ----------------
    # Sudoku rows from train-flop*.json; maze rows from the files of the SGD-trained free 3x3 kernel (train-flop*-maze-sgd.json),
    # which replace the maze rows of the first files
    tn = {k: v for k, v in jl(ROOT / 'results/alt-training/train-flop-fc2.json')['rows'].items() if not k.startswith('maze/')}
    to = {k: v for k, v in jl(ROOT / 'results/alt-training/train-flop.json')['rows'].items() if not k.startswith('maze/')}
    tn |= jl(ROOT / 'results/alt-training/train-flop-fc2-maze-sgd.json')['rows']
    to |= jl(ROOT / 'results/alt-training/train-flop-maze-sgd.json')['rows']
    training = {}
    for k, v in tn.items():
        assert v['matmul_class_equals_flopcountermode'] and v['flopcountermode_next_update'] == to[k]['flop_per_update'], k
        training[k] = dict(old=to[k]['flop_per_update'], new=v['flop_per_update'], new_with_compare=v['total_with_compare'], ratio=v['flop_per_update'] / to[k]['flop_per_update'])
    comps['training_per_update'] = dict(rows=training, sources={rel(ROOT / f'results/alt-training/{f}'): sha(ROOT / f'results/alt-training/{f}') for f in ('train-flop-fc2.json', 'train-flop.json', 'train-flop-fc2-maze-sgd.json', 'train-flop-maze-sgd.json')},
                                        note='one training update (loss, gradient, step) of each CRAM training method at its tuned choice; FullCounter')
    doc = dict(description=__doc__.split('\n\n')[0],
               convention=' '.join(__doc__.split('Convention (baselines/flops/fullcount.py): ')[1].split()),
               counter_sha256={'baselines/flops/fullcount.py': sha(ROOT / 'baselines/flops/fullcount.py'), 'baselines/flops/cram_kernels.py': sha(ROOT / 'baselines/flops/cram_kernels.py'),
                               'baselines/flops/cram_kernels_extra.py': sha(ROOT / 'baselines/flops/cram_kernels_extra.py')},
               script=rel(__file__), script_sha256=sha(__file__), created_utc=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
               cells=cells, components=comps)
    OUT.write_text(json.dumps(doc, indent=1) + '\n')
    for c in cells:
        print(f"{c['benchmark']:18s} {c['row']:18s} {c['old']:11.4g} -> {c['new']:11.4g}  x{c['ratio']:.3f}  (+cmp {c['new_with_compare']:.4g})")


if __name__ == '__main__':
    main()
