"""Collect the matmul-only inference flop cells of HRM and TRMP into results/flop-counts.json.

These are the counts under the first convention (matrix products only, torch's FlopCounterMode; a multiply-add is 2
flop). baselines/sweep/build.py reads their components; the counts under the all-arithmetic convention are in
results/op-counts.json (fc2_export.py). Reads runs/flops/hrm-sudoku.json and hrm-mazeunique.json (hrm_count.py) and
runs/flops/trmp.json (trmp_count.py). This script calls no baseline code.

  ../.venv/bin/python flops/report.py   # from baselines/
"""
import hashlib
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINES = HERE.parent
ROOT = BASELINES.parent
RUNS = BASELINES / 'runs/flops'
OUT = ROOT / 'results/flop-counts.json'


def sha(p):
    """SHA-256 hex digest of the file at `p`."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    """`p` as a path relative to the repository root."""
    return str(Path(p).resolve().relative_to(ROOT))


def scripts(*names):
    """{path: SHA-256} of the named scripts of baselines/flops/."""
    return {f'baselines/flops/{n}': sha(HERE / n) for n in names}


def hrm(task='sudoku', benchmark='Sudoku-Extreme'):
    """The HRM cell of one benchmark, from the record of hrm_count.py."""
    r = json.loads((RUNS / f'hrm-{task}.json').read_text())
    assert r['analytic_matches'] and r['act_steps'] == r['halt_max_steps'] == 16
    return dict(benchmark=benchmark, method='HRM', flop_per_puzzle=r['flop_per_puzzle'],
                components=dict(flop_per_act_step=r['flop_per_act_step'], act_steps=r['act_steps'],
                                attention_per_puzzle=r['puzzles_checked'][0]['by_op'].get('aten.bmm', 0),
                                flop_seen_by_counter_with_fa3=r['puzzles_checked'][0]['flop_seen_with_fa3'],
                                analytic_per_step=r['analytic']),
                convention=r['convention'] + '. FlashAttention 3 is invisible to the counter; flash_attn_func was '
                'replaced by SDPA (math kernel) for counting only, and the count equals the analytic matmul count. '
                'HRM evaluation halts every puzzle at halt_max_steps = 16 (the Q-halt test applies only in training), so '
                'the count is the same for every puzzle and every checkpoint.',
                sampled=False, checkpoint=rel(r['checkpoint']), checkpoint_sha256=r['checkpoint_sha256'],
                script=scripts('hrm_count.py'), record=rel(RUNS / f'hrm-{task}.json'),
                record_sha256=sha(RUNS / f'hrm-{task}.json'), slurm_job=r['slurm_job'])


def trmp():
    """The TRMP cells (one per benchmark), from the record of trmp_count.py."""
    r = json.loads((RUNS / 'trmp.json').read_text())
    cells = []
    for task, t in r['tasks'].items():
        p = t['per_rollout']
        assert p['step_matches_formula']
        cells.append(dict(benchmark=t['benchmark'], method='TRMP', flop_per_puzzle=t['flop_per_puzzle'],
                          components=dict(per_step=p['step'], halt_head_per_step=p['halt_head'],
                                          embedding_per_rollout=p['embedding'], readout_per_rollout=p['readout'],
                                          rollouts=p['rollouts'], step_cap=p['step_cap'],
                                          per_seed={k: (None if v is None else {x: v[x] for x in (
                                              'mean_flop_per_puzzle', 'sd_flop_per_puzzle', 'mean_rollout_steps',
                                              'mean_steps_per_puzzle', 'capped_rollouts', 'puzzles', 'solved', 'sigma',
                                              'model_sha256', 'test_json_sha256')}) for k, v in t['per_seed'].items()}),
                          status='complete' if len(t['seeds_done']) == 3 else (
                              f"pending: test step counts exist for seeds {t['seeds_done']} only; rerun trmp_count.py "
                              'and report.py when baselines/runs/trmp/<task>-s<k>/score/test.json exist'),
                          convention=r['convention'] + '; per puzzle ' + r['formula'] + '; the cell is the mean over the '
                          'three seed models of the mean over all test puzzles',
                          sampled=False, checkpoint=[f'baselines/runs/trmp/{task}-s{k}/model.pt' for k in range(3)],
                          script=scripts('trmp_count.py'), record=rel(RUNS / 'trmp.json'),
                          record_sha256=sha(RUNS / 'trmp.json')))
    return cells


def main():
    """Write results/flop-counts.json and print one line per cell."""
    cells = [hrm(), hrm('mazeunique', 'Maze-Unique'), *trmp()]
    out = dict(description='Inference flop per test puzzle of HRM and TRMP, matrix products only. '
               'A multiply-add is 2 flop.', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               report_script_sha256=sha(__file__), cells=cells)
    OUT.write_text(json.dumps(out, indent=1) + '\n')
    for c in cells:
        print(c['benchmark'], c['method'], c['flop_per_puzzle'], c.get('ci95'), c.get('status', ''))


if __name__ == '__main__':
    main()
