import os
import subprocess
from pathlib import Path

import pytest


SLURM = Path(__file__).resolve().parents[1] / 'slurm'


@pytest.fixture
def submit(tmp_path):
    """Run a batch script against a temporary project whose Python only records worker outcomes."""
    (tmp_path / 'slurm').symlink_to(SLURM, target_is_directory=True)
    (tmp_path / 'experiments/runs/ALT').mkdir(parents=True)
    (tmp_path / 'experiments/runs/ALT-sgd').mkdir(parents=True)
    trace = tmp_path / 'worker-events'
    trace.touch()
    python = tmp_path / '.venv/bin/python'
    python.parent.mkdir(parents=True)
    python.write_text('''#!/bin/bash
set -euo pipefail
if [ "$1" = -u ]; then shift; fi
phase=$2
if [ "$phase" = plan ]; then
    printf 'train --out run0\\ntrain --out run1\\n'
    exit
fi
exec >>"$TRACE" 2>&1
worker=${CUDA_VISIBLE_DEVICES:-}
if [ "$phase" = probe ]; then worker=$4; fi
if [ "$phase:$worker" = "$FAIL_AT" ]; then
    printf '%s failed\\n' "$phase:$worker"
    exit 23
fi
sleep 0.05
printf '%s finished\\n' "$phase:$worker"
''')
    python.chmod(0o755)

    def run(script, *args, fail_at='', probes=None):
        if probes is not None:
            path = tmp_path / 'experiments/runs/ALT/many/plan-probe.txt'
            path.parent.mkdir(parents=True)
            path.write_text(probes)
        env = dict(os.environ, SLURM_SUBMIT_DIR=str(tmp_path), SLURM_ARRAY_TASK_ID='0',
                   SLURM_ARRAY_TASK_COUNT='1', PROBES_PER_GPU='3', PLAN_DIR='runs/ALT/many',
                   TRACE=str(trace), FAIL_AT=fail_at)
        env.pop('CUDA_VISIBLE_DEVICES', None)
        result = subprocess.run(['bash', str(SLURM / 'training_methods' / script), *args],
                                env=env, cwd=tmp_path, capture_output=True, text=True, timeout=10)
        return result, trace.read_text().splitlines()

    return run


@pytest.mark.parametrize('script, args', [('tune.sbatch', ()), ('sgd.sbatch', ('tune',))])
@pytest.mark.parametrize('fail_at', ['', 'train:0', 'train:1'])
def test_tuning_waits_for_workers_and_selects_only_on_success(submit, script, args, fail_at):
    result, events = submit(script, *args, fail_at=fail_at)
    if fail_at:
        peer = 'train:1' if fail_at == 'train:0' else 'train:0'
        assert sorted(events) == sorted([f'{peer} finished', f'{fail_at} failed'])
        assert result.returncode == 23
        assert 'exited with status 23' in result.stderr
    else:
        assert sorted(events[:-1]) == ['train:0 finished', 'train:1 finished']
        assert events[-1] == 'select: finished'
        assert result.returncode == 0


def test_final_job_reports_failed_test_and_finishes_peers(submit):
    result, events = submit('final.sbatch', fail_at='test:0')
    assert result.returncode == 23
    assert sorted(events) == ['test:0 failed', 'test:1 finished', 'train:0 finished', 'train:1 finished']


@pytest.mark.parametrize('probes, fail_at, expected_status, expected_events', [
    ('', '', 0, []),
    ('run0\n', 'probe:run0', 23, ['probe:run0 failed']),
])
def test_probe_waits_only_on_spawned_workers(submit, probes, fail_at, expected_status, expected_events):
    result, events = submit('many.sbatch', 'probe', probes=probes, fail_at=fail_at)
    assert result.returncode == expected_status, result.stderr
    assert events == expected_events
