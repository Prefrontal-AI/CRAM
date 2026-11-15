"""CPU checks for the published checkpoint manifest and loader."""

import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import checkpoints.load as checkpoints
from checkpoints import evaluate


@pytest.mark.parametrize(
    ('name', 'seed'),
    [(name, seed) for name, entry in checkpoints.MANIFEST.items() for seed in range(len(entry['models']))],
)
def test_load_shipped_weights(name, seed):
    filename = checkpoints.path(name, seed).name
    contents = io.BytesIO(checkpoints.read_artifact(name, filename, seed))
    if filename.endswith('.pt'):
        weights = torch.load(contents, map_location='cpu', weights_only=True)
        expected = {k: v.numpy() if torch.is_tensor(v) else np.asarray(v) for k, v in weights.items()}
    else:
        with np.load(contents, allow_pickle=False) as weights:
            expected = dict(weights)

    actual = checkpoints.load(name, seed)
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], strict=True)


@pytest.fixture
def local_checkpoint(tmp_path, monkeypatch):
    folder = tmp_path / 'checkpoints' / 'fixture'
    folder.mkdir(parents=True)
    weights = folder / 'model.pt'
    torch.save({'value': torch.tensor([1, 2, 3]), 'label': 'fixture'}, weights)
    manifest = {
        'fixture': {
            'weights': weights.name,
            'models': [{
                'folder': 'checkpoints/fixture',
                'sha256': {weights.name: hashlib.sha256(weights.read_bytes()).hexdigest()},
            }],
        },
    }
    monkeypatch.setattr(checkpoints, 'ROOT', tmp_path)
    monkeypatch.setattr(checkpoints, 'HERE', folder.parent)
    monkeypatch.setattr(checkpoints, 'MANIFEST', manifest)
    return weights


@pytest.mark.parametrize('problem', ['mismatch', 'missing', 'unlisted', 'outside'])
def test_invalid_checkpoint_is_rejected_before_decoding(local_checkpoint, problem, monkeypatch):
    weights = local_checkpoint
    expected_error = ValueError
    if problem == 'mismatch':
        weights.write_bytes(b'changed fixture')
    elif problem == 'missing':
        weights.unlink()
        expected_error = FileNotFoundError
    elif problem == 'unlisted':
        checkpoints.MANIFEST['fixture']['models'][0]['sha256'].clear()
    else:
        outside = checkpoints.ROOT / weights.name
        weights.rename(outside)
        weights.symlink_to(outside)

    def unexpected_load(*args, **kwargs):
        pytest.fail('decoder reached before integrity verification')

    monkeypatch.setattr(torch, 'load', unexpected_load)
    with pytest.raises(expected_error):
        checkpoints.load_torch('fixture')


def test_nqueens_evaluator_rejects_mismatched_weights(local_checkpoint, monkeypatch):
    checkpoints.MANIFEST['nqueens-8'] = checkpoints.MANIFEST['fixture']
    monkeypatch.setattr(evaluate, 'MANIFEST', checkpoints.MANIFEST)
    monkeypatch.setitem(sys.modules, 'nqueens_free', SimpleNamespace())
    local_checkpoint.write_bytes(b'changed fixture')
    with pytest.raises(ValueError, match='SHA-256 differs'):
        evaluate.nqueens(8, 0)


def test_maze_evaluator_rejects_mismatched_config(local_checkpoint, monkeypatch):
    checkpoints.MANIFEST['maze-unique'] = checkpoints.MANIFEST['fixture']
    model = checkpoints.MANIFEST['maze-unique']['models'][0]
    model['sha256']['config.json'] = hashlib.sha256(b'{"window": 1}').hexdigest()
    (local_checkpoint.parent / 'config.json').write_text('{"window": 2}')
    monkeypatch.setattr(evaluate, 'MANIFEST', checkpoints.MANIFEST)

    def unexpected_config(**kwargs):
        pytest.fail('model configuration reached before integrity verification')

    monkeypatch.setitem(sys.modules, 'maze_unique', SimpleNamespace(Train=unexpected_config))
    with pytest.raises(ValueError, match='SHA-256 differs'):
        evaluate.maze(0)


def test_checker_exits_before_displaying_mismatched_weights(local_checkpoint):
    script = checkpoints.HERE / 'load.py'
    script.write_bytes(Path(checkpoints.__file__).read_bytes())
    (checkpoints.HERE / 'manifest.json').write_text(json.dumps(checkpoints.MANIFEST))
    torch.save({'value': torch.tensor([4, 5, 6])}, local_checkpoint)

    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False, timeout=15)
    assert run.returncode == 1
    assert 'SHA-256 differs from the manifest' in run.stdout
    assert '1 checkpoint file(s) failed verification' in run.stdout
    assert 'shape' not in run.stdout
    assert not run.stderr
