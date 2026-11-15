"""CPU-only fixtures for dataset provenance checks."""
import hashlib
import json
import subprocess

import numpy as np
import pytest

from cram import maze_unique
from scripts import prepare_data


def test_maze_unique_checks_pinned_array(tmp_path, monkeypatch):
    rel = 'all__inputs.npy'
    path = tmp_path / rel
    expected = np.array([[1, 2, 3, 4]], dtype=np.uint8)
    np.save(path, expected)
    monkeypatch.setattr(maze_unique, 'EQR', tmp_path)
    monkeypatch.setattr(maze_unique, 'SHA256', {rel: hashlib.sha256(path.read_bytes()).hexdigest()})

    np.testing.assert_array_equal(maze_unique._read(rel), expected)
    np.save(path, expected + 1)
    with pytest.raises(ValueError, match='not the pinned file'):
        maze_unique._read(rel)


def test_sudoku_rejects_unverified_source_before_writing_split(tmp_path, monkeypatch):
    csv_path = tmp_path / 'train.csv'
    csv_path.write_text('source,question,answer,rating\nfixture,.,1,1000\n')
    index = dict(dataset='fixture', revision='fixture', train_csv_rows=1, train_csv_content_sha256='0' * 64)
    (tmp_path / 'sudoku_extreme_splits.json').write_text(json.dumps(index))
    data = tmp_path / 'data'
    monkeypatch.setattr(prepare_data, 'HERE', tmp_path)
    monkeypatch.setattr(prepare_data, 'DATA', data)
    monkeypatch.setattr(prepare_data, 'hf_hub_download', lambda *args, **kwargs: csv_path)

    with pytest.raises(ValueError, match='train.csv is not the pinned file'):
        prepare_data.sudoku()
    assert not data.exists()


@pytest.mark.parametrize('valid_full_order', [True, False])
def test_sudoku_validates_full_order_before_writing_shards(tmp_path, monkeypatch, valid_full_order):
    solutions = np.array([[1], [2], [3]], dtype=np.uint8)
    clues = np.zeros_like(solutions)
    ratings = np.array([1001, 1002, 1003])

    def digest(rows):
        return prepare_data.content_sha256(solutions[rows], clues[rows], ratings[rows])

    dtypes = dict(solutions='uint8', clue_mask='uint8', ratings='int32')
    index = dict(dataset='fixture', revision='fixture', files={
        'frm/dev10k/dev.npz': dict(csv='test', rows=[1], content_sha256=digest([1]), dtypes=dtypes),
        'frm/split/dev.npz': dict(csv='test', rows=[1], content_sha256=digest([1]), dtypes=dtypes),
        'frm/split/test.npz': dict(csv='test', rows=[0], content_sha256=digest([0]), dtypes=dtypes),
    }, full_all_content_sha256=digest([1, 0, 2]) if valid_full_order else digest([0, 1, 2]))
    for name in ('train', 'test'):
        (tmp_path / f'{name}.csv').write_text(
            'source,question,answer,rating\nfixture,.,1,1001\nfixture,.,2,1002\nfixture,.,3,1003\n')
        index[f'{name}_csv_rows'] = 3
        index[f'{name}_csv_content_sha256'] = digest([0, 1, 2])
    (tmp_path / 'sudoku_extreme_splits.json').write_text(json.dumps(index))
    data = tmp_path / 'data'
    monkeypatch.setattr(prepare_data, 'HERE', tmp_path)
    monkeypatch.setattr(prepare_data, 'DATA', data)
    monkeypatch.setattr(prepare_data, 'hf_hub_download', lambda _, name, **kwargs: tmp_path / name)

    if not valid_full_order:
        with pytest.raises(ValueError, match='differs from the paper'):
            prepare_data.sudoku()
        assert not (data / 'full-test').exists()
        return
    prepare_data.sudoku()
    with np.load(data / 'full-test' / 'full-all.npz') as arrays:
        np.testing.assert_array_equal(arrays['solutions'], solutions[[1, 0, 2]].astype(np.int64))


@pytest.mark.parametrize('checkout', ['different_revision', 'modified', 'untracked'])
def test_gram_rejects_unverified_checkout(tmp_path, monkeypatch, checkout):
    source = tmp_path / 'gram'
    source.mkdir()
    git = ['git', '-C', str(source), '-c', 'core.hooksPath=/dev/null']
    subprocess.run([*git, 'init', '-q', '--template='], check=True, stdin=subprocess.DEVNULL)
    fixture = source / 'README.md'
    fixture.write_text('Fixture repository\n')
    subprocess.run([*git, 'add', 'README.md'], check=True, stdin=subprocess.DEVNULL)
    subprocess.run([*git, '-c', 'user.name=CRAM fixture', '-c', 'user.email=fixture@example.invalid',
                    '-c', 'commit.gpgSign=false', 'commit', '-qm', 'Create fixture'],
                   check=True, stdin=subprocess.DEVNULL)
    head = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True, stdin=subprocess.DEVNULL).strip()
    if checkout == 'modified':
        fixture.write_text('Local modification\n')
    if checkout == 'untracked':
        (source / 'local-notes.txt').write_text('Local notes\n')
    monkeypatch.setattr(prepare_data, 'GRAM_COMMIT', '0' * len(head) if checkout == 'different_revision' else head)
    monkeypatch.setattr(prepare_data, 'DATA', tmp_path / 'data')

    expected = 'is at' if checkout == 'different_revision' else 'use a clean GRAM checkout'
    with pytest.raises(ValueError, match=expected):
        prepare_data.multi_solution(prepare_data.MultiSolution(gram_src=source))
