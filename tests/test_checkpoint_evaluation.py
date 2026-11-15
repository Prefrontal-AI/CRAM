import sys
from types import SimpleNamespace

import pytest

import cram
from checkpoints import evaluate


@pytest.fixture
def mock_sudoku(monkeypatch):
    """Supply small stand-ins for the dataset, weights and GPU search."""
    def setup(expected_slice, rows, scores):
        def load_puzzles(split, start=0, count=None):
            if split == 'train-s0':
                return [0], None, None
            assert (split, start, count) == expected_slice
            return range(rows), range(rows), None

        def load_weights(*args, **kwargs):
            assert rows == expected_slice[2], 'short cohort reached model loading'
            return {'W': None, 'B': None}

        stats = SimpleNamespace(load_puzzles=load_puzzles, summarize=lambda result: result)
        search = SimpleNamespace(GeneralSearch=lambda *args, **kwargs: SimpleNamespace(run=lambda report: scores))
        frozen = dict(C=1, tol=0.0, cap=1, check=1, anneal=(0, 0, 1))
        ready = dict(S=None, hbar=1.0, tau_train=1.0, ebar=1.0)
        monkeypatch.setattr(cram, 'cram_sudoku_search', stats, raising=False)
        monkeypatch.setattr(cram, 'cram_sudoku_general', search, raising=False)
        monkeypatch.setitem(sys.modules, 'sudoku', SimpleNamespace(prepare=lambda *args: ready, FROZEN=frozen))
        monkeypatch.setattr(evaluate.torch, 'load', load_weights)
        monkeypatch.setattr(evaluate.torch, 'as_tensor', lambda values, device: values)

    return setup


@pytest.mark.parametrize('mode, split, start, count', [
    ('dev', 'dev10k', 0, 10_000),
    ('test', 'full-all', 10_000, 412_786),
])
def test_complete_cohort_requires_exact_scores(mock_sudoku, mode, split, start, count):
    record = evaluate.entry('sudoku-extreme', 0)['expected']
    scores = dict(record['dev'] if mode == 'dev' else record)
    mock_sudoku((split, start, count), count, scores)
    assert evaluate.sudoku(0, mode, count)[2]

    scores['mean_flop'] += 1
    assert not evaluate.sudoku(0, mode, count)[2]


@pytest.mark.parametrize('mode, split, start, count', [
    ('dev', 'dev10k', 0, 10_000),
    ('test', 'full-all', 10_000, 412_786),
    ('test', 'full-all', 10_000, 3_000),
])
def test_missing_puzzles_fail_before_model_loading(mock_sudoku, mode, split, start, count):
    mock_sudoku((split, start, count), count - 1, None)
    with pytest.raises(ValueError, match=f'requires {count:,} puzzles'):
        evaluate.sudoku(0, mode, count)


@pytest.mark.parametrize('count, solved, wrong_stops, status, code', [
    (3_000, 2_999, 0, 'PASSES', 0),
    (3_000, 2_990, 0, 'FAILS', 1),
    (3_000, 2_999, 1, 'FAILS', 1),
    (2_999, 2_999, 0, 'FAILS', 1),
])
def test_partial_cli_rate_check(mock_sudoku, capsys, count, solved, wrong_stops, status, code):
    scores = dict(n=count, solved=solved, wrong_stops=wrong_stops, budget_failures=0, mean_flop=1.0)
    mock_sudoku(('full-all', 10_000, 3_000), 3_000, scores)
    args = evaluate.tyro.cli(evaluate.Args, args=[
        '--tasks', 'sudoku-extreme', '--sudoku-set', 'test', '--sudoku-n', '3000',
    ])
    with pytest.raises(SystemExit) as exc:
        evaluate.main(args)
    assert exc.value.code == code
    out = capsys.readouterr().out
    assert f'{status} the partial test solve-rate check (3,000 puzzles)' in out
    assert 'matches its record' not in out


@pytest.mark.parametrize('count', [-1, 0, 412_787])
def test_sample_count_must_be_in_range(count):
    with pytest.raises(ValueError, match='must be between 1 and 412786'):
        evaluate.sudoku(0, 'test', count)
