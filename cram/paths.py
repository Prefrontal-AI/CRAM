"""Where the repository keeps its data and run outputs.

`CRAM_DATA` overrides the dataset folder (default `<repository>/data`, written by `scripts/prepare_data.py`).
Run outputs go under `experiments/runs/` and `baselines/runs/`; neither is tracked, except the table records in
`experiments/runs/table/`.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('CRAM_DATA', REPO / 'data'))
EXPERIMENTS = REPO / 'experiments'
RUNS = EXPERIMENTS / 'runs'
