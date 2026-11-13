"""GRAM Sudoku on the 412,786 unseen puzzles: every row of score.right.npz (gram/score.py merge, one flag per row of
TRM's 422,786-row test split in loader order) restricted to the rows outside dev10k, with baselines/trm/unseen.py's checks
(TRM's test split equals the official test row by row; dev10k sits at its indices). CPU only.

  python gram/unseen.py <score folder>   # -> <folder>/score-unseen.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trm'))
from unseen import digest, unseen_mask  # noqa: E402


def main(folder):
    folder = Path(folder)
    mask, idx = unseen_mask()
    score = json.loads((folder / 'score.json').read_text())
    flags = np.load(folder / 'score.right.npz')
    table = {}
    for key in flags.files:
        right = flags[key]
        assert right.dtype == bool and right.size == score['n'] == mask.size
        table[key] = dict(solved=int(right[mask].sum()), n=int(mask.sum()), exact=float(right[mask].mean()),
                          all=dict(solved=int(right.sum()), n=int(right.size)), dev10k=dict(solved=int(right[idx].sum()), n=int(idx.size)))
    res = dict(method='GRAM', split='official Sudoku-Extreme test rows outside dev10k (412,786 unseen)', table=table,
               source=str(folder / 'score.right.npz'), source_sha256=digest(folder / 'score.right.npz'),
               script=str(Path(__file__).resolve()), script_sha256=digest(__file__))
    (folder / 'score-unseen.json').write_text(json.dumps(res, indent=1) + '\n')
    print(folder, {k: round(v['exact'], 5) for k, v in table.items() if k.startswith('steps16')}, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
