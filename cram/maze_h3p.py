"""Maze-Hard diagnostic: the path cells at which the shortest path is not unique one step ahead.

`tie_cells` marks the cells of the labeled path that have two or more side neighbours one step closer to the goal.
At such a cell the distance to the goal does not decide which neighbour continues the path.
`experiments/maze_hard_latent.py` reports the model's accuracy on these cells; the function is never a training target.

Dimension names: b mazes, h and w grid rows and columns.
"""

import torch

from cram.maze_learned import read

SIDES = ((1, 0), (0, -1), (0, 1), (-1, 0))  # the four side neighbours of a cell, as (row offset, column offset)


def tie_cells(tok, y):
    """(b, h, w) bool, diagnostic only: path cells (not G) with two or more side-neighbours one step closer to G by
    breadth-first search (BFS). tok (b, h, w): tokens (0 wall, 1 open, 2 S, 3 G); y (b, h, w): the path label (0 or 1)."""
    from cram.maze_h3 import bfs
    d = bfs(tok, 3)
    nb = torch.stack([read(d, di, dj, float('inf')) for di, dj in SIDES])
    return ((nb == d[None] - 1).sum(0) >= 2) & (y > 0.5) & (tok != 3)
