"""Successor labels of the Maze-Hard paths, for the pointer diagnostic of maze_hard_latent.py.

Dimension names: b mazes, h w grid rows and columns, k offsets of a window (self first).
"""
import numpy as np


def successors(tok,path,offs):
    """Each path cell's successor, read from the labeled path.

    tok: (b, h, w) int tokens (0 wall, 1 open, 2 S, 3 G). path: (b, h, w), nonzero on the labeled path (S and G too).
    offs: the offsets (row, column) of a window, the self offset (0, 0) first.
    Returns (b, h, w) int64: at a path cell other than G, the index k >= 1 into `offs` of the offset to the next path
    cell when the path is walked from S to G; 0 at G, at every cell off the path and at every wall.
    The walk moves to the one path cell in the window that is not the previous cell. It raises ValueError when that
    cell is not unique, when the walk returns to a visited cell, or when it ends without visiting every path cell.
    Maze-Hard evaluation uses this only to report how often the largest attention weight of a path cell is at its
    successor (`ptr_acc_path` in maze_hard_latent.evaluate); no training loss of the paper's recipe reads it."""
    out=np.zeros(tok.shape,dtype=np.int64)
    n,h,w=tok.shape
    for b in range(n):
        (si,sj),=np.argwhere(tok[b]==2)
        previous,current=None,(si,sj)
        seen=set()
        while tok[b][current]!=3:
            if current in seen:
                raise ValueError('learned support forms a cycle in labeled path')
            seen.add(current)
            candidates=[]
            for k,(dr,dc) in enumerate(offs[1:],1):
                nxt=(current[0]+dr,current[1]+dc)
                if 0<=nxt[0]<h and 0<=nxt[1]<w and path[b][nxt] and nxt!=previous:
                    candidates.append((k,nxt))
            if len(candidates)!=1:
                raise ValueError(f'learned support cannot order path {b}: {current}, {candidates}')
            k,nxt=candidates[0]
            out[b][current]=k
            previous,current=current,nxt
        if len(seen)+1!=int(path[b].sum()):
            raise ValueError('learned support misses labeled path cells')
    return out
