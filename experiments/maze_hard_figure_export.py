"""Data for the Maze-Hard figure of the paper's appendix: one test maze solved by the CRAM model with a hidden layer.

  uv run python maze_hard_figure_export.py --maze 0     # writes ../results/maze-hard-figure.json

Model: the seed-0 final model of the paper's Maze-Hard recipe (recipes/maze-hard-latent.json; the file
runs/CRAM-HE/final-triton/CHE8-mu/final-s0.pt, where CHE8-mu is only the label of the run folder). Its hidden layer
solves A z = e_G with A = (4 + m) I - Adj on the open cells, and its answer energy is the Maze-Unique energy plus the
links gated by the attention (maze_hard_latent.py describes both). The model was already scored once on the test set.
This script runs its inference again on one test maze and checks that the result equals the scored one (the per-maze
record in test-scores.json).

Exported: the tokens, the label, the settled hidden states log z*_i, each cell's pointer (the offset with the largest
attention weight), the rounded answers of the damped descent (as maze_h3.descend: Euler steps 0.2 from logits 3) at a
few steps and at the stop, and the answer energy (`model.energy`, entropy included) of the continuous state sigmoid(u)
at every step. The descent is continued for `after` steps past the stop, to show that the stop is a fixed point of the
flow. The field `log_m` holds log(4 + m), the logarithm of the diagonal of A.
Dimension names: h w grid rows and columns, k offsets of the attention window (self first).
"""
import dataclasses
import hashlib
import json
from pathlib import Path

import torch
import tyro

import maze_hard_latent as H
from cram import maze_h3 as m3

HERE = Path(__file__).resolve().parent
FIN = HERE / 'runs' / 'CRAM-HE' / 'final-triton' / 'CHE8-mu'  # the final fits of the paper's Maze-Hard model (the folder name is only a label)


@dataclasses.dataclass
class Args:
    maze: int = 0  # test maze index (must be one the scored model solved)
    seed: int = 0  # training seed of the final model
    snapshots: tuple[int, ...] = (0, 16, 32, 64)  # descent steps whose rounded boards are exported (0: the all-on start)
    after: int = 48  # descent steps past the stop whose energy is also exported
    out: str = str(HERE.parent / 'results' / 'maze-hard-figure.json')  # output path


def main(a: Args):
    """Run the inference of the scored model on one test maze again, check it against the record, and write the JSON."""
    rec = json.loads((FIN / f'final-s{a.seed}.json').read_text())
    scores = json.loads((FIN / 'test-scores.json').read_text())
    assert scores['seeds'][str(a.seed)]['per_maze']['exact'][a.maze], 'choose a maze the scored model solved'
    model = H.build(H.Recipe(**rec['recipe']))
    model.load_state_dict(torch.load(FIN / f'final-s{a.seed}.pt', map_location='cpu'))
    tok, y = m3.load('test')
    T = torch.as_tensor(tok[a.maze:a.maze + 1])
    Y = torch.as_tensor(y[a.maze:a.maze + 1], dtype=torch.float32)
    with torch.no_grad():
        psi, steps = model.settle(T)  # psi = -log z
        att = model.couplings(T, psi)
        ref, used, done = m3.descend(model, T, att, steps=2048, eta=0.2)
        u = torch.full(T.shape, 3.0)
        boards = {0: (u > 0).int()[0].tolist()}
        energy = [float(model.energy(T, torch.sigmoid(u), att)[0])]
        for t in range(1, int(used[0]) + 1):
            u = u + 0.2 * (model.field(T, torch.sigmoid(u), att) - u)
            energy.append(float(model.energy(T, torch.sigmoid(u), att)[0]))
            if t in a.snapshots:
                boards[t] = (u > 0).int()[0].tolist()
        final = (u > 0).float()
        v, energy_after, moved = u.clone(), [], []
        for t in range(a.after):
            new = v + 0.2 * (model.field(T, torch.sigmoid(v), att) - v)
            moved.append(float((torch.sigmoid(new) - torch.sigmoid(v)).abs().max()))
            v = new
            energy_after.append(float(model.energy(T, torch.sigmoid(v), att)[0]))
        assert torch.equal((v > 0).float(), final), 'the rounded board must not change after the stop'
    assert bool(done[0]) and torch.equal(final, ref) and torch.equal(final, Y), 'the replayed descent must reproduce the scored solution'
    logz = (-psi[0]).tolist()
    out = {'maze': a.maze, 'split': 'test', 'seed': a.seed, 'model': str((FIN / f'final-s{a.seed}.pt').relative_to(HERE.parent)),
           'model_sha256': hashlib.sha256((FIN / f'final-s{a.seed}.pt').read_bytes()).hexdigest(),
           'tokens': tok[a.maze].tolist(), 'label': y[a.maze].astype(int).tolist(),
           'log_z': [[None if v == float('-inf') else v for v in row] for row in logz],
           'pointer_offset': [[list(model.aoffs[int(k)]) for k in row] for row in att[0].argmax(0).tolist()],
           'pointer_weight': att[0].amax(0).tolist(), 'hidden_steps': int(steps[0]), 'descent_steps': int(used[0]),
           'boards': {str(k): v for k, v in sorted(boards.items())}, 'final': final[0].int().tolist(),
           'energy': energy, 'energy_after_stop': energy_after, 'max_answer_change_after_stop': moved,
           'energy_of_rounded_final': float(model.energy(T, final, att)[0]),
           'log_m': float(torch.logaddexp(torch.tensor(1.3862943611198906), model.mu.detach()[0])),
           'note': 'tokens: 0 wall, 1 open, 2 S, 3 G. log_z: natural log of the settled hidden output (null where z = 0). '
                   'pointer_offset: (row, column) offset of the largest attention weight; [0, 0] = the cell itself. '
                   'energy[t]: answer energy (entropy included) of sigmoid(u) after t descent steps, t = 0..descent_steps; '
                   'energy_after_stop: the same for the steps after the stop.'}
    Path(a.out).write_text(json.dumps(out) + '\n')
    print('wrote', a.out, {k: out[k] for k in ('maze', 'hidden_steps', 'descent_steps', 'log_m')}, 'snapshots', list(out['boards']))


if __name__ == '__main__':
    main(tyro.cli(Args))
