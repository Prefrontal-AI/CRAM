"""Score the authors' FPRM Maze-Hard checkpoint with the evaluation setting of their published 87.0%.

The loop is the authors' (maze/reproduce/singlez_eval_sweep.py in the checkpoint repository): batches of 768 from their
test loader (the second batch holds 232 mazes and 536 pad rows), a fresh random state per batch after
torch.manual_seed(4), bf16 autocast, 20 iterations per compiled call, and a stop when the model reports the batch
halted, or every row has residue < fp_thresh or step size < 1e-3, or 35,000 iterations have run. A maze therefore runs
the number of iterations of its batch, a multiple of 20.

Writes <out>/batch<k>.json after each batch (with the CUDA RNG state, so a requeued job continues the same random
stream) and <out>/final.json. A run whose final.json exists exits at once.

Run (one GPU): cd $FPRM_SRC && .venv/bin/python <repo>/baselines/fprm/score.py <out dir> [flop record of fprm_count.py]

Dimension names: b rows of a batch, n tokens of a row.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
from common import PUBLISHED, load, sha256, test_batches  # noqa: E402

N_BULK = 20


def main(out, flop_record=None):
    out = Path(out)
    if (out / 'final.json').exists():
        print('final.json exists; nothing to do')
        return
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(PUBLISHED['seed'])
    torch.cuda.manual_seed_all(PUBLISHED['seed'])
    model, meta, rec = load(batch_size=PUBLISHED['batch_size'])
    fp_thresh = model.config.fp_thresh
    autocast = torch.autocast(device_type='cuda', dtype=torch.bfloat16)

    def _bulk_step(carry, batch):
        for _ in range(N_BULK):
            with autocast:
                carry, outputs = model(carry=carry, batch=batch)
        return carry, outputs

    bulk_step = torch.compile(_bulk_step, mode='default', dynamic=False)
    batches = []
    with torch.inference_mode():
        for index, (batch, real) in enumerate(test_batches(PUBLISHED['batch_size'])):
            done_file, state_file = out / f'batch{index}.json', out / f'batch{index}.rng.pt'
            if done_file.exists():
                batches.append(json.loads(done_file.read_text()))
                torch.cuda.set_rng_state(torch.load(state_file))
                continue
            with torch.device('cuda'):
                carry = model.initial_carry(batch)
            steps, start = 0, time.time()
            while True:
                carry, outputs = bulk_step(carry, batch)
                steps += N_BULK
                if carry.halted.all():
                    reason = 'model halted'
                    break
                state = carry.inner_carry.z_L_state
                row_done = (state['residues'] < fp_thresh) | (state['stepsize'].view(-1) < 1e-3)  # [b]
                if row_done.all() or steps >= model.max_iter:
                    reason = 'every row done' if row_done.all() else 'iteration cap'
                    break
                if steps % 1000 == 0:
                    print(f'batch {index} step {steps} rows done {int(row_done.sum())}/{len(row_done)} {time.time() - start:.0f}s', flush=True)
            preds = outputs['logits'].argmax(dim=-1)  # [b, n]
            valid = batch['labels'] != -100  # [b, n]
            row_real = valid.any(dim=1)  # [b]
            solved = (((preds == batch['labels']) | ~valid).all(dim=1) & row_real)
            state = carry.inner_carry.z_L_state
            b = dict(batch=index, rows=len(row_real), mazes=real, steps=steps, stop=reason, solved=int(solved.sum()),
                     token_correct=int(((preds == batch['labels']) & valid).sum()), tokens=int(valid.sum()),
                     real_rows_below_fp_thresh=int((state['residues'] < fp_thresh)[row_real].sum()),
                     real_rows_stepsize_out=int((state['stepsize'].view(-1) < 1e-3)[row_real].sum()), seconds=time.time() - start)
            assert int(row_real.sum()) == real
            torch.save(torch.cuda.get_rng_state(), state_file)
            done_file.write_text(json.dumps(b, indent=1) + '\n')
            batches.append(b)
            print(json.dumps(b), flush=True)
    mazes = sum(b['mazes'] for b in batches)
    assert mazes == 1000
    steps_per_maze = sum(b['mazes'] * b['steps'] for b in batches) / mazes
    rec.update(method='FPRM', benchmark='maze-hard', batches=batches, mazes=mazes, solved=sum(b['solved'] for b in batches),
               solve_rate=sum(b['solved'] for b in batches) / mazes, token_accuracy=sum(b['token_correct'] for b in batches) / sum(b['tokens'] for b in batches),
               steps_per_maze=steps_per_maze, steps_per_maze_note='each maze runs the iterations of its batch; mean over the 1,000 mazes (pad rows excluded)',
               steps_per_batch_unweighted=sum(b['steps'] for b in batches) / len(batches),
               published=dict(solve_rate=0.870, avg_outer_steps_per_batch=23900.0, source='fixed-point-reasoners/fprm maze/eval_results/conv1d_s4_decay0.996_pat10.json'),
               n_bulk=N_BULK, compiled=True, slurm_job=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
               script=str(HERE), script_sha256=sha256(HERE), common_sha256=sha256(HERE.parent / 'common.py'),
               created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    if flop_record:
        f = json.loads(Path(flop_record).read_text())
        assert f['checkpoint_sha256'] == rec['checkpoint_sha256']
        rec.update(flop_record=str(flop_record), flop_record_sha256=sha256(flop_record), flop_per_iteration=f['flop_per_iteration'],
                   flop_per_puzzle=f['flop_setup'] + steps_per_maze * f['flop_per_iteration'] + f['flop_decode'],
                   flop_per_puzzle_with_compare=steps_per_maze * f['flop_per_iteration_with_compare'],
                   flop_per_puzzle_old=steps_per_maze * f['flop_per_iteration_old'])
    (out / 'final.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k != 'params_by_tensor'}, indent=1), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
