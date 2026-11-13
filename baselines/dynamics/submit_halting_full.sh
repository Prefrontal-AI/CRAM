#!/bin/bash
# Halting of HRM, TRM, EqR and TRMP on the full Sudoku-Extreme test set (results/halting-full.json):
# one rollout per puzzle for the trained depth (16 ACT steps; TRMP 32 supervision steps) on all 412,786 test puzzles
# (full-all rows 10,000 and up), three seeds. Freezes the code as dynamics/submit.sh does and submits one job per pass
# (one H200 each), then a CPU job for EBR (ebr/halting.py, from saved records) and a CPU job that exports
# results/halting-full.json after every pass has ended.
# Usage (from baselines/): bash dynamics/submit_halting_full.sh
set -euo pipefail
T=$PWD; REPO=$(realpath $T/..); EXPERIMENTS=$REPO/experiments; R=$T/runs; O=$R/looped-dynamics; B=$O/halting-full
[[ -f dynamics/run.sbatch && -d $R/compute-sweep ]] || { echo "run from baselines/"; exit 1; }
STAMP=$(date -u +%Y%m%dT%H%M%SZ); CODE=$O/code/$STAMP
mkdir -p $CODE/baselines $O/passes logs $B
for d in dynamics sweep hrm trm eqr trmp frm ebr; do rsync -a --exclude __pycache__ $T/$d $CODE/baselines/; done
rsync -a --exclude __pycache__ $REPO/cram $CODE/
cp $T/clock.py $CODE/baselines/
chmod -R a-w $CODE

# official test rows of full-all[10000:] (FRM's 1,000 test puzzles, then shards 00-15), the order of EqR's `unseen` split and of TRMP's test split
[[ -f $O/rows-sudoku-unseen.npy ]] || $REPO/.venv/bin/python -c "
import json, numpy as np
i = json.load(open('$REPO/data/full-test/index.json'))
names = [n for n in i['full_all_order'] if n != 'full-dev10k']
rows = np.concatenate([np.array(i['shards'][n]['official_rows'], dtype=np.int64) for n in names])
assert len(rows) == 412786 == len(set(rows.tolist()))
np.save('$O/rows-sudoku-unseen.npy', rows)"

HRM_CK=(sudoku-e40000-s0/step_31248 sudoku-e40000-s1/step_46872 sudoku-e40000-s2/step_31248)
HRM_COMMON="lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0"
HRM_S="data_path=$EXPERIMENTS/runs/trm/data/sudoku-extreme-1k-aug-1000 epochs=40000 eval_interval=4000 $HRM_COMMON"
TRM_COMMON="arch=trm evaluators=[] lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.L_layers=2 ema=True"
TRM_S="$TRM_COMMON data_paths=[$EXPERIMENTS/runs/trm/data/sudoku-extreme-1k-aug-1000] epochs=50000 eval_interval=5000 arch.mlp_t=True arch.pos_encodings=none arch.H_cycles=3 arch.L_cycles=6"
IDS=""
for k in 0 1 2; do
  declare -A P=()
  P[hrm]="hrm $B/hrm/sudoku-extreme-s$k/act16 $R/hrm/${HRM_CK[$k]} --rows $O/rows-sudoku-unseen.npy --ref $R/compute-sweep/hrm/sudoku-extreme-s$k/act32.npz $HRM_S arch.halt_max_steps=16 +seed=$k"
  P[trm]="trm $B/trm/sudoku-extreme-s$k/act16 $R/trm/sudoku-s$k/step_65100 --rows $O/rows-sudoku-unseen.npy --ref $R/compute-sweep/trm/sudoku-extreme-s$k/act32.npz $TRM_S seed=$k arch.halt_max_steps=16"
  P[eqr]="eqr --ckpt $R/eqr/sudoku-s$k/model.pt --data $EXPERIMENTS/runs/eqr/data/sudoku-eval --split unseen --depth 16 --noise 0.01 --batch 1000 --stop 412786 --out $B/eqr/sudoku-extreme-s$k/d16-noise0p01"
  P[trmp]="trmp --run $R/trmp/sudoku-extreme-s$k --steps 32 --stop 412786 --batch 1000 --out $B/trmp/sudoku-extreme-s$k/steps32"
  for g in hrm trm eqr trmp; do
    f=$O/passes/halting-full-$g-s$k-$STAMP.txt
    printf '%s\n' "${P[$g]}" > $f
    id=$(sbatch --parsable -J ld-halt-$g-s$k --time=${TIME:-03:00:00} --export=ALL,CODE=$CODE dynamics/run.sbatch $f)
    echo "$id ld-halt-$g-s$k $(date -u +%FT%TZ) code $CODE passes $f" | tee -a $O/jobs.txt
    IDS+=":$id"
  done
done
id=$(sbatch --parsable -J ld-halt-ebr --export=ALL,CODE=$CODE dynamics/halting_ebr.sbatch)
echo "$id ld-halt-ebr $(date -u +%FT%TZ) code $CODE" | tee -a $O/jobs.txt
IDS+=":$id"
id=$(sbatch --parsable -J ld-halt-fill --dependency=afterany$IDS dynamics/halting_fill.sbatch)
echo "$id ld-halt-fill $(date -u +%FT%TZ) after$IDS" | tee -a $O/jobs.txt
