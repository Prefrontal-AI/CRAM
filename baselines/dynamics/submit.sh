#!/bin/bash
# Looped-model dynamics: freeze the code into runs/looped-dynamics/code/<stamp>/, write the pass files and submit one
# job per pass file (dynamics/run.sbatch). This script only writes files and calls sbatch.
# It needs the trained baselines in runs/<method>/ and the compute-sweep passes of HRM and TRM in runs/compute-sweep/
# (sweep/submit.sh), which the HRM and TRM passes use as a reference.
# Usage (from baselines/; ONLY="<groups>" restricts the jobs):
#   bash dynamics/submit.sh bench    a short slice to measure the run time, outputs in runs/looped-dynamics/bench/
#   bash dynamics/submit.sh full     the passes behind results/looped-dynamics.json, outputs in runs/looped-dynamics/<method>/
set -euo pipefail
MODE=$1
T=$PWD; REPO=$(realpath $T/..); EXPERIMENTS=$REPO/experiments; R=$T/runs; O=$R/looped-dynamics
[[ -f dynamics/run.sbatch && -d $R/compute-sweep ]] || { echo "run from baselines/"; exit 1; }
STAMP=$(date -u +%Y%m%dT%H%M%SZ); CODE=$O/code/$STAMP
mkdir -p $CODE/baselines $O/passes logs
for d in dynamics sweep hrm trm eqr trmp frm; do rsync -a --exclude __pycache__ $T/$d $CODE/baselines/; done
rsync -a --exclude __pycache__ $REPO/cram $CODE/
cp $T/clock.py $CODE/baselines/  # imported by trmp/fit.py
chmod -R a-w $CODE

# Puzzle rows of the HRM/TRM test splits (official order): Sudoku = full-frmtest (FRM's 1,000 test puzzles =
# full-all rows 10,000-10,999 = EqR first10k[:1000] = TRMP test[:1000]); Maze-Unique = all 1,000 test mazes.
[[ -f $O/rows-sudoku.npy ]] || $REPO/.venv/bin/python -c "
import json, numpy as np
i = json.load(open('$REPO/data/full-test/index.json'))['shards']['full-frmtest']['official_rows']
np.save('$O/rows-sudoku.npy', np.array(i, dtype=np.int64)); np.save('$O/rows-maze.npy', np.arange(1000, dtype=np.int64))"

HRM_CK=(sudoku-e40000-s0/step_31248 sudoku-e40000-s1/step_46872 sudoku-e40000-s2/step_31248)
HRM_CKM=(mazeunique-s0/step_26040 mazeunique-s1/step_20832 mazeunique-s2/step_15624)
HRM_COMMON="lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0"
HRM_S="data_path=$EXPERIMENTS/runs/trm/data/sudoku-extreme-1k-aug-1000 epochs=40000 eval_interval=4000 $HRM_COMMON"
BASELINES=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); THIRD_PARTY=${THIRD_PARTY:-$BASELINES/../third_party}  # locations: baselines/README.md
HRM_M="data_path=$BASELINES/../data/eqr/maze-30x30-unique-1k epochs=20000 eval_interval=2000 $HRM_COMMON"
TRM_COMMON="arch=trm evaluators=[] lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.L_layers=2 ema=True"
TRM_S="$TRM_COMMON data_paths=[$EXPERIMENTS/runs/trm/data/sudoku-extreme-1k-aug-1000] epochs=50000 eval_interval=5000 arch.mlp_t=True arch.pos_encodings=none arch.H_cycles=3 arch.L_cycles=6"
TRM_M="$TRM_COMMON data_paths=[$EXPERIMENTS/runs/trm/data/maze-unique-aug8] epochs=50000 eval_interval=5000 arch.H_cycles=3 arch.L_cycles=4"
FRM_ROOT=(frm-compiled frm-compiled-s1 frm-compiled-s2)
K_ACT=128  # 8 x the trained halt_max_steps of 16 (HRM, TRM, EqR)

if [[ $MODE == bench ]]; then
  B=$O/bench; SEEDS="0"
else
  B=$O; SEEDS="0 1 2"
fi
declare -A P
for k in $SEEDS; do
  P[hrm]+="hrm $B/hrm/sudoku-extreme-s$k/act$K_ACT $R/hrm/${HRM_CK[$k]} --rows $O/rows-sudoku.npy --ref $R/compute-sweep/hrm/sudoku-extreme-s$k/act32.npz $HRM_S arch.halt_max_steps=$K_ACT +seed=$k"$'\n'
  P[hrm]+="hrm $B/hrm/maze-unique-s$k/act$K_ACT $R/hrm/${HRM_CKM[$k]} --rows $O/rows-maze.npy --ref $R/compute-sweep/hrm/maze-unique-s$k/act32.npz $HRM_M arch.halt_max_steps=$K_ACT +seed=$k"$'\n'
  P[trm]+="trm $B/trm/sudoku-extreme-s$k/act$K_ACT $R/trm/sudoku-s$k/step_65100 --rows $O/rows-sudoku.npy --ref $R/compute-sweep/trm/sudoku-extreme-s$k/act32.npz $TRM_S seed=$k arch.halt_max_steps=$K_ACT"$'\n'
  P[trm]+="trm $B/trm/maze-unique-s$k/act$K_ACT $R/trm/mazeunique-s$k/step_65100 --rows $O/rows-maze.npy --ref $R/compute-sweep/trm/maze-unique-s$k/act32.npz $TRM_M seed=$k arch.halt_max_steps=$K_ACT"$'\n'
  for nz in 0 0.01; do
    P[eqr]+="eqr --ckpt $R/eqr/sudoku-s$k/model.pt --data $EXPERIMENTS/runs/eqr/data/sudoku-eval --split first10k --depth $K_ACT --noise $nz --batch 250 --stop 1000 --out $B/eqr/sudoku-extreme-s$k/d$K_ACT-noise${nz/./p}"$'\n'
    P[eqr]+="eqr-nocompile --ckpt $R/eqr/maze-repo-s$k/model.pt --data $EXPERIMENTS/runs/eqr/data/maze-unique --split test --depth $K_ACT --noise $nz --batch 250 --stop 1000 --out $B/eqr/maze-unique-s$k/d$K_ACT-noise${nz/./p}"$'\n'
  done
  if [[ $MODE == bench ]]; then
    P[trmp]+="trmp --run $R/trmp/sudoku-extreme-s$k --steps 1024 --stop 250 --out $B/trmp/sudoku-extreme-s$k/steps1024"$'\n'
    P[trmp]+="trmp --run $R/trmp/maze-unique-s$k --steps 256 --stop 250 --out $B/trmp/maze-unique-s$k/steps256"$'\n'
    P[frm]+="frm sudoku $R/${FRM_ROOT[$k]}/sudoku --seed $k --variant table --iters 4096 --out $B/frm/sudoku-extreme-s$k/table-iters4096"$'\n'
    P[frm]+="frm maze $R/${FRM_ROOT[$k]}/mazeunique --seed $k --calls 300 --out $B/frm/maze-unique-s$k/calls300"$'\n'
  else
    P[trmp]+="trmp --run $R/trmp/sudoku-extreme-s$k --steps 1024 --stop 1000 --out $B/trmp/sudoku-extreme-s$k/steps1024"$'\n'
    P[trmp]+="trmp --run $R/trmp/maze-unique-s$k --steps 256 --stop 1000 --out $B/trmp/maze-unique-s$k/steps256"$'\n'
    P[frms]+="frm sudoku $R/${FRM_ROOT[$k]}/sudoku --seed $k --variant table --iters 32768 --out $B/frm/sudoku-extreme-s$k/table-iters32768"$'\n'
    P[frms]+="frm sudoku $R/${FRM_ROOT[$k]}/sudoku --seed $k --variant det --iters 32768 --out $B/frm/sudoku-extreme-s$k/det-iters32768"$'\n'
    P[frmm]+="frm maze $R/${FRM_ROOT[$k]}/mazeunique --seed $k --calls 2056 --out $B/frm/maze-unique-s$k/calls2056"$'\n'
    # float32 controls: is a small relative change a bf16 rounding floor? (HRM's FlashAttention 3 needs bf16: no control)
    P[trm32]+="trm $B/trm/sudoku-extreme-s$k/act$K_ACT-fp32 $R/trm/sudoku-s$k/step_65100 --rows $O/rows-sudoku.npy --ref $R/compute-sweep/trm/sudoku-extreme-s$k/act32.npz $TRM_S seed=$k arch.halt_max_steps=$K_ACT arch.forward_dtype=float32"$'\n'
    P[trm32]+="trm $B/trm/maze-unique-s$k/act$K_ACT-fp32 $R/trm/mazeunique-s$k/step_65100 --rows $O/rows-maze.npy --ref $R/compute-sweep/trm/maze-unique-s$k/act32.npz $TRM_M seed=$k arch.halt_max_steps=$K_ACT arch.forward_dtype=float32"$'\n'
    P[eqr32]+="eqr --ckpt $R/eqr/sudoku-s$k/model.pt --data $EXPERIMENTS/runs/eqr/data/sudoku-eval --split first10k --depth $K_ACT --noise 0 --fp32 --batch 250 --stop 1000 --out $B/eqr/sudoku-extreme-s$k/d$K_ACT-noise0-fp32"$'\n'
    P[eqr32]+="eqr-nocompile --ckpt $R/eqr/maze-repo-s$k/model.pt --data $EXPERIMENTS/runs/eqr/data/maze-unique --split test --depth $K_ACT --noise 0 --fp32 --batch 250 --stop 1000 --out $B/eqr/maze-unique-s$k/d$K_ACT-noise0-fp32"$'\n'
    P[trmp32]+="trmp --run $R/trmp/sudoku-extreme-s$k --steps 1024 --stop 1000 --fp32 --out $B/trmp/sudoku-extreme-s$k/steps1024-fp32"$'\n'
    P[trmp32]+="trmp --run $R/trmp/maze-unique-s$k --steps 256 --stop 1000 --fp32 --out $B/trmp/maze-unique-s$k/steps256-fp32"$'\n'
  fi
done
for g in "${!P[@]}"; do
  [[ -n ${ONLY:-} && " $ONLY " != *" $g "* ]] && continue  # ONLY="trmp frm": submit these groups only
  f=$O/passes/$MODE-$g-$STAMP.txt
  printf '%s' "${P[$g]}" > $f
  id=$(sbatch --parsable -J ld-$MODE-$g --time=${TIME:-03:00:00} --export=ALL,CODE=$CODE dynamics/run.sbatch $f)
  echo "$id ld-$MODE-$g $(date -u +%FT%TZ) code $CODE passes $f" | tee -a $O/jobs.txt
done
