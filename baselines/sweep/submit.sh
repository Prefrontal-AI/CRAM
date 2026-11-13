#!/bin/bash
# Submit the compute sweep for one task: each trained baseline is rescored at a grid of its inference settings
# (sweep/run.sbatch). Job ids are appended to runs/compute-sweep/jobs.txt. This script only calls sbatch.
# It needs the trained baselines in runs/<method>/ (the fit and score scripts of each baseline folder).
# Usage (from baselines/): bash sweep/submit.sh sudoku [method...]    all passes behind results/compute-sweep/sudoku-extreme.json
#                          bash sweep/submit.sh maze hrm trm          the Maze-Unique passes of HRM and TRM, which dynamics/submit.sh uses as a reference
# The TRMP step-cap passes (sweep/trmp_caps.py through run.sbatch trmp-caps) and the TRMP lattice sweep (sweep/trmp_lattice.sbatch) are submitted separately.
set -euo pipefail
BASELINES=$PWD; EXPERIMENTS=$BASELINES/../experiments; R=$BASELINES/runs; O=$R/compute-sweep
TASK=$1; shift; METHODS=${*:-hrm trm eqr trmp frm}
mkdir -p $O logs
TRM_DATA=$EXPERIMENTS/runs/trm/data; source $BASELINES/trm/trm_configs.sh
EQR_MAZE=${CRAM_DATA:-${SLURM_SUBMIT_DIR:-$PWD}/../data}/eqr/maze-30x30-unique-1k
HRM_COMMON=(lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.halt_max_steps=32)
sub() {  # sub <tag> [sbatch options --] <method> <args...>
  local tag=$1; shift; local opts=()
  while [[ $1 == --* ]]; do opts+=("$1"); shift; done
  local id; id=$(sbatch --parsable -J cs-$tag "${opts[@]}" sweep/run.sbatch "$@")
  echo "$id cs-$tag $(date -u +%FT%TZ) $*" | tee -a $O/jobs.txt
}
fdir() { [[ $1 == 0 ]] && echo $R/frm-compiled || echo $R/frm-compiled-s$1; }
for k in 0 1 2; do
  for m in $METHODS; do
    case $TASK-$m in
      maze-hrm)
        ck=$(python3 -c "import json;print(json.load(open('$R/hrm/mazeunique-s$k/score.json'))['checkpoint'])")
        sub hrm-maze-s$k hrm $O/hrm/maze-unique-s$k/act32 $ck --ref $R/hrm/mazeunique-s$k/score.right.npy \
          data_path=$EQR_MAZE epochs=20000 eval_interval=2000 "${HRM_COMMON[@]}" +seed=$k ;;
      sudoku-hrm)
        ck=$(python3 -c "import json;print(json.load(open('$R/hrm/sudoku-e40000-s$k/score.json'))['checkpoint'])")
        sub hrm-sudoku-s$k hrm $O/hrm/sudoku-extreme-s$k/act32 $ck --ref $R/hrm/sudoku-e40000-s$k/score.right.npy \
          data_path=$TRM_DATA/sudoku-extreme-1k-aug-1000 epochs=40000 eval_interval=4000 "${HRM_COMMON[@]}" +seed=$k ;;
      maze-trm)
        sub trm-maze-s$k trm $O/trm/maze-unique-s$k/act32 $R/trm/mazeunique-s$k/step_65100 --ref $R/trm/mazeunique-s$k/score.right.npy \
          "${TRM_MAZEUNIQUE[@]}" seed=$k arch.halt_max_steps=32 ;;
      sudoku-trm)
        sub trm-sudoku-s$k trm $O/trm/sudoku-extreme-s$k/act32 $R/trm/sudoku-s$k/step_65100 --ref $R/trm/sudoku-s$k/score.right.npy \
          "${TRM_SUDOKU[@]}" seed=$k arch.halt_max_steps=32 ;;
      sudoku-eqr)
        E=$R/eqr/sudoku-s$k; D=$EXPERIMENTS/runs/eqr/data/sudoku-eval
        sub eqr-sudoku-s$k-b1 eqr --ckpt $E/model.pt --data $D --split unseen --depth 64 --breadth 1 --noise 0.01 --batch 2048 --stop 10240 --out $O/eqr/sudoku-extreme-s$k/d64-b1
        for g in 0 1 2 3; do
          lo=$((g * 2560)); hi=$(( g == 3 ? 10000 : lo + 2560 ))
          sub eqr-sudoku-s$k-b64-g$g eqr --ckpt $E/model.pt --data $D --split first10k --depth 64 --breadth 64 --noise 0.5 --batch 128 --start $lo --stop $hi --out $O/eqr/sudoku-extreme-s$k/d64-b64-g$g
        done
        [[ $k == 0 ]] && sub eqr-sudoku-check eqr --ckpt $E/model.pt --data $D --split first10k --depth 16 --breadth 64 --noise 0.5 --batch 128 --stop 128 --out $O/eqr/sudoku-extreme-s0/check-d16-b64 ;;
      sudoku-trmp)
        for g in 0 1 2 3; do
          sub trmp-sudoku-s$k-g$g trmp --run runs/trmp/sudoku-extreme-s$k --start $((g * 2560)) --stop $((g * 2560 + 2560)) --out $O/trmp/sudoku-extreme-s$k/rollouts-g$g
        done ;;
      sudoku-frm)
        sub frm-sudoku-s$k-a frm sudoku $(fdir $k)/sudoku --seed $k --iters 8 32 128 512 1024 2048 --check4096 --out $O/frm/sudoku-extreme-s$k
        sub frm-sudoku-s$k-b frm sudoku $(fdir $k)/sudoku --seed $k --iters 8192 --out $O/frm/sudoku-extreme-s$k ;;
    esac
  done
done
