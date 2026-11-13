# GRAM's hydra overrides for each benchmark (ahn-ml/gram-release f2e32b9). Source this file; it defines GRAM_<BENCH>
# arrays. Sudoku data under $TRM_DATA (GRAM's Sudoku builder is TRM's, byte for byte); other data under $GRAM_DATA.
D=${TRM_DATA:?}
GRAM_COMMON=(arch=gram evaluators="[]" lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 arch.L_layers=2 arch.kl_balance_alpha=0.8 ema=True)
# Sudoku-Extreme: the authors' scripts/sudoku.sh (K=6, T=3, MLP mixer, value head, beta 0.1 by default).
GRAM_SUDOKU=("${GRAM_COMMON[@]}" global_batch_size=768 data_paths="[$D/sudoku-extreme-1k-aug-1000]" epochs=50000 eval_interval=5000 arch.mlp_t=True arch.pos_encodings=none arch.H_cycles=3 arch.L_cycles=6 arch.use_value_head=true)
# Maze-Unique: no official recipe. TRM's Maze-Unique row (its official Maze-Hard attention recipe on EqR's split) with
# GRAM's settings for its other tasks (K=4, T=3, beta 0.1, KL balance 0.8, no value head).
# Global batch 384 instead of TRM's 768, so the fit runs on one H200: at 768 GRAM's float32
# latent heads over 900 tokens need more than one H200's memory. lr and epochs are kept, so the fit takes twice
# the updates (130,208) on the same number of training samples.
# GRAM's loss head treats any data path containing "sudoku" as Sudoku (rule check for its halting and value targets);
# this repository's path contains it, so the maze data is read through a symlink without it (GRAM_DATA).
GD=${GRAM_DATA:-${THIRD_PARTY:-${SLURM_SUBMIT_DIR:-$PWD}/../third_party}/cache/gram-data}
GRAM_MAZEUNIQUE=("${GRAM_COMMON[@]}" global_batch_size=384 data_paths="[$GD/maze-unique-aug8]" epochs=50000 eval_interval=5000 arch.H_cycles=3 arch.L_cycles=4)
# N-Queens and graph coloring: the authors' scripts/nqueens_{8,10}.sh and
# graph_coloring_{8,10}.sh (global batch 768, attention, H=3, L=4, per-task epochs, eval_interval and KL weight), unchanged
# except the seed (0-2; theirs 42) and checkpoint_every_eval (no in-run evaluation, as every fit here). Data: GRAM's own
# builders, the same files the CRAM runs read. GRAM switches its loss-head
# checker on the data path ("nqueen", "graph-coloring", "sudoku"), so the paths below contain the task's word and not "sudoku".
GRAM_TASK=("${GRAM_COMMON[@]}" global_batch_size=768 arch.H_cycles=3 arch.L_cycles=4)
GRAM_NQUEENS8=("${GRAM_TASK[@]}" data_paths="[$GD/nqueens-8]" epochs=5000 eval_interval=50 arch.kl_coef=0.07)
GRAM_NQUEENS10=("${GRAM_TASK[@]}" data_paths="[$GD/nqueens-10]" epochs=1000 eval_interval=5 arch.kl_coef=0.045)
GRAM_GC8=("${GRAM_TASK[@]}" data_paths="[$GD/graph-coloring-8]" epochs=5000 eval_interval=50 arch.kl_coef=0.5)
GRAM_GC10=("${GRAM_TASK[@]}" data_paths="[$GD/graph-coloring-10]" epochs=5000 eval_interval=50 arch.kl_coef=0.45)
# TRM and HRM baselines on N-Queens and graph coloring. GRAM's paper (Baek et al. 2026,
# Sec. 4) reports them with "their official code and configurations" and gives no per-task recipe; GRAM's tree ships both
# architectures (config/arch/trm.yaml: H=3, L=6, two layers, 7M; hrm.yaml: H=2, L=2, 4 + 4 layers, 27M) with its own
# loss head and loader. Recipe: the architecture's yaml unchanged, and GRAM's per-task training settings above (global
# batch 768, lr 1e-4, weight decay 1.0, EMA, per-task epochs), without GRAM's KL settings. Deterministic at evaluation
# (sample_init off): N = 1 only, as the published rows.
BASE_COMMON=(evaluators="[]" lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 ema=True global_batch_size=768)
for A in TRM HRM; do
  a=${A,,}
  declare -a "GRAM_${A}_NQUEENS8=(arch=$a \"\${BASE_COMMON[@]}\" data_paths=\"[\$GD/nqueens-8]\" epochs=5000 eval_interval=50)"
  declare -a "GRAM_${A}_NQUEENS10=(arch=$a \"\${BASE_COMMON[@]}\" data_paths=\"[\$GD/nqueens-10]\" epochs=1000 eval_interval=5)"
  declare -a "GRAM_${A}_GC8=(arch=$a \"\${BASE_COMMON[@]}\" data_paths=\"[\$GD/graph-coloring-8]\" epochs=5000 eval_interval=50)"
  declare -a "GRAM_${A}_GC10=(arch=$a \"\${BASE_COMMON[@]}\" data_paths=\"[\$GD/graph-coloring-10]\" epochs=5000 eval_interval=50)"
done
