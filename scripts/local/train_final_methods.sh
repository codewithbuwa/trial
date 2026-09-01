#!/usr/bin/env bash
set -euo pipefail

: "${SCRATCH_DIR:=/scratch/jordan/trial/outputs}"

TRAIN_ROOT="${TRAIN_ROOT:-experiments/E5_cluster_ablation/embedding_4}"
EVAL_FILE="${EVAL_FILE:-$TRAIN_ROOT/dpo/validation.jsonl}"
REFERENCE_MODEL="${REFERENCE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

mkdir -p "$SCRATCH_DIR"

COMMON_ARGS=(
  --train-root "$TRAIN_ROOT"
  --learning-rates 1e-5
  --betas 0.005
  --max-grad-norms 0.3
  --num-train-epochs 1.0
  --max-seq-length 512
  --per-device-train-batch-size 4
  --gradient-accumulation-steps 4
  --logging-steps 5
  --terminal-log-steps 500
  --save-steps 500
  --save-total-limit 1
  --seed 42
  --run
  --eval
  --eval-file "$EVAL_FILE"
  --reference-model-name-or-path "$REFERENCE_MODEL"
)

# poetry run python scripts/experiments/run_preference_sweeps.py \
#   --methods dpo \
#   --output-dir "$SCRATCH_DIR/dpo_final" \
#   "${COMMON_ARGS[@]}"

poetry run python scripts/experiments/run_preference_sweeps.py \
  --methods kto \
  --output-dir "$SCRATCH_DIR/kto_final" \
  "${COMMON_ARGS[@]}"

poetry run python scripts/experiments/run_preference_sweeps.py \
  --methods cpo \
  --output-dir "$SCRATCH_DIR/cpo_unary_final" \
  --alphas 0.0 \
  --z-baselines token_kl \
  "${COMMON_ARGS[@]}"

poetry run python scripts/experiments/run_preference_sweeps.py \
  --methods cpo \
  --output-dir "$SCRATCH_DIR/cpo_alpha_sweep" \
  --alphas 0.3,0.5,0.7 \
  --z-baselines token_kl \
  "${COMMON_ARGS[@]}"
