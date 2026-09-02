#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${SCRATCH_DIR:=/scratch/jordan/trial/outputs}"

TRAIN_ROOT="${TRAIN_ROOT:-experiments/E5_cluster_ablation/embedding_4}"
EVAL_FILE="${EVAL_FILE:-$TRAIN_ROOT/dpo/validation.jsonl}"
REFERENCE_MODEL="${REFERENCE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRATCH_DIR/cpo_kl_sweep}"

# At the observed KL scale (~50-200), this spans negligible through strong
# regularization without immediately overwhelming the ~0.5 base objective.
KL_COEFS="${KL_COEFS:-1e-4,3e-3,1e-2}"
ALPHAS="${ALPHAS:-0.0,0.3}"
SEED="${SEED:-42}"

mkdir -p "$OUTPUT_DIR"

poetry run python scripts/experiments/run_preference_sweeps.py \
  --methods cpo \
  --train-root "$TRAIN_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --learning-rates 1e-5 \
  --betas 0.005 \
  --alphas "$ALPHAS" \
  --kl-coefs "$KL_COEFS" \
  --z-baselines token_kl \
  --max-grad-norms 0.3 \
  --num-train-epochs 1.0 \
  --max-seq-length 512 \
  --per-device-train-batch-size 4 \
  --gradient-accumulation-steps 4 \
  --logging-steps 5 \
  --terminal-log-steps 500 \
  --save-steps 500 \
  --save-total-limit 1 \
  --seed "$SEED" \
  --run \
  --eval \
  --eval-file "$EVAL_FILE" \
  --reference-model-name-or-path "$REFERENCE_MODEL" \
  --score-metric normalized_reward_accuracy \
  --stop-on-failure

echo "CPO KL sweep complete: $OUTPUT_DIR"
