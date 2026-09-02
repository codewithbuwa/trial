#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${SCRATCH_DIR:=/scratch/jordan/trial/outputs}"

EVAL_FILE="${EVAL_FILE:-experiments/E5_cluster_ablation/embedding_4/dpo/validation.jsonl}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH_DIR/prometheus_final_models_5_prompts}"
PROMETHEUS_BASE_URL="${PROMETHEUS_BASE_URL:-http://localhost:8000/v1}"
PROMETHEUS_JUDGE_MODEL="${PROMETHEUS_JUDGE_MODEL:-prometheus-eval/prometheus-7b-v2.0}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-120}"
SEED="${SEED:-42}"

MODELS=(
  "DPO=$SCRATCH_DIR/dpo_final/dpo_lr1em05_b0p005_gn0p3"
  "KTO=$SCRATCH_DIR/kto_final/kto_lr1em05_b0p005_gn0p3"
  "CPO_UNARY=$SCRATCH_DIR/cpo_unary_final/cpo_lr1em05_b0p005_gn0p3_a0_token-kl"
  "CPO_A03=$SCRATCH_DIR/cpo_alpha_sweep/cpo_lr1em05_b0p005_gn0p3_a0p3_token-kl"
  "CPO_A05=$SCRATCH_DIR/cpo_alpha_sweep/cpo_lr1em05_b0p005_gn0p3_a0p5_token-kl"
  "CPO_A07=$SCRATCH_DIR/cpo_alpha_sweep/cpo_lr1em05_b0p005_gn0p3_a0p7_token-kl"
)

if [[ ! -s "$EVAL_FILE" ]]; then
  echo "Missing evaluation file: $EVAL_FILE" >&2
  exit 1
fi

for spec in "${MODELS[@]}"; do
  name="${spec%%=*}"
  path="${spec#*=}"
  if [[ ! -d "$path" ]]; then
    echo "Missing model directory for $name: $path" >&2
    exit 1
  fi
done

mkdir -p "$RUN_ROOT"

# Five prompts x 15 unique model pairs = 75 Prometheus judgments.
poetry run python scripts/evaluate/evaluate_judge.py \
  --eval-file "$EVAL_FILE" \
  --models "${MODELS[@]}" \
  --max-prompts 5 \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature 0 \
  --batch-size "$GEN_BATCH_SIZE" \
  --seed "$SEED" \
  --judge-provider prometheus \
  --judge-model "$PROMETHEUS_JUDGE_MODEL" \
  --openai-base-url "$PROMETHEUS_BASE_URL" \
  --judge-timeout "$JUDGE_TIMEOUT" \
  --randomize-positions \
  --output-jsonl "$RUN_ROOT/prometheus_pairwise.jsonl" \
  --summary-json "$RUN_ROOT/prometheus_summary.json"

echo "Prometheus smoke evaluation complete: $RUN_ROOT"
