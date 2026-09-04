#!/usr/bin/env bash
# Evaluate the lambda_kl (kl_coef) sweep — CPO unary (alpha=0) and CPO alpha=0.3 —
# under all four scorers on a SINGLE shared set of generations:
#   1. teacher-forced pairwise accuracy + drift   (per checkpoint, logprob-based)
#   2. PairRM        (pairwise Bradley-Terry judge)
#   3. Skywork       (reward-model judge)
#   4. Prometheus    (LLM-rubric judge, non-BT)
#
# Generations are produced ONCE (phase 1) and every judge reads that file, so the
# four scorers are strictly comparable (same prompts, same responses).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# --- Paths / config -----------------------------------------------------------
CKPT_ROOT="${CKPT_ROOT:-outputs/cpo_kl_sweep}"
REFERENCE_MODEL="${REFERENCE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
EVAL_FILE="${EVAL_FILE:-experiments/E5_cluster_ablation/embedding_4/dpo/validation.jsonl}"
RUN_ROOT="${RUN_ROOT:-outputs/lambda_kl_all_scorers}"

MAX_PROMPTS="${MAX_PROMPTS:-500}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
BETA="${BETA:-0.005}"              # teacher-forced reward beta (matches training)
SEED="${SEED:-42}"

# Judge models / endpoint.
PAIRRM_JUDGE_MODEL="${PAIRRM_JUDGE_MODEL:-llm-blender/PairRM}"
SKYWORK_JUDGE_MODEL="${SKYWORK_JUDGE_MODEL:-Skywork/Skywork-Reward-Llama-3.1-8B-v0.2}"
PROMETHEUS_JUDGE_MODEL="${PROMETHEUS_JUDGE_MODEL:-prometheus-eval/prometheus-7b-v2.0}"
PROMETHEUS_BASE_URL="${PROMETHEUS_BASE_URL:-http://localhost:8000/v1}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-120}"
JUDGE_MAX_LENGTH="${JUDGE_MAX_LENGTH:-4096}"
# Concurrent in-flight requests for network judges (Prometheus). Ignored for the
# local judges (PairRM/Skywork). vLLM batches concurrent requests server-side.
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"

# Prometheus/PairRM are order-sensitive; --position-balanced cancels position bias
# per-pair (recommended for headline numbers, 2x judge cost). Skywork ignores it.
POSITION_MODE="${POSITION_MODE:---randomize-positions}"   # or --position-balanced

# Toggles.
RUN_GENERATION="${RUN_GENERATION:-1}"
RUN_TEACHER_FORCED="${RUN_TEACHER_FORCED:-1}"
RUN_PAIRRM="${RUN_PAIRRM:-1}"
RUN_SKYWORK="${RUN_SKYWORK:-1}"
RUN_PROMETHEUS="${RUN_PROMETHEUS:-1}"
FORCE="${FORCE:-0}"

# --- Checkpoints under evaluation (NAME=PATH) ---------------------------------
# The six lambda_kl sweep runs: alpha in {0 (unary), 0.3} x kl_coef in {1e-4,3e-3,1e-2}.
MODELS=(
  "U_kl1e-4=$CKPT_ROOT/cpo_lr1em05_b0p005_gn0p3_a0_token-kl_kl0p0001"
  "U_kl3e-3=$CKPT_ROOT/cpo_lr1em05_b0p005_gn0p3_a0_token-kl_kl0p003"
  "U_kl1e-2=$CKPT_ROOT/cpo_lr1em05_b0p005_gn0p3_a0_token-kl_kl0p01"
  "C03_kl1e-4=$CKPT_ROOT/cpo_lr1em05_b0p005_gn0p3_a0p3_token-kl_kl0p0001"
  "C03_kl3e-3=$CKPT_ROOT/cpo_lr1em05_b0p005_gn0p3_a0p3_token-kl_kl0p003"
  "C03_kl1e-2=$CKPT_ROOT/cpo_lr1em05_b0p005_gn0p3_a0p3_token-kl_kl0p01"
)
# To add the kl=0 baselines, append e.g.
#   "U_kl0=<path to cpo ... a0 ... (no kl suffix)>"
#   "C03_kl0=<path to cpo ... a0p3 ... (no kl suffix)>"

GEN_DIR="$RUN_ROOT/generations"
GEN_JSONL="$GEN_DIR/lambda_kl_generations.jsonl"
TF_DIR="$RUN_ROOT/teacher_forced"
JUDGE_DIR="$RUN_ROOT/judges"
mkdir -p "$GEN_DIR" "$TF_DIR" "$JUDGE_DIR"

# --- Pre-flight ---------------------------------------------------------------
if [[ ! -s "$EVAL_FILE" ]]; then
  echo "Missing eval file: $EVAL_FILE" >&2; exit 1
fi
for spec in "${MODELS[@]}"; do
  path="${spec#*=}"
  [[ -d "$path" ]] || { echo "Missing checkpoint: ${spec%%=*} -> $path" >&2; exit 1; }
done

# Build a prompt subset capped at MAX_PROMPTS (jsonl: one prompt per line).
PROMPTS_FILE="$GEN_DIR/prompts_${MAX_PROMPTS}.jsonl"
if [[ "$FORCE" == "1" || ! -s "$PROMPTS_FILE" ]]; then
  head -n "$MAX_PROMPTS" "$EVAL_FILE" > "$PROMPTS_FILE"
fi
echo "Prompts: $(wc -l < "$PROMPTS_FILE")  |  Models: ${#MODELS[@]}  |  Run root: $RUN_ROOT"

# --- Phase 1: generate ONCE (shared across all judges) ------------------------
if [[ "$RUN_GENERATION" == "1" ]]; then
  if [[ "$FORCE" == "1" || ! -s "$GEN_JSONL" ]]; then
    echo ">>> Phase 1: generating outputs for ${#MODELS[@]} checkpoints -> $GEN_JSONL"
    poetry run python scripts/evaluate/generate_from_prompts.py \
      --prompts-file "$PROMPTS_FILE" \
      --models "${MODELS[@]}" \
      --max-prompt-length "$MAX_PROMPT_LENGTH" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature 0 \
      --batch-size "$GEN_BATCH_SIZE" \
      --seed "$SEED" \
      --output-jsonl "$GEN_JSONL" \
      --output-md "$GEN_DIR/lambda_kl_generations.md"
  else
    echo ">>> Phase 1: reusing existing generations: $GEN_JSONL"
  fi
fi

# --- Phase 2: teacher-forced pairwise accuracy + drift (per checkpoint) --------
if [[ "$RUN_TEACHER_FORCED" == "1" ]]; then
  echo ">>> Phase 2: teacher-forced pairwise accuracy (vs $REFERENCE_MODEL)"
  for spec in "${MODELS[@]}"; do
    name="${spec%%=*}"; path="${spec#*=}"
    out="$TF_DIR/${name}_pairwise_accuracy.json"
    if [[ "$FORCE" != "1" && -s "$out" ]]; then
      echo "  skip $name (exists)"; continue
    fi
    poetry run python scripts/evaluate/evaluate_pairwise_accuracy.py \
      --eval-file "$EVAL_FILE" \
      --model-name-or-path "$path" \
      --reference-model-name-or-path "$REFERENCE_MODEL" \
      --beta "$BETA" \
      --limit "$MAX_PROMPTS" \
      --output-json "$out"
  done
fi

# --- Phase 3: judges, all reading the SAME generations file -------------------
run_judge() {
  local provider="$1" model="$2"; shift 2
  local out_jsonl="$JUDGE_DIR/${provider}_pairwise.jsonl"
  local out_summary="$JUDGE_DIR/${provider}_summary.json"
  if [[ "$FORCE" != "1" && -s "$out_summary" ]]; then
    echo "  skip $provider (exists: $out_summary)"; return
  fi
  echo ">>> Phase 3: judge=$provider  model=$model"
  poetry run python scripts/evaluate/evaluate_judge.py \
    --generations-file "$GEN_JSONL" \
    --judge-provider "$provider" \
    --judge-model "$model" \
    --judge-timeout "$JUDGE_TIMEOUT" \
    --judge-max-length "$JUDGE_MAX_LENGTH" \
    --judge-concurrency "$JUDGE_CONCURRENCY" \
    --seed "$SEED" \
    "$POSITION_MODE" \
    --output-jsonl "$out_jsonl" \
    --summary-json "$out_summary" \
    "$@"
}

[[ "$RUN_PAIRRM"     == "1" ]] && run_judge pairrm  "$PAIRRM_JUDGE_MODEL"
[[ "$RUN_SKYWORK"    == "1" ]] && run_judge skywork "$SKYWORK_JUDGE_MODEL"
[[ "$RUN_PROMETHEUS" == "1" ]] && run_judge prometheus "$PROMETHEUS_JUDGE_MODEL" \
  --openai-base-url "$PROMETHEUS_BASE_URL"

echo "Done. Results under: $RUN_ROOT"
echo "  generations:     $GEN_JSONL"
echo "  teacher-forced:  $TF_DIR/*_pairwise_accuracy.json"
echo "  judge summaries: $JUDGE_DIR/{pairrm,skywork,prometheus}_summary.json"
