#!/usr/bin/env bash
set -euo pipefail

: "${SCRATCH_DIR:=/scratch/jordan/trial/outputs}"

TRAIN_ROOT="${TRAIN_ROOT:-experiments/E5_cluster_ablation/embedding_4}"
PROMPTS_FILE="${PROMPTS_FILE:-$TRAIN_ROOT/dpo/validation.jsonl}"
REFERENCE_MODEL="${REFERENCE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH_DIR/final_method_eval}"
PROMETHEUS_BASE_URL="${PROMETHEUS_BASE_URL:-http://localhost:8000/v1}"
PROMETHEUS_JUDGE_MODEL="${PROMETHEUS_JUDGE_MODEL:-prometheus-eval/prometheus-7b-v2.0}"
PAIRRM_JUDGE_MODEL="${PAIRRM_JUDGE_MODEL:-llm-blender/PairRM}"
SKYWORK_JUDGE_MODEL="${SKYWORK_JUDGE_MODEL:-Skywork/Skywork-Reward-Llama-3.1-8B-v0.2}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-120}"
SEED="${SEED:-42}"

DPO="$SCRATCH_DIR/dpo_final/dpo_lr1em05_b0p005_gn0p3"
KTO="$SCRATCH_DIR/kto_final/kto_lr1em05_b0p005_gn0p3"
CPO_UNARY="$SCRATCH_DIR/cpo_unary_final/cpo_lr1em05_b0p005_gn0p3_a0_token-kl"
CPO_A03="$SCRATCH_DIR/cpo_alpha_sweep/cpo_lr1em05_b0p005_gn0p3_a0p3_token-kl"
CPO_A05="$SCRATCH_DIR/cpo_alpha_sweep/cpo_lr1em05_b0p005_gn0p3_a0p5_token-kl"
CPO_A07="$SCRATCH_DIR/cpo_alpha_sweep/cpo_lr1em05_b0p005_gn0p3_a0p7_token-kl"

mkdir -p "$RUN_ROOT/teacher_forced" "$RUN_ROOT/generations" "$RUN_ROOT/judges" "$RUN_ROOT/analysis"

MODELS=(
  "DPO=$DPO"
  "KTO=$KTO"
  "CPO_UNARY=$CPO_UNARY"
  "CPO_A03=$CPO_A03"
  "CPO_A05=$CPO_A05"
  "CPO_A07=$CPO_A07"
)

for spec in "${MODELS[@]}"; do
  name="${spec%%=*}"
  path="${spec#*=}"
  if [[ ! -d "$path" ]]; then
    echo "Missing model directory for $name: $path" >&2
    exit 1
  fi
done

for spec in "${MODELS[@]}"; do
  name="${spec%%=*}"
  path="${spec#*=}"
  poetry run python scripts/evaluate/evaluate_pairwise_accuracy.py \
    --eval-file "$PROMPTS_FILE" \
    --model-name-or-path "$path" \
    --reference-model-name-or-path "$REFERENCE_MODEL" \
    --beta 0.005 \
    --batch-size "$EVAL_BATCH_SIZE" \
    --output-json "$RUN_ROOT/teacher_forced/${name}_pairwise_accuracy.json"
done

GENERATIONS_JSONL="$RUN_ROOT/generations/final_methods_generations.jsonl"
GENERATIONS_MD="$RUN_ROOT/generations/final_methods_generations.md"

poetry run python scripts/evaluate/generate_from_prompts.py \
  --prompts-file "$PROMPTS_FILE" \
  --models "${MODELS[@]}" \
  --temperature 0 \
  --batch-size "$GEN_BATCH_SIZE" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --seed "$SEED" \
  --output-jsonl "$GENERATIONS_JSONL" \
  --output-md "$GENERATIONS_MD"

poetry run python scripts/evaluate/evaluate_judge.py \
  --generations-file "$GENERATIONS_JSONL" \
  --judge-provider prometheus \
  --judge-model "$PROMETHEUS_JUDGE_MODEL" \
  --openai-base-url "$PROMETHEUS_BASE_URL" \
  --judge-timeout "$JUDGE_TIMEOUT" \
  --seed "$SEED" \
  --randomize-positions \
  --output-jsonl "$RUN_ROOT/judges/prometheus_pairwise.jsonl" \
  --summary-json "$RUN_ROOT/judges/prometheus_summary.json"

poetry run python scripts/evaluate/evaluate_judge.py \
  --generations-file "$GENERATIONS_JSONL" \
  --judge-provider pairrm \
  --judge-model "$PAIRRM_JUDGE_MODEL" \
  --judge-timeout "$JUDGE_TIMEOUT" \
  --seed "$SEED" \
  --randomize-positions \
  --output-jsonl "$RUN_ROOT/judges/pairrm_pairwise.jsonl" \
  --summary-json "$RUN_ROOT/judges/pairrm_summary.json"

poetry run python scripts/evaluate/evaluate_judge.py \
  --generations-file "$GENERATIONS_JSONL" \
  --judge-provider skywork \
  --judge-model "$SKYWORK_JUDGE_MODEL" \
  --judge-timeout "$JUDGE_TIMEOUT" \
  --seed "$SEED" \
  --randomize-positions \
  --output-jsonl "$RUN_ROOT/judges/skywork_pairwise.jsonl" \
  --summary-json "$RUN_ROOT/judges/skywork_summary.json"

poetry run python - "$GENERATIONS_JSONL" "$RUN_ROOT/analysis/response_stats.json" <<'PY'
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

generations_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])


def entropy(words: list[str]) -> float:
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
with generations_path.open(encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        by_model[str(record["model"])].append(record)

summary = {}
for model, records in sorted(by_model.items()):
    lengths = [len(str(record.get("response", "")).split()) for record in records]
    entropies = [
        entropy(str(record.get("response", "")).lower().split())
        for record in records
    ]
    summary[model] = {
        "count": len(records),
        "mean_response_words": statistics.fmean(lengths) if lengths else 0.0,
        "median_response_words": statistics.median(lengths) if lengths else 0.0,
        "response_word_variance": statistics.pvariance(lengths) if len(lengths) > 1 else 0.0,
        "min_response_words": min(lengths) if lengths else 0,
        "max_response_words": max(lengths) if lengths else 0,
        "mean_word_entropy": statistics.fmean(entropies) if entropies else 0.0,
        "word_entropy_variance": statistics.pvariance(entropies) if len(entropies) > 1 else 0.0,
    }

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "Evaluation complete: $RUN_ROOT"
