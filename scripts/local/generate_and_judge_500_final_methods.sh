#!/usr/bin/env bash
set -euo pipefail

: "${SCRATCH_DIR:=/scratch/jordan/trial/outputs}"

TRAIN_ROOT="${TRAIN_ROOT:-experiments/E5_cluster_ablation/embedding_4}"
FULL_PROMPTS_FILE="${FULL_PROMPTS_FILE:-$TRAIN_ROOT/dpo/validation.jsonl}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH_DIR/final_method_eval_500}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
SUBSET_SEED="${SUBSET_SEED:-42}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"

PROMETHEUS_BASE_URL="${PROMETHEUS_BASE_URL:-http://localhost:8000/v1}"
PROMETHEUS_JUDGE_MODEL="${PROMETHEUS_JUDGE_MODEL:-prometheus-eval/prometheus-7b-v2.0}"
PAIRRM_JUDGE_MODEL="${PAIRRM_JUDGE_MODEL:-llm-blender/PairRM}"
SKYWORK_JUDGE_MODEL="${SKYWORK_JUDGE_MODEL:-Skywork/Skywork-Reward-Llama-3.1-8B-v0.2}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-4}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-120}"
JUDGE_MAX_LENGTH="${JUDGE_MAX_LENGTH:-4096}"

RUN_PROMETHEUS="${RUN_PROMETHEUS:-1}"
RUN_PAIRRM="${RUN_PAIRRM:-1}"
RUN_SKYWORK="${RUN_SKYWORK:-1}"

DPO="$SCRATCH_DIR/dpo_final/dpo_lr1em05_b0p005_gn0p3"
KTO="$SCRATCH_DIR/kto_final/kto_lr1em05_b0p005_gn0p3"
CPO_UNARY="$SCRATCH_DIR/cpo_unary_final/cpo_lr1em05_b0p005_gn0p3_a0_token-kl"
CPO_A03="$SCRATCH_DIR/cpo_alpha_sweep/cpo_lr1em05_b0p005_gn0p3_a0p3_token-kl"
CPO_A05="$SCRATCH_DIR/cpo_alpha_sweep/cpo_lr1em05_b0p005_gn0p3_a0p5_token-kl"
CPO_A07="$SCRATCH_DIR/cpo_alpha_sweep/cpo_lr1em05_b0p005_gn0p3_a0p7_token-kl"

MODELS=(
  "DPO=$DPO"
  "KTO=$KTO"
  "CPO_UNARY=$CPO_UNARY"
  "CPO_A03=$CPO_A03"
  "CPO_A05=$CPO_A05"
  "CPO_A07=$CPO_A07"
)

MODEL_COUNT="${#MODELS[@]}"
EXPECTED_GENERATION_LINES="$((SUBSET_SIZE * MODEL_COUNT))"

mkdir -p "$RUN_ROOT/subsets" "$RUN_ROOT/generations" "$RUN_ROOT/judges" "$RUN_ROOT/analysis"

for spec in "${MODELS[@]}"; do
  name="${spec%%=*}"
  path="${spec#*=}"
  if [[ ! -d "$path" ]]; then
    echo "Missing model directory for $name: $path" >&2
    exit 1
  fi
done

SUBSET_FILE="$RUN_ROOT/subsets/validation_${SUBSET_SIZE}_seed${SUBSET_SEED}.jsonl"
GENERATIONS_JSONL="$RUN_ROOT/generations/final_methods_${SUBSET_SIZE}_generations.jsonl"
GENERATIONS_MD="$RUN_ROOT/generations/final_methods_${SUBSET_SIZE}_generations.md"

if [[ "$FORCE" == "1" || ! -s "$SUBSET_FILE" ]]; then
  poetry run python - "$FULL_PROMPTS_FILE" "$SUBSET_FILE" "$SUBSET_SIZE" "$SUBSET_SEED" <<'PY'
import random
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
n = int(sys.argv[3])
seed = int(sys.argv[4])

rows = src.read_text(encoding="utf-8").splitlines()
if len(rows) < n:
    raise SystemExit(f"Only {len(rows)} rows available, cannot sample {n}")

rng = random.Random(seed)
idxs = sorted(rng.sample(range(len(rows)), n))
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(rows[i] for i in idxs) + "\n", encoding="utf-8")
print(f"Wrote {n} sampled rows from {len(rows)} validation rows to {dst}")
PY
else
  echo "Using existing subset: $SUBSET_FILE"
fi

generation_lines=0
if [[ -s "$GENERATIONS_JSONL" ]]; then
  generation_lines="$(wc -l < "$GENERATIONS_JSONL" | tr -d ' ')"
fi

if [[ "$FORCE" == "1" || "$generation_lines" != "$EXPECTED_GENERATION_LINES" ]]; then
  poetry run python scripts/evaluate/generate_from_prompts.py \
    --prompts-file "$SUBSET_FILE" \
    --models "${MODELS[@]}" \
    --temperature 0 \
    --batch-size "$GEN_BATCH_SIZE" \
    --max-prompt-length "$MAX_PROMPT_LENGTH" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --seed "$SEED" \
    --output-jsonl "$GENERATIONS_JSONL" \
    --output-md "$GENERATIONS_MD"
else
  echo "Skipping generation; found $generation_lines/$EXPECTED_GENERATION_LINES lines: $GENERATIONS_JSONL"
fi

run_judge() {
  local provider="$1"
  local model="$2"
  local extra_arg="${3:-}"
  local output_jsonl="$RUN_ROOT/judges/${provider}_pairwise.jsonl"
  local summary_json="$RUN_ROOT/judges/${provider}_summary.json"

  if [[ "$FORCE" != "1" && -s "$summary_json" ]]; then
    echo "Skipping $provider judge; found $summary_json"
    return
  fi

  poetry run python scripts/evaluate/evaluate_judge.py \
    --generations-file "$GENERATIONS_JSONL" \
    --judge-provider "$provider" \
    --judge-model "$model" \
    $extra_arg \
    --judge-timeout "$JUDGE_TIMEOUT" \
    --judge-max-length "$JUDGE_MAX_LENGTH" \
    --seed "$SEED" \
    --randomize-positions \
    --output-jsonl "$output_jsonl" \
    --summary-json "$summary_json"
}

if [[ "$RUN_PROMETHEUS" == "1" ]]; then
  run_judge "prometheus" "$PROMETHEUS_JUDGE_MODEL" "--openai-base-url $PROMETHEUS_BASE_URL"
fi

if [[ "$RUN_PAIRRM" == "1" ]]; then
  run_judge "pairrm" "$PAIRRM_JUDGE_MODEL"
fi

if [[ "$RUN_SKYWORK" == "1" ]]; then
  run_judge "skywork" "$SKYWORK_JUDGE_MODEL"
fi

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

echo "Generation and judge evaluation complete: $RUN_ROOT"
