#!/usr/bin/env bash
set -euo pipefail

: "${SCRATCH_DIR:=/scratch/jordan/trial/outputs}"

FULL_GENERATIONS_JSONL="${FULL_GENERATIONS_JSONL:-$SCRATCH_DIR/final_method_eval/generations/final_methods_generations.jsonl}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH_DIR/final_method_eval_500_from_full}"
SUBSET_SIZE="${SUBSET_SIZE:-500}"
SUBSET_SEED="${SUBSET_SEED:-42}"
SEED="${SEED:-42}"
FORCE="${FORCE:-0}"

PROMETHEUS_BASE_URL="${PROMETHEUS_BASE_URL:-http://localhost:8000/v1}"
PROMETHEUS_JUDGE_MODEL="${PROMETHEUS_JUDGE_MODEL:-prometheus-eval/prometheus-7b-v2.0}"
PAIRRM_JUDGE_MODEL="${PAIRRM_JUDGE_MODEL:-llm-blender/PairRM}"
SKYWORK_JUDGE_MODEL="${SKYWORK_JUDGE_MODEL:-Skywork/Skywork-Reward-Llama-3.1-8B-v0.2}"

JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-120}"
JUDGE_MAX_LENGTH="${JUDGE_MAX_LENGTH:-4096}"

RUN_PROMETHEUS="${RUN_PROMETHEUS:-1}"
RUN_PAIRRM="${RUN_PAIRRM:-1}"
RUN_SKYWORK="${RUN_SKYWORK:-1}"

mkdir -p "$RUN_ROOT/subsets" "$RUN_ROOT/generations" "$RUN_ROOT/judges" "$RUN_ROOT/analysis"

if [[ ! -s "$FULL_GENERATIONS_JSONL" ]]; then
  echo "Missing full generations file: $FULL_GENERATIONS_JSONL" >&2
  exit 1
fi

SUBSET_GENERATIONS_JSONL="$RUN_ROOT/generations/final_methods_${SUBSET_SIZE}_seed${SUBSET_SEED}_generations.jsonl"

if [[ "$FORCE" == "1" || ! -s "$SUBSET_GENERATIONS_JSONL" ]]; then
  poetry run python - "$FULL_GENERATIONS_JSONL" "$SUBSET_GENERATIONS_JSONL" "$SUBSET_SIZE" "$SUBSET_SEED" <<'PY'
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
n = int(sys.argv[3])
seed = int(sys.argv[4])

by_prompt = defaultdict(list)
prompt_order = []
with src.open(encoding="utf-8") as handle:
    for line in handle:
        record = json.loads(line)
        prompt_id = str(record["prompt_id"])
        if prompt_id not in by_prompt:
            prompt_order.append(prompt_id)
        by_prompt[prompt_id].append(record)

if len(prompt_order) < n:
    raise SystemExit(f"Only {len(prompt_order)} prompts available, cannot sample {n}")

model_names = sorted({str(record["model"]) for rows in by_prompt.values() for record in rows})
missing = {
    prompt_id: sorted(set(model_names) - {str(record["model"]) for record in by_prompt[prompt_id]})
    for prompt_id in prompt_order
}
missing = {prompt_id: values for prompt_id, values in missing.items() if values}
if missing:
    preview = dict(list(missing.items())[:5])
    raise SystemExit(f"Full generations file has prompts with missing model responses: {preview}")

rng = random.Random(seed)
sampled = set(rng.sample(prompt_order, n))
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as handle:
    for prompt_id in prompt_order:
        if prompt_id in sampled:
            for record in by_prompt[prompt_id]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

print(
    f"Wrote {n} sampled prompts x {len(model_names)} models "
    f"from {len(prompt_order)} prompts to {dst}"
)
PY
else
  echo "Using existing sampled generations: $SUBSET_GENERATIONS_JSONL"
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
    --generations-file "$SUBSET_GENERATIONS_JSONL" \
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

write_response_stats() {
  local generations_jsonl="$1"
  local output_json="$2"

  poetry run python - "$generations_jsonl" "$output_json" <<'PY'
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
}

write_response_stats "$FULL_GENERATIONS_JSONL" "$RUN_ROOT/analysis/response_stats_full.json"
write_response_stats "$SUBSET_GENERATIONS_JSONL" "$RUN_ROOT/analysis/response_stats_${SUBSET_SIZE}_seed${SUBSET_SEED}.json"

echo "Judge evaluation complete: $RUN_ROOT"
