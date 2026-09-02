#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

: "${SCRATCH_DIR:=/scratch/jordan/trial/outputs}"

GENERATIONS_JSONL="${GENERATIONS_JSONL:-outputs/generations/analysis/non_truncated_prompt_matched_703_per_model.jsonl}"
RUN_ROOT="${RUN_ROOT:-$SCRATCH_DIR/generations/non_truncated_judge_eval_703}"

PAIRRM_JUDGE_MODEL="${PAIRRM_JUDGE_MODEL:-llm-blender/PairRM}"
SKYWORK_JUDGE_MODEL="${SKYWORK_JUDGE_MODEL:-Skywork/Skywork-Reward-Llama-3.1-8B-v0.2}"
PROMETHEUS_JUDGE_MODEL="${PROMETHEUS_JUDGE_MODEL:-prometheus-eval/prometheus-7b-v2.0}"
PROMETHEUS_BASE_URL="${PROMETHEUS_BASE_URL:-http://localhost:8000/v1}"

JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-120}"
JUDGE_MAX_LENGTH="${JUDGE_MAX_LENGTH:-4096}"
SEED="${SEED:-42}"
POSITION_BALANCED="${POSITION_BALANCED:-0}"
FORCE="${FORCE:-0}"

RUN_PAIRRM="${RUN_PAIRRM:-1}"
RUN_SKYWORK="${RUN_SKYWORK:-1}"
RUN_PROMETHEUS="${RUN_PROMETHEUS:-0}"

if [[ ! -s "$GENERATIONS_JSONL" ]]; then
  echo "Missing filtered generations file: $GENERATIONS_JSONL" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT/judges"

# Validate the complete 703-prompt x 6-model matrix before loading a judge.
poetry run python - "$GENERATIONS_JSONL" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

path = Path(sys.argv[1])
counts = Counter()
models_by_prompt = defaultdict(set)
seen = set()
maximum_tokens = 0

with path.open(encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
        record = json.loads(line)
        required = {"prompt_id", "model", "instruction", "response", "response_tokens"}
        missing = required - record.keys()
        if missing:
            raise SystemExit(f"Line {line_number} is missing fields: {sorted(missing)}")
        key = (str(record["prompt_id"]), str(record["model"]))
        if key in seen:
            raise SystemExit(f"Duplicate prompt/model record at line {line_number}: {key}")
        seen.add(key)
        model = str(record["model"])
        prompt_id = str(record["prompt_id"])
        response_tokens = int(record["response_tokens"])
        counts[model] += 1
        models_by_prompt[prompt_id].add(model)
        maximum_tokens = max(maximum_tokens, response_tokens)

expected_models = {"CPO_A03", "CPO_A05", "CPO_A07", "CPO_UNARY", "DPO", "KTO"}
if set(counts) != expected_models:
    raise SystemExit(f"Unexpected model set: {sorted(counts)}")
if any(counts[model] != 703 for model in expected_models):
    raise SystemExit(f"Expected 703 rows per model, found: {dict(sorted(counts.items()))}")
if len(models_by_prompt) != 703:
    raise SystemExit(f"Expected 703 prompts, found: {len(models_by_prompt)}")
if any(models != expected_models for models in models_by_prompt.values()):
    raise SystemExit("At least one prompt does not contain all six models")
if maximum_tokens >= 256:
    raise SystemExit(f"Found a response at or above the token limit: {maximum_tokens}")

print(
    f"Validated {len(seen)} rows: {len(models_by_prompt)} prompts x "
    f"{len(expected_models)} models; maximum response length={maximum_tokens} tokens"
)
PY

if [[ "$POSITION_BALANCED" == "1" ]]; then
  POSITION_ARGS=(--position-balanced)
else
  POSITION_ARGS=(--randomize-positions)
fi

run_judge() {
  local provider="$1"
  local model="$2"
  shift 2
  local output_jsonl="$RUN_ROOT/judges/${provider}_pairwise.jsonl"
  local summary_json="$RUN_ROOT/judges/${provider}_summary.json"

  if [[ "$FORCE" != "1" && -s "$summary_json" ]]; then
    echo "Skipping $provider; existing result found: $summary_json"
    return
  fi

  poetry run python scripts/evaluate/evaluate_judge.py \
    --generations-file "$GENERATIONS_JSONL" \
    --judge-provider "$provider" \
    --judge-model "$model" \
    --judge-timeout "$JUDGE_TIMEOUT" \
    --judge-max-length "$JUDGE_MAX_LENGTH" \
    --seed "$SEED" \
    "${POSITION_ARGS[@]}" \
    --output-jsonl "$output_jsonl" \
    --summary-json "$summary_json" \
    "$@"
}

if [[ "$RUN_PAIRRM" == "1" ]]; then
  run_judge "pairrm" "$PAIRRM_JUDGE_MODEL"
fi

if [[ "$RUN_SKYWORK" == "1" ]]; then
  run_judge "skywork" "$SKYWORK_JUDGE_MODEL"
fi

if [[ "$RUN_PROMETHEUS" == "1" ]]; then
  run_judge \
    "prometheus" \
    "$PROMETHEUS_JUDGE_MODEL" \
    --openai-base-url "$PROMETHEUS_BASE_URL"
fi

echo "Non-truncated judge evaluation complete: $RUN_ROOT"
