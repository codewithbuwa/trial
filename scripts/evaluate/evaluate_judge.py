from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from cpo_trl.data.datasets import load_jsonl, validate_rows
from cpo_trl.evaluation.judges import (
    format_reward_model_input,
    heuristic_judge,
    load_pairrm_ranker,
    load_skywork_reward_model,
    openai_chat_judge,
    pairrm_judge,
    parse_judge_json,
    parse_pairwise_winner_text,
    prometheus_judge,
    prometheus_prompt,
    skywork_judge,
    skywork_reward_score,
)

JUDGE_EVALUATOR_VERSION = 2
# Providers that judge over the network (one HTTP request per comparison) and so
# benefit from concurrent in-flight requests; local model judges do not.
NETWORK_JUDGE_PROVIDERS = {"openai", "prometheus"}


def response_length_stats(responses: list[str]) -> dict[str, float | int | None]:
    counts = sorted(len(response.split()) for response in responses)
    if not counts:
        return {
            "count": 0,
            "mean_words": None,
            "median_words": None,
            "p95_words": None,
            "min_words": None,
            "max_words": None,
        }
    mid = len(counts) // 2
    median = counts[mid] if len(counts) % 2 else (counts[mid - 1] + counts[mid]) / 2
    p95_index = min(len(counts) - 1, math.ceil(0.95 * (len(counts) - 1)))
    return {
        "count": len(counts),
        "mean_words": sum(counts) / len(counts),
        "median_words": median,
        "p95_words": counts[p95_index],
        "min_words": counts[0],
        "max_words": counts[-1],
    }


def parse_model_specs(values: list[str]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(value).name or value
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"invalid model spec: {value!r}")
        if name in seen:
            raise ValueError(f"duplicate model name: {name}")
        seen.add(name)
        specs.append((name, path))
    if len(specs) < 2:
        raise ValueError("judge evaluation requires at least two models")
    return specs


def generate_model_outputs(
    *,
    rows: list[dict[str, Any]],
    model_name: str,
    model_path: str,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    seed: int,
) -> list[str]:
    import torch
    from transformers import AutoTokenizer, set_seed

    from cpo_trl.data.formatting import format_prompt
    from cpo_trl.models.peft import load_causal_lm_for_training

    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = load_causal_lm_for_training(model_path, use_lora=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    prompts = [format_prompt(tokenizer, row) for row in rows]
    generations: list[str] = []
    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            prompt_batch = prompts[start : start + batch_size]
            encoded = tokenizer(
                prompt_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_length,
            ).to(device)
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            generated = model.generate(**encoded, **generation_kwargs)
            continuation = generated[:, encoded["input_ids"].shape[1] :]
            generations.extend(
                tokenizer.batch_decode(continuation, skip_special_tokens=True)
            )
    print(f"Generated {len(generations)} outputs for {model_name}")
    return generations


def build_comparisons(
    rows: list[dict[str, Any]],
    generations: dict[str, list[str]],
    *,
    seed: int,
    position_balanced: bool,
    randomize_positions: bool = True,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    comparisons: list[dict[str, Any]] = []
    model_names = list(generations)
    position_strategy = "position_balanced" if position_balanced else (
        "randomized" if randomize_positions else "fixed"
    )
    for row_index, row in enumerate(rows):
        for left, right in itertools.combinations(model_names, 2):
            swap = randomize_positions and rng.random() < 0.5
            first_a, first_b = (right, left) if swap else (left, right)
            ordered_pairs = [(first_a, first_b)]
            if position_balanced:
                ordered_pairs.append((first_b, first_a))
            for judge_order, (model_a, model_b) in enumerate(ordered_pairs):
                comparisons.append(
                    {
                        "row_index": row_index,
                        "prompt_id": row.get("prompt_id", str(row_index)),
                        "cluster_id": row.get("cluster_id", "unknown"),
                        "instruction": row["instruction"],
                        "input": row.get("input", ""),
                        "model_a": model_a,
                        "model_b": model_b,
                        "response_a": generations[model_a][row_index],
                        "response_b": generations[model_b][row_index],
                        "response_a_words": len(generations[model_a][row_index].split()),
                        "response_b_words": len(generations[model_b][row_index].split()),
                        "position_balanced": position_balanced,
                        "position_randomized": randomize_positions,
                        "position_strategy": position_strategy,
                        "position_seed": seed,
                        "position_swapped": model_a != left,
                        "judge_order": judge_order,
                    }
                )
    return comparisons


def summarize_judgments(records: list[dict[str, Any]]) -> dict[str, Any]:
    wins: dict[str, float] = defaultdict(float)
    totals: dict[str, float] = defaultdict(float)
    pairwise: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    clusters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    status_counts = Counter(str(record.get("status", "ok")) for record in records)
    scored_records = [record for record in records if record.get("status", "ok") == "ok"]
    for record in scored_records:
        model_a = record["model_a"]
        model_b = record["model_b"]
        cluster_id = record.get("cluster_id", "unknown")
        winner_model = record["winner_model"]
        pair_key = "__vs__".join(sorted((model_a, model_b)))
        totals[model_a] += 1
        totals[model_b] += 1
        pairwise[pair_key]["total"] += 1
        if winner_model == "tie":
            wins[model_a] += 0.5
            wins[model_b] += 0.5
            pairwise[pair_key]["ties"] += 1
            clusters[f"{cluster_id}:{model_a}"]["wins"] += 0.5
            clusters[f"{cluster_id}:{model_b}"]["wins"] += 0.5
        else:
            wins[winner_model] += 1
            pairwise[pair_key][winner_model] += 1
            clusters[f"{cluster_id}:{winner_model}"]["wins"] += 1
        clusters[f"{cluster_id}:{model_a}"]["total"] += 1
        clusters[f"{cluster_id}:{model_b}"]["total"] += 1
    return {
        "requested_comparisons": len(records),
        "total_comparisons": len(scored_records),
        "failed_comparisons": len(records) - len(scored_records),
        "judgment_status_counts": dict(sorted(status_counts.items())),
        "models": {
            model: {
                "comparisons": int(total),
                "judge_score": wins[model] / total if total else 0.0,
            }
            for model, total in sorted(totals.items())
        },
        "clusters": {
            key: {
                "comparisons": int(value["total"]),
                "judge_score": value["wins"] / value["total"] if value["total"] else 0.0,
            }
            for key, value in sorted(clusters.items())
        },
        "pairwise": {
            key: {metric: int(count) for metric, count in sorted(value.items())}
            for key, value in sorted(pairwise.items())
        },
    }


def generation_length_summary(generations: dict[str, list[str]]) -> dict[str, dict[str, float | int | None]]:
    return {
        model: response_length_stats(responses)
        for model, responses in sorted(generations.items())
    }


def load_generation_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    records = load_jsonl(path)
    by_prompt: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    prompt_order: list[str] = []
    for record in records:
        prompt_id = str(record["prompt_id"])
        model = str(record["model"])
        if prompt_id not in by_prompt:
            prompt_order.append(prompt_id)
        if model in by_prompt[prompt_id]:
            raise ValueError(f"duplicate generation for prompt_id={prompt_id!r}, model={model!r}")
        by_prompt[prompt_id][model] = record
    model_names = sorted({str(record["model"]) for record in records})
    if len(model_names) < 2:
        raise ValueError("judge evaluation requires generations for at least two models")
    rows: list[dict[str, Any]] = []
    generations = {model: [] for model in model_names}
    for prompt_id in prompt_order:
        prompt_records = by_prompt[prompt_id]
        missing = [model for model in model_names if model not in prompt_records]
        if missing:
            raise ValueError(f"prompt_id={prompt_id!r} is missing generation(s): {missing}")
        first = prompt_records[model_names[0]]
        expected_metadata = (
            first["instruction"],
            first.get("input", ""),
            first.get("cluster_id", "unknown"),
        )
        for model in model_names[1:]:
            record = prompt_records[model]
            metadata = (
                record["instruction"],
                record.get("input", ""),
                record.get("cluster_id", "unknown"),
            )
            if metadata != expected_metadata:
                raise ValueError(
                    f"inconsistent prompt metadata for prompt_id={prompt_id!r}: "
                    f"{model_names[0]!r} and {model!r} differ"
                )
        rows.append(
            {
                "prompt_id": prompt_id,
                "cluster_id": first.get("cluster_id", "unknown"),
                "instruction": first["instruction"],
                "input": first.get("input", ""),
            }
        )
        for model in model_names:
            generations[model].append(str(prompt_records[model].get("response", "")))
    return rows, generations


def comparison_id(
    comparison: dict[str, Any],
    *,
    judge_provider: str,
    judge_model: str | None,
    judge_config: dict[str, Any] | None = None,
) -> str:
    identity = {
        "prompt_id": comparison["prompt_id"],
        "cluster_id": comparison.get("cluster_id", "unknown"),
        "instruction": comparison["instruction"],
        "input": comparison.get("input", ""),
        "model_a": comparison["model_a"],
        "model_b": comparison["model_b"],
        "response_a": comparison["response_a"],
        "response_b": comparison["response_b"],
        "judge_order": comparison["judge_order"],
        "position_seed": comparison["position_seed"],
        "judge_provider": judge_provider,
        "judge_model": judge_model,
        "judge_config": judge_config or {},
        "evaluator_version": JUDGE_EVALUATOR_VERSION,
    }
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_existing_judgments(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    contents = path.read_bytes()
    lines = contents.splitlines(keepends=True)
    byte_offset = 0
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            byte_offset += len(raw_line)
            continue
        try:
            record = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if line_number == len(lines):
                with path.open("r+b") as handle:
                    handle.truncate(byte_offset)
                print(f"Warning: removed incomplete final record from resume file {path}")
                break
            raise ValueError(
                f"invalid JSON in resume file {path} at line {line_number}"
            ) from exc
        record_id = record.get("comparison_id")
        if not record_id:
            raise ValueError(
                f"resume file {path} contains legacy records without comparison_id; "
                "use --no-resume or choose a new output path"
            )
        records[str(record_id)] = record
        byte_offset += len(raw_line)
    if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return records


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def judge_configuration(args: argparse.Namespace) -> dict[str, Any]:
    configuration: dict[str, Any] = {}
    if args.judge_provider in {"openai", "prometheus"}:
        configuration["base_url"] = args.openai_base_url
    if args.judge_provider == "skywork":
        configuration.update(
            {
                "max_length": args.judge_max_length,
                "tie_threshold": args.skywork_tie_threshold,
            }
        )
    return configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge saved generations from multiple models.")
    parser.add_argument("--generations-file", type=Path, help="JSONL produced by generate_from_prompts.py")
    parser.add_argument("--eval-file", type=Path, default=Path("data/processed/dpo/validation.jsonl"))
    parser.add_argument("--models", nargs="+", help="Legacy NAME=PATH entries for generating then judging.")
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/judge/pairwise.jsonl"))
    parser.add_argument("--summary-json", type=Path, default=Path("outputs/judge/summary.json"))
    parser.add_argument("--max-prompts", type=int, default=None, help="Legacy model-generation mode only.")
    parser.add_argument("--max-prompt-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--judge-provider",
        choices=("heuristic", "openai", "prometheus", "pairrm", "skywork"),
        required=True,
    )
    parser.add_argument("--judge-model", default=os.environ.get("OPENAI_JUDGE_MODEL"))
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    parser.add_argument("--judge-max-length", type=int, default=4096)
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--judge-retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--judge-parse-retries", type=int, default=1)
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=1,
        help=(
            "Number of concurrent in-flight requests for network judges "
            "(openai/prometheus). Ignored for local judges (heuristic/pairrm/skywork)."
        ),
    )
    parser.add_argument("--skywork-tie-threshold", type=float, default=0.0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume successful comparisons from output JSONL (default: enabled).",
    )
    position_group = parser.add_mutually_exclusive_group()
    position_group.add_argument(
        "--position-balanced",
        action="store_true",
        help="Judge every model pair in both A/B orders to cancel response-position bias.",
    )
    position_group.add_argument(
        "--randomize-positions",
        action="store_true",
        help=(
            "Judge each pair once with seeded random A/B presentation (the default)."
        ),
    )
    position_group.add_argument(
        "--fixed-positions",
        action="store_true",
        help="Judge each model pair once in deterministic model insertion order.",
    )
    return parser.parse_args()


def validate_judge_settings(args: argparse.Namespace) -> str:
    position_flags = (
        bool(getattr(args, "position_balanced", False)),
        bool(getattr(args, "randomize_positions", False)),
        bool(getattr(args, "fixed_positions", False)),
    )
    if sum(position_flags) > 1:
        raise ValueError("position selection flags cannot be used together")
    if getattr(args, "judge_max_retries", 0) < 0:
        raise ValueError("--judge-max-retries must be non-negative")
    if getattr(args, "judge_retry_base_seconds", 0.0) < 0:
        raise ValueError("--judge-retry-base-seconds must be non-negative")
    if getattr(args, "judge_parse_retries", 0) < 0:
        raise ValueError("--judge-parse-retries must be non-negative")
    if getattr(args, "checkpoint_every", 1) < 1:
        raise ValueError("--checkpoint-every must be at least 1")
    if getattr(args, "judge_concurrency", 1) < 1:
        raise ValueError("--judge-concurrency must be at least 1")
    if getattr(args, "skywork_tie_threshold", 0.0) < 0:
        raise ValueError("--skywork-tie-threshold must be non-negative")
    if args.judge_provider == "skywork" and getattr(args, "position_balanced", False):
        print(
            "Warning: --position-balanced is redundant for independent Skywork scores; "
            "evaluating each pair once with randomized positions."
        )
        args.position_balanced = False
        args.randomize_positions = True
    if args.judge_provider == "openai":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"{args.api_key_env} must be set for --judge-provider openai")
        if not args.judge_model:
            raise ValueError("--judge-model is required for --judge-provider openai")
        return api_key
    if args.judge_provider == "prometheus" and not args.judge_model:
        raise ValueError("--judge-model is required for --judge-provider prometheus")
    if args.judge_provider == "pairrm" and not args.judge_model:
        args.judge_model = "llm-blender/PairRM"
    if args.judge_provider == "skywork" and not args.judge_model:
        args.judge_model = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"
    return ""


def evaluate_comparison(
    comparison: dict[str, Any],
    *,
    args: argparse.Namespace,
    api_key: str,
    pairrm_ranker: Any,
    skywork_reward_model: dict[str, Any] | None,
) -> dict[str, Any]:
    parse_attempts = 0
    max_parse_attempts = 1 + (
        args.judge_parse_retries
        if args.judge_provider in {"openai", "prometheus"}
        else 0
    )
    while parse_attempts < max_parse_attempts:
        parse_attempts += 1
        if args.judge_provider == "openai":
            judgment = openai_chat_judge(
                instruction=comparison["instruction"],
                input_text=comparison["input"],
                response_a=comparison["response_a"],
                response_b=comparison["response_b"],
                model=args.judge_model,
                base_url=args.openai_base_url,
                api_key=api_key,
                timeout=args.judge_timeout,
                max_retries=args.judge_max_retries,
                retry_base_seconds=args.judge_retry_base_seconds,
            )
        elif args.judge_provider == "prometheus":
            judgment = prometheus_judge(
                instruction=comparison["instruction"],
                input_text=comparison["input"],
                response_a=comparison["response_a"],
                response_b=comparison["response_b"],
                model=args.judge_model,
                base_url=args.openai_base_url,
                timeout=args.judge_timeout,
                max_retries=args.judge_max_retries,
                retry_base_seconds=args.judge_retry_base_seconds,
            )
        elif args.judge_provider == "pairrm":
            judgment = pairrm_judge(
                ranker=pairrm_ranker,
                instruction=comparison["instruction"],
                input_text=comparison["input"],
                response_a=comparison["response_a"],
                response_b=comparison["response_b"],
            )
        elif args.judge_provider == "skywork":
            if skywork_reward_model is None:
                raise RuntimeError("Skywork reward model was not loaded")
            judgment = skywork_judge(
                reward_model=skywork_reward_model,
                instruction=comparison["instruction"],
                input_text=comparison["input"],
                response_a=comparison["response_a"],
                response_b=comparison["response_b"],
                tie_threshold=args.skywork_tie_threshold,
            )
        else:
            judgment = heuristic_judge(
                comparison["response_a"],
                comparison["response_b"],
            )
        if judgment.get("status", "ok") != "parse_error":
            break
    judgment["parse_attempts"] = parse_attempts
    return judgment


def main() -> None:
    args = parse_args()
    api_key = validate_judge_settings(args)
    if args.generations_file:
        rows, generations = load_generation_records(args.generations_file)
    else:
        if not args.models:
            raise ValueError("--generations-file is required unless --models is provided")
        model_specs = parse_model_specs(args.models)
        rows = validate_rows(load_jsonl(args.eval_file), "dpo")
        if args.max_prompts:
            rows = rows[: args.max_prompts]
        generations = {
            name: generate_model_outputs(
                rows=rows,
                model_name=name,
                model_path=path,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            for name, path in model_specs
        }
    pairrm_ranker = (
        load_pairrm_ranker(args.judge_model)
        if args.judge_provider == "pairrm"
        else None
    )
    skywork_reward_model = (
        load_skywork_reward_model(args.judge_model, args.judge_max_length)
        if args.judge_provider == "skywork"
        else None
    )
    randomize_positions = not args.position_balanced and not args.fixed_positions
    comparisons = build_comparisons(
        rows,
        generations,
        seed=args.seed,
        position_balanced=args.position_balanced,
        randomize_positions=randomize_positions,
    )
    active_judge_configuration = judge_configuration(args)
    planned = {
        comparison_id(
            comparison,
            judge_provider=args.judge_provider,
            judge_model=args.judge_model,
            judge_config=active_judge_configuration,
        ): comparison
        for comparison in comparisons
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    existing = load_existing_judgments(args.output_jsonl) if args.resume else {}
    unmatched_existing = set(existing) - set(planned)
    if unmatched_existing:
        raise ValueError(
            f"resume file {args.output_jsonl} contains {len(unmatched_existing)} "
            "comparison(s) outside the current evaluation; use --no-resume or "
            "choose a new output path"
        )
    records_by_id = {
        record_id: record
        for record_id, record in existing.items()
        if record_id in planned
    }
    resumed_comparisons = sum(
        record.get("status", "ok") == "ok" for record in records_by_id.values()
    )
    if args.resume and records_by_id:
        # Compact the resume file to one record per comparison, dropping stale
        # duplicates and the non-ok records that the loop below re-attempts (and
        # would otherwise re-append), so repeated resumes cannot grow the file.
        retained = [
            records_by_id[record_id]
            for record_id in planned
            if record_id in records_by_id
            and records_by_id[record_id].get("status", "ok") == "ok"
        ]
        temporary = args.output_jsonl.with_name(
            f".{args.output_jsonl.name}.{os.getpid()}.compact.tmp"
        )
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for record in retained:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, args.output_jsonl)
        finally:
            if temporary.exists():
                temporary.unlink()
    output_mode = "a" if args.resume else "w"
    newly_attempted = 0

    def build_record(record_id: str, comparison: dict[str, Any]) -> dict[str, Any]:
        """Judge one comparison and assemble its output record (thread-safe)."""
        try:
            judgment = evaluate_comparison(
                comparison,
                args=args,
                api_key=api_key,
                pairrm_ranker=pairrm_ranker,
                skywork_reward_model=skywork_reward_model,
            )
        except Exception as exc:
            judgment = {
                "winner": None,
                "winner_model": None,
                "reason": str(exc),
                "status": "error",
                "error_type": type(exc).__name__,
            }
        winner = judgment.get("winner")
        winner_model = None
        if judgment.get("status", "ok") == "ok":
            if winner == "A":
                winner_model = comparison["model_a"]
            elif winner == "B":
                winner_model = comparison["model_b"]
            elif winner == "tie":
                winner_model = "tie"
            else:
                judgment = {
                    **judgment,
                    "status": "parse_error",
                    "reason": f"evaluator returned invalid winner: {winner!r}",
                }
        return {
            **comparison,
            **judgment,
            "winner_model": winner_model,
            "comparison_id": record_id,
            "judge_provider": args.judge_provider,
            "judge_model": args.judge_model,
            "judge_configuration": active_judge_configuration,
            "evaluator_version": JUDGE_EVALUATOR_VERSION,
        }

    pending = [
        (record_id, comparison)
        for record_id, comparison in planned.items()
        if not (
            (prior := records_by_id.get(record_id))
            and prior.get("status", "ok") == "ok"
        )
    ]
    # Only network judges gain from concurrency; local model judges share one GPU
    # and are left serial.
    concurrency = (
        args.judge_concurrency
        if args.judge_provider in NETWORK_JUDGE_PROVIDERS
        else 1
    )
    with args.output_jsonl.open(output_mode, encoding="utf-8") as handle:

        def persist(record: dict[str, Any]) -> None:
            nonlocal newly_attempted
            records_by_id[record["comparison_id"]] = record
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            newly_attempted += 1
            if newly_attempted % args.checkpoint_every == 0:
                os.fsync(handle.fileno())

        if concurrency > 1 and len(pending) > 1:
            # Judge concurrently; results are persisted on the main thread as each
            # future completes, so writes/checkpoints stay single-threaded and
            # resume-safe even though judgments finish out of order.
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(build_record, record_id, comparison)
                    for record_id, comparison in pending
                ]
                for future in as_completed(futures):
                    persist(future.result())
        else:
            for record_id, comparison in pending:
                persist(build_record(record_id, comparison))
        if newly_attempted % args.checkpoint_every:
            os.fsync(handle.fileno())
    records = [records_by_id[record_id] for record_id in planned]
    if args.position_balanced:
        position_strategy = "position_balanced"
    elif randomize_positions:
        position_strategy = "randomized"
    else:
        position_strategy = "fixed"
    summary = {
        "eval_file": str(args.eval_file) if not args.generations_file else None,
        "generations_file": str(args.generations_file) if args.generations_file else None,
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "judge_configuration": active_judge_configuration,
        "judge_base_url": args.openai_base_url
        if args.judge_provider in {"openai", "prometheus"}
        else None,
        "position_balanced": args.position_balanced,
        "position_randomized": randomize_positions,
        "position_strategy": position_strategy,
        "position_seed": args.seed,
        "judge_concurrency": (
            args.judge_concurrency
            if args.judge_provider in NETWORK_JUDGE_PROVIDERS
            else 1
        ),
        "resume_enabled": args.resume,
        "resumed_comparisons": resumed_comparisons,
        "newly_attempted_comparisons": newly_attempted,
        "generation_lengths": generation_length_summary(generations),
        **summarize_judgments(records),
    }
    write_json_atomic(args.summary_json, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
