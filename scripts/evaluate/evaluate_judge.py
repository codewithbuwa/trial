from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from collections import defaultdict
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
) -> list[str]:
    import torch
    from transformers import AutoTokenizer

    from cpo_trl.data.formatting import format_prompt
    from cpo_trl.models.peft import load_causal_lm_for_training

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
    for record in records:
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
        "total_comparisons": len(records),
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
        default="heuristic",
    )
    parser.add_argument("--judge-model", default=os.environ.get("OPENAI_JUDGE_MODEL"))
    parser.add_argument("--openai-base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    parser.add_argument("--judge-max-length", type=int, default=4096)
    parser.add_argument(
        "--position-balanced",
        action="store_true",
        help="Judge every model pair in both A/B orders to cancel response-position bias.",
    )
    parser.add_argument(
        "--randomize-positions",
        action="store_true",
        help=(
            "Judge each model pair once with seeded random A/B presentation per prompt. "
            "This is mutually exclusive with --position-balanced."
        ),
    )
    return parser.parse_args()


def validate_judge_settings(args: argparse.Namespace) -> str:
    if getattr(args, "position_balanced", False) and getattr(args, "randomize_positions", False):
        raise ValueError("--randomize-positions cannot be used with --position-balanced")
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


def main() -> None:
    args = parse_args()
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
            )
            for name, path in model_specs
        }
    api_key = validate_judge_settings(args)
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
    records: list[dict[str, Any]] = []
    randomize_positions = args.randomize_positions or not args.position_balanced
    for comparison in build_comparisons(
        rows,
        generations,
        seed=args.seed,
        position_balanced=args.position_balanced,
        randomize_positions=randomize_positions,
    ):
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
            judgment = skywork_judge(
                reward_model=skywork_reward_model,
                instruction=comparison["instruction"],
                input_text=comparison["input"],
                response_a=comparison["response_a"],
                response_b=comparison["response_b"],
            )
        else:
            judgment = heuristic_judge(comparison["response_a"], comparison["response_b"])
        winner = judgment["winner"]
        winner_model = "tie"
        if winner == "A":
            winner_model = comparison["model_a"]
        elif winner == "B":
            winner_model = comparison["model_b"]
        records.append({**comparison, **judgment, "winner_model": winner_model})
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "eval_file": str(args.eval_file) if not args.generations_file else None,
        "generations_file": str(args.generations_file) if args.generations_file else None,
        "judge_provider": args.judge_provider,
        "judge_model": args.judge_model,
        "judge_base_url": args.openai_base_url
        if args.judge_provider in {"openai", "prometheus"}
        else None,
        "position_balanced": args.position_balanced,
        "position_randomized": randomize_positions,
        "position_strategy": "position_balanced" if args.position_balanced else "randomized",
        "position_seed": args.seed,
        "generation_lengths": generation_length_summary(generations),
        **summarize_judgments(records),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
