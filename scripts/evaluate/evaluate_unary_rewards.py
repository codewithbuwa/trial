from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cpo_trl.data.datasets import RowKind, load_jsonl, validate_rows
from cpo_trl.evaluation.teacher_forced import collate_unary_batch, encode_unary, sequence_logp_sums_and_counts
from cpo_trl.models.peft import load_causal_lm_for_training


def infer_row_kind(rows: list[dict[str, Any]]) -> RowKind:
    if rows and "cluster_id" in rows[0]:
        return "cpo"
    return "kto"


def mean_or_none(total: float, count: int) -> float | None:
    return total / count if count else None


def summarize_records(records: list[dict[str, Any]], *, include_clusters: bool = True) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "total": len(records),
        "desirable_count": 0,
        "undesirable_count": 0,
        "desirable_reward_sum": 0.0,
        "undesirable_reward_sum": 0.0,
        "normalized_desirable_reward_sum": 0.0,
        "normalized_undesirable_reward_sum": 0.0,
        "desirable_sampled_kl_sum": 0.0,
        "undesirable_sampled_kl_sum": 0.0,
        "normalized_desirable_sampled_kl_sum": 0.0,
        "normalized_undesirable_sampled_kl_sum": 0.0,
        "desirable_length_sum": 0,
        "undesirable_length_sum": 0,
    }
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        label = bool(record["label"])
        prefix = "desirable" if label else "undesirable"
        stats[f"{prefix}_count"] += 1
        stats[f"{prefix}_reward_sum"] += float(record["reward"])
        stats[f"normalized_{prefix}_reward_sum"] += float(record["normalized_reward"])
        stats[f"{prefix}_sampled_kl_sum"] += float(record["sampled_kl"])
        stats[f"normalized_{prefix}_sampled_kl_sum"] += float(record["normalized_sampled_kl"])
        stats[f"{prefix}_length_sum"] += int(record["completion_length"])
        if include_clusters:
            clusters[str(record.get("cluster_id", "unknown"))].append(record)

    desirable_count = int(stats["desirable_count"])
    undesirable_count = int(stats["undesirable_count"])
    mean_desirable_reward = mean_or_none(stats["desirable_reward_sum"], desirable_count)
    mean_undesirable_reward = mean_or_none(stats["undesirable_reward_sum"], undesirable_count)
    mean_normalized_desirable_reward = mean_or_none(
        stats["normalized_desirable_reward_sum"], desirable_count
    )
    mean_normalized_undesirable_reward = mean_or_none(
        stats["normalized_undesirable_reward_sum"], undesirable_count
    )
    result: dict[str, Any] = {
        "total": len(records),
        "desirable_count": desirable_count,
        "undesirable_count": undesirable_count,
        "mean_desirable_reward": mean_desirable_reward,
        "mean_undesirable_reward": mean_undesirable_reward,
        "reward_separation": (
            mean_desirable_reward - mean_undesirable_reward
            if mean_desirable_reward is not None and mean_undesirable_reward is not None
            else None
        ),
        "mean_normalized_desirable_reward": mean_normalized_desirable_reward,
        "mean_normalized_undesirable_reward": mean_normalized_undesirable_reward,
        "normalized_reward_separation": (
            mean_normalized_desirable_reward - mean_normalized_undesirable_reward
            if (
                mean_normalized_desirable_reward is not None
                and mean_normalized_undesirable_reward is not None
            )
            else None
        ),
        "mean_desirable_sampled_kl": mean_or_none(
            stats["desirable_sampled_kl_sum"], desirable_count
        ),
        "mean_undesirable_sampled_kl": mean_or_none(
            stats["undesirable_sampled_kl_sum"], undesirable_count
        ),
        "mean_normalized_desirable_sampled_kl": mean_or_none(
            stats["normalized_desirable_sampled_kl_sum"], desirable_count
        ),
        "mean_normalized_undesirable_sampled_kl": mean_or_none(
            stats["normalized_undesirable_sampled_kl_sum"], undesirable_count
        ),
        "mean_desirable_length": mean_or_none(stats["desirable_length_sum"], desirable_count),
        "mean_undesirable_length": mean_or_none(
            stats["undesirable_length_sum"], undesirable_count
        ),
    }
    if include_clusters:
        result["clusters"] = {
            cluster_id: summarize_records(cluster_records, include_clusters=False)
            for cluster_id, cluster_records in sorted(clusters.items())
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate KTO/CPO unary desirable-vs-undesirable reward separation."
    )
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--row-kind", choices=("kto", "cpo"))
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--reference-model-name-or-path", required=True)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-records-jsonl", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded_rows = load_jsonl(args.eval_file)
    row_kind = args.row_kind or infer_row_kind(loaded_rows)
    rows = validate_rows(loaded_rows, row_kind)
    if args.limit:
        rows = rows[: args.limit]
    rows = [{**row, "row_index": index} for index, row in enumerate(rows)]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_causal_lm_for_training(args.model_name_or_path, use_lora=False)
    ref_model = (
        model
        if args.reference_model_name_or_path == args.model_name_or_path
        else load_causal_lm_for_training(args.reference_model_name_or_path, use_lora=False)
    )
    model.eval()
    ref_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ref_model.to(device)

    encoded = [encode_unary(tokenizer, row, args.max_seq_length) for row in rows]
    dataloader = DataLoader(
        encoded,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_unary_batch(tokenizer, batch),
    )
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            policy_logps, counts = sequence_logp_sums_and_counts(
                model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["response_start"],
            )
            reference_logps, _reference_counts = sequence_logp_sums_and_counts(
                ref_model,
                batch["input_ids"],
                batch["attention_mask"],
                batch["response_start"],
            )
            sampled_kl = policy_logps - reference_logps
            normalized_sampled_kl = (policy_logps / counts) - (reference_logps / counts)
            rewards = args.beta * sampled_kl
            normalized_rewards = args.beta * normalized_sampled_kl
            for (
                row_index,
                prompt_id,
                cluster_id,
                label,
                sampled_kl_value,
                normalized_sampled_kl_value,
                reward,
                normalized_reward,
                completion_length,
            ) in zip(
                batch["row_indices"],
                batch["prompt_ids"],
                batch["cluster_ids"],
                batch["labels"].detach().cpu().tolist(),
                sampled_kl.detach().cpu().tolist(),
                normalized_sampled_kl.detach().cpu().tolist(),
                rewards.detach().cpu().tolist(),
                normalized_rewards.detach().cpu().tolist(),
                counts.detach().cpu().tolist(),
                strict=True,
            ):
                records.append(
                    {
                        "row_index": row_index,
                        "prompt_id": prompt_id,
                        "cluster_id": cluster_id,
                        "label": bool(label),
                        "sampled_kl": float(sampled_kl_value),
                        "normalized_sampled_kl": float(normalized_sampled_kl_value),
                        "reward": float(reward),
                        "normalized_reward": float(normalized_reward),
                        "completion_length": int(completion_length),
                    }
                )

    result = {
        "model": args.model_name_or_path,
        "reference_model": args.reference_model_name_or_path,
        "beta": args.beta,
        "eval_file": str(args.eval_file),
        "row_kind": row_kind,
        **summarize_records(records),
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    records_path = args.output_records_jsonl
    if records_path is None and args.output_json:
        records_path = args.output_json.with_name(f"{args.output_json.stem}_records.jsonl")
    if records_path:
        records_path.parent.mkdir(parents=True, exist_ok=True)
        with records_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
