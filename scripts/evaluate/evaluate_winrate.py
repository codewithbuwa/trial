from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from cpo_trl.data.datasets import load_jsonl, validate_rows
from cpo_trl.evaluation.teacher_forced import (
    collate_pair_batch,
    encode_pair,
    pair_reward_margins,
    sequence_logp_sums_and_counts,
)
from cpo_trl.models.peft import load_causal_lm_for_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate preference winrate on chosen/rejected pairs.")
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--reference-model-name-or-path")
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-margins-jsonl", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.eval_file)
    if not rows or "chosen" not in rows[0] or "rejected" not in rows[0]:
        raise ValueError(
            "winrate evaluation requires chosen/rejected pair rows. "
            "Use data/processed/dpo/validation.jsonl after running prepare_ultrafeedback.py."
        )
    rows = validate_rows(rows, "dpo")
    if args.limit:
        rows = rows[: args.limit]
    rows = [{**row, "row_index": index} for index, row in enumerate(rows)]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_causal_lm_for_training(args.model_name_or_path, use_lora=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ref_model = None
    if args.reference_model_name_or_path:
        if args.reference_model_name_or_path == args.model_name_or_path:
            ref_model = model
        else:
            ref_model = load_causal_lm_for_training(args.reference_model_name_or_path, use_lora=False)
            ref_model.eval()
            ref_model.to(device)

    encoded = [encode_pair(tokenizer, row, args.max_seq_length) for row in rows]
    dataloader = DataLoader(
        encoded,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_pair_batch(tokenizer, batch),
    )

    total = 0
    wins = 0
    ties = 0
    margin_sum = 0.0
    normalized_wins = 0
    normalized_ties = 0
    normalized_margin_sum = 0.0
    reward_wins = 0
    reward_ties = 0
    reward_margin_sum = 0.0
    normalized_reward_wins = 0
    normalized_reward_ties = 0
    normalized_reward_margin_sum = 0.0
    chosen_sampled_kl_sum = 0.0
    rejected_sampled_kl_sum = 0.0
    normalized_chosen_sampled_kl_sum = 0.0
    normalized_rejected_sampled_kl_sum = 0.0
    chosen_length_sum = 0
    rejected_length_sum = 0
    margin_records: list[dict[str, object]] = []
    cluster_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "total": 0,
            "wins": 0,
            "ties": 0,
            "margin_sum": 0.0,
            "normalized_wins": 0,
            "normalized_ties": 0,
            "normalized_margin_sum": 0.0,
            "reward_wins": 0,
            "reward_ties": 0,
            "reward_margin_sum": 0.0,
            "normalized_reward_wins": 0,
            "normalized_reward_ties": 0,
            "normalized_reward_margin_sum": 0.0,
            "chosen_sampled_kl_sum": 0.0,
            "rejected_sampled_kl_sum": 0.0,
            "normalized_chosen_sampled_kl_sum": 0.0,
            "normalized_rejected_sampled_kl_sum": 0.0,
            "chosen_length_sum": 0,
            "rejected_length_sum": 0,
        }
    )
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            chosen, chosen_counts = sequence_logp_sums_and_counts(
                model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_response_start"],
            )
            rejected, rejected_counts = sequence_logp_sums_and_counts(
                model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_response_start"],
            )
            margins = chosen - rejected
            normalized_chosen = chosen / chosen_counts
            normalized_rejected = rejected / rejected_counts
            normalized_margins = normalized_chosen - normalized_rejected
            batch_wins = margins > 0
            batch_ties = margins == 0
            batch_normalized_wins = normalized_margins > 0
            batch_normalized_ties = normalized_margins == 0
            reward_margins = None
            normalized_reward_margins = None
            chosen_rewards = None
            rejected_rewards = None
            normalized_chosen_rewards = None
            normalized_rejected_rewards = None
            chosen_sampled_kl = None
            rejected_sampled_kl = None
            normalized_chosen_sampled_kl = None
            normalized_rejected_sampled_kl = None
            batch_reward_wins = None
            batch_reward_ties = None
            batch_normalized_reward_wins = None
            batch_normalized_reward_ties = None
            if ref_model is not None:
                ref_chosen, _ref_chosen_counts = sequence_logp_sums_and_counts(
                    ref_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_response_start"],
                )
                ref_rejected, _ref_rejected_counts = sequence_logp_sums_and_counts(
                    ref_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_response_start"],
                )
                chosen_sampled_kl = chosen - ref_chosen
                rejected_sampled_kl = rejected - ref_rejected
                normalized_chosen_sampled_kl = normalized_chosen - (ref_chosen / chosen_counts)
                normalized_rejected_sampled_kl = normalized_rejected - (ref_rejected / rejected_counts)
                chosen_rewards, rejected_rewards, reward_margins = pair_reward_margins(
                    policy_chosen_logps=chosen,
                    policy_rejected_logps=rejected,
                    reference_chosen_logps=ref_chosen,
                    reference_rejected_logps=ref_rejected,
                    beta=args.beta,
                )
                normalized_chosen_rewards, normalized_rejected_rewards, normalized_reward_margins = (
                    pair_reward_margins(
                        policy_chosen_logps=normalized_chosen,
                        policy_rejected_logps=normalized_rejected,
                        reference_chosen_logps=ref_chosen / chosen_counts,
                        reference_rejected_logps=ref_rejected / rejected_counts,
                        beta=args.beta,
                    )
                )
                batch_reward_wins = reward_margins > 0
                batch_reward_ties = reward_margins == 0
                batch_normalized_reward_wins = normalized_reward_margins > 0
                batch_normalized_reward_ties = normalized_reward_margins == 0
                reward_wins += int(batch_reward_wins.sum().item())
                reward_ties += int(batch_reward_ties.sum().item())
                reward_margin_sum += float(reward_margins.sum().item())
                normalized_reward_wins += int(batch_normalized_reward_wins.sum().item())
                normalized_reward_ties += int(batch_normalized_reward_ties.sum().item())
                normalized_reward_margin_sum += float(normalized_reward_margins.sum().item())
                chosen_sampled_kl_sum += float(chosen_sampled_kl.sum().item())
                rejected_sampled_kl_sum += float(rejected_sampled_kl.sum().item())
                normalized_chosen_sampled_kl_sum += float(normalized_chosen_sampled_kl.sum().item())
                normalized_rejected_sampled_kl_sum += float(normalized_rejected_sampled_kl.sum().item())
            total += margins.numel()
            wins += int(batch_wins.sum().item())
            ties += int(batch_ties.sum().item())
            margin_sum += float(margins.sum().item())
            normalized_wins += int(batch_normalized_wins.sum().item())
            normalized_ties += int(batch_normalized_ties.sum().item())
            normalized_margin_sum += float(normalized_margins.sum().item())
            chosen_length_sum += int(chosen_counts.sum().item())
            rejected_length_sum += int(rejected_counts.sum().item())
            for (
                row_index,
                prompt_id,
                cluster_id,
                margin,
                normalized_margin,
                chosen_length,
                rejected_length,
                win,
                tie,
                normalized_win,
                normalized_tie,
                reward_margin,
                normalized_reward_margin,
                chosen_reward,
                rejected_reward,
                normalized_chosen_reward,
                normalized_rejected_reward,
                reward_win,
                reward_tie,
                normalized_reward_win,
                normalized_reward_tie,
                chosen_kl,
                rejected_kl,
                normalized_chosen_kl,
                normalized_rejected_kl,
            ) in zip(
                batch["row_indices"],
                batch["prompt_ids"],
                batch["cluster_ids"],
                margins.detach().cpu().tolist(),
                normalized_margins.detach().cpu().tolist(),
                chosen_counts.detach().cpu().tolist(),
                rejected_counts.detach().cpu().tolist(),
                batch_wins.detach().cpu().tolist(),
                batch_ties.detach().cpu().tolist(),
                batch_normalized_wins.detach().cpu().tolist(),
                batch_normalized_ties.detach().cpu().tolist(),
                (
                    reward_margins.detach().cpu().tolist()
                    if reward_margins is not None
                    else [None] * margins.numel()
                ),
                (
                    normalized_reward_margins.detach().cpu().tolist()
                    if normalized_reward_margins is not None
                    else [None] * margins.numel()
                ),
                (
                    chosen_rewards.detach().cpu().tolist()
                    if chosen_rewards is not None
                    else [None] * margins.numel()
                ),
                (
                    rejected_rewards.detach().cpu().tolist()
                    if rejected_rewards is not None
                    else [None] * margins.numel()
                ),
                (
                    normalized_chosen_rewards.detach().cpu().tolist()
                    if normalized_chosen_rewards is not None
                    else [None] * margins.numel()
                ),
                (
                    normalized_rejected_rewards.detach().cpu().tolist()
                    if normalized_rejected_rewards is not None
                    else [None] * margins.numel()
                ),
                (
                    batch_reward_wins.detach().cpu().tolist()
                    if batch_reward_wins is not None
                    else [None] * margins.numel()
                ),
                (
                    batch_reward_ties.detach().cpu().tolist()
                    if batch_reward_ties is not None
                    else [None] * margins.numel()
                ),
                (
                    batch_normalized_reward_wins.detach().cpu().tolist()
                    if batch_normalized_reward_wins is not None
                    else [None] * margins.numel()
                ),
                (
                    batch_normalized_reward_ties.detach().cpu().tolist()
                    if batch_normalized_reward_ties is not None
                    else [None] * margins.numel()
                ),
                (
                    chosen_sampled_kl.detach().cpu().tolist()
                    if chosen_sampled_kl is not None
                    else [None] * margins.numel()
                ),
                (
                    rejected_sampled_kl.detach().cpu().tolist()
                    if rejected_sampled_kl is not None
                    else [None] * margins.numel()
                ),
                (
                    normalized_chosen_sampled_kl.detach().cpu().tolist()
                    if normalized_chosen_sampled_kl is not None
                    else [None] * margins.numel()
                ),
                (
                    normalized_rejected_sampled_kl.detach().cpu().tolist()
                    if normalized_rejected_sampled_kl is not None
                    else [None] * margins.numel()
                ),
                strict=True,
            ):
                stats = cluster_stats[cluster_id]
                stats["total"] += 1
                stats["wins"] += int(win)
                stats["ties"] += int(tie)
                stats["margin_sum"] += float(margin)
                stats["normalized_wins"] += int(normalized_win)
                stats["normalized_ties"] += int(normalized_tie)
                stats["normalized_margin_sum"] += float(normalized_margin)
                if reward_margin is not None:
                    stats["reward_wins"] += int(reward_win)
                    stats["reward_ties"] += int(reward_tie)
                    stats["reward_margin_sum"] += float(reward_margin)
                    stats["normalized_reward_wins"] += int(normalized_reward_win)
                    stats["normalized_reward_ties"] += int(normalized_reward_tie)
                    stats["normalized_reward_margin_sum"] += float(normalized_reward_margin)
                    stats["chosen_sampled_kl_sum"] += float(chosen_kl)
                    stats["rejected_sampled_kl_sum"] += float(rejected_kl)
                    stats["normalized_chosen_sampled_kl_sum"] += float(normalized_chosen_kl)
                    stats["normalized_rejected_sampled_kl_sum"] += float(normalized_rejected_kl)
                stats["chosen_length_sum"] += int(chosen_length)
                stats["rejected_length_sum"] += int(rejected_length)
                record = {
                    "row_index": row_index,
                    "prompt_id": prompt_id,
                    "cluster_id": cluster_id,
                    "margin": float(margin),
                    "normalized_margin": float(normalized_margin),
                    "chosen_length": int(chosen_length),
                    "rejected_length": int(rejected_length),
                    "win": bool(win),
                    "tie": bool(tie),
                    "normalized_win": bool(normalized_win),
                    "normalized_tie": bool(normalized_tie),
                }
                if reward_margin is not None:
                    record.update(
                        {
                            "chosen_reward": float(chosen_reward),
                            "rejected_reward": float(rejected_reward),
                            "reward_margin": float(reward_margin),
                            "reward_win": bool(reward_win),
                            "reward_tie": bool(reward_tie),
                            "normalized_chosen_reward": float(normalized_chosen_reward),
                            "normalized_rejected_reward": float(normalized_rejected_reward),
                            "normalized_reward_margin": float(normalized_reward_margin),
                            "normalized_reward_win": bool(normalized_reward_win),
                            "normalized_reward_tie": bool(normalized_reward_tie),
                            "chosen_eval_logratio": float(chosen_kl),
                            "rejected_eval_logratio": float(rejected_kl),
                            "mean_eval_logratio": float((chosen_kl + rejected_kl) / 2.0),
                            "chosen_token_eval_logratio": float(normalized_chosen_kl),
                            "rejected_token_eval_logratio": float(normalized_rejected_kl),
                            "mean_token_eval_logratio": float(
                                (normalized_chosen_kl + normalized_rejected_kl) / 2.0
                            ),
                            # Backward-compatible aliases for older plots/reports.
                            "chosen_sampled_kl": float(chosen_kl),
                            "rejected_sampled_kl": float(rejected_kl),
                            "sampled_mean_kl": float((chosen_kl + rejected_kl) / 2.0),
                            "chosen_normalized_sampled_kl": float(normalized_chosen_kl),
                            "rejected_normalized_sampled_kl": float(normalized_rejected_kl),
                            "normalized_sampled_mean_kl": float(
                                (normalized_chosen_kl + normalized_rejected_kl) / 2.0
                            ),
                        }
                    )
                margin_records.append(record)

    chosen_eval_logratio = chosen_sampled_kl_sum / total if total and ref_model is not None else None
    rejected_eval_logratio = rejected_sampled_kl_sum / total if total and ref_model is not None else None
    mean_eval_logratio = (
        (chosen_sampled_kl_sum + rejected_sampled_kl_sum) / (2 * total)
        if total and ref_model is not None
        else None
    )
    chosen_token_eval_logratio = (
        normalized_chosen_sampled_kl_sum / total if total and ref_model is not None else None
    )
    rejected_token_eval_logratio = (
        normalized_rejected_sampled_kl_sum / total if total and ref_model is not None else None
    )
    mean_token_eval_logratio = (
        (normalized_chosen_sampled_kl_sum + normalized_rejected_sampled_kl_sum) / (2 * total)
        if total and ref_model is not None
        else None
    )

    result = {
        "model": args.model_name_or_path,
        "reference_model": args.reference_model_name_or_path,
        "beta": args.beta,
        "eval_file": str(args.eval_file),
        "total": total,
        "wins": wins,
        "ties": ties,
        "losses": total - wins - ties,
        "winrate": wins / total if total else 0.0,
        "tie_rate": ties / total if total else 0.0,
        "mean_margin": margin_sum / total if total else 0.0,
        "normalized_wins": normalized_wins,
        "normalized_ties": normalized_ties,
        "normalized_losses": total - normalized_wins - normalized_ties,
        "normalized_winrate": normalized_wins / total if total else 0.0,
        "normalized_tie_rate": normalized_ties / total if total else 0.0,
        "mean_normalized_margin": normalized_margin_sum / total if total else 0.0,
        "reward_accuracy": reward_wins / total if total and ref_model is not None else None,
        "reward_tie_rate": reward_ties / total if total and ref_model is not None else None,
        "mean_reward_margin": reward_margin_sum / total if total and ref_model is not None else None,
        "normalized_reward_accuracy": (
            normalized_reward_wins / total if total and ref_model is not None else None
        ),
        "normalized_reward_tie_rate": (
            normalized_reward_ties / total if total and ref_model is not None else None
        ),
        "mean_normalized_reward_margin": (
            normalized_reward_margin_sum / total if total and ref_model is not None else None
        ),
        "chosen_eval_logratio": chosen_eval_logratio,
        "rejected_eval_logratio": rejected_eval_logratio,
        "mean_eval_logratio": mean_eval_logratio,
        "chosen_token_eval_logratio": chosen_token_eval_logratio,
        "rejected_token_eval_logratio": rejected_token_eval_logratio,
        "mean_token_eval_logratio": mean_token_eval_logratio,
        # Backward-compatible aliases for older plots/reports.
        "chosen_sampled_kl": chosen_eval_logratio,
        "rejected_sampled_kl": rejected_eval_logratio,
        "sampled_mean_kl": mean_eval_logratio,
        "chosen_normalized_sampled_kl": chosen_token_eval_logratio,
        "rejected_normalized_sampled_kl": rejected_token_eval_logratio,
        "normalized_sampled_mean_kl": mean_token_eval_logratio,
        "mean_chosen_length": chosen_length_sum / total if total else 0.0,
        "mean_rejected_length": rejected_length_sum / total if total else 0.0,
        "clusters": {
            cluster_id: {
                "total": int(stats["total"]),
                "wins": int(stats["wins"]),
                "ties": int(stats["ties"]),
                "losses": int(stats["total"] - stats["wins"] - stats["ties"]),
                "winrate": stats["wins"] / stats["total"] if stats["total"] else 0.0,
                "mean_margin": stats["margin_sum"] / stats["total"] if stats["total"] else 0.0,
                "normalized_wins": int(stats["normalized_wins"]),
                "normalized_ties": int(stats["normalized_ties"]),
                "normalized_losses": int(
                    stats["total"] - stats["normalized_wins"] - stats["normalized_ties"]
                ),
                "normalized_winrate": (
                    stats["normalized_wins"] / stats["total"] if stats["total"] else 0.0
                ),
                "normalized_tie_rate": (
                    stats["normalized_ties"] / stats["total"] if stats["total"] else 0.0
                ),
                "mean_normalized_margin": (
                    stats["normalized_margin_sum"] / stats["total"] if stats["total"] else 0.0
                ),
                "reward_accuracy": (
                    stats["reward_wins"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "reward_tie_rate": (
                    stats["reward_ties"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "mean_reward_margin": (
                    stats["reward_margin_sum"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "normalized_reward_accuracy": (
                    stats["normalized_reward_wins"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "normalized_reward_tie_rate": (
                    stats["normalized_reward_ties"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "mean_normalized_reward_margin": (
                    stats["normalized_reward_margin_sum"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "chosen_eval_logratio": (
                    stats["chosen_sampled_kl_sum"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "rejected_eval_logratio": (
                    stats["rejected_sampled_kl_sum"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "mean_eval_logratio": (
                    (stats["chosen_sampled_kl_sum"] + stats["rejected_sampled_kl_sum"])
                    / (2 * stats["total"])
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "chosen_token_eval_logratio": (
                    stats["normalized_chosen_sampled_kl_sum"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "rejected_token_eval_logratio": (
                    stats["normalized_rejected_sampled_kl_sum"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "mean_token_eval_logratio": (
                    (
                        stats["normalized_chosen_sampled_kl_sum"]
                        + stats["normalized_rejected_sampled_kl_sum"]
                    )
                    / (2 * stats["total"])
                    if stats["total"] and ref_model is not None
                    else None
                ),
                # Backward-compatible aliases for older plots/reports.
                "chosen_sampled_kl": (
                    stats["chosen_sampled_kl_sum"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "rejected_sampled_kl": (
                    stats["rejected_sampled_kl_sum"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "sampled_mean_kl": (
                    (stats["chosen_sampled_kl_sum"] + stats["rejected_sampled_kl_sum"])
                    / (2 * stats["total"])
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "chosen_normalized_sampled_kl": (
                    stats["normalized_chosen_sampled_kl_sum"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "rejected_normalized_sampled_kl": (
                    stats["normalized_rejected_sampled_kl_sum"] / stats["total"]
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "normalized_sampled_mean_kl": (
                    (
                        stats["normalized_chosen_sampled_kl_sum"]
                        + stats["normalized_rejected_sampled_kl_sum"]
                    )
                    / (2 * stats["total"])
                    if stats["total"] and ref_model is not None
                    else None
                ),
                "mean_chosen_length": (
                    stats["chosen_length_sum"] / stats["total"] if stats["total"] else 0.0
                ),
                "mean_rejected_length": (
                    stats["rejected_length_sum"] / stats["total"] if stats["total"] else 0.0
                ),
            }
            for cluster_id, stats in sorted(cluster_stats.items())
        },
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    margin_path = args.output_margins_jsonl
    if margin_path is None and args.output_json:
        margin_path = args.output_json.with_name(f"{args.output_json.stem}_margins.jsonl")
    if margin_path:
        margin_path.parent.mkdir(parents=True, exist_ok=True)
        with margin_path.open("w", encoding="utf-8") as handle:
            for record in margin_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
