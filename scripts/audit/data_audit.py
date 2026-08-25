from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cpo_trl.data.datasets import load_jsonl
from cpo_trl.sampling.pair_sampler import CPOPairAwareBatchSampler


SPLITS = ("train", "validation", "test")
KINDS = ("sft", "dpo", "kto", "cpo")
EXPECTED_CLUSTERS = ("coding", "math", "writing", "general")


def prompt_ids(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("prompt_id")) for row in rows if row.get("prompt_id") is not None}


def cluster_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("cluster_id", "unknown")) for row in rows).items()))


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(bool(row.get("label")) for row in rows if "label" in row)
    return {"desirable": counts[True], "undesirable": counts[False]}


def dpo_pair_count(rows: list[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("chosen") and row.get("rejected"))


def cpo_pair_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], set[bool]] = defaultdict(set)
    response_count = 0
    for row in rows:
        if "label" not in row:
            continue
        response_count += 1
        key = (str(row.get("cluster_id", "unknown")), str(row.get("prompt_id")))
        grouped[key].add(bool(row["label"]))
    eligible = sum(labels == {True, False} for labels in grouped.values())
    return {
        "responses": response_count,
        "prompt_cluster_groups": len(grouped),
        "pair_eligible_prompt_cluster_groups": eligible,
        "pair_eligible_rate": eligible / len(grouped) if grouped else 0.0,
    }


def sampler_report(rows: list[dict[str, Any]], *, batch_size: int, seed: int) -> dict[str, Any]:
    if not rows:
        return {"available": False, "reason": "no CPO rows"}
    if batch_size < 2:
        return {"available": False, "reason": "batch_size must be >= 2"}
    sampler = CPOPairAwareBatchSampler(rows, batch_size=batch_size, seed=seed)
    dataset_clusters = Counter(str(row.get("cluster_id", "unknown")) for row in rows)
    sampler_pairs = Counter(sampler.stats.cluster_pairs)
    total_dataset = sum(dataset_clusters.values())
    total_pairs = sum(sampler_pairs.values())
    cluster_distribution = {}
    for cluster_id in sorted(set(dataset_clusters) | set(sampler_pairs)):
        dataset_p = dataset_clusters[cluster_id] / total_dataset if total_dataset else 0.0
        sampler_p = sampler_pairs[cluster_id] / total_pairs if total_pairs else 0.0
        cluster_distribution[cluster_id] = {
            "dataset_probability": dataset_p,
            "sampler_pair_probability": sampler_p,
            "absolute_delta": abs(dataset_p - sampler_p),
        }
    return {
        "available": True,
        "batch_size": batch_size,
        "seed": seed,
        "stats": vars(sampler.stats),
        "cluster_distribution_check": cluster_distribution,
        "max_cluster_probability_delta": max(
            (entry["absolute_delta"] for entry in cluster_distribution.values()),
            default=0.0,
        ),
    }


def split_report(path: Path, kind: str) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    rows = load_jsonl(path)
    report: dict[str, Any] = {
        "exists": True,
        "path": str(path),
        "rows": len(rows),
        "prompts": len(prompt_ids(rows)),
        "clusters": cluster_counts(rows),
    }
    if kind in {"kto", "cpo"}:
        report["labels"] = label_counts(rows)
    if kind == "dpo":
        report["pairs"] = dpo_pair_count(rows)
    if kind == "cpo":
        report["pairing"] = cpo_pair_stats(rows)
    return report


def prompt_overlap_report(reports: dict[str, dict[str, dict[str, Any]]], data_root: Path) -> dict[str, Any]:
    overlap: dict[str, Any] = {}
    for kind in KINDS:
        split_ids = {}
        for split in SPLITS:
            path = data_root / kind / f"{split}.jsonl"
            split_ids[split] = prompt_ids(load_jsonl(path)) if path.exists() else set()
        overlap[kind] = {
            f"{left}_vs_{right}": sorted(split_ids[left] & split_ids[right])
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        }
    return overlap


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    splits = {
        kind: {
            split: split_report(args.data_root / kind / f"{split}.jsonl", kind)
            for split in SPLITS
        }
        for kind in KINDS
    }
    cpo_train = args.data_root / "cpo" / "train.jsonl"
    cpo_rows = load_jsonl(cpo_train) if cpo_train.exists() else []
    present_clusters = set(cluster_counts(cpo_rows))
    return {
        "experiment": "E1_data_audit",
        "data_root": str(args.data_root),
        "expected_clusters": list(EXPECTED_CLUSTERS),
        "missing_expected_clusters": sorted(set(EXPECTED_CLUSTERS) - present_clusters),
        "splits": splits,
        "prompt_overlap": prompt_overlap_report(splits, args.data_root),
        "sampler": sampler_report(cpo_rows, batch_size=args.batch_size, seed=args.seed),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the E1 data and sampler audit report.")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/E1_data_audit/results/data_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
