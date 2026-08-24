from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from cpo_trl.data.datasets import load_jsonl


def unique_prompt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    prompts: list[dict[str, Any]] = []
    for row in rows:
        prompt_id = str(row.get("prompt_id", len(prompts)))
        if prompt_id in seen:
            continue
        seen.add(prompt_id)
        prompts.append(
            {
                "prompt_id": prompt_id,
                "instruction": str(row.get("instruction", "")),
                "input": str(row.get("input", "")),
                "cluster_id": str(row.get("cluster_id", "unknown")),
            }
        )
    return prompts


def balanced_prompt_rows(
    prompts: list[dict[str, Any]],
    *,
    per_cluster: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prompts:
        grouped[str(row.get("cluster_id", "unknown"))].append(row)
    if not grouped:
        return []
    target = per_cluster or min(len(rows) for rows in grouped.values())
    rng = random.Random(seed)
    balanced: list[dict[str, Any]] = []
    for cluster_id in sorted(grouped):
        rows = list(grouped[cluster_id])
        rng.shuffle(rows)
        balanced.extend(rows[:target])
    balanced.sort(key=lambda row: (str(row["cluster_id"]), str(row["prompt_id"])))
    return balanced


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clusters: dict[str, int] = defaultdict(int)
    for row in rows:
        clusters[str(row.get("cluster_id", "unknown"))] += 1
    return {
        "num_prompts": len(rows),
        "cluster_counts": dict(sorted(clusters.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evaluation prompt manifests.")
    parser.add_argument("--source-file", type=Path, default=Path("data/processed/dpo/validation.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--balanced-per-cluster", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = unique_prompt_rows(load_jsonl(args.source_file))
    natural = sorted(prompts, key=lambda row: str(row["prompt_id"]))
    balanced = balanced_prompt_rows(
        natural,
        per_cluster=args.balanced_per_cluster,
        seed=args.seed,
    )
    write_jsonl(args.output_dir / "eval_manifest_natural.jsonl", natural)
    write_jsonl(args.output_dir / "eval_manifest_balanced.jsonl", balanced)
    write_jsonl(args.output_dir / "cluster_gold_set.jsonl", balanced)
    manifest = {
        "source_file": str(args.source_file),
        "seed": args.seed,
        "natural": split_manifest(natural),
        "balanced": split_manifest(balanced),
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
