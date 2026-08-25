from __future__ import annotations

import argparse
import json
from pathlib import Path


CONDITIONS = (
    ("semantic_4", "keyword", "Meaningful keyword clusters: coding/math/writing/general."),
    ("random_4", "random4", "Four prompt-stable random clusters with balanced sizes."),
    ("random_4_matched", "random4_matched", "Randomized prompt membership preserving semantic cluster sizes."),
    ("single_cluster", "single_cluster", "All prompts share one global z reference."),
    ("alternative_clusters", "keyword", "Placeholder condition for a future alternative partitioning source."),
)


def prepare_e5(
    *,
    experiment_dir: Path,
    data_root: Path,
    output_root: Path,
    limit: int | None,
    seed: int,
) -> Path:
    manifest_path = experiment_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for name, cluster_mode, description in CONDITIONS:
            condition_dir = experiment_dir / name
            condition_dir.mkdir(parents=True, exist_ok=True)
            condition_data_root = data_root / name
            condition_output = output_root / f"cpo_{name}"
            data_command = [
                "poetry",
                "run",
                "python",
                "scripts/data/prepare_ultrafeedback.py",
                "--cluster-mode",
                cluster_mode,
                "--seed",
                str(seed),
                "--output-root",
                str(condition_data_root),
            ]
            if limit is not None:
                data_command.extend(["--limit", str(limit)])
            train_command = [
                "poetry",
                "run",
                "python",
                "scripts/train/train_cpo.py",
                "--config",
                "configs/cpo/cpo_controlled.yaml",
                "--train-file",
                str(condition_data_root / "cpo" / "train.jsonl"),
                "--output-dir",
                str(condition_output),
            ]
            manifest.write(
                json.dumps(
                    {
                        "experiment": "E5_cluster_ablation",
                        "condition": name,
                        "cluster_mode": cluster_mode,
                        "description": description,
                        "data_root": str(condition_data_root),
                        "output_dir": str(condition_output),
                        "data_command": data_command,
                        "train_command": train_command,
                    }
                )
                + "\n"
            )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare E5 cluster-ablation commands.")
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/E5_cluster_ablation"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_e5"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = prepare_e5(
        experiment_dir=args.experiment_dir,
        data_root=args.data_root,
        output_root=args.output_root,
        limit=args.limit,
        seed=args.seed,
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
