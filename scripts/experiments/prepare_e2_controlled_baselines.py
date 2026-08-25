from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


RUNS = (
    ("sft", "configs/sft/sft.yaml", "scripts/train/train_sft.py", "sft"),
    ("dpo", "configs/dpo/dpo_controlled.yaml", "scripts/train/train_dpo.py", "dpo"),
    ("kto", "configs/kto/kto_controlled.yaml", "scripts/train/train_kto.py", "kto"),
    ("cpo_unary", "configs/cpo/cpo_unary.yaml", "scripts/train/train_cpo.py", "cpo_unary"),
    ("cpo", "configs/cpo/cpo_controlled.yaml", "scripts/train/train_cpo.py", "cpo"),
)


def read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return dict(loaded)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def build_runs(*, experiment_dir: Path, output_root: Path) -> list[dict[str, Any]]:
    runs = []
    config_dir = experiment_dir / "configs"
    for name, source_config, script, output_slug in RUNS:
        config = read_yaml(Path(source_config))
        config["output_dir"] = str(output_root / output_slug)
        target_config = config_dir / f"{name}.yaml"
        command = ["poetry", "run", "python", script, "--config", str(target_config)]
        runs.append(
            {
                "experiment": "E2_controlled_baselines",
                "name": name,
                "source_config": source_config,
                "config_path": str(target_config),
                "output_dir": str(output_root / output_slug),
                "command": command,
                "controlled_fields": {
                    key: config.get(key)
                    for key in (
                        "model_name_or_path",
                        "max_seq_length",
                        "per_device_train_batch_size",
                        "gradient_accumulation_steps",
                        "learning_rate",
                        "num_train_epochs",
                        "seed",
                        "use_lora",
                    )
                },
            }
        )
    return runs


def write_manifest(runs: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run) + "\n")


def prepare_e2(*, experiment_dir: Path, output_root: Path) -> Path:
    runs = build_runs(experiment_dir=experiment_dir, output_root=output_root)
    for run in runs:
        config = read_yaml(Path(run["source_config"]))
        config["output_dir"] = run["output_dir"]
        write_yaml(Path(run["config_path"]), config)
    manifest_path = experiment_dir / "manifest.jsonl"
    write_manifest(runs, manifest_path)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare E2 controlled baseline configs and commands.")
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/E2_controlled_baselines"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/checkpoints"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = prepare_e2(experiment_dir=args.experiment_dir, output_root=args.output_root)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
