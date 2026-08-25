from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


SEEDS = (42, 123, 456)
METHODS = {
    "sft": ("configs/sft/sft.yaml", "scripts/train/train_sft.py"),
    "dpo": ("configs/dpo/dpo_controlled.yaml", "scripts/train/train_dpo.py"),
    "kto": ("configs/kto/kto_controlled.yaml", "scripts/train/train_kto.py"),
    "cpo_unary": ("configs/cpo/cpo_unary.yaml", "scripts/train/train_cpo.py"),
    "cpo": ("configs/cpo/cpo_controlled.yaml", "scripts/train/train_cpo.py"),
}


def load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return dict(loaded)


def prepare_e7(*, experiment_dir: Path, output_root: Path) -> Path:
    manifest_path = experiment_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for seed in SEEDS:
            seed_dir = experiment_dir / f"seed_{seed}"
            for method, (source_config, script) in METHODS.items():
                config = load_config(Path(source_config))
                config["seed"] = seed
                config["output_dir"] = str(output_root / f"{method}_seed_{seed}")
                config_path = seed_dir / f"{method}.yaml"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                manifest.write(
                    json.dumps(
                        {
                            "experiment": "E7_robustness",
                            "seed": seed,
                            "method": method,
                            "config_path": str(config_path),
                            "output_dir": config["output_dir"],
                            "command": ["poetry", "run", "python", script, "--config", str(config_path)],
                        }
                    )
                    + "\n"
                )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare E7 robustness runs over seeds.")
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/E7_robustness"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/checkpoints"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = prepare_e7(experiment_dir=args.experiment_dir, output_root=args.output_root)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
