from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


ALPHAS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)


def alpha_dir_name(alpha: float) -> str:
    return f"alpha_{alpha:.2f}"


def load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return dict(loaded)


def prepare_e4(
    *,
    experiment_dir: Path,
    output_root: Path,
    base_config_path: Path = Path("configs/cpo/cpo_controlled.yaml"),
) -> Path:
    manifest_path = experiment_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    base_config = load_config(base_config_path)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for alpha in ALPHAS:
            run_dir = experiment_dir / alpha_dir_name(alpha)
            config_path = run_dir / "config.yaml"
            config = {
                **base_config,
                "alpha": alpha,
                "output_dir": str(output_root / f"cpo_alpha_{alpha:.2f}".replace(".", "p")),
            }
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            record = {
                "experiment": "E4_alpha_sweep",
                "alpha": alpha,
                "config_path": str(config_path),
                "output_dir": config["output_dir"],
                "command": [
                    "poetry",
                    "run",
                    "python",
                    "scripts/train/train_cpo.py",
                    "--config",
                    str(config_path),
                ],
            }
            manifest.write(json.dumps(record) + "\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare E4 CPO alpha sweep configs.")
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/E4_alpha_sweep"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/cpo/cpo_controlled.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = prepare_e4(
        experiment_dir=args.experiment_dir,
        output_root=args.output_root,
        base_config_path=args.base_config,
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
