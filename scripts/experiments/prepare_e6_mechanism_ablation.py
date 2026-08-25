from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


REFERENCE_BASELINES = ("token_kl", "same_completion_logratio", "kto_mismatched_logratio")
EMA_MOMENTA = (0.0, 0.5, 0.9, 0.99)
PAIRWISE_ALPHAS = (0.0, 0.3, 1.0)
SAMPLER_SETTINGS = (True, False)


def load_config(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return dict(loaded)


def slug_value(value: object) -> str:
    return str(value).replace(".", "p").replace("_", "-").lower()


def add_run(
    runs: list[dict[str, Any]],
    *,
    family: str,
    name: str,
    config: dict[str, Any],
    experiment_dir: Path,
    output_root: Path,
) -> None:
    run_dir = experiment_dir / family / name
    config_path = run_dir / "config.yaml"
    config = {**config, "output_dir": str(output_root / f"cpo_{family}_{name}")}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    runs.append(
        {
            "experiment": "E6_mechanism_ablation",
            "family": family,
            "name": name,
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
    )


def prepare_e6(
    *,
    experiment_dir: Path,
    output_root: Path,
    base_config_path: Path = Path("configs/cpo/cpo_controlled.yaml"),
) -> Path:
    base_config = load_config(base_config_path)
    runs: list[dict[str, Any]] = []
    for baseline in REFERENCE_BASELINES:
        add_run(
            runs,
            family="reference",
            name=slug_value(baseline),
            config={**base_config, "z_baseline": baseline},
            experiment_dir=experiment_dir,
            output_root=output_root,
        )
    for momentum in EMA_MOMENTA:
        add_run(
            runs,
            family="ema",
            name=f"momentum_{slug_value(momentum)}",
            config={**base_config, "z_momentum": momentum},
            experiment_dir=experiment_dir,
            output_root=output_root,
        )
    for alpha in PAIRWISE_ALPHAS:
        add_run(
            runs,
            family="pairwise",
            name=f"alpha_{slug_value(alpha)}",
            config={**base_config, "alpha": alpha},
            experiment_dir=experiment_dir,
            output_root=output_root,
        )
    for enabled in SAMPLER_SETTINGS:
        add_run(
            runs,
            family="sampler",
            name="pair_aware" if enabled else "plain_batching",
            config={**base_config, "pair_aware_batching": enabled},
            experiment_dir=experiment_dir,
            output_root=output_root,
        )
    manifest_path = experiment_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run) + "\n")
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare E6 CPO mechanism-ablation configs.")
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/E6_mechanism_ablation"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/cpo/cpo_controlled.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = prepare_e6(
        experiment_dir=args.experiment_dir,
        output_root=args.output_root,
        base_config_path=args.base_config,
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
