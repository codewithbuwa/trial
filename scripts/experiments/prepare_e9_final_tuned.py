from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


METHOD_CONFIGS = {
    "sft": ("configs/sft/sft.yaml", "scripts/train/train_sft.py"),
    "dpo": ("configs/dpo/dpo_tuned.yaml", "scripts/train/train_dpo.py"),
    "kto": ("configs/kto/kto_tuned.yaml", "scripts/train/train_kto.py"),
    "cpo_unary": ("configs/cpo/cpo_unary.yaml", "scripts/train/train_cpo.py"),
    "cpo": ("configs/cpo/cpo_tuned.yaml", "scripts/train/train_cpo.py"),
}
TUNED_KEYS = ("learning_rate", "beta", "alpha", "max_grad_norm", "z_baseline")


def load_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return dict(loaded)


def load_best(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"best_by_method must be a JSON object: {path}")
    return {str(key).lower(): dict(value) for key, value in loaded.items() if isinstance(value, dict)}


def apply_tuned_values(config: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    tuned = dict(config)
    for key in TUNED_KEYS:
        if best.get(key) is not None:
            tuned[key] = best[key]
    return tuned


def prepare_e9(
    *,
    experiment_dir: Path,
    output_root: Path,
    best_by_method: Path | None,
    data_root: Path,
    test_split: str,
) -> Path:
    best = load_best(best_by_method)
    manifest_path = experiment_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for method, (source_config, script) in METHOD_CONFIGS.items():
            config = apply_tuned_values(load_yaml(Path(source_config)), best.get(method, {}))
            config["output_dir"] = str(output_root / f"final_{method}")
            config_path = experiment_dir / "configs" / f"{method}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            eval_kind = "dpo" if method in {"sft", "dpo"} else ("kto" if method == "kto" else "cpo")
            manifest.write(
                json.dumps(
                    {
                        "experiment": "E9_final_tuned",
                        "method": method,
                        "source_config": source_config,
                        "selected_from": str(best_by_method) if best_by_method else None,
                        "config_path": str(config_path),
                        "output_dir": config["output_dir"],
                        "train_command": ["poetry", "run", "python", script, "--config", str(config_path)],
                        "test_eval_command": [
                            "poetry",
                            "run",
                            "python",
                            "scripts/evaluate/evaluate_pairwise_accuracy.py",
                            "--eval-file",
                            str(data_root / eval_kind / test_split),
                            "--model-name-or-path",
                            config["output_dir"],
                            "--output-json",
                            str(Path(config["output_dir"]) / "test_pairwise_accuracy.json"),
                        ],
                    }
                )
                + "\n"
            )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare E9 final tuned train/test comparison.")
    parser.add_argument("--experiment-dir", type=Path, default=Path("experiments/E9_final_tuned"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--best-by-method", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--test-split", default="test.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = prepare_e9(
        experiment_dir=args.experiment_dir,
        output_root=args.output_root,
        best_by_method=args.best_by_method,
        data_root=args.data_root,
        test_split=args.test_split,
    )
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
