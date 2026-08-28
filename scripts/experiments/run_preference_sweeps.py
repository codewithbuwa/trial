from __future__ import annotations

import argparse
import json
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

import yaml


METHODS = ("dpo", "kto", "cpo")
DEFAULT_LRS = {
    "dpo": [1e-6, 5e-6, 1e-5],
    "kto": [1e-6, 5e-6, 1e-5],
    "cpo": [1e-6, 5e-6, 1e-5],
}
DEFAULT_BETAS = [0.01, 0.02, 0.05]
DEFAULT_MAX_GRAD_NORMS = [0.3, 1.0]
DEFAULT_ALPHAS = [0.3]
DEFAULT_CPO_Z_BASELINES = ["token_kl"]
BASE_CONFIG_FILES = ("model.yaml", "data.yaml", "lora.yaml", "training.yaml")


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated float")
    return values


def parse_str_list(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def load_base_config(method: str, config_dir: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for name in BASE_CONFIG_FILES:
        path = config_dir / "base" / name
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"config must be a mapping: {path}")
            config.update(loaded)

    candidates = [
        config_dir / method / f"{method}_controlled.yaml",
        config_dir / f"{method}.yaml",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        checked = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"missing base config for {method}; checked: {checked}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    config.update(loaded)
    return config


def build_sweep_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for method in args.methods:
        base_config = load_base_config(method, args.config_dir)
        lrs = args.learning_rates or DEFAULT_LRS[method]
        betas = args.betas or DEFAULT_BETAS
        alphas = args.alphas or DEFAULT_ALPHAS
        max_grad_norms = args.max_grad_norms or DEFAULT_MAX_GRAD_NORMS
        z_baselines = args.z_baselines or DEFAULT_CPO_Z_BASELINES
        alpha_values: list[float | None] = alphas if method == "cpo" else [None]
        z_baseline_values: list[str | None] = z_baselines if method == "cpo" else [None]
        for learning_rate, beta, max_grad_norm, alpha, z_baseline in product(
            lrs, betas, max_grad_norms, alpha_values, z_baseline_values
        ):
            name_parts = [
                method,
                f"lr{slug_float(learning_rate)}",
                f"b{slug_float(beta)}",
                f"gn{slug_float(max_grad_norm)}",
            ]
            if alpha is not None:
                name_parts.append(f"a{slug_float(alpha)}")
            if z_baseline is not None:
                name_parts.append(z_baseline.replace("_", "-"))
            run_name = "_".join(name_parts)
            output_dir = args.output_dir / run_name
            config_path = args.output_dir / "configs" / f"{run_name}.yaml"
            train_file = args.train_root / method / args.split
            config = {
                **base_config,
                "train_file": str(train_file),
                "model_name_or_path": str(args.model_name_or_path),
                "output_dir": str(output_dir),
                "learning_rate": learning_rate,
                "max_grad_norm": max_grad_norm,
                "weight_decay": (
                    args.weight_decay
                    if args.weight_decay is not None
                    else base_config.get("weight_decay", 0.0)
                ),
                "beta": beta,
                "num_train_epochs": args.num_train_epochs,
                "max_seq_length": args.max_seq_length,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "logging_steps": args.logging_steps,
                "save_steps": args.save_steps,
                "save_total_limit": args.save_total_limit,
                "seed": args.seed,
                "use_lora": True,
            }
            if args.warmup_steps is not None:
                config["warmup_steps"] = args.warmup_steps
                config["warmup_ratio"] = 0.0
            if alpha is not None:
                config["alpha"] = alpha
            if z_baseline is not None:
                config["z_baseline"] = z_baseline
            command = [
                "poetry",
                "run",
                "python",
                f"scripts/train/train_{method}.py",
                "--config",
                str(config_path),
            ]
            runs.append(
                {
                    "method": method,
                    "run_name": run_name,
                    "config": config,
                    "config_path": config_path,
                    "output_dir": output_dir,
                    "command": command,
                    "learning_rate": learning_rate,
                    "beta": beta,
                    "alpha": alpha,
                    "max_grad_norm": max_grad_norm,
                    "z_baseline": z_baseline,
                }
            )
    return runs


def write_sweep(runs: list[dict[str, Any]], output_dir: Path) -> Path:
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for run in runs:
            config_path = Path(run["config_path"])
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(run["config"], handle, sort_keys=False)
            record = {
                "method": run["method"],
                "run_name": run["run_name"],
                "config_path": str(config_path),
                "output_dir": str(run["output_dir"]),
                "command": run["command"],
                "learning_rate": run["learning_rate"],
                "beta": run["beta"],
                "alpha": run["alpha"],
                "max_grad_norm": run["max_grad_norm"],
                "z_baseline": run["z_baseline"],
            }
            manifest.write(json.dumps(record) + "\n")
    return manifest_path


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object: {path}")
    return loaded


def write_results_summary(
    results: list[dict[str, Any]],
    output_dir: Path,
    score_metric: str,
) -> tuple[Path, Path]:
    results_path = output_dir / "sweep_results.jsonl"
    best_path = output_dir / "best_by_method.json"
    with results_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result) + "\n")

    best_by_method: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        candidates = [
            result
            for result in results
            if result["method"] == method and result["status"] == "ok" and result.get(score_metric) is not None
        ]
        candidates.sort(
            key=lambda result: (
                result.get(score_metric) if result.get(score_metric) is not None else float("-inf"),
                result.get("mean_normalized_reward_margin")
                if result.get("mean_normalized_reward_margin") is not None
                else float("-inf"),
                result.get("normalized_pairwise_accuracy")
                if result.get("normalized_pairwise_accuracy") is not None
                else float("-inf"),
            ),
            reverse=True,
        )
        if candidates:
            best_by_method[method] = candidates[0]
    best_path.write_text(json.dumps(best_by_method, indent=2) + "\n", encoding="utf-8")
    return results_path, best_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and optionally run short DPO/KTO/CPO hyperparameter sweeps."
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--train-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", default="train.jsonl")
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sweeps"))
    parser.add_argument("--learning-rates", type=parse_float_list)
    parser.add_argument("--betas", type=parse_float_list)
    parser.add_argument("--alphas", type=parse_float_list)
    parser.add_argument("--max-grad-norms", type=parse_float_list)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument(
        "--z-baselines",
        type=parse_str_list,
        help="Comma-separated CPO z baselines. Defaults to token_kl.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", action="store_true", help="Run commands after writing configs.")
    parser.add_argument("--eval", action="store_true", help="Evaluate each completed run with evaluate_pairwise_accuracy.py.")
    parser.add_argument("--eval-file", type=Path, default=Path("data/processed/dpo/validation.jsonl"))
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument(
        "--reference-model-name-or-path",
        help="Reference model for reward accuracy/KL during eval. Defaults to --model-name-or-path.",
    )
    parser.add_argument("--score-metric", default="normalized_reward_accuracy")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = build_sweep_configs(args)
    manifest_path = write_sweep(runs, args.output_dir)
    print(f"Wrote {len(runs)} sweep configs")
    print(f"Manifest: {manifest_path}")
    for run in runs:
        print(" ".join(run["command"]))
    if not args.run:
        return
    results: list[dict[str, Any]] = []
    for run in runs:
        print(f"\n=== Running {run['run_name']} ===", flush=True)
        completed = subprocess.run(run["command"], check=False)
        result: dict[str, Any] = {
            "method": run["method"],
            "run_name": run["run_name"],
            "output_dir": str(run["output_dir"]),
            "config_path": str(run["config_path"]),
            "learning_rate": run["learning_rate"],
            "beta": run["beta"],
            "alpha": run["alpha"],
            "max_grad_norm": run["max_grad_norm"],
            "z_baseline": run["z_baseline"],
            "train_returncode": completed.returncode,
            "status": "ok" if completed.returncode == 0 else "train_failed",
        }
        if completed.returncode and args.stop_on_failure:
            results.append(result)
            write_results_summary(results, args.output_dir, args.score_metric)
            sys.exit(completed.returncode)
        if completed.returncode == 0 and args.eval:
            eval_path = Path(run["output_dir"]) / "pairwise_accuracy.json"
            eval_command = [
                "poetry",
                "run",
                "python",
                "scripts/evaluate/evaluate_pairwise_accuracy.py",
                "--eval-file",
                str(args.eval_file),
                "--model-name-or-path",
                str(run["output_dir"]),
                "--reference-model-name-or-path",
                str(args.reference_model_name_or_path or args.model_name_or_path),
                "--beta",
                str(run["beta"]),
                "--batch-size",
                str(args.eval_batch_size),
                "--output-json",
                str(eval_path),
            ]
            if args.eval_limit:
                eval_command.extend(["--limit", str(args.eval_limit)])
            print(" ".join(eval_command), flush=True)
            eval_completed = subprocess.run(eval_command, check=False)
            result["eval_returncode"] = eval_completed.returncode
            if eval_completed.returncode == 0:
                metrics = read_json(eval_path)
                for key in (
                    "pairwise_accuracy",
                    "normalized_pairwise_accuracy",
                    "reward_accuracy",
                    "normalized_reward_accuracy",
                    "mean_margin",
                    "mean_normalized_margin",
                    "mean_reward_margin",
                    "mean_normalized_reward_margin",
                    "mean_eval_logratio",
                    "mean_token_eval_logratio",
                    "sampled_mean_kl",
                    "normalized_sampled_mean_kl",
                ):
                    result[key] = metrics.get(key)
            else:
                result["status"] = "eval_failed"
                if args.stop_on_failure:
                    results.append(result)
                    write_results_summary(results, args.output_dir, args.score_metric)
                    sys.exit(eval_completed.returncode)
        results.append(result)
        write_results_summary(results, args.output_dir, args.score_metric)
    if results:
        results_path, best_path = write_results_summary(results, args.output_dir, args.score_metric)
        print(f"Sweep results: {results_path}")
        print(f"Best by method: {best_path}")


if __name__ == "__main__":
    main()
