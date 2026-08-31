from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "train_file": None,
    "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
    "output_dir": None,
    "max_seq_length": 1024,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
    "max_grad_norm": 0.3,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "warmup_steps": 0,
    "num_train_epochs": 1.0,
    "logging_steps": 10,
    "terminal_log_steps": None,
    "save_steps": 500,
    "save_total_limit": 1,
    "seed": 42,
    "use_lora": False,
}


BASE_CONFIG_FILES = (
    Path("configs/base/model.yaml"),
    Path("configs/base/data.yaml"),
    Path("configs/base/lora.yaml"),
    Path("configs/base/training.yaml"),
)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return dict(loaded)


def load_base_configs() -> dict[str, Any]:
    values: dict[str, Any] = {}
    for path in BASE_CONFIG_FILES:
        if path.exists():
            values.update(load_yaml_mapping(path))
    return values


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--train-file", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--model-name-or-path", default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--max-seq-length", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--per-device-train-batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--max-grad-norm", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--warmup-ratio", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--warmup-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-train-epochs", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--logging-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--terminal-log-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--save-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--save-total-limit", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--use-lora", action="store_true", default=argparse.SUPPRESS)


def parse_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = parser.parse_args()
    values: dict[str, Any] = dict(DEFAULTS)
    values.update(load_base_configs())
    if args.config:
        values.update(load_yaml_mapping(args.config))
    cli_values = {key: value for key, value in vars(args).items() if key != "config"}
    values.update(cli_values)
    if values["train_file"] is None:
        parser.error("--train-file is required unless provided by --config")
    if values["output_dir"] is None:
        parser.error("--output-dir is required unless provided by --config")
    return argparse.Namespace(**values)


def training_args_dict(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_grad_norm": args.max_grad_norm,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "report_to": ["tensorboard"],
        "remove_unused_columns": False,
    }
    if args.warmup_steps:
        values["warmup_steps"] = args.warmup_steps
    else:
        values["warmup_ratio"] = args.warmup_ratio
    return values
