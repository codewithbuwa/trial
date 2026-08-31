from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from scripts.experiments.run_preference_sweeps import (
    build_sweep_configs,
    parse_float_list,
    write_results_summary,
    write_sweep,
)


def write_config(path: Path, method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "train_file": f"data/processed/{method}/train.jsonl",
                "model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
                "output_dir": f"outputs/{method}",
                "learning_rate": 1e-6,
                "beta": 0.02,
                "use_lora": True,
            }
        ),
        encoding="utf-8",
    )


def test_parse_float_list() -> None:
    assert parse_float_list("1e-6, 2e-6") == [1e-6, 2e-6]


def test_build_sweep_configs_adds_cpo_alpha_only(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    (config_dir / "base").mkdir(parents=True)
    (config_dir / "base" / "training.yaml").write_text("weight_decay: 0.123\n", encoding="utf-8")
    for method in ("dpo", "cpo"):
        write_config(config_dir / method / f"{method}_controlled.yaml", method)
    args = argparse.Namespace(
        methods=["dpo", "cpo"],
        config_dir=config_dir,
        train_root=Path("data/processed"),
        split="train.jsonl",
        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir=tmp_path / "sweeps",
        learning_rates=[1e-6],
        betas=[0.02],
        alphas=[0.3],
        max_grad_norms=[0.3, 1.0],
        weight_decay=None,
        z_baselines=["token_kl"],
        num_train_epochs=0.1,
        max_seq_length=512,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        logging_steps=5,
        terminal_log_steps=None,
        save_steps=500,
        save_total_limit=1,
        warmup_steps=10,
        seed=42,
    )

    runs = build_sweep_configs(args)

    assert len(runs) == 4
    dpo_runs = [run for run in runs if run["method"] == "dpo"]
    cpo_runs = [run for run in runs if run["method"] == "cpo"]
    assert all(run["alpha"] is None for run in dpo_runs)
    assert all("alpha" not in run["config"] for run in dpo_runs)
    assert {run["max_grad_norm"] for run in dpo_runs} == {0.3, 1.0}
    assert {run["alpha"] for run in cpo_runs} == {0.3}
    assert {run["z_baseline"] for run in cpo_runs} == {"token_kl"}
    assert all(f"scripts/train/train_{run['method']}.py" in run["command"] for run in runs)
    assert all(run["config"]["weight_decay"] == 0.123 for run in runs)
    assert all("terminal_log_steps" not in run["config"] for run in runs)


def test_build_sweep_configs_can_set_sparse_terminal_logging(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    (config_dir / "base").mkdir(parents=True)
    write_config(config_dir / "kto" / "kto_controlled.yaml", "kto")
    args = argparse.Namespace(
        methods=["kto"],
        config_dir=config_dir,
        train_root=Path("data/processed"),
        split="train.jsonl",
        model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
        output_dir=tmp_path / "sweeps",
        learning_rates=[1e-5],
        betas=[0.01],
        alphas=None,
        max_grad_norms=[0.3],
        weight_decay=None,
        z_baselines=None,
        num_train_epochs=1.0,
        max_seq_length=512,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=5,
        terminal_log_steps=500,
        save_steps=500,
        save_total_limit=1,
        warmup_steps=10,
        seed=42,
    )

    runs = build_sweep_configs(args)

    assert len(runs) == 1
    assert runs[0]["config"]["logging_steps"] == 5
    assert runs[0]["config"]["terminal_log_steps"] == 500


def test_write_sweep_creates_manifest_and_configs(tmp_path: Path) -> None:
    runs = [
        {
            "method": "dpo",
            "run_name": "dpo_lr1em6_b0p02",
            "config": {"train_file": "data/processed/dpo/train.jsonl"},
            "config_path": tmp_path / "configs" / "dpo_lr1em6_b0p02.yaml",
            "output_dir": tmp_path / "dpo_lr1em6_b0p02",
            "command": ["poetry", "run", "python", "scripts/train/train_dpo.py", "--config", "x"],
            "learning_rate": 1e-6,
            "beta": 0.02,
            "alpha": None,
            "max_grad_norm": 0.3,
            "z_baseline": None,
        }
    ]

    manifest_path = write_sweep(runs, tmp_path)

    assert manifest_path.exists()
    assert (tmp_path / "configs" / "dpo_lr1em6_b0p02.yaml").exists()
    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == 1


def test_write_results_summary_ranks_best_by_method(tmp_path: Path) -> None:
    results = [
        {
            "method": "dpo",
            "run_name": "dpo_low",
            "status": "ok",
            "normalized_reward_accuracy": 0.51,
            "mean_normalized_reward_margin": 0.01,
            "normalized_pairwise_accuracy": 0.52,
        },
        {
            "method": "dpo",
            "run_name": "dpo_high",
            "status": "ok",
            "normalized_reward_accuracy": 0.54,
            "mean_normalized_reward_margin": 0.005,
            "normalized_pairwise_accuracy": 0.53,
        },
        {
            "method": "cpo",
            "run_name": "cpo_failed",
            "status": "train_failed",
            "normalized_reward_accuracy": 0.9,
        },
    ]

    results_path, best_path = write_results_summary(results, tmp_path, "normalized_reward_accuracy")

    assert len(results_path.read_text(encoding="utf-8").splitlines()) == 3
    best = json.loads(best_path.read_text(encoding="utf-8"))
    assert best["dpo"]["run_name"] == "dpo_high"
    assert "cpo" not in best
