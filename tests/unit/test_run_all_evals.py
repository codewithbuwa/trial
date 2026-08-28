from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.experiments.run_all_evals import build_commands, validate_inputs


def test_build_commands_runs_all_local_eval_steps() -> None:
    args = argparse.Namespace(
        output_root=Path("output"),
        data_root=Path("data/processed"),
        eval_split="validation.jsonl",
        beta=0.02,
        batch_size=4,
        max_prompts=30,
        judge_timeout=60.0,
        openai_judge=False,
        openai_judge_model=None,
        prometheus_judge=False,
        prometheus_judge_model="prometheus-eval/prometheus-7b-v2.0",
        prometheus_base_url="http://localhost:8000/v1",
    )

    commands = build_commands(args)

    assert len(commands) == 11
    pairwise_accuracy_commands = [command for command in commands if "scripts/evaluate/evaluate_pairwise_accuracy.py" in command]
    assert len(pairwise_accuracy_commands) == 5
    assert all("--reference-model-name-or-path" in command for command in pairwise_accuracy_commands)
    unary_commands = [command for command in commands if "scripts/evaluate/evaluate_unary_rewards.py" in command]
    assert len(unary_commands) == 3
    judge_command = next(command for command in commands if "scripts/evaluate/evaluate_judge.py" in command)
    assert "SFT=output/sft" in judge_command
    assert "CPO_UNARY=output/cpo_unary" in judge_command
    assert "--judge-provider" in judge_command
    assert judge_command[judge_command.index("--judge-provider") + 1] == "heuristic"
    assert judge_command[judge_command.index("--max-prompts") + 1] == "30"
    assert judge_command[judge_command.index("--judge-timeout") + 1] == "60.0"
    assert any("scripts/experiments/plot_evals.py" in command for command in commands)
    assert any("scripts/evaluate/build_eval_report.py" in command for command in commands)
    report_command = commands[-1]
    assert report_command[report_command.index("--judge-summary") + 1] == (
        "output/judge/heuristic_summary.json"
    )


def test_build_commands_can_switch_to_openai_judge() -> None:
    args = argparse.Namespace(
        output_root=Path("output"),
        data_root=Path("data/processed"),
        eval_split="validation.jsonl",
        beta=0.02,
        batch_size=4,
        max_prompts=100,
        judge_timeout=120.0,
        openai_judge=True,
        openai_judge_model="gpt-test",
        prometheus_judge=False,
        prometheus_judge_model="prometheus-eval/prometheus-7b-v2.0",
        prometheus_base_url="http://localhost:8000/v1",
    )

    commands = build_commands(args)

    assert len(commands) == 11
    judge_commands = [command for command in commands if "scripts/evaluate/evaluate_judge.py" in command]
    assert len(judge_commands) == 1
    openai_command = judge_commands[0]
    assert openai_command[openai_command.index("--judge-provider") + 1] == "openai"
    assert openai_command[openai_command.index("--judge-model") + 1] == "gpt-test"
    assert openai_command[openai_command.index("--max-prompts") + 1] == "100"
    assert openai_command[openai_command.index("--judge-timeout") + 1] == "120.0"
    assert "heuristic" not in openai_command
    report_command = commands[-1]
    assert report_command[report_command.index("--judge-summary") + 1] == (
        "output/judge/summary.json"
    )


def test_build_commands_can_switch_to_prometheus_judge() -> None:
    args = argparse.Namespace(
        output_root=Path("output"),
        data_root=Path("data/processed"),
        eval_split="validation.jsonl",
        beta=0.02,
        batch_size=4,
        max_prompts=100,
        judge_timeout=500.0,
        openai_judge=False,
        openai_judge_model=None,
        prometheus_judge=True,
        prometheus_judge_model="prometheus-eval/prometheus-7b-v2.0",
        prometheus_base_url="http://localhost:8000/v1",
    )

    commands = build_commands(args)

    judge_command = next(command for command in commands if "scripts/evaluate/evaluate_judge.py" in command)
    assert judge_command[judge_command.index("--judge-provider") + 1] == "prometheus"
    assert judge_command[judge_command.index("--judge-model") + 1] == (
        "prometheus-eval/prometheus-7b-v2.0"
    )
    assert judge_command[judge_command.index("--openai-base-url") + 1] == (
        "http://localhost:8000/v1"
    )
    assert judge_command[judge_command.index("--judge-timeout") + 1] == "500.0"
    report_command = commands[-1]
    assert report_command[report_command.index("--judge-summary") + 1] == (
        "output/judge/prometheus_summary.json"
    )


def test_build_commands_rejects_multiple_remote_judges() -> None:
    args = argparse.Namespace(
        output_root=Path("output"),
        data_root=Path("data/processed"),
        eval_split="validation.jsonl",
        beta=0.02,
        batch_size=4,
        max_prompts=100,
        judge_timeout=60.0,
        openai_judge=True,
        openai_judge_model="gpt-test",
        prometheus_judge=True,
        prometheus_judge_model="prometheus-eval/prometheus-7b-v2.0",
        prometheus_base_url="http://localhost:8000/v1",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_commands(args)


def test_validate_inputs_fails_before_huggingface_fallback(tmp_path: Path) -> None:
    args = argparse.Namespace(
        output_root=tmp_path / "output",
        data_root=tmp_path / "data" / "ultrafeedback",
        eval_split="validation.jsonl",
    )

    with pytest.raises(FileNotFoundError, match="--output-root outputs"):
        validate_inputs(args)
