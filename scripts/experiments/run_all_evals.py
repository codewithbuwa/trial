from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


MODELS = ("dpo", "kto", "cpo")


def validate_inputs(args: argparse.Namespace) -> None:
    missing_paths = []
    for model in MODELS:
        checkpoint_dir = args.output_root / model
        if not checkpoint_dir.is_dir():
            missing_paths.append(str(checkpoint_dir))
    for kind in ("dpo", "kto", "cpo"):
        eval_file = args.data_root / kind / args.eval_split
        if not eval_file.is_file():
            missing_paths.append(str(eval_file))
    if missing_paths:
        joined = "\n  - ".join(missing_paths)
        raise FileNotFoundError(
            "run_all_evals.py could not find required local checkpoints/data files:\n"
            f"  - {joined}\n\n"
            "Check the root directories, for example:\n"
            f"  ls -lh {args.output_root}\n"
            f"  ls -lh {args.data_root}\n\n"
            "If your checkpoints are under outputs/ instead of output/, rerun with:\n"
            "  --output-root outputs"
        )


def command_to_text(command: list[str]) -> str:
    return " ".join(command)


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    output_root = args.output_root
    data_root = args.data_root
    artifact_root = output_root.parent if output_root.name == "checkpoints" else output_root
    reference_model_name_or_path = getattr(args, "reference_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    if args.openai_judge and args.prometheus_judge:
        raise ValueError("--openai-judge and --prometheus-judge are mutually exclusive")
    if args.openai_judge:
        judge_provider = "openai"
        judge_summary = artifact_root / "judge" / "summary.json"
        judge_pairwise = artifact_root / "judge" / "pairwise.jsonl"
    elif args.prometheus_judge:
        judge_provider = "prometheus"
        judge_summary = artifact_root / "judge" / "prometheus_summary.json"
        judge_pairwise = artifact_root / "judge" / "prometheus_pairwise.jsonl"
    else:
        judge_provider = "heuristic"
        judge_summary = artifact_root / "judge" / "heuristic_summary.json"
        judge_pairwise = artifact_root / "judge" / "heuristic_pairwise.jsonl"
    commands: list[list[str]] = []
    for model in MODELS:
        commands.append(
            [
                "poetry",
                "run",
                "python",
                "scripts/evaluate/evaluate_winrate.py",
                "--eval-file",
                str(data_root / "dpo" / args.eval_split),
                "--model-name-or-path",
                str(output_root / model),
                "--reference-model-name-or-path",
                reference_model_name_or_path,
                "--beta",
                str(args.beta),
                "--batch-size",
                str(args.batch_size),
                "--output-json",
                str(output_root / model / "winrate.json"),
            ]
        )
    commands.extend(
        [
            [
                "poetry",
                "run",
                "python",
                "scripts/evaluate/evaluate_unary_rewards.py",
                "--eval-file",
                str(data_root / "kto" / args.eval_split),
                "--row-kind",
                "kto",
                "--model-name-or-path",
                str(output_root / "kto"),
                "--reference-model-name-or-path",
                reference_model_name_or_path,
                "--beta",
                str(args.beta),
                "--batch-size",
                str(args.batch_size),
                "--output-json",
                str(output_root / "kto" / "unary_rewards.json"),
            ],
            [
                "poetry",
                "run",
                "python",
                "scripts/evaluate/evaluate_unary_rewards.py",
                "--eval-file",
                str(data_root / "cpo" / args.eval_split),
                "--row-kind",
                "cpo",
                "--model-name-or-path",
                str(output_root / "cpo"),
                "--reference-model-name-or-path",
                reference_model_name_or_path,
                "--beta",
                str(args.beta),
                "--batch-size",
                str(args.batch_size),
                "--output-json",
                str(output_root / "cpo" / "unary_rewards.json"),
            ],
        ]
    )
    judge_command = [
        "poetry",
        "run",
        "python",
        "scripts/evaluate/evaluate_judge.py",
        "--eval-file",
        str(data_root / "dpo" / args.eval_split),
        "--models",
        f"DPO={output_root / 'dpo'}",
        f"KTO={output_root / 'kto'}",
        f"CPO={output_root / 'cpo'}",
        "--max-prompts",
        str(args.max_prompts),
        "--judge-provider",
        judge_provider,
        "--output-jsonl",
        str(judge_pairwise),
        "--summary-json",
        str(judge_summary),
        "--judge-timeout",
        str(args.judge_timeout),
        "--position-balanced",
    ]
    if args.openai_judge:
        judge_model = args.openai_judge_model or os.environ.get("OPENAI_JUDGE_MODEL")
        if not judge_model:
            raise ValueError(
                "--openai-judge-model or OPENAI_JUDGE_MODEL is required with --openai-judge"
            )
        judge_command.extend(["--judge-model", judge_model])
    if args.prometheus_judge:
        if not args.prometheus_judge_model:
            raise ValueError("--prometheus-judge-model is required with --prometheus-judge")
        judge_command.extend(
            [
                "--judge-model",
                args.prometheus_judge_model,
                "--openai-base-url",
                args.prometheus_base_url,
            ]
        )
    commands.append(judge_command)
    commands.extend(
        [
            [
                "poetry",
                "run",
                "python",
                "scripts/experiments/plot_evals.py",
                "--result",
                f"DPO={output_root / 'dpo' / 'winrate.json'}",
                "--result",
                f"KTO={output_root / 'kto' / 'winrate.json'}",
                "--result",
                f"CPO={output_root / 'cpo' / 'winrate.json'}",
                "--training-dir",
                f"DPO={output_root / 'dpo'}",
                "--training-dir",
                f"KTO={output_root / 'kto'}",
                "--training-dir",
                f"CPO={output_root / 'cpo'}",
                "--output-dir",
                str(artifact_root / "metrics"),
            ],
            [
                "poetry",
                "run",
                "python",
                "scripts/evaluate/build_eval_report.py",
                "--output-root",
                str(output_root),
                "--judge-summary",
                str(judge_summary),
                "--judge-pairwise",
                str(judge_pairwise),
                "--output-json",
                str(artifact_root / "metrics" / "evaluation_report.json"),
            ],
        ]
    )
    return commands


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local evaluation steps, with an optional OpenAI judge pass."
    )
    parser.add_argument("--output-root", type=Path, default=Path("outputs/checkpoints"))
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--eval-split", default="validation.jsonl")
    parser.add_argument("--reference-model-name-or-path", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=30)
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    parser.add_argument("--openai-judge", action="store_true")
    parser.add_argument("--openai-judge-model", default=os.environ.get("OPENAI_JUDGE_MODEL"))
    parser.add_argument("--prometheus-judge", action="store_true")
    parser.add_argument(
        "--prometheus-judge-model",
        default=os.environ.get("PROMETHEUS_JUDGE_MODEL", "prometheus-eval/prometheus-7b-v2.0"),
    )
    parser.add_argument(
        "--prometheus-base-url",
        default=os.environ.get("PROMETHEUS_BASE_URL", "http://localhost:8000/v1"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        validate_inputs(args)
    commands = build_commands(args)
    for index, command in enumerate(commands, start=1):
        print(f"\n[{index}/{len(commands)}] {command_to_text(command)}", flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, check=False)
        if completed.returncode and not args.continue_on_failure:
            sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
