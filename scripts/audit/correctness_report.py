from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CorrectnessCheck:
    name: str
    question: str
    pytest_target: str


CHECKS = (
    CorrectnessCheck(
        name="alpha_endpoints",
        question="CPO loss reduces to unary at alpha=0 and pairwise at alpha=1.",
        pytest_target="tests/unit/test_losses.py",
    ),
    CorrectnessCheck(
        name="pair_construction",
        question="Pairwise examples require same prompt, same cluster, one desirable, one undesirable.",
        pytest_target="tests/unit/test_losses.py::test_derived_pair_indices_group_by_prompt_and_cluster",
    ),
    CorrectnessCheck(
        name="ema_state",
        question="Cluster reference z_k updates and round-trips through state dicts.",
        pytest_target="tests/unit/test_cpo_trainer.py::test_cpo_loss_computer_state_roundtrip",
    ),
    CorrectnessCheck(
        name="finite_values",
        question="Losses and gradients fail fast on NaN/Inf.",
        pytest_target="tests/unit/test_finite.py",
    ),
    CorrectnessCheck(
        name="pair_aware_sampler",
        question="Pair-aware batching preserves valid pairs and reports coverage.",
        pytest_target="tests/unit/test_sampling.py",
    ),
)


def run_check(target: str) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", target],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "target": target,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def build_report(*, run: bool) -> dict[str, Any]:
    checks = []
    for check in CHECKS:
        record = asdict(check)
        if run:
            record["result"] = run_check(check.pytest_target)
        checks.append(record)
    return {
        "experiment": "E0_correctness",
        "purpose": "Verify that the implementation matches the intended CPO formulation before training.",
        "checks": checks,
        "all_passed": all(check.get("result", {}).get("passed", True) for check in checks),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the E0 CPO correctness report.")
    parser.add_argument("--run", action="store_true", help="Execute the pytest checks.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/E0_correctness/results/correctness_report.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(run=args.run)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.run and not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
