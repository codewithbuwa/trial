from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.evaluate.build_eval_report import build_report, distribution_stats, parse_named_paths


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_distribution_stats_reports_percentiles_and_rates() -> None:
    stats = distribution_stats([-1.0, 0.0, 1.0, 3.0])

    assert stats["count"] == 4
    assert stats["mean"] == pytest.approx(0.75)
    assert stats["median"] == pytest.approx(0.5)
    assert stats["positive_rate"] == pytest.approx(0.5)
    assert stats["tie_rate"] == pytest.approx(0.25)


def test_parse_named_paths_overrides_defaults() -> None:
    parsed = parse_named_paths(["SFT=/tmp/sft.json"], {"DPO": Path("dpo.json")})

    assert parsed == {"SFT": Path("/tmp/sft.json")}


def test_build_report_aggregates_eval_training_judge_and_flags(tmp_path: Path) -> None:
    output = tmp_path / "output"
    sft_dir = output / "sft"
    cpo_dir = output / "cpo"
    write_json(
        sft_dir / "pairwise_accuracy.json",
        {
            "model": "output/sft",
            "eval_file": "data/processed/dpo/validation.jsonl",
            "total": 2,
            "normalized_pairwise_accuracy": 0.4,
            "mean_normalized_margin": -0.1,
            "pairwise_accuracy": 0.5,
            "mean_margin": -2.0,
            "mean_chosen_length": 10.0,
            "mean_rejected_length": 20.0,
            "reward_accuracy": 0.5,
            "mean_reward_margin": 0.1,
            "normalized_reward_accuracy": 0.5,
            "mean_normalized_reward_margin": 0.01,
            "sampled_mean_kl": 5.0,
            "normalized_sampled_mean_kl": 0.2,
            "clusters": {
                "coding": {
                    "total": 2,
                    "normalized_pairwise_accuracy": 0.4,
                    "mean_normalized_margin": -0.1,
                    "reward_accuracy": 0.5,
                    "mean_reward_margin": 0.1,
                    "sampled_mean_kl": 5.0,
                }
            },
        },
    )
    write_jsonl(
        sft_dir / "pairwise_accuracy_margins.jsonl",
        [
            {
                "cluster_id": "coding",
                "margin": -2.0,
                "normalized_margin": -0.1,
                "reward_margin": -0.2,
                "normalized_reward_margin": -0.02,
                "sampled_mean_kl": 4.0,
                "normalized_sampled_mean_kl": 0.1,
                "chosen_length": 10,
                "rejected_length": 20,
            },
            {
                "cluster_id": "coding",
                "margin": 1.0,
                "normalized_margin": 0.2,
                "reward_margin": 0.4,
                "normalized_reward_margin": 0.04,
                "sampled_mean_kl": 6.0,
                "normalized_sampled_mean_kl": 0.3,
                "chosen_length": 11,
                "rejected_length": 21,
            },
        ],
    )
    write_json(
        sft_dir / "trainer_state.json",
        {
            "global_step": 2,
            "log_history": [
                {"step": 1, "loss": 1.2, "grad_norm": 0.5, "learning_rate": 1e-6},
                {"step": 2, "loss": 1.0, "grad_norm": 0.7, "learning_rate": 0.0},
            ],
        },
    )
    write_json(
        cpo_dir / "pairwise_accuracy.json",
        {
            "model": "output/cpo",
            "total": 2,
            "normalized_pairwise_accuracy": 0.6,
            "mean_normalized_margin": 0.2,
            "clusters": {},
        },
    )
    write_jsonl(cpo_dir / "pairwise_accuracy_margins.jsonl", [])
    write_jsonl(
        cpo_dir / "train_metrics.jsonl",
        [
            {
                "step": 1,
                "loss": 0.5,
                "unary_loss": 0.4,
                "pair_loss": 0.1,
                "num_pairs": 0,
                "grad_norm": 0.2,
            }
        ],
    )
    write_jsonl(
        cpo_dir / "cpo_diagnostics.jsonl",
        [
            {"event": "sampler", "baseline_ready_rate": 0.5},
            {"event": "final", "total_observed_pairs": 0, "z_k": {"coding": 0.1}},
        ],
    )
    write_json(
        output / "judge" / "summary.json",
        {"models": {"SFT": {"judge_score": 0.7}, "CPO": {"judge_score": 0.3}}},
    )
    write_json(
        output / "cpo" / "unary_rewards.json",
        {"reward_separation": -0.1, "normalized_reward_separation": -0.01},
    )

    report = build_report(
        result_paths={
            "SFT": sft_dir / "pairwise_accuracy.json",
            "CPO": cpo_dir / "pairwise_accuracy.json",
        },
        training_dirs={"SFT": sft_dir, "CPO": cpo_dir},
        unary_reward_paths={"CPO": output / "cpo" / "unary_rewards.json"},
        judge_summary=output / "judge" / "summary.json",
        judge_pairwise=output / "judge" / "pairwise.jsonl",
        output_root=output,
    )

    assert report["model_metrics"]["SFT"]["normalized_pairwise_accuracy"] == 0.4
    assert report["model_metrics"]["SFT"]["reward_accuracy"] == 0.5
    assert report["cluster_metrics"]["coding"]["SFT"]["normalized_pairwise_accuracy"] == 0.4
    assert report["margin_distribution"]["SFT"]["normalized_margin"]["count"] == 2
    assert report["margin_distribution"]["SFT"]["reward_margin"]["mean"] == pytest.approx(0.1)
    assert report["margin_distribution"]["SFT"]["normalized_reward_margin"]["mean"] == pytest.approx(0.01)
    assert report["margin_distribution"]["SFT"]["sampled_mean_kl"]["mean"] == pytest.approx(5.0)
    assert report["margin_distribution"]["SFT"]["normalized_sampled_mean_kl"]["mean"] == pytest.approx(0.2)
    assert report["unary_reward_metrics"]["CPO"]["summary"]["reward_separation"] == -0.1
    assert report["training_diagnostics"]["SFT"]["final_loss"] == 1.0
    assert report["training_diagnostics"]["CPO"]["final_diagnostic"]["z_k"] == {"coding": 0.1}
    flag_names = {flag["name"] for flag in report["diagnostic_flags"]}
    assert "normalized_pairwise_accuracy_below_half" in flag_names
    assert "large_length_gap" in flag_names
    assert "cpo_no_pairs" in flag_names
    assert "low_cpo_baseline_ready_rate" in flag_names
    assert "judge_logprob_disagreement" in flag_names
    assert "non_positive_unary_reward_separation" in flag_names
