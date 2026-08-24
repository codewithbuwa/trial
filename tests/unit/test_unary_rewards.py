from __future__ import annotations

import pytest

from scripts.evaluate.evaluate_unary_rewards import infer_row_kind, summarize_records


def test_infer_row_kind_uses_cluster_presence() -> None:
    assert infer_row_kind([{"completion": "yes", "label": True}]) == "kto"
    assert infer_row_kind([{"completion": "yes", "label": True, "cluster_id": "coding"}]) == "cpo"


def test_summarize_records_reports_reward_separation_and_lengths() -> None:
    summary = summarize_records(
        [
            {
                "label": True,
                "cluster_id": "coding",
                "reward": 0.4,
                "normalized_reward": 0.04,
                "sampled_kl": 20.0,
                "normalized_sampled_kl": 2.0,
                "completion_length": 10,
            },
            {
                "label": False,
                "cluster_id": "coding",
                "reward": -0.2,
                "normalized_reward": -0.02,
                "sampled_kl": -10.0,
                "normalized_sampled_kl": -1.0,
                "completion_length": 5,
            },
        ]
    )

    assert summary["mean_desirable_reward"] == pytest.approx(0.4)
    assert summary["mean_undesirable_reward"] == pytest.approx(-0.2)
    assert summary["reward_separation"] == pytest.approx(0.6)
    assert summary["normalized_reward_separation"] == pytest.approx(0.06)
    assert summary["mean_desirable_length"] == pytest.approx(10.0)
    assert summary["mean_undesirable_length"] == pytest.approx(5.0)
    assert summary["clusters"]["coding"]["reward_separation"] == pytest.approx(0.6)
