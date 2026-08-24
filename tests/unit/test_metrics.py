from __future__ import annotations

from cpo_trl.metrics import grouped_preference_record


def test_grouped_preference_record_groups_dpo_logs() -> None:
    record = grouped_preference_record(
        method="dpo",
        beta=0.02,
        global_step=10,
        epoch=0.5,
        logs={
            "loss": "0.693",
            "grad_norm": "5.987",
            "learning_rate": "1.992e-07",
            "entropy": "1.378",
            "num_tokens": "6.471e+04",
            "rewards/chosen": "0.001099",
            "rewards/rejected": "-7.138e-05",
            "rewards/accuracies": "0.4",
            "rewards/margins": "0.001171",
            "logps/chosen": "-375.8",
            "logps/rejected": "-340.8",
            "mean_token_accuracy": "0.6492",
        },
    )

    assert record["method"] == "dpo"
    assert record["step"] == 10
    assert record["objective"]["loss"] == 0.693
    assert record["objective"]["beta"] == 0.02
    assert record["preference_signal"]["reward_margin"] == 0.001171
    assert record["preference_signal"]["chosen_logp_mean"] == -375.8
    assert record["optimization"]["learning_rate"] == 1.992e-07
    assert record["run_state"]["global_step"] == 10


def test_grouped_preference_record_omits_missing_optional_metrics() -> None:
    record = grouped_preference_record(
        method="kto",
        beta=0.1,
        global_step=3,
        epoch=None,
        logs={"loss": 0.5},
    )

    assert record["method"] == "kto"
    assert record["objective"] == {"loss": 0.5, "beta": 0.1}
    assert record["preference_signal"] == {}
    assert record["optimization"] == {}
    assert record["run_state"] == {"global_step": 3}
