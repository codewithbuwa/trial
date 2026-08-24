"""Grouped training metric logging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def maybe_float(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def grouped_preference_record(
    *,
    method: str,
    logs: dict[str, Any],
    global_step: int,
    epoch: float | None,
    beta: float,
) -> dict[str, Any]:
    """Convert flat TRL DPO/KTO logs into a grouped JSONL record."""

    return {
        "step": global_step,
        "epoch": maybe_float(logs.get("epoch", epoch)),
        "method": method,
        "objective": compact_dict(
            {
                "loss": maybe_float(logs.get("loss")),
                "beta": beta,
                "entropy": maybe_float(logs.get("entropy")),
                "num_tokens": maybe_float(logs.get("num_tokens")),
            }
        ),
        "preference_signal": compact_dict(
            {
                "reward_margin": maybe_float(logs.get("rewards/margins")),
                "reward_accuracy": maybe_float(logs.get("rewards/accuracies")),
                "chosen_reward_mean": maybe_float(logs.get("rewards/chosen")),
                "rejected_reward_mean": maybe_float(logs.get("rewards/rejected")),
                "chosen_logp_mean": maybe_float(logs.get("logps/chosen")),
                "rejected_logp_mean": maybe_float(logs.get("logps/rejected")),
                "chosen_logit_mean": maybe_float(logs.get("logits/chosen")),
                "rejected_logit_mean": maybe_float(logs.get("logits/rejected")),
                "mean_token_accuracy": maybe_float(logs.get("mean_token_accuracy")),
            }
        ),
        "optimization": compact_dict(
            {
                "grad_norm": maybe_float(logs.get("grad_norm")),
                "learning_rate": maybe_float(logs.get("learning_rate")),
            }
        ),
        "run_state": compact_dict(
            {
                "global_step": global_step,
                "epoch": maybe_float(logs.get("epoch", epoch)),
            }
        ),
    }


try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - transformers may be absent in unit-only envs
    TrainerCallback = object  # type: ignore[assignment,misc]


class GroupedPreferenceMetricsCallback(TrainerCallback):
    """Write grouped DPO/KTO train metrics from Transformers log events."""

    def __init__(
        self,
        *,
        method: str,
        beta: float,
        filename: str = "train_metrics_grouped.jsonl",
    ) -> None:
        self.method = method
        self.beta = beta
        self.filename = filename

    def on_log(self, args, state, control, logs=None, **kwargs):  # type: ignore[no-untyped-def]
        del control, kwargs
        if logs is None or "loss" not in logs:
            return
        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            return
        path = Path(output_dir) / self.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        record = grouped_preference_record(
            method=self.method,
            logs=dict(logs),
            global_step=int(getattr(state, "global_step", 0)),
            epoch=getattr(state, "epoch", None),
            beta=self.beta,
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
