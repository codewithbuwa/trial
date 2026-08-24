"""Fail-fast finite checks for losses and gradients."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch


class NonFiniteError(FloatingPointError):
    """Raised when a tensor contains NaN or infinite values."""


def assert_finite_tensor(tensor: torch.Tensor, name: str) -> None:
    """Raise when ``tensor`` contains a non-finite value."""

    if not torch.isfinite(tensor).all().item():
        raise NonFiniteError(f"non-finite values detected in {name}")


def assert_finite_loss(loss: torch.Tensor) -> None:
    assert_finite_tensor(loss.detach(), "loss")


def assert_finite_gradients(
    parameters: Iterable[torch.nn.Parameter] | Iterable[tuple[str, torch.nn.Parameter]],
) -> None:
    """Raise when any materialized gradient contains NaN or inf."""

    for index, item in enumerate(parameters):
        if isinstance(item, tuple):
            label, parameter = item
        else:
            label, parameter = f"gradient[{index}]", item
        if parameter.grad is not None:
            assert_finite_tensor(parameter.grad.detach(), label)


def non_finite_gradient_names(
    parameters: Iterable[torch.nn.Parameter] | Iterable[tuple[str, torch.nn.Parameter]],
) -> list[str]:
    """Return names for gradients containing NaN or inf."""

    names: list[str] = []
    for index, item in enumerate(parameters):
        if isinstance(item, tuple):
            label, parameter = item
        else:
            label, parameter = f"gradient[{index}]", item
        if parameter.grad is not None and not torch.isfinite(parameter.grad.detach()).all().item():
            names.append(label)
    return names


def zero_gradients(parameters: Iterable[torch.nn.Parameter]) -> None:
    """Zero materialized gradients in-place."""

    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.detach().zero_()


try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - transformers may be absent in unit-only envs
    TrainerCallback = object  # type: ignore[assignment,misc]


class FiniteTrainingCallback(TrainerCallback):
    """Transformers callback that handles non-finite gradients."""

    def __init__(
        self,
        *,
        fail_fast: bool = True,
        log_filename: str = "non_finite_gradients.jsonl",
    ) -> None:
        self.fail_fast = fail_fast
        self.log_filename = log_filename
        self.skipped_steps = 0

    def _write_record(self, args, record: dict[str, object]) -> None:  # type: ignore[no-untyped-def]
        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            return
        path = Path(output_dir) / self.log_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):  # type: ignore[no-untyped-def]
        model = kwargs.get("model")
        if model is None:
            return control
        names = non_finite_gradient_names(model.named_parameters())
        if not names:
            return control
        if self.fail_fast:
            raise NonFiniteError(f"non-finite values detected in {names[0]}")
        self.skipped_steps += 1
        joined = ", ".join(names[:3])
        if len(names) > 3:
            joined += f", ... +{len(names) - 3} more"
        record = {
            "warning": "skipping optimizer step with non-finite gradients",
            "step": state.global_step,
            "epoch": getattr(state, "epoch", None),
            "count": len(names),
            "parameters": names,
            "parameters_preview": joined,
            "skipped_steps": self.skipped_steps,
        }
        print(record)
        self._write_record(args, record)
        zero_gradients(model.parameters())
        return control
