from __future__ import annotations

import json
from argparse import Namespace

import torch

from cpo_trl.finite import FiniteTrainingCallback


def test_finite_callback_persists_skipped_step(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    next(model.parameters()).grad = torch.full_like(next(model.parameters()), float("nan"))
    callback = FiniteTrainingCallback(fail_fast=False)

    callback.on_pre_optimizer_step(
        Namespace(output_dir=str(tmp_path)),
        Namespace(global_step=7, epoch=0.5),
        object(),
        model=model,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "non_finite_gradients.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["step"] == 7
    assert records[0]["skipped_steps"] == 1
    assert records[0]["parameters"] == ["weight"]
