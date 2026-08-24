from __future__ import annotations

from cpo_trl.peft import ensure_transformers_warning_state


class DummyModel:
    pass


def test_ensure_transformers_warning_state_adds_missing_attribute() -> None:
    model = DummyModel()

    returned = ensure_transformers_warning_state(model)

    assert returned is model
    assert model.warnings_issued == {}


def test_ensure_transformers_warning_state_preserves_existing_attribute() -> None:
    model = DummyModel()
    model.warnings_issued = {"estimate_tokens": True}

    ensure_transformers_warning_state(model)

    assert model.warnings_issued == {"estimate_tokens": True}
