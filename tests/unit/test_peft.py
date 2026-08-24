from __future__ import annotations

from argparse import Namespace

from cpo_trl.peft import LoraSettings, ensure_transformers_warning_state, lora_settings_from_config


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


def test_lora_settings_defaults_match_base_config_shape() -> None:
    settings = LoraSettings()

    assert settings.r == 16
    assert settings.lora_alpha == 32
    assert settings.lora_dropout == 0.05
    assert settings.target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def test_lora_settings_from_config_accepts_yaml_lists() -> None:
    settings = lora_settings_from_config(
        Namespace(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
    )

    assert settings.r == 8
    assert settings.lora_alpha == 16
    assert settings.lora_dropout == 0.1
    assert settings.target_modules == ("q_proj", "v_proj")
