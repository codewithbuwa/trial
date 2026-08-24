"""PEFT configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoraSettings:
    r: int = 32
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: str | tuple[str, ...] = "all-linear"
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


def build_lora_config(settings: LoraSettings | None = None):
    """Build a PEFT LoRA config from local defaults."""

    from peft import LoraConfig

    settings = settings or LoraSettings()
    return LoraConfig(
        r=settings.r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=(
            settings.target_modules
            if isinstance(settings.target_modules, str)
            else list(settings.target_modules)
        ),
        bias=settings.bias,
        task_type=settings.task_type,
    )


def is_adapter_checkpoint(model_name_or_path: str | Path) -> bool:
    """Return whether a path looks like a PEFT adapter checkpoint."""

    return (Path(model_name_or_path) / "adapter_config.json").exists()


def load_causal_lm_for_training(
    model_name_or_path: str | Path,
    *,
    use_lora: bool,
    create_lora: bool = False,
):
    """Load a causal LM, including PEFT adapter checkpoints.

    When ``model_name_or_path`` points at a LoRA adapter directory, this loads
    the base model named by the adapter config and attaches the adapter. For a
    plain model path or Hub id, this loads the model normally. Set
    ``create_lora=True`` only for custom training loops that need this helper to
    wrap the model directly instead of passing ``peft_config`` to TRL.
    """

    if is_adapter_checkpoint(model_name_or_path):
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_name_or_path,
            is_trainable=use_lora,
        )
        ensure_transformers_warning_state(model)
        return model

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    if use_lora and create_lora:
        from peft import get_peft_model

        model = get_peft_model(model, build_lora_config())
    ensure_transformers_warning_state(model)
    return model


def peft_config_for_new_adapter(model_name_or_path: str | Path, *, use_lora: bool):
    """Return a LoRA config only when training should create a new adapter."""

    if not use_lora or is_adapter_checkpoint(model_name_or_path):
        return None
    return build_lora_config()


def ensure_transformers_warning_state(model):
    """Attach the Transformers warning state expected by current TRL trainers."""

    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    return model
