"""Formatting helpers for chat-style TRL datasets."""

from __future__ import annotations

from typing import Any


def build_user_content(instruction: str, input_text: str | None = None) -> str:
    """Join instruction and optional input into a single user message."""

    input_text = (input_text or "").strip()
    if not input_text:
        return instruction.strip()
    return f"{instruction.strip()}\n\n{input_text}"


def chat_messages(
    instruction: str,
    response: str | None = None,
    *,
    input_text: str | None = None,
) -> list[dict[str, str]]:
    """Build OpenAI-style chat messages for tokenizer templates."""

    messages = [{"role": "user", "content": build_user_content(instruction, input_text)}]
    if response is not None:
        messages.append({"role": "assistant", "content": response.strip()})
    return messages


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = False,
    tokenize: bool = False,
) -> Any:
    """Apply a tokenizer chat template with a deterministic fallback."""

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
    if tokenize:
        raise TypeError("tokenize=True requires a tokenizer with apply_chat_template")
    rendered = ""
    for message in messages:
        rendered += f"{message['role'].upper()}: {message['content']}\n"
    if add_generation_prompt:
        rendered += "ASSISTANT: "
    return rendered


def format_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    """Format the prompt side of a preference row."""

    return apply_chat_template(
        tokenizer,
        chat_messages(row["instruction"], input_text=row.get("input", "")),
        add_generation_prompt=True,
        tokenize=False,
    )


def format_sft_text(tokenizer: Any, row: dict[str, Any]) -> str:
    """Format a supervised fine-tuning row as prompt plus assistant answer."""

    return apply_chat_template(
        tokenizer,
        chat_messages(row["instruction"], row["chosen"], input_text=row.get("input", "")),
        add_generation_prompt=False,
        tokenize=False,
    )


def format_preference_row(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Format a chosen/rejected preference row for TRL preference trainers."""

    formatted = {
        "prompt": format_prompt(tokenizer, row),
        "chosen": row["chosen"],
        "rejected": row["rejected"],
    }
    if "cluster_id" in row:
        formatted["cluster_id"] = row["cluster_id"]
    if "prompt_id" in row:
        formatted["prompt_id"] = row["prompt_id"]
    return formatted


def format_kto_row(tokenizer: Any, row: dict[str, Any]) -> dict[str, Any]:
    """Format a KTO unary row for TRL."""

    return {
        "prompt": format_prompt(tokenizer, row),
        "completion": row["completion"],
        "label": row["label"],
    }
