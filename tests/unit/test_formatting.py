from __future__ import annotations

import pytest

from cpo_trl.data import JsonlSchemaError, validate_rows
from cpo_trl.formatting import format_preference_row, format_sft_text


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        rendered = "|".join(f"{message['role']}:{message['content']}" for message in messages)
        if add_generation_prompt:
            rendered += "|assistant:"
        return rendered


def test_sft_formatting_uses_chat_template() -> None:
    text = format_sft_text(
        DummyTokenizer(),
        {"instruction": "Say hi", "input": "", "chosen": "Hi"},
    )
    assert text == "user:Say hi|assistant:Hi"


def test_preference_formatting_preserves_cluster() -> None:
    row = {
        "prompt_id": "1",
        "instruction": "Say hi",
        "input": "",
        "chosen": "Hi",
        "rejected": "No",
        "cluster_id": "general",
    }
    formatted = format_preference_row(DummyTokenizer(), row)
    assert formatted["prompt"] == "user:Say hi|assistant:"
    assert formatted["chosen"] == "Hi"
    assert formatted["rejected"] == "No"
    assert formatted["cluster_id"] == "general"


def test_row_schema_validation() -> None:
    validate_rows([{"instruction": "x", "chosen": "y"}], "sft")
    validate_rows([{"instruction": "x", "chosen": "y", "rejected": "z"}], "dpo")
    validate_rows([{"instruction": "x", "completion": "y", "label": True}], "kto")
    validate_rows(
        [{"prompt_id": "1", "instruction": "x", "completion": "y", "label": True, "cluster_id": "c"}],
        "cpo",
    )
    with pytest.raises(JsonlSchemaError):
        validate_rows([{"instruction": "x", "completion": "y", "label": "true"}], "kto")
    with pytest.raises(JsonlSchemaError):
        validate_rows(
            [{"prompt_id": "1", "instruction": "x", "completion": "y", "label": "true", "cluster_id": "c"}],
            "cpo",
        )
