from __future__ import annotations

import torch

from cpo_trl.eval import (
    collate_mismatched_unary_batch,
    collate_pair_batch,
    collate_unary_batch,
    encode_pair,
    encode_unary,
    pair_reward_margins,
    sequence_logp_sums_and_counts,
    sequence_logps_with_token_kl,
)


class DummyTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(
        self,
        text: str,
        *,
        truncation: bool = False,
        max_length: int | None = None,
        add_special_tokens: bool = False,
    ) -> dict[str, list[int]]:
        del add_special_tokens
        ids = [ord(char) % 251 + 1 for char in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids}


def test_mismatched_batch_rotates_within_cluster_across_prompts() -> None:
    batch = collate_mismatched_unary_batch(
        DummyTokenizer(),
        prompt_texts=["prompt-a:", "prompt-a:", "prompt-b:", "prompt-b:"],
        completions=["a+", "a-", "b+", "b-"],
        prompt_ids=["a", "a", "b", "b"],
        cluster_ids=["coding", "coding", "coding", "coding"],
        max_length=64,
    )
    assert batch is not None
    assert batch["input_ids"].shape[0] == 4
    assert batch["cluster_ids"] == ["coding", "coding", "coding", "coding"]


def test_mismatched_batch_skips_clusters_without_different_prompt() -> None:
    batch = collate_mismatched_unary_batch(
        DummyTokenizer(),
        prompt_texts=["prompt-a:", "prompt-a:"],
        completions=["a+", "a-"],
        prompt_ids=["a", "a"],
        cluster_ids=["coding", "coding"],
        max_length=64,
    )
    assert batch is None


def test_unary_attention_mask_uses_lengths_not_pad_token_values() -> None:
    batch = collate_unary_batch(
        DummyTokenizer(),
        [
            {
                "input_ids": [7, 0, 8],
                "response_start": 1,
                "label": True,
                "cluster_id": "coding",
                "row_index": 0,
                "prompt_id": "p0",
                "prompt_text": "p",
                "completion": "c",
            },
            {
                "input_ids": [9],
                "response_start": 0,
                "label": False,
                "cluster_id": "coding",
                "row_index": 1,
                "prompt_id": "p1",
                "prompt_text": "p",
                "completion": "c",
            },
        ],
    )

    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_unary_collator_infers_adjacent_pair_indices() -> None:
    rows = [
        {
            "input_ids": [1, 2],
            "response_start": 1,
            "label": True,
            "cluster_id": "coding",
            "row_index": 10,
            "prompt_id": "p0",
            "prompt_text": "p",
            "completion": "chosen",
        },
        {
            "input_ids": [3, 4],
            "response_start": 1,
            "label": False,
            "cluster_id": "coding",
            "row_index": 11,
            "prompt_id": "p0",
            "prompt_text": "p",
            "completion": "rejected",
        },
        {
            "input_ids": [5, 6],
            "response_start": 1,
            "label": False,
            "cluster_id": "math",
            "row_index": 12,
            "prompt_id": "p1",
            "prompt_text": "p",
            "completion": "rejected",
        },
        {
            "input_ids": [7, 8],
            "response_start": 1,
            "label": True,
            "cluster_id": "math",
            "row_index": 13,
            "prompt_id": "p1",
            "prompt_text": "p",
            "completion": "chosen",
        },
    ]

    batch = collate_unary_batch(DummyTokenizer(), rows)

    assert batch["pair_indices"].tolist() == [[0, 1], [3, 2]]


def test_pair_attention_mask_uses_lengths_not_pad_token_values() -> None:
    batch = collate_pair_batch(
        DummyTokenizer(),
        [
            {
                "chosen_input_ids": [7, 0, 8],
                "chosen_response_start": 1,
                "rejected_input_ids": [9, 0],
                "rejected_response_start": 1,
                "cluster_id": "coding",
                "row_index": 0,
                "prompt_id": "p0",
            },
            {
                "chosen_input_ids": [10],
                "chosen_response_start": 0,
                "rejected_input_ids": [11],
                "rejected_response_start": 0,
                "cluster_id": "coding",
                "row_index": 1,
                "prompt_id": "p1",
            },
        ],
    )

    assert batch["chosen_attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert batch["rejected_attention_mask"].tolist() == [[1, 1], [1, 0]]


def test_encode_unary_preserves_response_token_when_prompt_is_truncated() -> None:
    row = {
        "instruction": "this prompt is intentionally much longer than the sequence budget",
        "input": "",
        "completion": "yes",
        "label": True,
        "prompt_id": "p0",
        "cluster_id": "general",
    }

    encoded = encode_unary(DummyTokenizer(), row, max_length=8)

    assert len(encoded["input_ids"]) == 8
    assert encoded["response_start"] < len(encoded["input_ids"])


def test_encode_pair_preserves_response_token_when_prompt_is_truncated() -> None:
    row = {
        "instruction": "this prompt is intentionally much longer than the sequence budget",
        "input": "",
        "chosen": "yes",
        "rejected": "no",
        "prompt_id": "p0",
        "cluster_id": "general",
    }

    encoded = encode_pair(DummyTokenizer(), row, max_length=8)

    assert encoded["chosen_response_start"] < len(encoded["chosen_input_ids"])
    assert encoded["rejected_response_start"] < len(encoded["rejected_input_ids"])


def test_encode_unary_appends_eos_to_completion() -> None:
    row = {
        "instruction": "prompt",
        "input": "",
        "completion": "yes",
        "label": True,
        "prompt_id": "p0",
        "cluster_id": "general",
    }

    encoded = encode_unary(DummyTokenizer(), row, max_length=128)
    eos_ids = DummyTokenizer()("<eos>", add_special_tokens=False)["input_ids"]

    assert encoded["input_ids"][-len(eos_ids) :] == eos_ids


def test_encode_unary_does_not_duplicate_existing_eos() -> None:
    row = {
        "instruction": "prompt",
        "input": "",
        "completion": "yes<eos>",
        "label": True,
        "prompt_id": "p0",
        "cluster_id": "general",
    }

    encoded = encode_unary(DummyTokenizer(), row, max_length=128)
    eos_ids = DummyTokenizer()("<eos>", add_special_tokens=False)["input_ids"]

    assert encoded["input_ids"][-len(eos_ids) :] == eos_ids
    assert encoded["input_ids"][-2 * len(eos_ids) : -len(eos_ids)] != eos_ids


def test_sequence_logp_sums_and_counts_returns_response_token_counts() -> None:
    class StaticModel(torch.nn.Module):
        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):  # type: ignore[override]
            del attention_mask
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 16)
            return type("Output", (), {"logits": logits})

    sums, counts = sequence_logp_sums_and_counts(
        StaticModel(),
        input_ids=torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        starts=torch.tensor([1, 1]),
    )

    assert counts.tolist() == [2, 1]
    token_logp = -torch.log(torch.tensor(16.0))
    assert torch.allclose(sums, torch.tensor([2.0 * token_logp, token_logp]))


def test_sequence_logps_with_token_kl_returns_summed_forward_kl() -> None:
    class StaticModel(torch.nn.Module):
        def __init__(self, logits: torch.Tensor) -> None:
            super().__init__()
            self.register_buffer("stored_logits", logits)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):  # type: ignore[override]
            del attention_mask
            logits = self.stored_logits.expand(input_ids.shape[0], input_ids.shape[1], -1)
            return type("Output", (), {"logits": logits})

    policy_logits = torch.log(torch.tensor([0.7, 0.2, 0.1])).view(1, 1, 3)
    ref_logits = torch.log(torch.tensor([0.2, 0.5, 0.3])).view(1, 1, 3)
    policy_sums, ref_sums, kl_values = sequence_logps_with_token_kl(
        StaticModel(policy_logits),
        StaticModel(ref_logits),
        input_ids=torch.tensor([[0, 1, 2]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        starts=torch.tensor([1]),
    )

    expected_kl = (torch.tensor([0.7, 0.2, 0.1]) * (policy_logits[0, 0] - ref_logits[0, 0])).sum()
    assert torch.allclose(policy_sums, torch.log(torch.tensor([0.2 * 0.1])))
    assert torch.allclose(ref_sums, torch.log(torch.tensor([0.5 * 0.3])))
    assert torch.allclose(kl_values, (2.0 * expected_kl).view(1))


def test_pair_reward_margins_uses_policy_reference_logratio() -> None:
    chosen_rewards, rejected_rewards, reward_margins = pair_reward_margins(
        policy_chosen_logps=torch.tensor([4.0, 2.0]),
        policy_rejected_logps=torch.tensor([1.0, 3.0]),
        reference_chosen_logps=torch.tensor([3.0, 2.0]),
        reference_rejected_logps=torch.tensor([2.0, 2.5]),
        beta=0.5,
    )

    assert torch.allclose(chosen_rewards, torch.tensor([0.5, 0.0]))
    assert torch.allclose(rejected_rewards, torch.tensor([-0.5, 0.25]))
    assert torch.allclose(reward_margins, torch.tensor([1.0, -0.25]))
