"""Pairwise preference evaluation helpers."""

from __future__ import annotations

from typing import Any

import torch

from cpo_trl.data.formatting import format_prompt


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def _completion_with_eos(tokenizer: Any, completion: str) -> str:
    eos = getattr(tokenizer, "eos_token", None)
    if eos and not completion.endswith(eos):
        return completion + eos
    return completion


def _encode_prompt_completion(
    tokenizer: Any,
    *,
    prompt: str,
    completion: str,
    max_length: int,
    min_response_tokens: int = 1,
) -> tuple[list[int], int]:
    """Tokenize prompt/completion while preserving response tokens under truncation."""

    if max_length < min_response_tokens:
        raise ValueError("max_length must leave room for at least one response token")
    completion = _completion_with_eos(tokenizer, completion)
    prompt_ids = _token_ids(tokenizer, prompt)
    completion_ids = _token_ids(tokenizer, completion)
    if not completion_ids:
        raise ValueError("completion must contain at least one token")
    if len(prompt_ids) + len(completion_ids) > max_length:
        prompt_budget = max_length - min_response_tokens
        prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
        completion_budget = max_length - len(prompt_ids)
        completion_ids = completion_ids[:completion_budget]
    input_ids = prompt_ids + completion_ids
    response_start = len(prompt_ids)
    if response_start >= len(input_ids):
        raise ValueError("truncated example contains no response tokens")
    return input_ids, response_start


def _pad_token_rows(rows: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad token rows and build masks from true sequence lengths, not token values."""

    width = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
    for index, row in enumerate(rows):
        length = len(row)
        input_ids[index, :length] = torch.tensor(row, dtype=torch.long)
        attention_mask[index, :length] = 1
    return input_ids, attention_mask


def encode_pair(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    """Tokenize chosen/rejected continuations for pairwise scoring."""

    prompt = format_prompt(tokenizer, row)
    chosen = row["chosen"]
    rejected = row["rejected"]
    chosen_input_ids, chosen_response_start = _encode_prompt_completion(
        tokenizer,
        prompt=prompt,
        completion=chosen,
        max_length=max_length,
    )
    rejected_input_ids, rejected_response_start = _encode_prompt_completion(
        tokenizer,
        prompt=prompt,
        completion=rejected,
        max_length=max_length,
    )
    return {
        "row_index": row.get("row_index"),
        "prompt_id": row.get("prompt_id"),
        "chosen_input_ids": chosen_input_ids,
        "chosen_response_start": chosen_response_start,
        "rejected_input_ids": rejected_input_ids,
        "rejected_response_start": rejected_response_start,
        "cluster_id": row.get("cluster_id", "unknown"),
    }


def encode_unary(tokenizer: Any, row: dict[str, Any], max_length: int) -> dict[str, Any]:
    """Tokenize a unary completion row for CPO/KTO-style scoring."""

    prompt = format_prompt(tokenizer, row)
    completion = row["completion"]
    input_ids, response_start = _encode_prompt_completion(
        tokenizer,
        prompt=prompt,
        completion=completion,
        max_length=max_length,
    )
    return {
        "row_index": row.get("row_index"),
        "prompt_id": row.get("prompt_id"),
        "input_ids": input_ids,
        "response_start": response_start,
        "label": row["label"],
        "cluster_id": row.get("cluster_id", "unknown"),
        "prompt_text": prompt,
        "completion": completion,
    }


def explicit_pair_indices_from_adjacent_rows(rows: list[dict[str, Any]]) -> torch.Tensor:
    """Infer sampler-defined pair positions from adjacent unary rows."""

    pair_indices: list[tuple[int, int]] = []
    for start in range(0, len(rows) - 1, 2):
        left = rows[start]
        right = rows[start + 1]
        if str(left.get("prompt_id")) != str(right.get("prompt_id")):
            continue
        if str(left.get("cluster_id", "unknown")) != str(right.get("cluster_id", "unknown")):
            continue
        left_label = bool(left["label"])
        right_label = bool(right["label"])
        if left_label == right_label:
            continue
        if left_label:
            pair_indices.append((start, start + 1))
        else:
            pair_indices.append((start + 1, start))
    return torch.tensor(pair_indices, dtype=torch.long)


def collate_unary_batch(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad tokenized unary rows into a batch."""

    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask = _pad_token_rows([row["input_ids"] for row in rows], pad_id)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_start": torch.tensor([row["response_start"] for row in rows]),
        "labels": torch.tensor([row["label"] for row in rows], dtype=torch.bool),
        "cluster_ids": [row["cluster_id"] for row in rows],
        "row_indices": [row.get("row_index") for row in rows],
        "prompt_ids": [row.get("prompt_id") for row in rows],
        "prompt_texts": [row["prompt_text"] for row in rows],
        "completions": [row["completion"] for row in rows],
        "pair_indices": explicit_pair_indices_from_adjacent_rows(rows),
    }


def collate_mismatched_unary_batch(
    tokenizer: Any,
    *,
    prompt_texts: list[str],
    completions: list[str],
    prompt_ids: list[str],
    cluster_ids: list[str],
    max_length: int,
) -> dict[str, Any] | None:
    """Build in-cluster mismatched prompt/completion rows for KTO-style z estimates."""

    grouped: dict[str, list[int]] = {}
    for index, cluster_id in enumerate(cluster_ids):
        grouped.setdefault(cluster_id, []).append(index)

    rows: list[dict[str, Any]] = []
    for cluster_id, indices in grouped.items():
        if len(indices) < 2:
            continue
        for index in indices:
            prompt_id = str(prompt_ids[index])
            candidates = [
                candidate
                for candidate in indices
                if candidate != index and str(prompt_ids[candidate]) != prompt_id
            ]
            if not candidates:
                continue
            try:
                position = indices.index(index)
            except ValueError:
                position = 0
            candidate = candidates[position % len(candidates)]
            input_ids, response_start = _encode_prompt_completion(
                tokenizer,
                prompt=prompt_texts[index],
                completion=completions[candidate],
                max_length=max_length,
            )
            rows.append(
                {
                    "input_ids": input_ids,
                    "response_start": response_start,
                    "cluster_id": cluster_id,
                }
            )

    if not rows:
        return None

    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask = _pad_token_rows([row["input_ids"] for row in rows], pad_id)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_start": torch.tensor([row["response_start"] for row in rows]),
        "cluster_ids": [row["cluster_id"] for row in rows],
    }


def collate_pair_batch(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad tokenized pair rows into a batch."""

    pad_id = tokenizer.pad_token_id

    chosen_ids, chosen_attention_mask = _pad_token_rows([row["chosen_input_ids"] for row in rows], pad_id)
    rejected_ids, rejected_attention_mask = _pad_token_rows([row["rejected_input_ids"] for row in rows], pad_id)
    return {
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": chosen_attention_mask,
        "chosen_response_start": torch.tensor([row["chosen_response_start"] for row in rows]),
        "rejected_input_ids": rejected_ids,
        "rejected_attention_mask": rejected_attention_mask,
        "rejected_response_start": torch.tensor([row["rejected_response_start"] for row in rows]),
        "cluster_ids": [row["cluster_id"] for row in rows],
        "row_indices": [row.get("row_index") for row in rows],
        "prompt_ids": [row.get("prompt_id") for row in rows],
    }


def sequence_logps(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    starts: torch.Tensor,
) -> torch.Tensor:
    """Return summed response-token log probabilities."""

    sums, _counts = sequence_logp_sums_and_counts(model, input_ids, attention_mask, starts)
    return sums


def sequence_logps_with_token_kl(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return response log-prob sums, reference sums, and summed token KL(policy || ref)."""

    policy_outputs = policy_model(input_ids=input_ids, attention_mask=attention_mask)
    with torch.no_grad():
        ref_outputs = ref_model(input_ids=input_ids, attention_mask=attention_mask)

    policy_logits = policy_outputs.logits[:, :-1, :]
    ref_logits = ref_outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    token_mask = attention_mask[:, 1:].bool()
    positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    response_mask = positions >= (starts.to(labels.device).unsqueeze(1) - 1)
    mask = token_mask & response_mask
    row_indices = mask.nonzero(as_tuple=False)[:, 0]
    if row_indices.numel() == 0:
        zeros = torch.zeros(input_ids.shape[0], dtype=policy_logits.dtype, device=input_ids.device)
        return zeros, zeros, zeros

    selected_labels = labels[mask]
    policy_log_probs = torch.log_softmax(policy_logits[mask].float(), dim=-1)
    ref_log_probs = torch.log_softmax(ref_logits[mask].float(), dim=-1)
    policy_token_logps = torch.gather(
        policy_log_probs,
        1,
        selected_labels.unsqueeze(1),
    ).squeeze(1)
    ref_token_logps = torch.gather(
        ref_log_probs,
        1,
        selected_labels.unsqueeze(1),
    ).squeeze(1)
    token_kl = (policy_log_probs.exp() * (policy_log_probs - ref_log_probs)).sum(dim=-1)

    policy_sums = torch.zeros(input_ids.shape[0], dtype=policy_token_logps.dtype, device=input_ids.device)
    ref_sums = torch.zeros_like(policy_sums)
    kl_sums = torch.zeros_like(policy_sums)
    policy_sums.scatter_add_(0, row_indices, policy_token_logps)
    ref_sums.scatter_add_(0, row_indices, ref_token_logps)
    kl_sums.scatter_add_(0, row_indices, token_kl.clamp_min(0.0))
    return (
        policy_sums.to(policy_logits.dtype),
        ref_sums.to(policy_logits.dtype),
        kl_sums.to(policy_logits.dtype),
    )


def pair_reward_margins(
    *,
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return DPO-style chosen/rejected rewards and chosen-minus-rejected margin."""

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    return chosen_rewards, rejected_rewards, chosen_rewards - rejected_rewards


def sequence_logp_sums_and_counts(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed response-token log probabilities and token counts."""

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    # Upcast to float32 before log_softmax so summed log-probs stay accurate when
    # the model runs in a reduced precision (matches sequence_logps_with_token_kl).
    token_logps = torch.gather(
        torch.log_softmax(logits.float(), dim=-1), 2, labels.unsqueeze(-1)
    ).squeeze(-1)
    token_mask = attention_mask[:, 1:].bool()
    positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    response_mask = positions >= (starts.to(labels.device).unsqueeze(1) - 1)
    mask = token_mask & response_mask
    counts = mask.sum(dim=1).clamp_min(1)
    return (token_logps * mask).sum(dim=1), counts
