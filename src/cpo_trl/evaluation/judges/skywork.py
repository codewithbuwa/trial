from __future__ import annotations

from typing import Any


def load_skywork_reward_model(model_name: str, max_length: int) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs: dict[str, Any] = {
        "num_labels": 1,
        "attn_implementation": "eager",
    }
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["device_map"] = "cuda:0"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    if not torch.cuda.is_available():
        model.to(device)
    model.eval()
    return {
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "max_length": max_length,
    }


def format_reward_model_input(
    tokenizer: Any,
    instruction: str,
    input_text: str,
    response: str,
) -> str:
    prompt = instruction if not input_text else f"{instruction}\n\n{input_text}"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False)
    except (AttributeError, ValueError):
        return f"User: {prompt}\n\nAssistant: {response}"


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded["input_ids"])


def _render_with_token_budget(
    *,
    tokenizer: Any,
    instruction: str,
    input_text: str,
    response: str,
    max_length: int,
) -> tuple[str, dict[str, Any]]:
    prompt = instruction if not input_text else f"{instruction}\n\n{input_text}"
    full_text = format_reward_model_input(tokenizer, instruction, input_text, response)
    full_tokens = tokenizer(full_text, add_special_tokens=True)["input_ids"]
    if len(full_tokens) <= max_length:
        return full_text, {
            "input_truncated": False,
            "full_input_tokens": len(full_tokens),
            "scored_input_tokens": len(full_tokens),
            "truncation_strategy": "none",
        }

    prompt_ids = _token_ids(tokenizer, prompt)
    response_ids = _token_ids(tokenizer, response)
    empty_text = format_reward_model_input(tokenizer, "", "", "")
    template_tokens = tokenizer(empty_text, add_special_tokens=True)["input_ids"]
    content_budget = max(2, max_length - len(template_tokens) - 8)
    prompt_budget = content_budget // 2
    response_budget = content_budget - prompt_budget
    if len(prompt_ids) < prompt_budget:
        response_budget += prompt_budget - len(prompt_ids)
        prompt_budget = len(prompt_ids)
    elif len(response_ids) < response_budget:
        prompt_budget += response_budget - len(response_ids)
        response_budget = len(response_ids)

    kept_prompt_ids = prompt_ids[:prompt_budget]
    kept_response_ids = response_ids[:response_budget]
    while True:
        truncated_prompt = tokenizer.decode(kept_prompt_ids, skip_special_tokens=True)
        truncated_response = tokenizer.decode(kept_response_ids, skip_special_tokens=True)
        text = format_reward_model_input(
            tokenizer,
            truncated_prompt,
            "",
            truncated_response,
        )
        scored_tokens = tokenizer(text, add_special_tokens=True)["input_ids"]
        overflow = len(scored_tokens) - max_length
        if overflow <= 0:
            break
        if len(kept_prompt_ids) >= len(kept_response_ids) and kept_prompt_ids:
            kept_prompt_ids = kept_prompt_ids[: max(0, len(kept_prompt_ids) - overflow)]
        elif kept_response_ids:
            kept_response_ids = kept_response_ids[: max(0, len(kept_response_ids) - overflow)]
        else:
            raise ValueError("judge max length is too small for the tokenizer template")

    return text, {
        "input_truncated": True,
        "full_input_tokens": len(full_tokens),
        "scored_input_tokens": len(scored_tokens),
        "truncation_strategy": "balanced_prompt_response",
        "prompt_tokens_kept": len(kept_prompt_ids),
        "prompt_tokens_total": len(prompt_ids),
        "response_tokens_kept": len(kept_response_ids),
        "response_tokens_total": len(response_ids),
    }


def skywork_reward_score(
    *,
    reward_model: dict[str, Any],
    instruction: str,
    input_text: str,
    response: str,
    return_diagnostics: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    import torch

    tokenizer = reward_model["tokenizer"]
    model = reward_model["model"]
    device = reward_model["device"]
    text, diagnostics = _render_with_token_budget(
        tokenizer=tokenizer,
        instruction=instruction,
        input_text=input_text,
        response=response,
        max_length=reward_model["max_length"],
    )
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
    ).to(device)
    with torch.no_grad():
        logits = model(**encoded).logits
    score = float(logits.squeeze().detach().float().cpu().item())
    if return_diagnostics:
        return score, diagnostics
    return score


def skywork_judge(
    *,
    reward_model: dict[str, Any],
    instruction: str,
    input_text: str,
    response_a: str,
    response_b: str,
    tie_threshold: float = 0.0,
) -> dict[str, Any]:
    if tie_threshold < 0:
        raise ValueError("tie_threshold must be non-negative")
    score_a, diagnostics_a = skywork_reward_score(
        reward_model=reward_model,
        instruction=instruction,
        input_text=input_text,
        response=response_a,
        return_diagnostics=True,
    )
    score_b, diagnostics_b = skywork_reward_score(
        reward_model=reward_model,
        instruction=instruction,
        input_text=input_text,
        response=response_b,
        return_diagnostics=True,
    )
    difference = score_a - score_b
    if abs(difference) <= tie_threshold:
        winner = "tie"
    elif difference > 0:
        winner = "A"
    else:
        winner = "B"
    return {
        "winner": winner,
        "reason": f"Skywork reward scores: A={score_a:.6f}, B={score_b:.6f}",
        "status": "ok",
        "score_a": score_a,
        "score_b": score_b,
        "score_difference": difference,
        "tie_threshold": tie_threshold,
        "response_a_truncation": diagnostics_a,
        "response_b_truncation": diagnostics_b,
    }
