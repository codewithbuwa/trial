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


def skywork_reward_score(
    *,
    reward_model: dict[str, Any],
    instruction: str,
    input_text: str,
    response: str,
) -> float:
    import torch

    tokenizer = reward_model["tokenizer"]
    model = reward_model["model"]
    device = reward_model["device"]
    text = format_reward_model_input(tokenizer, instruction, input_text, response)
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=reward_model["max_length"],
    ).to(device)
    with torch.no_grad():
        logits = model(**encoded).logits
    return float(logits.squeeze().detach().float().cpu().item())


def skywork_judge(
    *,
    reward_model: dict[str, Any],
    instruction: str,
    input_text: str,
    response_a: str,
    response_b: str,
) -> dict[str, str]:
    score_a = skywork_reward_score(
        reward_model=reward_model,
        instruction=instruction,
        input_text=input_text,
        response=response_a,
    )
    score_b = skywork_reward_score(
        reward_model=reward_model,
        instruction=instruction,
        input_text=input_text,
        response=response_b,
    )
    if score_a > score_b:
        winner = "A"
    elif score_b > score_a:
        winner = "B"
    else:
        winner = "tie"
    return {
        "winner": winner,
        "reason": f"Skywork reward scores: A={score_a:.6f}, B={score_b:.6f}",
    }
