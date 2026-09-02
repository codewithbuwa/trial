from __future__ import annotations

from cpo_trl.evaluation.judges.common import parse_judge_json
from cpo_trl.evaluation.judges.http import request_chat_completion


def prometheus_prompt(
    *,
    instruction: str,
    input_text: str,
    response_a: str,
    response_b: str,
) -> str:
    rubric = (
        "Choose the better assistant response based on instruction-following, "
        "correctness, helpfulness, clarity, completeness, and safety. "
        "Do not prefer longer answers unless the extra detail improves quality."
    )
    return (
        "You are a fair evaluator language model.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Input:\n{input_text}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        f"Rubric:\n{rubric}\n\n"
        'Return JSON only: {"winner": "A"|"B"|"tie", "reason": "..."}'
    )


def prometheus_judge(
    *,
    instruction: str,
    input_text: str,
    response_a: str,
    response_b: str,
    model: str,
    base_url: str,
    timeout: float,
    max_retries: int = 3,
    retry_base_seconds: float = 1.0,
) -> dict[str, object]:
    prompt = prometheus_prompt(
        instruction=instruction,
        input_text=input_text,
        response_a=response_a,
        response_b=response_b,
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    content = request_chat_completion(
        payload=payload,
        base_url=base_url,
        authorization="Bearer dummy",
        timeout=timeout,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        evaluator_name="Prometheus judge",
    )
    return parse_judge_json(content)
