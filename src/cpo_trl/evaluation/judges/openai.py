from __future__ import annotations

from cpo_trl.evaluation.judges.common import parse_judge_json
from cpo_trl.evaluation.judges.http import request_chat_completion


def openai_chat_judge(
    *,
    instruction: str,
    input_text: str,
    response_a: str,
    response_b: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float,
    max_retries: int = 3,
    retry_base_seconds: float = 1.0,
) -> dict[str, object]:
    user_content = (
        "You are judging two assistant responses to the same user request. "
        "Choose the response that is more helpful, correct, complete, and safe. "
        "Ignore formatting differences unless they affect usefulness. "
        'Return only JSON with keys "winner" and "reason"; winner must be "A", "B", or "tie".\n\n'
        f"Instruction:\n{instruction}\n\n"
        f"Input:\n{input_text}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": "You are a strict pairwise evaluator."},
            {"role": "user", "content": user_content},
        ],
    }
    content = request_chat_completion(
        payload=payload,
        base_url=base_url,
        authorization=f"Bearer {api_key}",
        timeout=timeout,
        max_retries=max_retries,
        retry_base_seconds=retry_base_seconds,
        evaluator_name="OpenAI judge",
    )
    return parse_judge_json(content)
