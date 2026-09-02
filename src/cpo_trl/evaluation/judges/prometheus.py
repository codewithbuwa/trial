from __future__ import annotations

import json
import urllib.error
import urllib.request

from cpo_trl.evaluation.judges.common import parse_judge_json


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
) -> dict[str, str]:
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
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer dummy",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Prometheus judge request failed with HTTP {exc.code}: {detail}"
        ) from exc
    content = body["choices"][0]["message"]["content"]
    return parse_judge_json(content)
