from __future__ import annotations

import json
import urllib.error
import urllib.request

from cpo_trl.evaluation.judges.common import parse_judge_json


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
) -> dict[str, str]:
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
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"judge request failed with HTTP {exc.code}: {detail}") from exc
    content = body["choices"][0]["message"]["content"]
    return parse_judge_json(content)
