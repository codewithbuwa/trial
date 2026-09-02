from __future__ import annotations

import json
import random
import socket
import time
import urllib.error
import urllib.request
from typing import Any

RETRYABLE_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def request_chat_completion(
    *,
    payload: dict[str, Any],
    base_url: str,
    authorization: str,
    timeout: float,
    max_retries: int,
    retry_base_seconds: float,
    evaluator_name: str,
) -> str:
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if retry_base_seconds < 0:
        raise ValueError("retry_base_seconds must be non-negative")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("response content is not a string")
            return content
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            error = RuntimeError(
                f"{evaluator_name} request failed with HTTP {exc.code}: {detail}"
            )
            retryable = exc.code in RETRYABLE_HTTP_CODES
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            error = RuntimeError(f"{evaluator_name} request failed: {exc}")
            retryable = True
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            error = RuntimeError(f"{evaluator_name} returned a malformed response: {exc}")
            retryable = False
        if not retryable or attempt == max_retries:
            raise error
        delay = retry_base_seconds * (2**attempt)
        delay += random.uniform(0.0, retry_base_seconds)
        time.sleep(delay)
    raise AssertionError("unreachable")
