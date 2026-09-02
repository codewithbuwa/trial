from __future__ import annotations

import json
import re
from typing import Any


def _normalize_winner(value: object) -> str | None:
    normalized = re.sub(r"[^a-z]", "", str(value).lower())
    aliases = {
        "a": "A",
        "answera": "A",
        "responsea": "A",
        "b": "B",
        "answerb": "B",
        "responseb": "B",
        "tie": "tie",
        "draw": "tie",
        "equal": "tie",
    }
    return aliases.get(normalized)


def _result(
    *,
    winner: str | None,
    reason: str,
    status: str,
    parse_method: str,
    raw_output: str,
) -> dict[str, Any]:
    return {
        "winner": winner,
        "reason": reason,
        "status": status,
        "parse_method": parse_method,
        "raw_judge_output": raw_output,
    }


def _json_objects(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            objects.append(parsed)
    return objects


def parse_judge_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        candidates = _json_objects(text)
        parsed = next((item for item in candidates if "winner" in item), None)
    if isinstance(parsed, dict):
        winner = _normalize_winner(parsed.get("winner"))
        if winner is not None:
            return _result(
                winner=winner,
                reason=str(parsed.get("reason", "")).strip(),
                status="ok",
                parse_method="json",
                raw_output=text,
            )
    fallback = parse_pairwise_winner_text(text)
    if fallback["status"] == "ok":
        fallback["parse_method"] = "text_fallback"
        return fallback
    reason = "judge returned no valid winner"
    if isinstance(parsed, dict) and "winner" in parsed:
        reason = f"judge returned unsupported winner: {parsed['winner']!r}"
    return _result(
        winner=None,
        reason=reason,
        status="parse_error",
        parse_method="failed",
        raw_output=text,
    )


def parse_pairwise_winner_text(text: str) -> dict[str, Any]:
    normalized = text.strip()
    label_patterns = (
        r"(?:winner|better response|preferred response|choice|result)\s*[:\-]\s*"
        r"(?:response\s+|answer\s+)?(A|B)\b",
        r"\[(?:winner|result|choice)\]\s*(?:response\s+|answer\s+)?(A|B)\b",
    )
    preference_patterns = (
        r"\b(?:response|answer)\s+([AB])\s+(?:is\s+)?(?:\w+\s+){0,3}"
        r"(?:better|stronger|preferred|superior|wins|outperforms|edges?\s+ahead)\b",
        r"\b([AB])\s+(?:is\s+)?(?:\w+\s+){0,3}"
        r"(?:better|stronger|preferred|superior|wins|outperforms|edges?\s+ahead)\b",
        r"\b(?:choose|select|prefer|pick)\s+(?:response\s+|answer\s+)?([AB])\b",
    )
    for pattern in (*label_patterns, *preference_patterns):
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return _result(
                winner=match.group(1).upper(),
                reason=normalized[:500],
                status="ok",
                parse_method="text",
                raw_output=text,
            )
    tie_patterns = (
        r"(?:winner|result|choice)\s*[:\-]\s*(tie|draw)\b",
        r"(?:^|[.!?]\s+)(?:it(?:'s| is)|this is|the result is)\s+(?:a\s+)?(tie|draw)\b",
        r"^(?:a\s+)?(tie|draw)[.!]?$",
        r"(?:^|[.!?]\s+)the (?:responses|answers) (?:are|were) (?:a\s+)?(tie|draw)\b",
    )
    for pattern in tie_patterns:
        tie_match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if tie_match:
            return _result(
                winner="tie",
                reason=normalized[:500],
                status="ok",
                parse_method="text",
                raw_output=text,
            )
    return _result(
        winner=None,
        reason="judge returned no parseable winner",
        status="parse_error",
        parse_method="failed",
        raw_output=text,
    )
