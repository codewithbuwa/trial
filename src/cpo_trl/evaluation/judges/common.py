from __future__ import annotations

import json
import re


def parse_judge_json(text: str) -> dict[str, str]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            fallback = parse_pairwise_winner_text(text)
            if fallback["winner"] != "tie":
                return fallback
            return {"winner": "tie", "reason": "judge returned non-JSON output"}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            fallback = parse_pairwise_winner_text(text)
            if fallback["winner"] != "tie":
                return fallback
            return {"winner": "tie", "reason": "judge returned invalid JSON"}
    winner = str(parsed.get("winner", "tie")).strip()
    if winner not in {"A", "B", "tie"}:
        winner = "tie"
    return {"winner": winner, "reason": str(parsed.get("reason", "")).strip()}


def parse_pairwise_winner_text(text: str) -> dict[str, str]:
    normalized = text.strip()
    patterns = (
        r"(?:winner|better response|preferred response|choice|result)\s*[:\-]\s*(A|B|tie)\b",
        r"\[(?:winner|result|choice)\]\s*(A|B|tie)\b",
        r"\b(response|answer)\s+(A|B)\s+(?:wins|is better|is preferred)\b",
        r"\b(A|B)\s+(?:wins|is better|is preferred)\b",
        r"\b(tie)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if not match:
            continue
        winner = match.group(match.lastindex or 1).upper()
        if winner == "TIE":
            winner = "tie"
        return {"winner": winner, "reason": normalized[:500]}
    return {"winner": "tie", "reason": "judge returned no parseable winner"}
