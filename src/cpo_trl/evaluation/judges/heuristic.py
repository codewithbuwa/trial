from __future__ import annotations


def heuristic_judge(response_a: str, response_b: str) -> dict[str, str]:
    words_a = len(response_a.split())
    words_b = len(response_b.split())
    if words_a == words_b:
        return {"winner": "tie", "reason": "heuristic tie on response length"}
    winner = "A" if words_a > words_b else "B"
    return {"winner": winner, "reason": "heuristic picked the longer response"}
