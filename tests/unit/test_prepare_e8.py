from __future__ import annotations

from scripts.experiments.prepare_e8_qualitative import select_cases


def test_select_cases_builds_qualitative_buckets() -> None:
    cpo_margins = [
        {"prompt_id": "p0", "normalized_margin": 0.9, "normalized_pairwise_correct": True},
        {"prompt_id": "p1", "normalized_margin": -0.4, "normalized_pairwise_correct": False},
        {"prompt_id": "p2", "normalized_margin": 0.2, "normalized_pairwise_correct": True},
    ]
    cpo_unary_margins = [
        {"prompt_id": "p0", "normalized_margin": 0.1, "normalized_pairwise_correct": True},
        {"prompt_id": "p1", "normalized_margin": 0.2, "normalized_pairwise_correct": True},
        {"prompt_id": "p2", "normalized_margin": -0.2, "normalized_pairwise_correct": False},
    ]
    judge_records = [{"prompt_id": "p0", "winner_model": "CPO"}]
    generations = [{"prompt_id": "p0", "model": "CPO", "response": "answer"}]

    cases = select_cases(
        cpo_margins=cpo_margins,
        cpo_unary_margins=cpo_unary_margins,
        judge_records=judge_records,
        generations=generations,
        n=2,
    )

    assert cases["experiment"] == "E8_qualitative"
    assert cases["cpo_strong_pairwise_correct"][0]["prompt_id"] == "p0"
    assert cases["cpo_strong_pairwise_incorrect"][0]["prompt_id"] == "p1"
    assert {case["prompt_id"] for case in cases["cpo_vs_cpo_unary_disagreements"]} == {"p1", "p2"}
    assert cases["judge_cpo_wins"] == judge_records
    assert "p0" in cases["generations_by_prompt"]
