from __future__ import annotations

import argparse
import json

import pytest

import scripts.evaluate.evaluate_judge as evaluate_judge
from scripts.evaluate.evaluate_judge import (
    build_comparisons,
    format_reward_model_input,
    generation_length_summary,
    pairrm_judge,
    parse_judge_json,
    parse_pairwise_winner_text,
    parse_model_specs,
    prometheus_judge,
    prometheus_prompt,
    response_length_stats,
    skywork_judge,
    summarize_judgments,
    validate_judge_settings,
)


def test_parse_model_specs_accepts_named_paths() -> None:
    assert parse_model_specs(["SFT=Qwen/Qwen2.5-1.5B-Instruct", "DPO=outputs/checkpoints/dpo"]) == [
        ("SFT", "Qwen/Qwen2.5-1.5B-Instruct"),
        ("DPO", "outputs/checkpoints/dpo"),
    ]


def test_parse_judge_json_extracts_json_from_text() -> None:
    parsed = parse_judge_json('Result:\n{"winner": "A", "reason": "more complete"}')

    assert parsed == {"winner": "A", "reason": "more complete"}


def test_parse_judge_json_falls_back_to_text_winner() -> None:
    parsed = parse_judge_json("Evaluation: response B is better because it is clearer.")

    assert parsed["winner"] == "B"
    assert "response B is better" in parsed["reason"]


def test_parse_pairwise_winner_text_handles_result_marker() -> None:
    parsed = parse_pairwise_winner_text("[RESULT] A")

    assert parsed["winner"] == "A"


def test_prometheus_prompt_contains_pairwise_rubric() -> None:
    prompt = prometheus_prompt(
        instruction="Explain DPO",
        input_text="",
        response_a="Answer A",
        response_b="Answer B",
    )

    assert "Instruction:\nExplain DPO" in prompt
    assert "Response A:\nAnswer A" in prompt
    assert "Response B:\nAnswer B" in prompt
    assert "Rubric:" in prompt
    assert '"winner": "A"|"B"|"tie"' in prompt


def test_prometheus_judge_calls_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {
                    "choices": [
                        {"message": {"content": '{"winner": "B", "reason": "more precise"}'}}
                    ]
                }
            ).encode("utf-8")

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        captured["timeout"] = timeout
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["authorization"] = request.headers["Authorization"]
        return FakeResponse()

    monkeypatch.setattr(evaluate_judge.urllib.request, "urlopen", fake_urlopen)

    judgment = prometheus_judge(
        instruction="Explain DPO",
        input_text="",
        response_a="short",
        response_b="clear answer",
        model="prometheus-eval/prometheus-7b-v2.0",
        base_url="http://localhost:8000/v1",
        timeout=12.0,
    )

    assert judgment == {"winner": "B", "reason": "more precise"}
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["authorization"] == "Bearer dummy"
    assert captured["timeout"] == 12.0
    payload = captured["payload"]
    assert payload["model"] == "prometheus-eval/prometheus-7b-v2.0"
    assert "Rubric:" in payload["messages"][0]["content"]


def test_pairrm_judge_prefers_lower_rank() -> None:
    class FakeRanker:
        def rank(
            self,
            inputs: list[str],
            candidates_texts: list[list[str]],
            return_scores: bool,
            batch_size: int,
        ) -> list[list[int]]:
            assert inputs == ["Explain DPO"]
            assert candidates_texts == [["short", "clear answer"]]
            assert return_scores is False
            assert batch_size == 1
            return [[2, 1]]

    judgment = pairrm_judge(
        ranker=FakeRanker(),
        instruction="Explain DPO",
        input_text="",
        response_a="short",
        response_b="clear answer",
    )

    assert judgment["winner"] == "B"
    assert "PairRM" in judgment["reason"]


def test_pairrm_judge_handles_equal_ranks_as_tie() -> None:
    class FakeRanker:
        def rank(
            self,
            inputs: list[str],
            candidates_texts: list[list[str]],
            return_scores: bool,
            batch_size: int,
        ) -> list[list[int]]:
            return [[1, 1]]

    judgment = pairrm_judge(
        ranker=FakeRanker(),
        instruction="Explain DPO",
        input_text="extra context",
        response_a="answer a",
        response_b="answer b",
    )

    assert judgment["winner"] == "tie"


def test_skywork_judge_prefers_higher_reward_score(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_reward_score(
        *,
        reward_model: dict[str, object],
        instruction: str,
        input_text: str,
        response: str,
    ) -> float:
        return {"short": 0.1, "clear answer": 0.9}[response]

    monkeypatch.setattr(evaluate_judge, "skywork_reward_score", fake_reward_score)

    judgment = skywork_judge(
        reward_model={},
        instruction="Explain DPO",
        input_text="",
        response_a="short",
        response_b="clear answer",
    )

    assert judgment["winner"] == "B"
    assert "Skywork reward scores" in judgment["reason"]


def test_format_reward_model_input_falls_back_without_chat_template() -> None:
    class FakeTokenizer:
        pass

    text = format_reward_model_input(
        FakeTokenizer(),
        instruction="Explain DPO",
        input_text="Use one paragraph.",
        response="DPO aligns a policy with pairwise preferences.",
    )

    assert "User: Explain DPO" in text
    assert "Use one paragraph." in text
    assert "Assistant: DPO aligns" in text


def test_validate_judge_settings_requires_prometheus_model() -> None:
    args = argparse.Namespace(
        judge_provider="prometheus",
        judge_model=None,
        api_key_env="OPENAI_API_KEY",
    )

    with pytest.raises(ValueError, match="--judge-model is required"):
        validate_judge_settings(args)


def test_validate_judge_settings_sets_default_skywork_model() -> None:
    args = argparse.Namespace(
        judge_provider="skywork",
        judge_model=None,
        api_key_env="OPENAI_API_KEY",
    )

    validate_judge_settings(args)

    assert args.judge_model == "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


def test_build_comparisons_creates_all_model_pairs() -> None:
    rows = [{"prompt_id": "p0", "cluster_id": "coding", "instruction": "Do x", "input": ""}]
    generations = {"SFT": ["a"], "DPO": ["b"], "CPO": ["c"]}

    comparisons = build_comparisons(rows, generations, seed=0, position_balanced=False)

    assert len(comparisons) == 3
    assert comparisons[0]["response_a_words"] >= 1
    assert comparisons[0]["response_b_words"] >= 1
    assert {frozenset((row["model_a"], row["model_b"])) for row in comparisons} == {
        frozenset(("SFT", "DPO")),
        frozenset(("SFT", "CPO")),
        frozenset(("DPO", "CPO")),
    }


def test_response_length_stats_counts_words() -> None:
    stats = response_length_stats(["one two", "one two three four"])

    assert stats["count"] == 2
    assert stats["mean_words"] == 3
    assert stats["median_words"] == 3
    assert stats["p95_words"] == 4


def test_generation_length_summary_groups_by_model() -> None:
    summary = generation_length_summary({"SFT": ["short"], "DPO": ["two words"]})

    assert summary["SFT"]["mean_words"] == 1
    assert summary["DPO"]["mean_words"] == 2


def test_summarize_judgments_counts_ties_as_half_win() -> None:
    summary = summarize_judgments(
        [
            {
                "model_a": "SFT",
                "model_b": "DPO",
                "winner_model": "DPO",
                "cluster_id": "general",
            },
            {
                "model_a": "SFT",
                "model_b": "CPO",
                "winner_model": "tie",
                "cluster_id": "general",
            },
        ]
    )

    assert summary["total_comparisons"] == 2
    assert summary["models"]["SFT"]["judge_score"] == 0.25
    assert summary["models"]["DPO"]["judge_score"] == 1.0
    assert summary["models"]["CPO"]["judge_score"] == 0.5
