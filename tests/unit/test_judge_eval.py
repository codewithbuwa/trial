from __future__ import annotations

import argparse
import json
import urllib.error

import pytest

import cpo_trl.evaluation.judges.http as judge_http
import cpo_trl.evaluation.judges.skywork as skywork_evaluator
import scripts.evaluate.evaluate_judge as evaluate_judge
from scripts.evaluate.evaluate_judge import (
    build_comparisons,
    comparison_id,
    evaluate_comparison,
    format_reward_model_input,
    generation_length_summary,
    load_existing_judgments,
    load_generation_records,
    pairrm_judge,
    parse_judge_json,
    parse_args,
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

    assert parsed["winner"] == "A"
    assert parsed["reason"] == "more complete"
    assert parsed["status"] == "ok"
    assert parsed["parse_method"] == "json"


def test_parse_judge_json_falls_back_to_text_winner() -> None:
    parsed = parse_judge_json("Evaluation: response B is better because it is clearer.")

    assert parsed["winner"] == "B"
    assert "response B is better" in parsed["reason"]


def test_parse_judge_json_normalizes_verbose_winner() -> None:
    parsed = parse_judge_json('{"winner": "Response A.", "reason": "clearer"}')

    assert parsed["winner"] == "A"
    assert parsed["status"] == "ok"


def test_parse_judge_json_finds_valid_object_among_multiple_braces() -> None:
    parsed = parse_judge_json(
        'Example: {"score": 1}\nDecision: {"winner": "B", "reason": "correct"}'
    )

    assert parsed["winner"] == "B"
    assert parsed["parse_method"] == "json"


def test_parse_judge_json_marks_unparseable_output_as_error() -> None:
    parsed = parse_judge_json("Both have merits; the decision is difficult.")

    assert parsed["winner"] is None
    assert parsed["status"] == "parse_error"
    assert parsed["raw_judge_output"]


def test_text_parser_handles_modifiers_without_false_tie() -> None:
    assert parse_pairwise_winner_text("Response A is slightly better.")["winner"] == "A"
    parsed = parse_pairwise_winner_text("Not really a tie, but B edges ahead.")

    assert parsed["winner"] == "B"
    assert parsed["status"] == "ok"


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

    monkeypatch.setattr(judge_http.urllib.request, "urlopen", fake_urlopen)

    judgment = prometheus_judge(
        instruction="Explain DPO",
        input_text="",
        response_a="short",
        response_b="clear answer",
        model="prometheus-eval/prometheus-7b-v2.0",
        base_url="http://localhost:8000/v1",
        timeout=12.0,
    )

    assert judgment["winner"] == "B"
    assert judgment["reason"] == "more precise"
    assert judgment["status"] == "ok"
    assert captured["url"] == "http://localhost:8000/v1/chat/completions"
    assert captured["authorization"] == "Bearer dummy"
    assert captured["timeout"] == 12.0
    payload = captured["payload"]
    assert payload["model"] == "prometheus-eval/prometheus-7b-v2.0"
    assert "Rubric:" in payload["messages"][0]["content"]


def test_prometheus_judge_retries_transient_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0
    sleeps: list[float] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"choices":[{"message":{"content":"{\\"winner\\":\\"A\\"}"}}]}'

    def fake_urlopen(request: object, timeout: float) -> FakeResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.URLError("temporary outage")
        return FakeResponse()

    monkeypatch.setattr(judge_http.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(judge_http.time, "sleep", sleeps.append)
    monkeypatch.setattr(judge_http.random, "uniform", lambda start, end: 0.0)

    judgment = prometheus_judge(
        instruction="Explain DPO",
        input_text="",
        response_a="a",
        response_b="b",
        model="prometheus",
        base_url="http://localhost:8000/v1",
        timeout=1.0,
        max_retries=1,
        retry_base_seconds=0.25,
    )

    assert judgment["winner"] == "A"
    assert attempts == 2
    assert sleeps == [0.25]


def test_evaluate_comparison_retries_parse_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_prometheus(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"winner": None, "status": "parse_error", "reason": "bad output"}
        return {"winner": "B", "status": "ok", "reason": "clearer"}

    monkeypatch.setattr(evaluate_judge, "prometheus_judge", fake_prometheus)
    args = argparse.Namespace(
        judge_provider="prometheus",
        judge_model="judge",
        openai_base_url="http://localhost:8000/v1",
        judge_timeout=1.0,
        judge_max_retries=0,
        judge_retry_base_seconds=0.0,
        judge_parse_retries=1,
        skywork_tie_threshold=0.0,
    )
    comparison = {
        "instruction": "prompt",
        "input": "",
        "response_a": "a",
        "response_b": "b",
    }

    judgment = evaluate_comparison(
        comparison,
        args=args,
        api_key="",
        pairrm_ranker=None,
        skywork_reward_model=None,
    )

    assert judgment["winner"] == "B"
    assert judgment["parse_attempts"] == 2


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
        return_diagnostics: bool = False,
    ) -> float | tuple[float, dict[str, object]]:
        score = {"short": 0.1, "clear answer": 0.9}[response]
        if return_diagnostics:
            return score, {"input_truncated": False}
        return score

    monkeypatch.setattr(skywork_evaluator, "skywork_reward_score", fake_reward_score)

    judgment = skywork_judge(
        reward_model={},
        instruction="Explain DPO",
        input_text="",
        response_a="short",
        response_b="clear answer",
    )

    assert judgment["winner"] == "B"
    assert "Skywork reward scores" in judgment["reason"]
    assert judgment["score_difference"] == pytest.approx(-0.8)


def test_skywork_tie_threshold_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_reward_score(**kwargs: object) -> tuple[float, dict[str, object]]:
        score = 1.0 if kwargs["response"] == "a" else 0.98
        return score, {"input_truncated": False}

    monkeypatch.setattr(skywork_evaluator, "skywork_reward_score", fake_reward_score)

    judgment = skywork_judge(
        reward_model={},
        instruction="prompt",
        input_text="",
        response_a="a",
        response_b="b",
        tie_threshold=0.05,
    )

    assert judgment["winner"] == "tie"
    assert judgment["tie_threshold"] == 0.05


def test_skywork_truncation_preserves_prompt_and_response_content() -> None:
    class CharacterTokenizer:
        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            tokenize: bool,
        ) -> str:
            return f"U:{messages[0]['content']}|A:{messages[1]['content']}"

        def __call__(
            self,
            text: str,
            add_special_tokens: bool = True,
        ) -> dict[str, list[int]]:
            return {"input_ids": [ord(character) for character in text]}

        def decode(self, values: list[int], skip_special_tokens: bool) -> str:
            return "".join(chr(value) for value in values)

    _, diagnostics = skywork_evaluator._render_with_token_budget(
        tokenizer=CharacterTokenizer(),
        instruction="p" * 80,
        input_text="",
        response="r" * 80,
        max_length=60,
    )

    assert diagnostics["input_truncated"] is True
    assert diagnostics["scored_input_tokens"] <= 60
    assert diagnostics["prompt_tokens_kept"] > 0
    assert diagnostics["response_tokens_kept"] > 0


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


def test_judge_provider_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["evaluate_judge.py"])

    with pytest.raises(SystemExit):
        parse_args()


def test_fixed_position_strategy_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate_judge.py", "--judge-provider", "heuristic", "--fixed-positions"],
    )

    args = parse_args()

    assert args.fixed_positions is True
    assert args.randomize_positions is False
    assert args.position_balanced is False


def test_validate_judge_settings_sets_default_skywork_model() -> None:
    args = argparse.Namespace(
        judge_provider="skywork",
        judge_model=None,
        api_key_env="OPENAI_API_KEY",
    )

    validate_judge_settings(args)

    assert args.judge_model == "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"


def test_validate_judge_settings_avoids_redundant_skywork_balancing() -> None:
    args = argparse.Namespace(
        judge_provider="skywork",
        judge_model="skywork",
        api_key_env="OPENAI_API_KEY",
        position_balanced=True,
        randomize_positions=False,
        fixed_positions=False,
    )

    validate_judge_settings(args)

    assert args.position_balanced is False
    assert args.randomize_positions is True


def test_validate_judge_settings_rejects_randomized_and_balanced_positions() -> None:
    args = argparse.Namespace(
        judge_provider="heuristic",
        judge_model=None,
        api_key_env="OPENAI_API_KEY",
        position_balanced=True,
        randomize_positions=True,
    )

    with pytest.raises(ValueError, match="cannot be used"):
        validate_judge_settings(args)


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


def test_build_comparisons_randomizes_positions_once_per_prompt_pair() -> None:
    rows = [
        {"prompt_id": "p0", "cluster_id": "coding", "instruction": "Do x", "input": ""},
        {"prompt_id": "p1", "cluster_id": "math", "instruction": "Do y", "input": ""},
    ]
    generations = {"KTO": ["kto0", "kto1"], "DPO": ["dpo0", "dpo1"]}

    comparisons = build_comparisons(
        rows,
        generations,
        seed=1,
        position_balanced=False,
        randomize_positions=True,
    )

    assert len(comparisons) == 2
    assert [row["position_strategy"] for row in comparisons] == ["randomized", "randomized"]
    assert [row["position_seed"] for row in comparisons] == [1, 1]
    assert [row["judge_order"] for row in comparisons] == [0, 0]
    assert [row["position_swapped"] for row in comparisons] == [True, False]
    assert [(row["model_a"], row["model_b"]) for row in comparisons] == [
        ("DPO", "KTO"),
        ("KTO", "DPO"),
    ]


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


def test_load_generation_records_groups_saved_generations(tmp_path) -> None:
    path = tmp_path / "generations.jsonl"
    records = [
        {
            "model": "DPO",
            "prompt_id": "p1",
            "cluster_id": "coding",
            "instruction": "Do x",
            "input": "",
            "response": "answer dpo",
        },
        {
            "model": "CPO",
            "prompt_id": "p1",
            "cluster_id": "coding",
            "instruction": "Do x",
            "input": "",
            "response": "answer cpo",
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    rows, generations = load_generation_records(path)

    assert rows == [
        {"prompt_id": "p1", "cluster_id": "coding", "instruction": "Do x", "input": ""}
    ]
    assert generations == {"CPO": ["answer cpo"], "DPO": ["answer dpo"]}


def test_load_generation_records_rejects_mismatched_prompt_metadata(tmp_path) -> None:
    path = tmp_path / "generations.jsonl"
    records = [
        {"model": "A", "prompt_id": "p1", "instruction": "one", "response": "a"},
        {"model": "B", "prompt_id": "p1", "instruction": "two", "response": "b"},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")

    with pytest.raises(ValueError, match="inconsistent prompt metadata"):
        load_generation_records(path)


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


def test_summarize_judgments_excludes_failed_records() -> None:
    summary = summarize_judgments(
        [
            {
                "model_a": "A",
                "model_b": "B",
                "winner_model": "A",
                "status": "ok",
            },
            {
                "model_a": "A",
                "model_b": "B",
                "winner_model": None,
                "status": "parse_error",
            },
        ]
    )

    assert summary["requested_comparisons"] == 2
    assert summary["total_comparisons"] == 1
    assert summary["failed_comparisons"] == 1
    assert summary["models"]["A"]["judge_score"] == 1.0


def test_comparison_ids_are_stable_and_resume_uses_latest_record(tmp_path) -> None:
    comparison = build_comparisons(
        [{"prompt_id": "p1", "instruction": "prompt", "input": ""}],
        {"A": ["a"], "B": ["b"]},
        seed=7,
        position_balanced=False,
    )[0]
    record_id = comparison_id(
        comparison,
        judge_provider="prometheus",
        judge_model="judge",
    )
    path = tmp_path / "judgments.jsonl"
    rows = [
        {"comparison_id": record_id, "status": "error"},
        {"comparison_id": record_id, "status": "ok", "winner": "A"},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    loaded = load_existing_judgments(path)

    assert len(record_id) == 64
    assert loaded[record_id]["status"] == "ok"


def test_resume_repairs_an_incomplete_final_record(tmp_path) -> None:
    path = tmp_path / "judgments.jsonl"
    complete = {"comparison_id": "complete", "status": "ok"}
    path.write_text(json.dumps(complete) + "\n" + '{"comparison_id": "partial"')

    loaded = load_existing_judgments(path)

    assert loaded == {"complete": complete}
    assert path.read_text() == json.dumps(complete) + "\n"


def test_main_checkpoints_and_resumes_completed_comparisons(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "pairwise.jsonl"
    summary_path = tmp_path / "summary.json"
    args = argparse.Namespace(
        generations_file=tmp_path / "generations.jsonl",
        eval_file=tmp_path / "validation.jsonl",
        models=None,
        output_jsonl=output_path,
        summary_json=summary_path,
        max_prompts=None,
        max_prompt_length=128,
        max_new_tokens=32,
        temperature=0.0,
        top_p=0.9,
        batch_size=1,
        seed=42,
        judge_provider="heuristic",
        judge_model=None,
        openai_base_url="http://unused",
        api_key_env="OPENAI_API_KEY",
        judge_timeout=1.0,
        judge_max_length=128,
        judge_max_retries=0,
        judge_retry_base_seconds=0.0,
        judge_parse_retries=0,
        skywork_tie_threshold=0.0,
        checkpoint_every=1,
        resume=True,
        position_balanced=False,
        randomize_positions=False,
        fixed_positions=False,
    )
    rows = [{"prompt_id": "p1", "instruction": "prompt", "input": ""}]
    generations = {"A": ["short"], "B": ["a longer response"]}
    calls = 0

    def fake_heuristic(response_a: str, response_b: str) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"winner": "B", "reason": "better", "status": "ok"}

    monkeypatch.setattr(evaluate_judge, "parse_args", lambda: args)
    monkeypatch.setattr(
        evaluate_judge,
        "load_generation_records",
        lambda path: (rows, generations),
    )
    monkeypatch.setattr(evaluate_judge, "heuristic_judge", fake_heuristic)

    evaluate_judge.main()
    evaluate_judge.main()

    assert calls == 1
    assert len(output_path.read_text().splitlines()) == 1
    summary = json.loads(summary_path.read_text())
    assert summary["resumed_comparisons"] == 1
    assert summary["newly_attempted_comparisons"] == 0
