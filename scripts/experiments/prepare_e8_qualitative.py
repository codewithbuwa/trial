from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def by_prompt(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("prompt_id", record.get("row_index", "")))].append(record)
    return grouped


def top_margin_cases(records: list[dict[str, Any]], *, metric: str, n: int, reverse: bool) -> list[dict[str, Any]]:
    scored = [record for record in records if isinstance(record.get(metric), int | float)]
    return sorted(scored, key=lambda record: float(record[metric]), reverse=reverse)[:n]


def select_cases(
    *,
    cpo_margins: list[dict[str, Any]],
    cpo_unary_margins: list[dict[str, Any]],
    judge_records: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    n: int,
) -> dict[str, Any]:
    cpo_by_prompt = by_prompt(cpo_margins)
    unary_by_prompt = by_prompt(cpo_unary_margins)
    disagreements = []
    for prompt_id, cpo_rows in cpo_by_prompt.items():
        unary_rows = unary_by_prompt.get(prompt_id, [])
        if not unary_rows:
            continue
        cpo_pairwise_correct = bool(
            cpo_rows[0].get("normalized_pairwise_correct", cpo_rows[0].get("pairwise_correct", False))
        )
        unary_pairwise_correct = bool(
            unary_rows[0].get("normalized_pairwise_correct", unary_rows[0].get("pairwise_correct", False))
        )
        if cpo_pairwise_correct != unary_pairwise_correct:
            disagreements.append({"prompt_id": prompt_id, "cpo": cpo_rows[0], "cpo_unary": unary_rows[0]})
    judge_cpo_wins = [
        record
        for record in judge_records
        if str(record.get("winner_model", "")).upper() == "CPO"
        or str(record.get("winner", "")).upper() == "CPO"
    ][:n]
    return {
        "experiment": "E8_qualitative",
        "cpo_strong_pairwise_correct": top_margin_cases(
            cpo_margins, metric="normalized_margin", n=n, reverse=True
        ),
        "cpo_strong_pairwise_incorrect": top_margin_cases(
            cpo_margins, metric="normalized_margin", n=n, reverse=False
        ),
        "cpo_vs_cpo_unary_disagreements": disagreements[:n],
        "judge_cpo_wins": judge_cpo_wins,
        "generations_by_prompt": by_prompt(generations),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare E8 qualitative case selections.")
    parser.add_argument("--cpo-margins", type=Path, default=Path("outputs/checkpoints/cpo/pairwise_accuracy_margins.jsonl"))
    parser.add_argument(
        "--cpo-unary-margins",
        type=Path,
        default=Path("outputs/checkpoints/cpo_unary/pairwise_accuracy_margins.jsonl"),
    )
    parser.add_argument("--judge-pairwise", type=Path, default=Path("outputs/judge/pairwise.jsonl"))
    parser.add_argument("--generations", type=Path, default=Path("outputs/generations/main.jsonl"))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--output-json", type=Path, default=Path("experiments/E8_qualitative/cases.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = select_cases(
        cpo_margins=read_jsonl(args.cpo_margins),
        cpo_unary_margins=read_jsonl(args.cpo_unary_margins),
        judge_records=read_jsonl(args.judge_pairwise),
        generations=read_jsonl(args.generations),
        n=args.n,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
