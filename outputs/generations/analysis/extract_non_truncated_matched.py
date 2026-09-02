#!/usr/bin/env python3
"""Extract the largest prompt-matched set of generations below the token cap."""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer


ANALYSIS_DIR = Path(__file__).resolve().parent
EVAL_DIR = ANALYSIS_DIR.parent
OUTPUTS_DIR = EVAL_DIR.parent
SOURCE = EVAL_DIR / "generations" / "final_methods_generations.jsonl"
TOKENIZER_PATH = (
    OUTPUTS_DIR
    / "dpo_final"
    / "dpo_lr1em05_b0p005_gn0p3"
    / "tokenizer.json"
)
OUTPUT = ANALYSIS_DIR / "non_truncated_prompt_matched_703_per_model.jsonl"
MANIFEST = ANALYSIS_DIR / "non_truncated_prompt_matched_703_per_model_manifest.json"
TOKEN_LIMIT = 256


def main() -> None:
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    rows = []
    prompt_ids_by_model = defaultdict(set)

    with SOURCE.open(encoding="utf-8") as source_file:
        for source_line, line in enumerate(source_file, start=1):
            row = json.loads(line)
            response_tokens = len(
                tokenizer.encode(row["response"], add_special_tokens=False).ids
            )
            enriched = {
                **row,
                "response_tokens": response_tokens,
                "truncation_inference": "non_truncated_below_256_tokens",
                "source_line": source_line,
            }
            rows.append(enriched)
            if response_tokens < TOKEN_LIMIT:
                prompt_ids_by_model[row["model"]].add(row["prompt_id"])

    models = sorted(prompt_ids_by_model)
    shared_prompt_ids = set.intersection(
        *(prompt_ids_by_model[model] for model in models)
    )
    selected = [
        row
        for row in rows
        if row["prompt_id"] in shared_prompt_ids
        and row["response_tokens"] < TOKEN_LIMIT
    ]

    counts = Counter(row["model"] for row in selected)
    expected = len(shared_prompt_ids)
    if any(counts[model] != expected for model in models):
        raise RuntimeError(f"Unbalanced output: {dict(counts)}")
    if len(selected) != expected * len(models):
        raise RuntimeError("Unexpected duplicate or missing prompt/model rows")

    with OUTPUT.open("w", encoding="utf-8") as output_file:
        for row in selected:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    manifest = {
        "source_file": str(SOURCE),
        "tokenizer_file": str(TOKENIZER_PATH),
        "criterion": "response_tokens < 256",
        "classification_limitation": (
            "Inferred from retokenized response length because finish_reason/EOS "
            "metadata is absent from the source generations."
        ),
        "models": models,
        "source_non_truncated_counts_by_model": {
            model: len(prompt_ids_by_model[model]) for model in models
        },
        "shared_prompt_count": expected,
        "rows_per_model": dict(sorted(counts.items())),
        "total_rows": len(selected),
        "output_file": str(OUTPUT),
        "output_sha256": digest,
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
