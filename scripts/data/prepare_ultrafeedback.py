from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cpo_trl.data.datasets import write_jsonl


def text_from_message(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("role") == "assistant":
                parts.append(str(item.get("content", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(value)


def cluster_for(instruction: str) -> str:
    lowered = instruction.lower()
    if any(word in lowered for word in ("code", "python", "javascript", "function", "bug")):
        return "coding"
    if any(word in lowered for word in ("math", "calculate", "equation", "proof")):
        return "math"
    if any(word in lowered for word in ("write", "draft", "email", "story")):
        return "writing"
    return "general"


def normalize_row(row: dict[str, Any]) -> dict[str, str]:
    instruction = str(row.get("prompt") or row.get("instruction") or "")
    chosen = text_from_message(row.get("chosen") or row.get("chosen_response") or "")
    rejected = text_from_message(row.get("rejected") or row.get("rejected_response") or "")
    prompt_id = str(row.get("prompt_id") or hashlib.sha1(instruction.encode("utf-8")).hexdigest())
    return {
        "prompt_id": prompt_id,
        "instruction": instruction,
        "input": str(row.get("input") or ""),
        "chosen": chosen,
        "rejected": rejected,
        "cluster_id": str(row.get("cluster_id") or cluster_for(instruction)),
    }


def assign_random_clusters(
    pair_rows: list[dict[str, str]],
    *,
    seed: int,
    n_clusters: int = 4,
    prefix: str = "random",
    matched: bool = False,
) -> list[dict[str, str]]:
    """Assign prompt-stable random cluster ids to pair rows."""

    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    prompt_ids = sorted({row["prompt_id"] for row in pair_rows})
    rng = random.Random(seed)
    if matched:
        prompt_to_cluster = {
            row["prompt_id"]: row["cluster_id"]
            for row in sorted(pair_rows, key=lambda row: row["prompt_id"])
        }
        labels = [prompt_to_cluster[prompt_id] for prompt_id in sorted(prompt_to_cluster)]
        rng.shuffle(labels)
        prompt_clusters = dict(zip(sorted(prompt_to_cluster), labels, strict=True))
    else:
        rng.shuffle(prompt_ids)
        prompt_clusters = {
            prompt_id: f"{prefix}_{index % n_clusters}"
            for index, prompt_id in enumerate(prompt_ids)
        }
    return [
        {
            **row,
            "cluster_id": prompt_clusters[row["prompt_id"]],
        }
        for row in pair_rows
    ]


def assign_single_cluster(pair_rows: list[dict[str, str]], *, cluster_id: str = "global") -> list[dict[str, str]]:
    """Assign every prompt to one global cluster for cluster-ablation controls."""

    return [{**row, "cluster_id": cluster_id} for row in pair_rows]


class PromptEmbedder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 32,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ) -> Any: ...


def assign_embedding_clusters(
    pair_rows: list[dict[str, str]],
    *,
    seed: int,
    n_clusters: int = 4,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    prefix: str = "embedding",
    embedder: PromptEmbedder | None = None,
) -> list[dict[str, str]]:
    """Cluster unique prompts using sentence embeddings and KMeans."""

    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    prompt_texts = {
        row["prompt_id"]: row["instruction"]
        for row in sorted(pair_rows, key=lambda row: row["prompt_id"])
    }
    if len(prompt_texts) < n_clusters:
        raise ValueError("n_clusters cannot exceed the number of unique prompts")

    prompt_ids = list(prompt_texts)
    texts = [prompt_texts[prompt_id] for prompt_id in prompt_ids]
    if embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "embedding4 clustering requires sentence-transformers. "
                "Install it with `poetry add sentence-transformers scikit-learn`."
            ) from exc
        embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    try:
        from sklearn.cluster import KMeans
    except ImportError as exc:
        raise RuntimeError(
            "embedding4 clustering requires scikit-learn. "
            "Install it with `poetry add sentence-transformers scikit-learn`."
        ) from exc
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit_predict(embeddings)
    prompt_clusters = {
        prompt_id: f"{prefix}_{label}"
        for prompt_id, label in zip(prompt_ids, labels, strict=True)
    }
    return [{**row, "cluster_id": prompt_clusters[row["prompt_id"]]} for row in pair_rows]


def split_by_prompt(
    pair_rows: list[dict[str, str]],
    *,
    eval_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Split pair rows by prompt id so a prompt cannot cross data splits."""

    if not 0.0 <= eval_ratio < 1.0:
        raise ValueError("eval_ratio must be in [0, 1)")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0, 1)")
    if eval_ratio + test_ratio >= 1.0:
        raise ValueError("eval_ratio + test_ratio must be less than 1")
    if eval_ratio == 0.0 and test_ratio == 0.0:
        return pair_rows, [], []
    prompt_ids = sorted({row["prompt_id"] for row in pair_rows})
    rng = random.Random(seed)
    rng.shuffle(prompt_ids)
    eval_count = max(1, round(len(prompt_ids) * eval_ratio)) if eval_ratio else 0
    test_count = max(1, round(len(prompt_ids) * test_ratio)) if test_ratio else 0
    if eval_count + test_count >= len(prompt_ids):
        overflow = eval_count + test_count - max(0, len(prompt_ids) - 1)
        if test_count >= overflow:
            test_count -= overflow
        else:
            eval_count = max(0, eval_count - (overflow - test_count))
            test_count = 0
    eval_ids = set(prompt_ids[:eval_count])
    test_ids = set(prompt_ids[eval_count : eval_count + test_count])
    train_rows = [
        row
        for row in pair_rows
        if row["prompt_id"] not in eval_ids and row["prompt_id"] not in test_ids
    ]
    eval_rows = [row for row in pair_rows if row["prompt_id"] in eval_ids]
    test_rows = [row for row in pair_rows if row["prompt_id"] in test_ids]
    return train_rows, eval_rows, test_rows


def build_outputs(pair_rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    """Build all training schemas from normalized pair rows."""

    sft_rows = [
        {
            "prompt_id": row["prompt_id"],
            "instruction": row["instruction"],
            "input": row["input"],
            "chosen": row["chosen"],
        }
        for row in pair_rows
    ]
    dpo_rows = [
        {
            "prompt_id": row["prompt_id"],
            "instruction": row["instruction"],
            "input": row["input"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "cluster_id": row["cluster_id"],
        }
        for row in pair_rows
    ]
    kto_rows = []
    cpo_rows = []
    for row in pair_rows:
        base = {
            "prompt_id": row["prompt_id"],
            "instruction": row["instruction"],
            "input": row["input"],
        }
        kto_base = {**base, "cluster_id": row["cluster_id"]}
        kto_rows.append({**kto_base, "completion": row["chosen"], "label": True})
        kto_rows.append({**kto_base, "completion": row["rejected"], "label": False})
        cpo_base = {**base, "cluster_id": row["cluster_id"]}
        cpo_rows.append({**cpo_base, "completion": row["chosen"], "label": True})
        cpo_rows.append({**cpo_base, "completion": row["rejected"], "label": False})
    return {"sft": sft_rows, "dpo": dpo_rows, "kto": kto_rows, "cpo": cpo_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrafeedback_binarized")
    parser.add_argument("--split", default="train_prefs")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cluster-mode",
        choices=("keyword", "single_cluster", "random4", "random4_matched", "embedding4"),
        default="keyword",
        help="How to assign cluster_id values for DPO/CPO rows.",
    )
    parser.add_argument(
        "--random-cluster-prefix",
        default="random",
        help="Prefix used when --cluster-mode random4 is selected.",
    )
    parser.add_argument("--output-root", default="data/processed")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("prepare_ultrafeedback.py requires datasets. Install project dependencies first.") from exc

    dataset = load_dataset(args.dataset, split=args.split)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    pair_rows = [normalize_row(dict(row)) for row in dataset]
    if args.cluster_mode == "single_cluster":
        pair_rows = assign_single_cluster(pair_rows)
    elif args.cluster_mode in {"random4", "random4_matched"}:
        pair_rows = assign_random_clusters(
            pair_rows,
            seed=args.seed,
            n_clusters=4,
            prefix=args.random_cluster_prefix,
            matched=args.cluster_mode == "random4_matched",
        )
    elif args.cluster_mode == "embedding4":
        pair_rows = assign_embedding_clusters(pair_rows, seed=args.seed, n_clusters=4)
    train_pairs, eval_pairs, test_pairs = split_by_prompt(
        pair_rows,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    train_outputs = build_outputs(train_pairs)
    eval_outputs = build_outputs(eval_pairs)
    test_outputs = build_outputs(test_pairs)

    root = args.output_root.rstrip("/")
    for kind, rows in train_outputs.items():
        write_jsonl(f"{root}/{kind}/train.jsonl", rows)
    for kind, rows in eval_outputs.items():
        write_jsonl(f"{root}/{kind}/validation.jsonl", rows)
    for kind, rows in test_outputs.items():
        write_jsonl(f"{root}/{kind}/test.jsonl", rows)
    print(
        {
            "pair_rows": len(pair_rows),
            "train_pairs": len(train_pairs),
            "eval_pairs": len(eval_pairs),
            "test_pairs": len(test_pairs),
            "eval_ratio": args.eval_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "cluster_mode": args.cluster_mode,
            "cluster_counts": dict(Counter(row["cluster_id"] for row in pair_rows)),
            "output_root": root,
        }
    )


if __name__ == "__main__":
    main()
