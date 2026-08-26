from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from cpo_trl.data.datasets import load_jsonl, write_jsonl
from scripts.data.prepare_ultrafeedback import assign_embedding_clusters


def unique_prompts(rows: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(row["prompt_id"]): str(row["instruction"])
        for row in sorted(rows, key=lambda row: str(row["prompt_id"]))
    }


def scatter_plot(
    xy: Any,
    labels: list[str],
    *,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for cluster_id in sorted(set(labels)):
        indices = [index for index, label in enumerate(labels) if label == cluster_id]
        ax.scatter(
            [xy[index, 0] for index in indices],
            [xy[index, 1] for index in indices],
            s=8,
            alpha=0.75,
            label=cluster_id,
        )
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster prompts with embeddings and visualize in 2D.")
    parser.add_argument("--input-jsonl", type=Path, default=Path("data/processed/dpo/train.jsonl"))
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/E5_cluster_ablation/embedding_4"))
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise RuntimeError(
            "Embedding clustering requires sentence-transformers and scikit-learn. "
            "Install them with `poetry add sentence-transformers scikit-learn`."
        ) from exc

    rows = load_jsonl(args.input_jsonl)
    if not rows:
        raise ValueError(f"{args.input_jsonl} contains no rows")

    model = SentenceTransformer(args.model_name)
    clustered_rows = assign_embedding_clusters(
        rows,
        seed=args.seed,
        n_clusters=args.n_clusters,
        model_name=args.model_name,
        batch_size=args.batch_size,
        embedder=model,
    )
    prompts = unique_prompts(clustered_rows)
    prompt_ids = list(prompts)
    texts = [prompts[prompt_id] for prompt_id in prompt_ids]
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    prompt_to_cluster = {
        str(row["prompt_id"]): str(row["cluster_id"])
        for row in clustered_rows
    }
    labels = [prompt_to_cluster[prompt_id] for prompt_id in prompt_ids]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl is not None:
        write_jsonl(args.output_jsonl, clustered_rows)

    cluster_counts = dict(sorted(Counter(labels).items()))
    (args.output_dir / "cluster_counts.json").write_text(
        json.dumps(
            {
                "input_jsonl": str(args.input_jsonl),
                "rows": len(clustered_rows),
                "prompts": len(prompt_ids),
                "model_name": args.model_name,
                "n_clusters": args.n_clusters,
                "seed": args.seed,
                "cluster_counts": cluster_counts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pca_xy = PCA(n_components=2, random_state=args.seed).fit_transform(embeddings)
    perplexity = min(args.tsne_perplexity, max(1.0, (len(prompt_ids) - 1) / 3))
    tsne_xy = TSNE(
        n_components=2,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(embeddings)

    scatter_plot(pca_xy, labels, title="PCA Prompt Clusters", output_path=args.output_dir / "pca_clusters.png")
    scatter_plot(tsne_xy, labels, title="t-SNE Prompt Clusters", output_path=args.output_dir / "tsne_clusters.png")

    with (args.output_dir / "prompt_clusters.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("prompt_id", "cluster_id", "pca_x", "pca_y", "tsne_x", "tsne_y", "instruction"),
        )
        writer.writeheader()
        for index, prompt_id in enumerate(prompt_ids):
            writer.writerow(
                {
                    "prompt_id": prompt_id,
                    "cluster_id": labels[index],
                    "pca_x": pca_xy[index, 0],
                    "pca_y": pca_xy[index, 1],
                    "tsne_x": tsne_xy[index, 0],
                    "tsne_y": tsne_xy[index, 1],
                    "instruction": prompts[prompt_id],
                }
            )

    print(json.dumps({"cluster_counts": cluster_counts, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
