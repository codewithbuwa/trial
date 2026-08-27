from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def load_cluster_dir(path: Path) -> list[dict[str, Any]]:
    if not path.is_dir():
        raise ValueError(f"{path} is not a directory")
    rows: list[dict[str, Any]] = []
    for jsonl_path in sorted(path.glob("*.jsonl")):
        rows.extend(load_jsonl(jsonl_path))
    if not rows:
        raise ValueError(f"{path} contains no .jsonl files")
    return rows


def prompt_text(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction") or row.get("prompt") or "").strip()
    input_text = str(row.get("input") or "").strip()
    if not input_text:
        return instruction
    return f"{instruction}\n\n{input_text}"


def unique_prompt_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    prompts: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        prompt = prompt_text(row)
        prompt_id = str(row.get("prompt_id") or index)
        if prompt_id not in prompts:
            prompts[prompt_id] = {
                "prompt_id": prompt_id,
                "cluster_id": str(row.get("cluster_id", "unknown")),
                "prompt": prompt,
            }
    return list(prompts.values())


def token_length_rows(rows: Iterable[dict[str, Any]], *, encoding_name: str) -> list[dict[str, Any]]:
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("prompt_token_lengths.py requires tiktoken. Install it with `poetry add tiktoken`.") from exc

    encoding = tiktoken.get_encoding(encoding_name)
    output = []
    for row in unique_prompt_rows(rows):
        tokens = len(encoding.encode(row["prompt"]))
        output.append(
            {
                "prompt_id": row["prompt_id"],
                "cluster_id": row["cluster_id"],
                "num_tokens": tokens,
            }
        )
    return output


def cluster_vectors(rows: Iterable[dict[str, Any]]) -> dict[str, list[int]]:
    vectors: dict[str, list[int]] = {}
    for row in rows:
        vectors.setdefault(str(row["cluster_id"]), []).append(int(row["num_tokens"]))
    return {cluster_id: sorted(tokens, reverse=True) for cluster_id, tokens in sorted(vectors.items())}


def write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        rows = payload["records"]
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Plotting requires matplotlib. Install project dependencies first.") from exc
    return plt


def plot_histogram(path: Path, vectors: dict[str, list[int]], *, bins: int) -> None:
    plt = load_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    cluster_ids = sorted(vectors)

    fig, ax = plt.subplots(figsize=(11, 6))
    for cluster_id in cluster_ids:
        ax.hist(
            vectors[cluster_id],
            bins=bins,
            alpha=0.42,
            label=f"{cluster_id} (n={len(vectors[cluster_id])})",
        )
    ax.set_title("Prompt Token Length Histogram by Cluster")
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("Prompt count")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_violin(path: Path, vectors: dict[str, list[int]]) -> None:
    plt = load_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    cluster_ids = sorted(vectors)
    values = [vectors[cluster_id] for cluster_id in cluster_ids]

    fig, ax = plt.subplots(figsize=(11, 6))
    parts = ax.violinplot(values, vert=False, showmeans=True, showmedians=True, showextrema=True)
    for body in parts["bodies"]:
        body.set_alpha(0.55)
    ax.set_title("Prompt Token Length Violin Plot by Cluster")
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("Cluster")
    ax.set_yticks(range(1, len(cluster_ids) + 1), labels=cluster_ids)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_boxplot(path: Path, vectors: dict[str, list[int]]) -> None:
    plt = load_pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    cluster_ids = sorted(vectors)
    values = [vectors[cluster_id] for cluster_id in cluster_ids]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.boxplot(
        values,
        tick_labels=cluster_ids,
        vert=False,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.35},
    )
    ax.set_title("Prompt Token Length Boxplot by Cluster")
    ax.set_xlabel("Prompt tokens")
    ax.set_ylabel("Cluster")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute tiktoken prompt lengths grouped with cluster ids.")
    parser.add_argument("--input-jsonl", type=Path, help="Optional single JSONL input file.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/E5_cluster_ablation/embedding_4/cluster_jsonl"),
        help="Directory containing per-cluster JSONL files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/E5_cluster_ablation/embedding_4/prompt_token_lengths.json"),
        help="Output path. .json writes cluster vectors plus records; .jsonl writes one record per prompt.",
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="tiktoken encoding name.",
    )
    parser.add_argument(
        "--histogram-output",
        type=Path,
        default=Path("experiments/E5_cluster_ablation/embedding_4/prompt_token_histogram_by_cluster.png"),
        help="PNG output path for the per-cluster histogram.",
    )
    parser.add_argument(
        "--violin-output",
        type=Path,
        default=Path("experiments/E5_cluster_ablation/embedding_4/prompt_token_violin_by_cluster.png"),
        help="PNG output path for the per-cluster violin plot.",
    )
    parser.add_argument(
        "--boxplot-output",
        type=Path,
        default=Path("experiments/E5_cluster_ablation/embedding_4/prompt_token_boxplot_by_cluster.png"),
        help="PNG output path for the per-cluster boxplot with outliers.",
    )
    parser.add_argument("--bins", type=int, default=60, help="Number of histogram bins.")
    parser.add_argument("--no-plot", action="store_true", help="Skip writing the PNG distribution plot.")
    args = parser.parse_args()

    input_rows = load_jsonl(args.input_jsonl) if args.input_jsonl else load_cluster_dir(args.input_dir)
    rows = token_length_rows(input_rows, encoding_name=args.encoding)
    rows.sort(key=lambda row: (str(row["cluster_id"]), -int(row["num_tokens"]), str(row["prompt_id"])))
    vectors = cluster_vectors(rows)
    payload = {
        "encoding": args.encoding,
        "input": str(args.input_jsonl or args.input_dir),
        "clusters": vectors,
        "records": rows,
    }
    write_output(args.output, payload)
    if not args.no_plot:
        plot_histogram(args.histogram_output, vectors, bins=args.bins)
        plot_violin(args.violin_output, vectors)
        plot_boxplot(args.boxplot_output, vectors)

    print(
        json.dumps(
            {
                "input": str(args.input_jsonl or args.input_dir),
                "output": str(args.output),
                "plot_outputs": None
                if args.no_plot
                else {
                    "histogram": str(args.histogram_output),
                    "violin": str(args.violin_output),
                    "boxplot": str(args.boxplot_output),
                },
                "encoding": args.encoding,
                "num_prompts": len(rows),
                "max_num_tokens": max((int(row["num_tokens"]) for row in rows), default=0),
                "clusters": {
                    cluster_id: {
                        "num_prompts": len(tokens),
                        "max_num_tokens": max(tokens) if tokens else 0,
                    }
                    for cluster_id, tokens in vectors.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
