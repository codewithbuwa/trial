from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cpo_trl_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cpo_trl_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_RESULTS = {
    "SFT": Path("outputs/checkpoints/sft/pairwise_accuracy.json"),
    "DPO": Path("outputs/checkpoints/dpo/pairwise_accuracy.json"),
    "KTO": Path("outputs/checkpoints/kto/pairwise_accuracy.json"),
    "CPO_UNARY": Path("outputs/checkpoints/cpo_unary/pairwise_accuracy.json"),
    "CPO": Path("outputs/checkpoints/cpo/pairwise_accuracy.json"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SFT/DPO/KTO/CPO evaluation comparisons.")
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="Model result as NAME=path/to/pairwise_accuracy.json. Defaults to outputs/{sft,dpo,kto,cpo}/pairwise_accuracy.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evals"))
    parser.add_argument(
        "--training-dir",
        action="append",
        default=[],
        help="Training log directory as NAME=path. Defaults to outputs/{sft,dpo,kto,cpo}.",
    )
    return parser.parse_args()


def parse_named_paths(values: list[str], defaults: dict[str, Path]) -> dict[str, Path]:
    if not values:
        return defaults
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=path, got: {value}")
        name, path = value.split("=", 1)
        parsed[name] = Path(path)
    return parsed


def load_results(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    results = {}
    for name, path in paths.items():
        if path.exists():
            results[name] = json.loads(path.read_text(encoding="utf-8"))
    if not results:
        missing = ", ".join(str(path) for path in paths.values())
        raise FileNotFoundError(f"no pairwise_accuracy JSON files found; looked for: {missing}")
    return results


def companion_margins_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}_margins.jsonl")


def load_margins(paths: dict[str, Path]) -> dict[str, list[float]]:
    margins: dict[str, list[float]] = {}
    for name, result_path in paths.items():
        path = companion_margins_path(result_path)
        if not path.exists():
            continue
        values = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    values.append(float(json.loads(line)["margin"]))
        margins[name] = values
    return margins


def label_bars_inside(ax, bars) -> None:  # type: ignore[no-untyped-def]
    """Place black vertical .8f labels in the middle of bars."""

    for bar in bars:
        value = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        y = value / 2
        ax.text(
            x,
            y,
            f"{value:.8f}",
            ha="center",
            va="center",
            rotation=90,
            color="black",
            fontsize=9,
        )


def save_bar(labels: list[str], values: list[float], title: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(labels)])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(1.0, max(values, default=0.0) * 1.15))
    label_bars_inside(ax, bars)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pairwise_accuracy_by_model(results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    labels = list(results)
    save_bar(
        labels,
        [float(results[name]["pairwise_accuracy"]) for name in labels],
        "Preference Pairwise Accuracy by Model",
        "Pairwise Accuracy",
        output_dir / "pairwise_accuracy_by_model.png",
    )


def plot_mean_margin_by_model(results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    labels = list(results)
    values = [float(results[name]["mean_margin"]) for name in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#B279A2"][: len(labels)])
    ax.axhline(0.0, color="#333333", linewidth=1)
    ax.set_title("Mean Chosen-Rejected Logprob Margin by Model")
    ax.set_ylabel("Mean margin")
    lower = min(0.0, min(values, default=0.0) * 1.15)
    upper = max(0.0, max(values, default=0.0) * 1.15)
    if lower == upper:
        lower, upper = -1.0, 1.0
    ax.set_ylim(lower, upper)
    label_bars_inside(ax, bars)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_margin_by_model.png", dpi=180)
    plt.close(fig)


def plot_cluster_pairwise_accuracy(results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    preferred = ["coding", "math", "writing", "general"]
    clusters = [
        cluster
        for cluster in preferred
        if any(cluster in result.get("clusters", {}) for result in results.values())
    ]
    extra = sorted(
        {
            cluster
            for result in results.values()
            for cluster in result.get("clusters", {})
            if cluster not in preferred
        }
    )
    clusters.extend(extra)
    if not clusters:
        return

    labels = list(results)
    width = 0.8 / max(1, len(labels))
    x_positions = list(range(len(clusters)))
    fig, ax = plt.subplots(figsize=(max(9, len(clusters) * 1.5), 5))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
    grouped_bars = []
    for model_index, name in enumerate(labels):
        values = [
            float(results[name].get("clusters", {}).get(cluster, {}).get("pairwise_accuracy", 0.0))
            for cluster in clusters
        ]
        offsets = [position - 0.4 + width / 2 + model_index * width for position in x_positions]
        bars = ax.bar(offsets, values, width=width, label=name, color=colors[model_index % len(colors)])
        grouped_bars.append(bars)
    ax.set_title("Cluster Pairwise Accuracy by Model")
    ax.set_ylabel("Pairwise Accuracy")
    ax.set_xticks(x_positions, clusters)
    ax.set_ylim(0, 1.0)
    for bars in grouped_bars:
        label_bars_inside(ax, bars)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cluster_pairwise_accuracy_by_model.png", dpi=180)
    plt.close(fig)


def plot_margin_histograms(margins: dict[str, list[float]], output_dir: Path) -> None:
    if not margins:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, values in margins.items():
        if values:
            ax.hist(values, bins=50, alpha=0.45, label=name)
    ax.axvline(0.0, color="#333333", linewidth=1)
    ax.set_title("Chosen-Rejected Logprob Margin Distribution")
    ax.set_xlabel("Margin")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "margin_histogram.png", dpi=180)
    plt.close(fig)


def find_trainer_state(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("**/trainer_state.json"), key=lambda path: len(path.parts))
    return candidates[-1] if candidates else None


def load_training_series(training_dirs: dict[str, Path]) -> tuple[dict[str, list[tuple[int, float]]], dict[str, list[tuple[int, float]]]]:
    losses: dict[str, list[tuple[int, float]]] = {}
    margins: dict[str, list[tuple[int, float]]] = {}
    for name, directory in training_dirs.items():
        state_path = find_trainer_state(directory)
        if state_path is None:
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for entry in state.get("log_history", []):
            step = int(entry.get("step", len(losses.get(name, []))))
            if "loss" in entry:
                losses.setdefault(name, []).append((step, float(entry["loss"])))
            if "rewards/margins" in entry:
                margins.setdefault(name, []).append((step, float(entry["rewards/margins"])))
            elif "reward_margin" in entry:
                margins.setdefault(name, []).append((step, float(entry["reward_margin"])))
    return losses, margins


def plot_series(series: dict[str, list[tuple[int, float]]], title: str, ylabel: str, path: Path) -> None:
    if not series:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, values in series.items():
        if values:
            xs, ys = zip(*values, strict=True)
            ax.plot(xs, ys, label=name)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_paths = parse_named_paths(args.result, DEFAULT_RESULTS)
    training_dirs = parse_named_paths(
        args.training_dir,
        {name: Path(f"outputs/{name.lower()}") for name in DEFAULT_RESULTS},
    )
    results = load_results(result_paths)
    plot_pairwise_accuracy_by_model(results, args.output_dir)
    plot_mean_margin_by_model(results, args.output_dir)
    plot_cluster_pairwise_accuracy(results, args.output_dir)
    plot_margin_histograms(load_margins(result_paths), args.output_dir)
    losses, reward_margins = load_training_series(training_dirs)
    plot_series(losses, "Training Loss over Steps", "Loss", args.output_dir / "training_loss.png")
    plot_series(
        reward_margins,
        "Reward Margin over Steps",
        "Reward margin",
        args.output_dir / "reward_margin.png",
    )
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
