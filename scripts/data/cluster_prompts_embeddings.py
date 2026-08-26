from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the 3D projection

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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


def scatter_plot_3d(
    xyz: Any,
    labels: list[str],
    *,
    title: str,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for cluster_id in sorted(set(labels)):
        indices = [index for index, label in enumerate(labels) if label == cluster_id]
        ax.scatter(
            [xyz[index, 0] for index in indices],
            [xyz[index, 1] for index in indices],
            [xyz[index, 2] for index in indices],
            s=8,
            alpha=0.75,
            label=cluster_id,
        )
    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.set_zlabel("component 3")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_interactive_3d_html(
    xyz: Any,
    labels: list[str],
    prompts: dict[str, str],
    prompt_ids: list[str],
    *,
    title: str,
    output_path: Path,
) -> None:
    cluster_ids = sorted(set(labels))
    palette = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#4f46e5",
        "#be123c",
    ]
    colors = {cluster_id: palette[index % len(palette)] for index, cluster_id in enumerate(cluster_ids)}
    points = [
        {
            "prompt_id": prompt_id,
            "cluster_id": labels[index],
            "x": float(xyz[index, 0]),
            "y": float(xyz[index, 1]),
            "z": float(xyz[index, 2]),
            "instruction": prompts[prompt_id],
            "color": colors[labels[index]],
        }
        for index, prompt_id in enumerate(prompt_ids)
    ]
    payload = json.dumps({"title": title, "points": points, "colors": colors}, ensure_ascii=True)
    html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    html, body {
      height: 100%;
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f8fafc;
      color: #0f172a;
    }
    body {
      display: grid;
      grid-template-rows: auto 1fr;
    }
    header {
      padding: 14px 18px;
      border-bottom: 1px solid #cbd5e1;
      background: #ffffff;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }
    .meta {
      margin-top: 4px;
      color: #475569;
      font-size: 13px;
    }
    main {
      position: relative;
      min-height: 0;
    }
    canvas {
      display: block;
      width: 100%;
      height: 100%;
      cursor: grab;
    }
    canvas:active {
      cursor: grabbing;
    }
    .legend {
      position: absolute;
      top: 14px;
      right: 14px;
      display: grid;
      gap: 6px;
      padding: 10px 12px;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      font-size: 13px;
    }
    .legend-row {
      display: flex;
      align-items: center;
      gap: 8px;
      white-space: nowrap;
    }
    .swatch {
      width: 10px;
      height: 10px;
      border-radius: 999px;
    }
    .tooltip {
      position: absolute;
      max-width: 420px;
      display: none;
      padding: 8px 10px;
      background: #0f172a;
      color: #f8fafc;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.35;
      pointer-events: none;
      box-shadow: 0 10px 24px rgba(15, 23, 42, 0.24);
    }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <div class="meta">Drag to rotate. Scroll to zoom. Hover points for prompt details.</div>
  </header>
  <main>
    <canvas id="plot"></canvas>
    <div class="legend" id="legend"></div>
    <div class="tooltip" id="tooltip"></div>
  </main>
  <script>
    const data = __DATA__;
    const canvas = document.getElementById("plot");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("tooltip");
    const legend = document.getElementById("legend");
    let width = 0;
    let height = 0;
    let scale = 1;
    let rotationX = -0.58;
    let rotationY = 0.72;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let projected = [];

    for (const [cluster, color] of Object.entries(data.colors)) {
      const row = document.createElement("div");
      row.className = "legend-row";
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = color;
      const label = document.createElement("span");
      label.textContent = cluster;
      row.append(swatch, label);
      legend.append(row);
    }

    const bounds = data.points.reduce((acc, point) => {
      acc.minX = Math.min(acc.minX, point.x);
      acc.maxX = Math.max(acc.maxX, point.x);
      acc.minY = Math.min(acc.minY, point.y);
      acc.maxY = Math.max(acc.maxY, point.y);
      acc.minZ = Math.min(acc.minZ, point.z);
      acc.maxZ = Math.max(acc.maxZ, point.z);
      return acc;
    }, {minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity});

    function normalize(point) {
      const cx = (bounds.minX + bounds.maxX) / 2;
      const cy = (bounds.minY + bounds.maxY) / 2;
      const cz = (bounds.minZ + bounds.maxZ) / 2;
      const span = Math.max(bounds.maxX - bounds.minX, bounds.maxY - bounds.minY, bounds.maxZ - bounds.minZ) || 1;
      return {
        x: (point.x - cx) / span,
        y: (point.y - cy) / span,
        z: (point.z - cz) / span,
      };
    }

    function rotate(point) {
      const cosX = Math.cos(rotationX);
      const sinX = Math.sin(rotationX);
      const cosY = Math.cos(rotationY);
      const sinY = Math.sin(rotationY);
      const y1 = point.y * cosX - point.z * sinX;
      const z1 = point.y * sinX + point.z * cosX;
      const x2 = point.x * cosY + z1 * sinY;
      const z2 = -point.x * sinY + z1 * cosY;
      return {x: x2, y: y1, z: z2};
    }

    function resize() {
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      width = rect.width;
      height = rect.height;
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      draw();
    }

    function drawAxes() {
      ctx.strokeStyle = "#cbd5e1";
      ctx.lineWidth = 1;
      const axes = [
        [{x: -0.55, y: 0, z: 0}, {x: 0.55, y: 0, z: 0}, "x"],
        [{x: 0, y: -0.55, z: 0}, {x: 0, y: 0.55, z: 0}, "y"],
        [{x: 0, y: 0, z: -0.55}, {x: 0, y: 0, z: 0.55}, "z"],
      ];
      for (const [from, to, label] of axes) {
        const a = project(from);
        const b = project(to);
        ctx.beginPath();
        ctx.moveTo(a.sx, a.sy);
        ctx.lineTo(b.sx, b.sy);
        ctx.stroke();
        ctx.fillStyle = "#64748b";
        ctx.fillText(label, b.sx + 5, b.sy + 5);
      }
    }

    function project(point) {
      const rotated = rotate(point);
      const depth = 1.8 + rotated.z;
      const perspective = scale / depth;
      const size = Math.min(width, height) * 0.9;
      return {
        sx: width / 2 + rotated.x * size * perspective,
        sy: height / 2 - rotated.y * size * perspective,
        depth: rotated.z,
      };
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#f8fafc";
      ctx.fillRect(0, 0, width, height);
      ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      drawAxes();
      projected = data.points.map((point) => ({...point, ...project(normalize(point))}));
      projected.sort((a, b) => a.depth - b.depth);
      for (const point of projected) {
        ctx.beginPath();
        ctx.arc(point.sx, point.sy, 4, 0, Math.PI * 2);
        ctx.fillStyle = point.color;
        ctx.globalAlpha = 0.78;
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    function nearestPoint(clientX, clientY) {
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      let best = null;
      let bestDistance = 12;
      for (const point of projected) {
        const distance = Math.hypot(point.sx - x, point.sy - y);
        if (distance < bestDistance) {
          best = point;
          bestDistance = distance;
        }
      }
      return best;
    }

    canvas.addEventListener("pointerdown", (event) => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (dragging) {
        rotationY += (event.clientX - lastX) * 0.008;
        rotationX += (event.clientY - lastY) * 0.008;
        lastX = event.clientX;
        lastY = event.clientY;
        tooltip.style.display = "none";
        draw();
        return;
      }
      const point = nearestPoint(event.clientX, event.clientY);
      if (!point) {
        tooltip.style.display = "none";
        return;
      }
      tooltip.style.display = "block";
      tooltip.style.left = `${event.clientX + 12}px`;
      tooltip.style.top = `${event.clientY + 12}px`;
      tooltip.textContent = `${point.prompt_id} | cluster ${point.cluster_id}: ${point.instruction}`;
    });
    canvas.addEventListener("pointerup", () => {
      dragging = false;
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      scale = Math.max(0.35, Math.min(3.5, scale * (event.deltaY < 0 ? 1.08 : 0.92)));
      draw();
    }, {passive: false});
    window.addEventListener("resize", resize);
    resize();
  </script>
</body>
</html>
"""
    html = html.replace("__TITLE__", html_lib.escape(title)).replace("__DATA__", payload)
    output_path.write_text(html, encoding="utf-8")


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
    pca_xyz = PCA(n_components=3, random_state=args.seed).fit_transform(embeddings)
    perplexity = min(args.tsne_perplexity, max(1.0, (len(prompt_ids) - 1) / 3))
    tsne_xy = TSNE(
        n_components=2,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(embeddings)
    tsne_xyz = TSNE(
        n_components=3,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    ).fit_transform(embeddings)

    scatter_plot(pca_xy, labels, title="PCA Prompt Clusters", output_path=args.output_dir / "pca_clusters.png")
    scatter_plot(tsne_xy, labels, title="t-SNE Prompt Clusters", output_path=args.output_dir / "tsne_clusters.png")
    scatter_plot_3d(
        pca_xyz,
        labels,
        title="3D PCA Prompt Clusters",
        output_path=args.output_dir / "pca_clusters_3d.png",
    )
    scatter_plot_3d(
        tsne_xyz,
        labels,
        title="3D t-SNE Prompt Clusters",
        output_path=args.output_dir / "tsne_clusters_3d.png",
    )
    write_interactive_3d_html(
        pca_xyz,
        labels,
        prompts,
        prompt_ids,
        title="Interactive 3D PCA Prompt Clusters",
        output_path=args.output_dir / "pca_clusters_3d.html",
    )
    write_interactive_3d_html(
        tsne_xyz,
        labels,
        prompts,
        prompt_ids,
        title="Interactive 3D t-SNE Prompt Clusters",
        output_path=args.output_dir / "tsne_clusters_3d.html",
    )

    with (args.output_dir / "prompt_clusters.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "prompt_id",
                "cluster_id",
                "pca_x",
                "pca_y",
                "pca_z",
                "tsne_x",
                "tsne_y",
                "tsne_z",
                "instruction",
            ),
        )
        writer.writeheader()
        for index, prompt_id in enumerate(prompt_ids):
            writer.writerow(
                {
                    "prompt_id": prompt_id,
                    "cluster_id": labels[index],
                    "pca_x": pca_xy[index, 0],
                    "pca_y": pca_xy[index, 1],
                    "pca_z": pca_xyz[index, 2],
                    "tsne_x": tsne_xy[index, 0],
                    "tsne_y": tsne_xy[index, 1],
                    "tsne_z": tsne_xyz[index, 2],
                    "instruction": prompts[prompt_id],
                }
            )

    print(json.dumps({"cluster_counts": cluster_counts, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
