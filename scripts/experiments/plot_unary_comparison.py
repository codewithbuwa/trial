"""Figures for the three-way unary comparison (KTO vs CPO-unary keyword vs random4).

Reads output_unary/ logs and writes PNGs into output_unary/plots/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/Users/jordanbuwa/Documents/cpo_trl/output_unary")
OUT = BASE / "plots"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("KTO", "kto", "tab:blue"),
    ("CPO-unary\nkeyword", "cpo_unary_keyword", "tab:green"),
    ("CPO-unary\nrandom4", "cpo_unary_random4", "tab:orange"),
]
N = 1000  # teacher-forced eval size


def wr(d):
    return json.load(open(BASE / d / "pairwise_accuracy.json"))


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def ci95(p, n):
    return 1.96 * math.sqrt(p * (1 - p) / n)


# 1. Teacher-forced metrics grouped bars with 95% CI on pairwise_accuracy
def plot_teacher_forced():
    data = {label: wr(d) for label, d, _ in MODELS}
    metrics = [
        ("normalized_pairwise_accuracy", "Norm. pairwise accuracy"),
        ("normalized_reward_accuracy", "Norm. reward acc."),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(metrics))
    w = 0.25
    for i, (label, d, color) in enumerate(MODELS):
        vals = [data[label][k] for k, _ in metrics]
        errs = [ci95(v, N) for v in vals]
        bars = ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=4,
                      color=color, label=label.replace("\n", " "))
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, (0.45 + v) / 2, f"{v:.4f}", ha="center", va="center",
                    rotation=90, color="white", fontweight="bold", fontsize=8)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x, [m[1] for m in metrics])
    ax.set_ylim(0.45, 0.72)
    ax.set_ylabel("rate")
    ax.set_title("Teacher-forced metrics (n=1000, error bars = 95% CI)\n"
                 "all pairwise gaps < CI — not significant")
    ax.legend()
    savefig(fig, "1_teacher_forced.png")


# 2. Judge scores
def plot_judge():
    j = json.load(open(BASE / "judge" / "prometheus_summary.json"))["models"]
    key = {"KTO": "KTO", "CPO-unary\nkeyword": "CPO_UNARY_KEYWORD",
           "CPO-unary\nrandom4": "CPO_UNARY_RANDOM4"}
    labels = [m[0] for m in MODELS]
    vals = [j[key[l]]["judge_score"] for l in labels]
    colors = [m[2] for m in MODELS]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(labels)), vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, (0.4 + v) / 2, f"{v:.4f}", ha="center", va="center",
                rotation=90, color="white", fontweight="bold", fontsize=9)
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="0.5 (tournament mean)")
    ax.set_xticks(range(len(labels)), [l.replace("\n", " ") for l in labels])
    ax.set_ylim(0.4, 0.56)
    ax.set_ylabel("Prometheus judge score")
    ax.set_title("Prometheus judge (200 prompts/pair, 400 comparisons/model)\n"
                 "head-to-head differences not significant")
    ax.legend()
    savefig(fig, "2_judge_score.png")


# 3. Head-to-head diverging bars with significance
def plot_head_to_head():
    pw = json.load(open(BASE / "judge" / "prometheus_summary.json"))["pairwise"]

    def binom_p(k, n):
        if n == 0:
            return 1.0
        z = (k - n / 2) / math.sqrt(n * 0.25)
        return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    rows = []
    for key, v in pw.items():
        a, b = key.split("__vs__")
        wa, wb, t = v.get(a, 0), v.get(b, 0), v.get("ties", 0)
        p = binom_p(max(wa, wb), wa + wb)
        rows.append((a.replace("CPO_UNARY_", "").replace("_", " "),
                     b.replace("CPO_UNARY_", "").replace("_", " "), wa, wb, t, p))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(rows))
    for i, (a, b, wa, wb, t, p) in enumerate(rows):
        ax.barh(i, -wa, color="tab:green", alpha=0.85)
        ax.barh(i, wb, color="tab:blue", alpha=0.85)
        ax.barh(i, t, left=-t / 2, color="lightgray", alpha=0.9)
        ax.text(-wa - 3, i, f"{a}: {wa}", ha="right", va="center", fontsize=8)
        ax.text(wb + 3, i, f"{b}: {wb}", ha="left", va="center", fontsize=8)
        ax.text(0, i + 0.28, f"ties {t}  ·  p={p:.2f}{' *' if p < 0.05 else ' (ns)'}",
                ha="center", va="bottom", fontsize=7, color="dimgray")
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, [f"{a}\nvs {b}" for a, b, *_ in rows], fontsize=8)
    ax.set_xlabel("← left model preferred      |      right model preferred →")
    ax.set_title("Prometheus head-to-head (200 prompts/pair)\n"
                 "keyword vs random4 = 93–93 (p=1.0); nothing significant")
    ax.set_xlim(-120, 120)
    savefig(fig, "3_head_to_head.png")


# 4. Metric disagreement: normalized pairwise accuracy vs judge score
def plot_disagreement():
    data = {label: wr(d) for label, d, _ in MODELS}
    j = json.load(open(BASE / "judge" / "prometheus_summary.json"))["models"]
    key = {"KTO": "KTO", "CPO-unary\nkeyword": "CPO_UNARY_KEYWORD",
           "CPO-unary\nrandom4": "CPO_UNARY_RANDOM4"}
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, d, color in MODELS:
        x = data[label]["normalized_pairwise_accuracy"]
        yv = j[key[label]]["judge_score"]
        ax.scatter(x, yv, s=160, color=color, zorder=3)
        ax.annotate(label.replace("\n", " "), (x, yv),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("normalized pairwise accuracy (teacher-forced)")
    ax.set_ylabel("Prometheus judge score")
    ax.set_title("Metric disagreement\n(log-prob best = keyword, judge best = KTO)")
    ax.grid(alpha=0.3)
    savefig(fig, "4_pairwise_accuracy_vs_judge.png")


# 5. Per-cluster normalized pairwise accuracy (grouped)
def plot_cluster():
    import collections
    clusters = ["coding", "general", "writing", "math"]
    data = {}
    for label, d, _ in MODELS:
        rows = [json.loads(l) for l in open(BASE / d / "pairwise_accuracy_margins.jsonl")]
        byc = collections.defaultdict(list)
        for r in rows:
            byc[r.get("cluster_id", "?")].append(bool(r.get("normalized_pairwise_correct")))
        data[label] = {c: (sum(byc[c]) / len(byc[c]) if byc.get(c) else np.nan) for c in clusters}
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(clusters))
    w = 0.25
    for i, (label, d, color) in enumerate(MODELS):
        vals = [data[label][c] for c in clusters]
        ax.bar(x + (i - 1) * w, vals, w, color=color, label=label.replace("\n", " "))
        for xi, v in zip(x + (i - 1) * w, vals):
            if not np.isnan(v):
                ax.text(xi, (0.45 + v) / 2, f"{v:.4f}", ha="center", va="center",
                        rotation=90, color="white", fontweight="bold", fontsize=7)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x, [f"{c}" for c in clusters])
    ax.set_ylim(0.45, 0.72)
    ax.set_ylabel("normalized pairwise accuracy")
    ax.set_title("Per-cluster normalized pairwise accuracy (eval clusters shared)\n"
                 "near-identical across models; math n=29 (noisy)")
    ax.legend()
    savefig(fig, "5_cluster_pairwise_accuracy.png")


# 6. Cluster reference z_k (active) + cluster sizes
def plot_zk():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for m, ax, title in [("cpo_unary_keyword", ax1, "keyword clusters"),
                         ("cpo_unary_random4", ax2, "random4 clusters")]:
        s = json.load(open(BASE / m / "cpo_state.json"))["loss_computer"]
        z = s["z_values"]
        cc = s["z_counts"]
        clusters = list(z.keys())
        zv = [z[c] for c in clusters]
        ax.bar(range(len(clusters)), zv, color="tab:purple", alpha=0.85)
        for i, (c, v) in enumerate(zip(clusters, zv)):
            ax.text(i, v / 2, f"{v:.4f}", ha="center", va="center",
                    rotation=90, color="white", fontweight="bold", fontsize=8)
            ax.text(i, v + 1, f"n={cc.get(c,0)}", ha="center", fontsize=7)
        ax.set_xticks(range(len(clusters)), clusters, rotation=20, fontsize=8)
        ax.set_ylabel("final $z_k$ (token-KL reference)")
        ax.set_title(f"{title}")
        ax.set_ylim(0, max(zv) * 1.25)
    fig.suptitle("Cluster reference $z_k$ is active (~44–78), yet outcomes tie")
    savefig(fig, "6_zk_reference.png")


def main():
    plot_teacher_forced()
    plot_judge()
    plot_head_to_head()
    plot_disagreement()
    plot_cluster()
    plot_zk()
    print("\nAll plots ->", OUT)


if __name__ == "__main__":
    main()
