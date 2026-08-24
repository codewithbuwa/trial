"""Generate cross-run plots for the method x hyperparameter sweep.

Reads output/sweeps/sweep_results.jsonl (54 runs: cpo/dpo/kto x lr x beta x
max_grad_norm) and writes PNGs into output/sweeps/sweep_plots/.
Everything is summary-level (one row per run); there are no per-step curves.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/Users/jordanbuwa/Documents/cpo_trl/output/sweeps")
OUT = BASE / "sweep_plots"
OUT.mkdir(parents=True, exist_ok=True)

METHODS = ["kto", "cpo", "dpo"]
MCOLOR = {"kto": "tab:blue", "cpo": "tab:green", "dpo": "tab:red"}
PRIMARY = "normalized_winrate"


def load():
    rows = [json.loads(l) for l in (BASE / "sweep_results.jsonl").open() if l.strip()]
    return [r for r in rows if r.get("status") == "ok"]


def savefig(fig, name, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(OUT / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def by_method(rows):
    d = {m: [r for r in rows if r["method"] == m] for m in METHODS}
    return d


def run_label(r):
    """Human-readable explicit-parameter label for a run."""
    parts = [f"{r['method'].upper():3s}", f"lr={r['learning_rate']:g}", f"β={r['beta']:g}",
             f"gn={r['max_grad_norm']:g}"]
    if r.get("alpha") is not None:
        parts.append(f"α={r['alpha']:g}")
    return "  ".join(parts)


# 1. best-by-method grouped bars
def plot_best_by_method(rows):
    dm = by_method(rows)
    metrics = ["normalized_winrate", "winrate", "reward_accuracy"]
    best = {m: max(dm[m], key=lambda r: r[PRIMARY]) for m in METHODS}
    x = np.arange(len(metrics))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, m in enumerate(METHODS):
        vals = [best[m][k] for k in metrics]
        ax.bar(x + (i - 1) * w, vals, w, label=run_label(best[m]), color=MCOLOR[m])
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(x, metrics)
    ax.set_ylabel("rate")
    ax.set_ylim(0.45, 0.65)
    ax.set_title("Best run per method")
    ax.legend(fontsize=7, loc="lower right")
    savefig(fig, "1_best_by_method.png")


# 2. lr x beta heatmaps, faceted method x grad_norm
def plot_heatmaps(rows):
    lrs = sorted({r["learning_rate"] for r in rows})
    betas = sorted({r["beta"] for r in rows})
    gns = sorted({r["max_grad_norm"] for r in rows})
    vmin = min(r[PRIMARY] for r in rows)
    vmax = max(r[PRIMARY] for r in rows)
    fig, axes = plt.subplots(len(METHODS), len(gns), figsize=(4 * len(gns) + 1.2, 3.2 * len(METHODS)),
                             constrained_layout=True)
    for i, m in enumerate(METHODS):
        for j, gn in enumerate(gns):
            ax = axes[i, j]
            mat = np.full((len(betas), len(lrs)), np.nan)
            for r in rows:
                if r["method"] == m and r["max_grad_norm"] == gn:
                    bi = betas.index(r["beta"])
                    li = lrs.index(r["learning_rate"])
                    mat[bi, li] = r[PRIMARY]
            im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(lrs)), [f"{x:g}" for x in lrs], fontsize=8)
            ax.set_yticks(range(len(betas)), [f"{b:g}" for b in betas], fontsize=8)
            ax.set_title(f"{m.upper()}  gn={gn:g}", fontsize=9)
            if i == len(METHODS) - 1:
                ax.set_xlabel("learning rate")
            if j == 0:
                ax.set_ylabel("beta")
            for bi in range(len(betas)):
                for li in range(len(lrs)):
                    if not np.isnan(mat[bi, li]):
                        ax.text(li, bi, f"{mat[bi, li]:.3f}", ha="center", va="center",
                                fontsize=7, color="white" if mat[bi, li] < (vmin + vmax) / 2 else "black")
    fig.colorbar(im, ax=axes, label=PRIMARY, shrink=0.6, location="right")
    fig.suptitle("Normalized winrate over lr x beta (faceted by method x grad-norm)")
    savefig(fig, "2_lr_beta_heatmaps.png", tight=False)


# 3. distribution of primary metric by method
def plot_distribution(rows):
    dm = by_method(rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, m in enumerate(METHODS):
        vals = [r[PRIMARY] for r in dm[m]]
        jitter = np.random.default_rng(0).normal(0, 0.05, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=MCOLOR[m], alpha=0.7, s=30)
        ax.hlines(np.median(vals), i - 0.25, i + 0.25, color="black", lw=2)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xticks(range(len(METHODS)), [m.upper() for m in METHODS])
    ax.set_ylabel(PRIMARY)
    ax.set_title("Spread of normalized winrate across the grid (bar = median)")
    ax.grid(alpha=0.3, axis="y")
    savefig(fig, "3_distribution_by_method.png")


# 4. winrate vs reward accuracy agreement
def plot_agreement(rows):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        ax.scatter([r["reward_accuracy"] for r in rs], [r[PRIMARY] for r in rs],
                   color=MCOLOR[m], label=m.upper(), alpha=0.75, s=35)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("reward accuracy (train-side)")
    ax.set_ylabel("normalized winrate (eval-side)")
    ax.set_title("Metric agreement across runs")
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, "4_winrate_vs_reward_accuracy.png")


# 5. quality vs drift
def plot_quality_vs_drift(rows):
    fig, ax = plt.subplots(figsize=(7, 5))
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m and r.get("sampled_mean_kl") is not None]
        ax.scatter([r["sampled_mean_kl"] for r in rs], [r[PRIMARY] for r in rs],
                   color=MCOLOR[m], label=m.upper(), alpha=0.75, s=35)
    ax.set_xlabel("sampled mean KL (signed drift from reference)")
    ax.set_ylabel("normalized winrate")
    ax.set_title("Quality vs drift across runs")
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, "5_quality_vs_drift.png")


# 6. grad-norm effect (0.3 vs 1.0), paired by (method, lr, beta)
def plot_gradnorm_effect(rows):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    idx = {}
    for r in rows:
        key = (r["method"], r["learning_rate"], r["beta"])
        idx.setdefault(key, {})[r["max_grad_norm"]] = r[PRIMARY]
    xs = {"kto": 0, "cpo": 1, "dpo": 2}
    for key, gns in idx.items():
        if 0.3 in gns and 1.0 in gns:
            m = key[0]
            x = xs[m]
            ax.plot([x - 0.15, x + 0.15], [gns[0.3], gns[1.0]], "-o", color=MCOLOR[m], alpha=0.5, ms=4)
    ax.set_xticks(list(xs.values()), [f"{m.upper()}\n(gn 0.3 -> 1.0)" for m in xs])
    ax.set_ylabel(PRIMARY)
    ax.set_title("Effect of max_grad_norm (each line = one lr/beta config)")
    ax.grid(alpha=0.3, axis="y")
    savefig(fig, "6_gradnorm_effect.png")


# 7. lr and beta marginal effects
def plot_hparam_effects(rows):
    lrs = sorted({r["learning_rate"] for r in rows})
    betas = sorted({r["beta"] for r in rows})
    fig, (axl, axb) = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in METHODS:
        rs = [r for r in rows if r["method"] == m]
        lr_mean = [np.mean([r[PRIMARY] for r in rs if r["learning_rate"] == lr]) for lr in lrs]
        b_mean = [np.mean([r[PRIMARY] for r in rs if r["beta"] == b]) for b in betas]
        axl.plot(range(len(lrs)), lr_mean, "o-", color=MCOLOR[m], label=m.upper())
        axb.plot(range(len(betas)), b_mean, "o-", color=MCOLOR[m], label=m.upper())
    axl.set_xticks(range(len(lrs)), [f"{x:g}" for x in lrs])
    axl.set_xlabel("learning rate")
    axl.set_ylabel(f"mean {PRIMARY}")
    axl.set_title("LR effect (averaged over beta, grad-norm)")
    axl.legend()
    axl.grid(alpha=0.3)
    axb.set_xticks(range(len(betas)), [f"{b:g}" for b in betas])
    axb.set_xlabel("beta")
    axb.set_title("Beta effect (averaged over lr, grad-norm)")
    axb.legend()
    axb.grid(alpha=0.3)
    savefig(fig, "7_lr_beta_effects.png")


# 8. top-N leaderboard
def plot_leaderboard(rows, n=15):
    top = sorted(rows, key=lambda r: r[PRIMARY], reverse=True)[:n]
    top = top[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * n + 1))
    ys = range(len(top))
    ax.barh(list(ys), [r[PRIMARY] for r in top], color=[MCOLOR[r["method"]] for r in top])
    ax.set_yticks(list(ys), [run_label(r) for r in top], fontsize=7, fontfamily="monospace")
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    for y, r in zip(ys, top):
        ax.text(r[PRIMARY] + 0.001, y, f"{r[PRIMARY]:.3f}", va="center", fontsize=7)
    ax.set_xlim(0.5, max(r[PRIMARY] for r in top) + 0.02)
    ax.set_xlabel(PRIMARY)
    ax.set_title(f"Top {n} runs (all methods)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MCOLOR[m]) for m in METHODS]
    ax.legend(handles, [m.upper() for m in METHODS], fontsize=8, loc="lower right")
    savefig(fig, "8_leaderboard.png")


def main():
    rows = load()
    print(f"loaded {len(rows)} ok runs; methods={sorted({r['method'] for r in rows})}")
    plot_best_by_method(rows)
    plot_heatmaps(rows)
    plot_distribution(rows)
    plot_agreement(rows)
    plot_quality_vs_drift(rows)
    plot_gradnorm_effect(rows)
    plot_hparam_effects(rows)
    plot_leaderboard(rows)
    print("\nAll plots written to", OUT)


if __name__ == "__main__":
    main()
