"""Generate alpha-sweep plots from CPO training/eval logs.

Reads the token-kl runs under output/alpha_sweeps/, keyed by their alpha value,
and writes PNGs into output/alpha_sweeps/sweep_plots/.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASE = Path("/Users/jordanbuwa/Documents/cpo_trl/output/alpha_sweeps")
OUT = BASE / "sweep_plots"
OUT.mkdir(parents=True, exist_ok=True)

CLUSTERS = ["math", "coding", "writing", "general"]


def alpha_from_name(name: str) -> float | None:
    m = re.search(r"_a([0-9p]+)_token-kl$", name)
    if not m:
        return None
    return float(m.group(1).replace("p", "."))


def load_runs() -> list[dict]:
    runs = []
    for d in sorted(BASE.glob("cpo_*_token-kl")):
        alpha = alpha_from_name(d.name)
        if alpha is None:
            continue
        tm_path = d / "train_metrics.jsonl"
        wr_path = d / "winrate.json"
        if not tm_path.exists():
            continue
        tm = [json.loads(l) for l in tm_path.open() if l.strip()]
        wr = json.load(wr_path.open()) if wr_path.exists() else None
        margins = None
        wm_path = d / "winrate_margins.jsonl"
        if wm_path.exists():
            margins = [json.loads(l) for l in wm_path.open() if l.strip()]
        runs.append({"alpha": alpha, "dir": d, "tm": tm, "wr": wr, "margins": margins})
    runs.sort(key=lambda r: r["alpha"])
    return runs


def series(tm, key):
    return [r[key] for r in tm if isinstance(r.get(key), (int, float))]


def steps(tm):
    return [r["step"] for r in tm]


def savefig(fig, name):
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------- A. vs-alpha summary ----------
def plot_summary_vs_alpha(runs):
    ev = [r for r in runs if r["wr"]]
    a = [r["alpha"] for r in ev]

    def g(key):
        return [r["wr"].get(key) for r in ev]

    # 1. winrate + reward accuracy vs alpha
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(a, g("winrate"), "o-", label="winrate")
    ax.plot(a, g("normalized_winrate"), "s-", label="normalized winrate")
    ax.plot(a, g("reward_accuracy"), "^--", label="reward accuracy")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("alpha (KTO=0  ->  DPO=1)")
    ax.set_ylabel("rate")
    ax.set_title("Win/accuracy vs alpha")
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, "A1_winrate_vs_alpha.png")

    # 2. margins vs alpha
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(a, g("mean_normalized_margin"), "o-", label="mean normalized margin")
    ax.set_xlabel("alpha")
    ax.set_ylabel("normalized margin")
    ax.set_title("Mean normalized preference margin vs alpha")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(a, g("mean_margin"), "s--", color="tab:red", label="mean raw margin")
    ax2.set_ylabel("raw margin", color="tab:red")
    savefig(fig, "A2_margin_vs_alpha.png")

    # 3. sampled KL vs alpha
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(a, g("sampled_mean_kl"), "o-", label="chosen+rejected mean")
    ax.plot(a, g("chosen_sampled_kl"), "s--", label="chosen")
    ax.plot(a, g("rejected_sampled_kl"), "^--", label="rejected")
    ax.set_xlabel("alpha")
    ax.set_ylabel("sampled KL from reference")
    ax.set_title("Policy drift (sampled KL) vs alpha")
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, "A3_sampled_kl_vs_alpha.png")

    # 4. generation length vs alpha
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(a, g("mean_chosen_length"), "o-", label="chosen")
    ax.plot(a, g("mean_rejected_length"), "s--", label="rejected")
    ax.set_xlabel("alpha")
    ax.set_ylabel("mean length (tokens)")
    ax.set_title("Response length vs alpha")
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, "A4_length_vs_alpha.png")


# ---------- C17. Pareto: quality vs drift ----------
def plot_pareto(runs):
    ev = [r for r in runs if r["wr"]]
    kl = [r["wr"]["sampled_mean_kl"] for r in ev]
    win = [r["wr"]["normalized_winrate"] for r in ev]
    a = [r["alpha"] for r in ev]
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(kl, win, c=a, cmap="viridis", s=90, zorder=3)
    for x, y, av in zip(kl, win, a):
        ax.annotate(f"a={av:g}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("sampled KL from reference (drift)")
    ax.set_ylabel("normalized winrate (quality)")
    ax.set_title("Quality vs drift across alpha")
    ax.grid(alpha=0.3)
    fig.colorbar(sc, label="alpha")
    savefig(fig, "C17_pareto_quality_vs_drift.png")


# ---------- B. training dynamics (small multiples) ----------
def plot_training_curves(runs):
    n = len(runs)
    cols = 3
    rows = (n + cols - 1) // cols

    # loss components
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True)
    for ax, r in zip(axes.ravel(), runs):
        st = steps(r["tm"])
        ax.plot(st, series(r["tm"], "loss"), label="total", lw=1.4)
        ax.plot(st, series(r["tm"], "unary_loss"), label="unary", lw=1)
        ax.plot(st, series(r["tm"], "pair_loss"), label="pair", lw=1)
        ax.set_title(f"alpha={r['alpha']:g}", fontsize=9)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Loss components vs step")
    savefig(fig, "B7_loss_components.png")

    # reward margin
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True)
    for ax, r in zip(axes.ravel(), runs):
        st = steps(r["tm"])
        ax.plot(st, series(r["tm"], "reward_margin"), color="tab:green", lw=1.2)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_title(f"alpha={r['alpha']:g}", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Training reward margin vs step")
    savefig(fig, "B8_reward_margin.png")

    # likelihood displacement: pos vs neg reward
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True)
    for ax, r in zip(axes.ravel(), runs):
        st = steps(r["tm"])
        ax.plot(st, series(r["tm"], "positive_reward_mean"), label="desirable", color="tab:blue", lw=1)
        ax.plot(st, series(r["tm"], "negative_reward_mean"), label="undesirable", color="tab:red", lw=1)
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.set_title(f"alpha={r['alpha']:g}", fontsize=9)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Reward by label vs step (likelihood displacement)")
    savefig(fig, "B9_reward_by_label.png")

    # grad norm
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True)
    for ax, r in zip(axes.ravel(), runs):
        st = steps(r["tm"])
        ax.plot(st, series(r["tm"], "grad_norm"), color="tab:purple", lw=1)
        ax.set_title(f"alpha={r['alpha']:g}", fontsize=9)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Gradient norm vs step")
    savefig(fig, "B10_grad_norm.png")


# ---------- B11. z_k dynamics ----------
def plot_zk(runs):
    n = len(runs)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), sharex=True)
    for ax, r in zip(axes.ravel(), runs):
        st = steps(r["tm"])
        for c in CLUSTERS:
            vals = [row.get("z_k", {}).get(c) for row in r["tm"]]
            if any(v is not None for v in vals):
                ax.plot(st, [v if v is not None else np.nan for v in vals], label=c, lw=1)
        ax.set_title(f"alpha={r['alpha']:g}", fontsize=9)
        ax.grid(alpha=0.3)
    axes.ravel()[0].legend(fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Cluster reference z_k vs step")
    savefig(fig, "B11_zk_dynamics.png")

    # final z_k heatmap: cluster x alpha
    a = [r["alpha"] for r in runs]
    mat = np.full((len(CLUSTERS), len(runs)), np.nan)
    for j, r in enumerate(runs):
        last = r["tm"][-1].get("z_k", {})
        for i, c in enumerate(CLUSTERS):
            if c in last:
                mat[i, j] = last[c]
    fig, ax = plt.subplots(figsize=(1.1 * len(runs) + 2, 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(runs)), [f"{x:g}" for x in a])
    ax.set_yticks(range(len(CLUSTERS)), CLUSTERS)
    ax.set_xlabel("alpha")
    ax.set_title("Final cluster reference z_k")
    for i in range(len(CLUSTERS)):
        for j in range(len(runs)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if mat[i, j] < np.nanmax(mat) * 0.6 else "black")
    fig.colorbar(im, label="z_k")
    savefig(fig, "D18_zk_final_heatmap.png")


# ---------- C15. per-cluster winrate vs alpha ----------
def plot_cluster_winrate(runs):
    ev = [r for r in runs if r["margins"]]
    a = [r["alpha"] for r in ev]
    data = {c: [] for c in CLUSTERS}
    for r in ev:
        by_c = {c: [] for c in CLUSTERS}
        for row in r["margins"]:
            c = row.get("cluster_id")
            if c in by_c:
                by_c[c].append(bool(row.get("normalized_win")))
        for c in CLUSTERS:
            data[c].append(np.mean(by_c[c]) if by_c[c] else np.nan)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in CLUSTERS:
        ax.plot(a, data[c], "o-", label=c)
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("alpha")
    ax.set_ylabel("normalized winrate")
    ax.set_title("Per-cluster winrate vs alpha")
    ax.legend()
    ax.grid(alpha=0.3)
    savefig(fig, "C15_cluster_winrate_vs_alpha.png")


# ---------- C13/16. margin distributions & length bias ----------
def plot_margin_distributions(runs):
    ev = [r for r in runs if r["margins"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = np.linspace(0, len(ev) - 1, len(ev))
    cmap = plt.get_cmap("viridis")
    for k, r in enumerate(ev):
        vals = np.array([row["normalized_margin"] for row in r["margins"]])
        vals = vals[np.isfinite(vals)]
        parts = ax.violinplot(vals, positions=[offsets[k]], widths=0.8, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(cmap(k / max(1, len(ev) - 1)))
            pc.set_alpha(0.7)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xticks(offsets, [f"{r['alpha']:g}" for r in ev])
    ax.set_xlabel("alpha")
    ax.set_ylabel("normalized margin (chosen - rejected)")
    ax.set_title("Per-example margin distribution vs alpha")
    ax.grid(alpha=0.3)
    savefig(fig, "C13_margin_distributions.png")


# Leaderboard: runs ranked by normalized winrate (alpha is the only varied knob)
def plot_leaderboard(runs):
    ev = [r for r in runs if r["wr"]]
    ev = sorted(ev, key=lambda r: r["wr"]["normalized_winrate"])  # ascending -> best on top
    a = [r["alpha"] for r in ev]
    vals = [r["wr"]["normalized_winrate"] for r in ev]
    cmap = plt.get_cmap("viridis")
    amax = max(a) or 1.0
    colors = [cmap(av / amax) for av in a]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(ev) + 1))
    ys = range(len(ev))
    ax.barh(list(ys), vals, color=colors)
    ax.set_yticks(list(ys), [f"α={av:g}" for av in a], fontsize=9, fontfamily="monospace")
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    for y, v in zip(ys, vals):
        ax.text(v + 0.0004, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.5, max(vals) + 0.006)
    ax.set_xlabel("normalized winrate")
    ax.set_title("Alpha sweep leaderboard\n(lr=1e-05, β=0.01, gn=0.3, z=token-kl)")
    savefig(fig, "L_leaderboard.png")


# Leaderboard by Prometheus judge score (round-robin win fraction per alpha)
def plot_judge_leaderboard():
    summary = BASE / "judge" / "prometheus_summary.json"
    if not summary.exists():
        print("no judge summary, skipping judge leaderboard")
        return
    d = json.load(summary.open())
    models = d.get("models", {})

    def alpha_of(key):  # "CPO_0p05" -> 0.05
        return float(key.replace("CPO_", "").replace("p", "."))

    items = sorted(
        ((alpha_of(k), v["judge_score"], v.get("comparisons")) for k, v in models.items()),
        key=lambda t: t[1],
    )
    a = [t[0] for t in items]
    vals = [t[1] for t in items]
    cmap = plt.get_cmap("viridis")
    amax = max(a) or 1.0
    colors = [cmap(av / amax) for av in a]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(items) + 1))
    ys = range(len(items))
    ax.barh(list(ys), vals, color=colors)
    ax.set_yticks(list(ys), [f"α={av:g}" for av in a], fontsize=9, fontfamily="monospace")
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    for y, v in zip(ys, vals):
        ax.text(v + 0.002, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(min(0.5, min(vals) - 0.02), max(vals) + 0.03)
    ax.set_xlabel("Prometheus judge score (round-robin win fraction)")
    ncmp = items[0][2] if items else None
    ax.set_title(f"Alpha sweep judge leaderboard\n({d.get('judge_model','judge')}, {ncmp} comparisons/model)")
    savefig(fig, "L_judge_leaderboard.png")


def main():
    runs = load_runs()
    print(f"loaded {len(runs)} token-kl runs: alphas={[r['alpha'] for r in runs]}")
    plot_summary_vs_alpha(runs)
    plot_pareto(runs)
    plot_training_curves(runs)
    plot_zk(runs)
    plot_cluster_winrate(runs)
    plot_margin_distributions(runs)
    plot_leaderboard(runs)
    plot_judge_leaderboard()
    print("\nAll plots written to", OUT)


if __name__ == "__main__":
    main()
