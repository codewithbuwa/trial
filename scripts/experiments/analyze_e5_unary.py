"""
Analyse and plot the E5 unary cluster-ablation results.

Compares three preference-optimization runs on Qwen2.5-1.5B-Instruct
(lr=1e-5, beta=0.01, grad-norm=0.3, ~1 epoch / 4000 steps), all evaluated on
the SAME 4000-example validation set (identical chosen/rejected lengths),
differing only in how the training/eval data is clustered:

  * cpo_embedding4 : CPO (token-KL z-baseline, alpha=0) on 4 embedding clusters
  * cpo_random4    : CPO (same config) on 4 random clusters
  * kto            : KTO baseline

Outputs a multi-panel PNG and prints a text analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results_e5_unary"
OUTDIR = RESULTS / "analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

RUNS = {
    "CPO embedding4": RESULTS / "cpo_unary_embedding4/cpo_lr1em05_b0p01_gn0p3_a0_token-kl",
    "CPO random4": RESULTS / "cpo_unary_random4/cpo_lr1em05_b0p01_gn0p3_a0_token-kl",
    "KTO": RESULTS / "kto/kto_lr1em05_b0p01_gn0p3",
}
COLORS = {"CPO embedding4": "#2166ac", "CPO random4": "#67a9cf", "KTO": "#b2182b"}


def load_winrate(run_dir: Path) -> dict:
    return json.loads((run_dir / "winrate.json").read_text())


def load_grouped(run_dir: Path) -> list[dict]:
    path = run_dir / "train_metrics_grouped.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


wr = {name: load_winrate(d) for name, d in RUNS.items()}
grouped = {name: load_grouped(d) for name, d in RUNS.items()}

# ----------------------------------------------------------------------------
# Text analysis
# ----------------------------------------------------------------------------
print("=" * 78)
print("E5 UNARY CLUSTER ABLATION  —  headline metrics (n=4000, shared eval set)")
print("=" * 78)
hdr = f"{'method':<16}{'winrate':>9}{'norm_wr':>9}{'rew_acc':>9}{'norm_rew':>10}" \
      f"{'mean_mrg':>10}{'rew_mrg':>9}{'tok_drift':>10}"
print(hdr)
for name, w in wr.items():
    print(f"{name:<16}{w['winrate']:>9.3f}{w['normalized_winrate']:>9.3f}"
          f"{w['reward_accuracy']:>9.3f}{w['normalized_reward_accuracy']:>10.3f}"
          f"{w['mean_margin']:>10.1f}{w['mean_reward_margin']:>9.3f}"
          f"{w['mean_token_eval_logratio']:>10.3f}")
print()
print("Reading:")
print("  * winrate      = P(chosen logp-sum > rejected logp-sum), length-sensitive")
print("  * norm_wr      = same, per-token normalized (length-controlled)")
print("  * rew_acc      = DPO-style reward-margin accuracy vs reference model")
print("  * tok_drift    = mean per-token logratio vs reference (more negative = more drift)")
print()

# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.30)

# --- Panel A: headline metrics grouped bars -------------------------------
axA = fig.add_subplot(gs[0, 0])
metrics = [
    ("winrate", "Winrate"),
    ("normalized_winrate", "Norm.\nwinrate"),
    ("reward_accuracy", "Reward\nacc."),
    ("normalized_reward_accuracy", "Norm.\nrew. acc."),
]
x = np.arange(len(metrics))
width = 0.25
for i, (name, w) in enumerate(wr.items()):
    vals = [w[k] for k, _ in metrics]
    bars = axA.bar(x + (i - 1) * width, vals, width, label=name, color=COLORS[name])
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7)
axA.axhline(0.5, ls="--", lw=0.8, color="grey", alpha=0.7)
axA.text(len(metrics) - 0.5, 0.505, "chance", fontsize=7, color="grey")
axA.set_xticks(x)
axA.set_xticklabels([lbl for _, lbl in metrics], fontsize=8)
axA.set_ylim(0.48, 0.72)
axA.set_ylabel("score")
axA.set_title("A. Headline accuracy metrics", fontweight="bold", fontsize=11)
axA.legend(fontsize=8, loc="upper left")
axA.grid(axis="y", alpha=0.3)

# --- Panel B: per-cluster winrate (raw vs normalized) ---------------------
axB = fig.add_subplot(gs[0, 1])
# Use the two runs on the SAME embedding_4 eval set for a fair per-cluster view.
for name, marker in [("KTO", "s"), ("CPO embedding4", "o")]:
    clusters = wr[name]["clusters"]
    keys = sorted(clusters)
    raw = [clusters[k]["winrate"] for k in keys]
    norm = [clusters[k]["normalized_winrate"] for k in keys]
    axB.plot(keys, raw, marker=marker, color=COLORS[name], label=f"{name} (raw)")
    axB.plot(keys, norm, marker=marker, ls="--", color=COLORS[name], alpha=0.6,
             label=f"{name} (norm)")
axB.axhline(0.5, ls=":", lw=0.8, color="grey")
axB.set_ylabel("winrate")
axB.set_title("B. Per-cluster winrate (embedding_4 eval)", fontweight="bold", fontsize=11)
axB.tick_params(axis="x", rotation=30)
axB.legend(fontsize=7)
axB.grid(alpha=0.3)

# --- Panel C: length bias — cluster winrate vs mean chosen length ----------
axC = fig.add_subplot(gs[0, 2])
for name in ["KTO", "CPO embedding4"]:
    clusters = wr[name]["clusters"]
    keys = sorted(clusters)
    lengths = [clusters[k]["mean_chosen_length"] for k in keys]
    raw = [clusters[k]["winrate"] for k in keys]
    norm = [clusters[k]["normalized_winrate"] for k in keys]
    axC.scatter(lengths, raw, color=COLORS[name], marker="o", s=60,
                label=f"{name} raw")
    axC.scatter(lengths, norm, color=COLORS[name], marker="x", s=60,
                label=f"{name} norm")
    for k, ln, r in zip(keys, lengths, raw):
        axC.annotate(k.split("_")[-1], (ln, r), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
axC.set_xlabel("mean chosen length (tokens)")
axC.set_ylabel("winrate")
axC.set_title("C. Length bias: raw winrate falls\non long-response clusters",
              fontweight="bold", fontsize=11)
axC.legend(fontsize=7)
axC.grid(alpha=0.3)

# --- Panel D: margins & reference drift ------------------------------------
axD = fig.add_subplot(gs[1, 0])
names = list(wr)
rew_margin = [wr[n]["mean_reward_margin"] for n in names]
drift = [-wr[n]["mean_token_eval_logratio"] for n in names]  # positive = drift magnitude
xx = np.arange(len(names))
axD2 = axD.twinx()
b1 = axD.bar(xx - 0.2, rew_margin, 0.4, color=[COLORS[n] for n in names],
             label="mean reward margin")
b2 = axD2.bar(xx + 0.2, drift, 0.4, color="lightgrey", edgecolor="k", hatch="//",
              label="ref. drift |token logratio|")
axD.set_xticks(xx)
axD.set_xticklabels(names, fontsize=8, rotation=10)
axD.set_ylabel("mean reward margin", color="#2166ac")
axD2.set_ylabel("per-token drift from reference", color="grey")
axD.set_title("D. Margin size vs reference drift", fontweight="bold", fontsize=11)
for b, v in zip(b1, rew_margin):
    axD.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
for b, v in zip(b2, drift):
    axD2.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
axD.grid(axis="y", alpha=0.3)

# --- Panel E: training reward-margin curves --------------------------------
axE = fig.add_subplot(gs[1, 1])
for name, rows in grouped.items():
    steps = [r["step"] for r in rows]
    if name == "KTO":
        rm = [r["preference_signal"]["reward_margin"] for r in rows]
    else:
        rm = [r["preference_movement"]["reward_margin"] for r in rows]
    axE.plot(steps, rm, marker="o", ms=3, color=COLORS[name], label=name)
axE.set_xlabel("training step")
axE.set_ylabel("train reward margin")
axE.set_title("E. Training preference signal", fontweight="bold", fontsize=11)
axE.legend(fontsize=8)
axE.grid(alpha=0.3)

# --- Panel F: training loss curves -----------------------------------------
axF = fig.add_subplot(gs[1, 2])
for name, rows in grouped.items():
    steps = [r["step"] for r in rows]
    loss = [r["objective"]["loss"] for r in rows]
    axF.plot(steps, loss, marker="o", ms=3, color=COLORS[name], label=name)
axF.set_xlabel("training step")
axF.set_ylabel("loss")
axF.set_title("F. Training loss", fontweight="bold", fontsize=11)
axF.legend(fontsize=8)
axF.grid(alpha=0.3)

fig.suptitle("E5 Unary Cluster Ablation — CPO (embedding vs random clusters) vs KTO  "
             "| Qwen2.5-1.5B-Instruct, shared 4000-example eval",
             fontsize=13, fontweight="bold")

out_png = OUTDIR / "e5_unary_analysis.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"[saved] {out_png}")
