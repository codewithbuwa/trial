"""Pareto plot for the 5-way final comparison: normalized winrate vs judge score.

Reads output_compare/<method>/winrate.json and
output_compare/judge/prometheus_summary_with_cpo_unary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path("/Users/jordanbuwa/Documents/cpo_trl/output_compare")
OUT = BASE / "comparison_plots"
OUT.mkdir(parents=True, exist_ok=True)

# method dir -> (display label, judge key, color)
METHODS = {
    "sft": ("SFT", "SFT", "tab:gray"),
    "dpo": ("DPO", "DPO", "tab:red"),
    "kto": ("KTO", "KTO", "tab:blue"),
    "cpo": ("CPO", "CPO", "tab:green"),
    "cpo_unary": ("CPO-unary", "CPO_UNARY", "tab:purple"),
}


def load():
    judge = json.load((BASE / "judge" / "prometheus_summary_with_cpo_unary.json").open())["models"]
    pts = []
    for d, (label, jkey, color) in METHODS.items():
        wr = json.load((BASE / d / "winrate.json").open())
        pts.append({
            "label": label,
            "color": color,
            "winrate": wr["normalized_winrate"],
            "judge": judge[jkey]["judge_score"],
        })
    return pts


def pareto_front(pts):
    """Points not dominated on (winrate, judge), both maximized."""
    front = []
    for p in pts:
        if not any(q is not p and q["winrate"] >= p["winrate"] and q["judge"] >= p["judge"]
                   and (q["winrate"] > p["winrate"] or q["judge"] > p["judge"]) for q in pts):
            front.append(p)
    return sorted(front, key=lambda p: p["winrate"])


def main():
    pts = load()
    front = pareto_front(pts)
    front_set = {id(p) for p in front}

    fig, ax = plt.subplots(figsize=(7.5, 6))
    # frontier line
    ax.plot([p["judge"] for p in front], [p["winrate"] for p in front],
            "--", color="black", lw=1.2, zorder=1, label="Pareto frontier")
    for p in pts:
        on = id(p) in front_set
        ax.scatter(p["judge"], p["winrate"], s=180 if on else 110,
                   color=p["color"], edgecolor="black" if on else "none",
                   linewidth=1.8, zorder=3)
        ax.annotate(
            f"{p['label']}\n({p['judge']:.3f}, {p['winrate']:.3f})",
            (p["judge"], p["winrate"]),
            textcoords="offset points", xytext=(9, 6), fontsize=8,
            fontweight="bold" if on else "normal",
        )
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("Prometheus judge score (generated-output)")
    ax.set_ylabel("normalized winrate (teacher-forced)")
    ax.set_title("Final comparison — quality Pareto\n(top-right = better on both metrics)")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "pareto_winrate_vs_judge.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("Pareto-optimal:", [p["label"] for p in front])
    print("wrote", path)


if __name__ == "__main__":
    main()
