"""Overlay the unary_loss (blue-curve) trajectory for CPO-mixed vs CPO-unary."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

BASE = Path("/Users/jordanbuwa/Documents/cpo_trl/output_compare")
OUT = BASE / "comparison_plots"
OUT.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("CPO-mixed (α=0.7)", BASE / "cpo" / "train_metrics.jsonl", "tab:blue"),
    ("CPO-unary (α=0)", BASE / "cpo_unary" / "train_metrics.jsonl", "tab:orange"),
]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def ema(xs, alpha=0.1):
    out, m = [], xs[0]
    for x in xs:
        m = alpha * x + (1 - alpha) * m
        out.append(m)
    return out


fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (label, path, color) in zip(axes, RUNS):
    r = load(path)
    steps = [x["step"] for x in r]
    u = [x["unary_loss"] for x in r]
    ax.plot(steps, u, color=color, alpha=0.3, lw=0.8, label="raw per-step")
    ax.plot(steps, ema(u), color=color, lw=2.2, label="EMA-smoothed")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="0.5 — sigmoid midpoint (loss when β·margin ≈ 0)")
    ax.set_xlabel("step")
    ax.set_title(label)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("unary_loss")
fig.suptitle("Unary loss — CPO-mixed vs CPO-unary  (bold = EMA-smoothed; β=0.01)")
fig.tight_layout()
path = OUT / "unary_loss_mixed_vs_unary.png"
fig.savefig(path, dpi=140, bbox_inches="tight")
plt.close(fig)
print("wrote", path)
