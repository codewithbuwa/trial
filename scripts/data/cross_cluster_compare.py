import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp

df = pd.read_csv("combined_per_row.csv")

# ---- 1. Clean comparison table (means + n per cluster) ----
metrics = [
    "prompt_len", "chosen_len", "rejected_len", "chosen_rejected_len_ratio",
    "chosen_distinct1", "chosen_distinct2", "chosen_ttr", "chosen_entropy",
    "chosen_has_code", "chosen_has_latex", "chosen_has_list",
]

summary_rows = []
for cid, g in df.groupby("cluster_id"):
    row = {"cluster_id": cid, "n": len(g), "pct_of_total": 100 * len(g) / len(df)}
    for m in metrics:
        if m.startswith("chosen_has_"):
            row[m + "_pct"] = 100 * g[m].mean()
        else:
            row[m + "_mean"] = g[m].mean()
    summary_rows.append(row)

summary = pd.DataFrame(summary_rows).sort_values("cluster_id")
summary.to_csv("comparison_table.csv", index=False)
print("=== Comparison table ===")
print(summary.round(2).to_string(index=False))

# ---- 2. Kruskal-Wallis across all 4 clusters ----
print("\n=== Kruskal-Wallis across all 4 clusters ===")
kw_results = {}
numeric_fields = ["prompt_len", "chosen_len", "rejected_len", "chosen_rejected_len_ratio",
                   "chosen_distinct1", "chosen_distinct2", "chosen_ttr", "chosen_entropy",
                   "rejected_entropy"]
for field in numeric_fields:
    groups = [g[field].dropna().values for _, g in df.groupby("cluster_id")]
    h, p = stats.kruskal(*groups)
    kw_results[field] = (h, p)
    flag = " *** SIGNIFICANT" if p < 0.001 else (" * significant" if p < 0.05 else "")
    print(f"  {field:<28s} H={h:9.2f}  p={p:.2e}{flag}")

# ---- 3. Post-hoc pairwise (Dunn's test, Holm correction) for significant fields ----
print("\n=== Post-hoc pairwise (Dunn's test, Holm-corrected p-values) ===")
for field in numeric_fields:
    h, p = kw_results[field]
    if p >= 0.05:
        continue
    print(f"\n-- {field} --")
    dunn = sp.posthoc_dunn(df, val_col=field, group_col="cluster_id", p_adjust="holm")
    print(dunn.round(4).to_string())

# ---- 4. Effect sizes (epsilon-squared for Kruskal-Wallis) ----
print("\n=== Effect sizes (eta-squared_H, 0-1 scale, analogous to R^2) ===")
n_total = len(df)
k_groups = df["cluster_id"].nunique()
print(f"(n={n_total}, k={k_groups} groups -- with n this large, p-values are ~always significant;")
print(" effect size tells you if the difference is actually large)\n")
print("  guide: <0.01 negligible | 0.01-0.06 small | 0.06-0.14 medium | >0.14 large\n")
for field in numeric_fields:
    h, p = kw_results[field]
    eta_sq_h = (h - k_groups + 1) / (n_total - k_groups)
    if eta_sq_h >= 0.14:
        tag = "LARGE"
    elif eta_sq_h >= 0.06:
        tag = "medium"
    elif eta_sq_h >= 0.01:
        tag = "small"
    else:
        tag = "negligible"
    print(f"  {field:<28s} eta_sq_H={eta_sq_h:.4f}  ({tag})")

# ---- 5. Sanity check: median vs mean for length ratio (means get skewed by outliers) ----
print("\n=== chosen_rejected_len_ratio: mean vs median (checking outlier skew) ===")
for cid, g in df.groupby("cluster_id"):
    print(f"  {cid:<14s} mean={g['chosen_rejected_len_ratio'].mean():7.2f}  "
          f"median={g['chosen_rejected_len_ratio'].median():.2f}  "
          f"max={g['chosen_rejected_len_ratio'].max():.1f}")
