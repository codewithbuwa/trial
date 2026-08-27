"""
cluster_metrics.py

Descriptive + comparative metrics per cluster, for prompt/completion data in
the UltraFeedback-style pairwise schema:

    {"prompt_id": ..., "instruction": ..., "input": ..., "chosen": ...,
     "rejected": ..., "cluster_id": ...}

Metrics computed per row:
  - prompt / chosen / rejected token length (tiktoken cl100k_base if
    available, else whitespace-token fallback)
  - chosen:rejected length ratio
  - distinct-1 / distinct-2 (unique n-grams / total n-grams) for chosen
    and rejected, as a repetitiveness proxy
  - type-token ratio (TTR) for chosen and rejected
  - word-level entropy for chosen and rejected
  - structural flags: contains code block, contains LaTeX/math, contains
    a bullet/numbered list

Then aggregated per cluster_id (mean, std, n), plus optional
Kruskal-Wallis tests across clusters for a few key fields to flag which
differences are unlikely to be noise.

Usage:
    python cluster_metrics.py --input data.jsonl --output cluster_summary.csv
    python cluster_metrics.py --input data.jsonl --output cluster_summary.csv \
        --row-output per_row_metrics.csv
"""
import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def n_tokens(text: str) -> int:
        return len(_enc.encode(text)) if text else 0
except ImportError:
    _enc = None

    def n_tokens(text: str) -> int:
        return len(re.findall(r"\S+", text)) if text else 0


CODE_BLOCK_RE = re.compile(r"```")
LATEX_RE = re.compile(r"\$[^$]+\$|\\\[|\\\(")
LIST_RE = re.compile(r"^\s*([-*]|\d+\.)\s+", re.MULTILINE)
WORD_RE = re.compile(r"\w+")


def words(text: str):
    return WORD_RE.findall(text.lower()) if text else []


def distinct_n(text: str, n: int) -> float:
    w = words(text)
    if len(w) < n:
        return 0.0
    ngrams = list(zip(*[w[i:] for i in range(n)]))
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def type_token_ratio(text: str) -> float:
    w = words(text)
    return len(set(w)) / len(w) if w else 0.0


def word_entropy(text: str) -> float:
    w = words(text)
    if not w:
        return 0.0
    counts = Counter(w)
    total = len(w)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def has_code(text: str) -> bool:
    return bool(CODE_BLOCK_RE.search(text or ""))


def has_latex(text: str) -> bool:
    return bool(LATEX_RE.search(text or ""))


def has_list(text: str) -> bool:
    return bool(LIST_RE.search(text or ""))


def load_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_row_metrics(row: dict) -> dict:
    prompt_text = f"{row.get('instruction', '')}\n{row.get('input', '')}".strip()
    chosen = row.get("chosen", "") or ""
    rejected = row.get("rejected", "") or ""

    chosen_len = n_tokens(chosen)
    rejected_len = n_tokens(rejected)

    return {
        "prompt_id": row.get("prompt_id"),
        "cluster_id": row.get("cluster_id"),
        "prompt_len": n_tokens(prompt_text),
        "chosen_len": chosen_len,
        "rejected_len": rejected_len,
        "chosen_rejected_len_ratio": (chosen_len / rejected_len) if rejected_len else float("nan"),
        "chosen_distinct1": distinct_n(chosen, 1),
        "chosen_distinct2": distinct_n(chosen, 2),
        "rejected_distinct1": distinct_n(rejected, 1),
        "rejected_distinct2": distinct_n(rejected, 2),
        "chosen_ttr": type_token_ratio(chosen),
        "rejected_ttr": type_token_ratio(rejected),
        "chosen_entropy": word_entropy(chosen),
        "rejected_entropy": word_entropy(rejected),
        "chosen_has_code": has_code(chosen),
        "chosen_has_latex": has_latex(chosen),
        "chosen_has_list": has_list(chosen),
    }


NUMERIC_FIELDS = [
    "prompt_len", "chosen_len", "rejected_len", "chosen_rejected_len_ratio",
    "chosen_distinct1", "chosen_distinct2", "rejected_distinct1", "rejected_distinct2",
    "chosen_ttr", "rejected_ttr", "chosen_entropy", "rejected_entropy",
]
BOOL_FIELDS = ["chosen_has_code", "chosen_has_latex", "chosen_has_list"]


def aggregate_by_cluster(rows_metrics: list) -> dict:
    by_cluster = defaultdict(list)
    for m in rows_metrics:
        by_cluster[m["cluster_id"]].append(m)

    summary = {}
    for cluster_id, items in by_cluster.items():
        s = {"n": len(items)}
        for field in NUMERIC_FIELDS:
            vals = [m[field] for m in items if not (isinstance(m[field], float) and math.isnan(m[field]))]
            s[f"{field}_mean"] = statistics.mean(vals) if vals else float("nan")
            s[f"{field}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        for field in BOOL_FIELDS:
            vals = [m[field] for m in items]
            s[f"{field}_pct"] = 100 * sum(vals) / len(vals) if vals else float("nan")
        summary[cluster_id] = s
    return summary


def kruskal_wallis_by_cluster(rows_metrics: list, field: str):
    try:
        from scipy import stats as scipy_stats
    except ImportError:
        return None
    by_cluster = defaultdict(list)
    for m in rows_metrics:
        v = m[field]
        if isinstance(v, float) and math.isnan(v):
            continue
        by_cluster[m["cluster_id"]].append(v)
    groups = [vals for vals in by_cluster.values() if len(vals) > 1]
    if len(groups) < 2:
        return None
    stat, p = scipy_stats.kruskal(*groups)
    return {"field": field, "H": stat, "p": p}


def write_summary_csv(summary: dict, path: str):
    all_fields = sorted({k for v in summary.values() for k in v})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id"] + all_fields)
        for cluster_id, vals in sorted(summary.items(), key=lambda kv: str(kv[0])):
            writer.writerow([cluster_id] + [vals.get(field, "") for field in all_fields])


def write_row_csv(rows_metrics: list, path: str):
    if not rows_metrics:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(rows_metrics)


def main():
    parser = argparse.ArgumentParser(description="Per-cluster prompt/completion metrics")
    parser.add_argument("--input", required=True, help="Path to JSONL file")
    parser.add_argument("--output", default="cluster_summary.csv", help="Per-cluster summary CSV")
    parser.add_argument("--row-output", default=None, help="Optional per-row metrics CSV")
    args = parser.parse_args()

    rows = load_rows(args.input)
    rows_metrics = [compute_row_metrics(r) for r in rows]

    summary = aggregate_by_cluster(rows_metrics)
    write_summary_csv(summary, args.output)
    print(f"[cluster_metrics] {len(rows)} rows -> {len(summary)} clusters -> {args.output}")

    if args.row_output:
        write_row_csv(rows_metrics, args.row_output)
        print(f"[cluster_metrics] per-row metrics -> {args.row_output}")

    print("\nKruskal-Wallis across clusters (needs >=2 clusters w/ >1 sample; requires scipy):")
    for field in ["prompt_len", "chosen_len", "chosen_distinct1", "chosen_entropy"]:
        result = kruskal_wallis_by_cluster(rows_metrics, field)
        if result:
            flag = " *" if result["p"] < 0.05 else ""
            print(f"  {field:<20s} H={result['H']:.3f}  p={result['p']:.4f}{flag}")
        else:
            print(f"  {field:<20s} skipped (need scipy + >=2 clusters with >1 sample each)")


if __name__ == "__main__":
    main()