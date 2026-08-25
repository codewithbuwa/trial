from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_MODELS = {
    "SFT": "sft",
    "DPO": "dpo",
    "KTO": "kto",
    "CPO_UNARY": "cpo_unary",
    "CPO": "cpo",
}
SCALAR_TYPES = (int, float, str, bool)


def parse_named_paths(values: list[str], defaults: dict[str, Path]) -> dict[str, Path]:
    if not values:
        return defaults
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=path, got: {value}")
        name, path = value.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"missing name in: {value}")
        parsed[name] = Path(path.strip())
    return parsed


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def companion_margins_path(result_path: Path) -> Path:
    return result_path.with_name(f"{result_path.stem}_margins.jsonl")


def finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    return converted


def first_finite(record: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = finite_float(record.get(key))
        if value is not None:
            return value
    return None


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def distribution_stats(values: list[float]) -> dict[str, Any]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "positive_rate": None,
            "tie_rate": None,
        }
    sorted_values = sorted(finite_values)
    mean = sum(sorted_values) / len(sorted_values)
    variance = sum((value - mean) ** 2 for value in sorted_values) / len(sorted_values)
    return {
        "count": len(sorted_values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": sorted_values[0],
        "p05": percentile(sorted_values, 0.05),
        "p25": percentile(sorted_values, 0.25),
        "median": percentile(sorted_values, 0.50),
        "p75": percentile(sorted_values, 0.75),
        "p95": percentile(sorted_values, 0.95),
        "max": sorted_values[-1],
        "positive_rate": sum(value > 0 for value in sorted_values) / len(sorted_values),
        "tie_rate": sum(value == 0 for value in sorted_values) / len(sorted_values),
    }


def scalar_metrics(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key != "clusters" and isinstance(value, SCALAR_TYPES)
    }


def margin_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [value for record in records if (value := finite_float(record.get("margin"))) is not None]
    normalized = [
        value
        for record in records
        if (value := finite_float(record.get("normalized_margin"))) is not None
    ]
    reward_margins = [
        value
        for record in records
        if (value := finite_float(record.get("reward_margin"))) is not None
    ]
    normalized_reward_margins = [
        value
        for record in records
        if (value := finite_float(record.get("normalized_reward_margin"))) is not None
    ]
    eval_logratios = [
        value
        for record in records
        if (value := first_finite(record, "mean_eval_logratio", "sampled_mean_kl"))
        is not None
    ]
    token_eval_logratios = [
        value
        for record in records
        if (
            value := first_finite(
                record,
                "mean_token_eval_logratio",
                "normalized_sampled_mean_kl",
            )
        )
        is not None
    ]
    chosen_lengths = [
        value
        for record in records
        if (value := finite_float(record.get("chosen_length"))) is not None
    ]
    rejected_lengths = [
        value
        for record in records
        if (value := finite_float(record.get("rejected_length"))) is not None
    ]
    by_cluster: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        cluster_id = str(record.get("cluster_id", "unknown"))
        margin = finite_float(record.get("margin"))
        normalized_margin = finite_float(record.get("normalized_margin"))
        reward_margin = finite_float(record.get("reward_margin"))
        normalized_reward_margin = finite_float(record.get("normalized_reward_margin"))
        eval_logratio = first_finite(record, "mean_eval_logratio", "sampled_mean_kl")
        token_eval_logratio = first_finite(
            record,
            "mean_token_eval_logratio",
            "normalized_sampled_mean_kl",
        )
        if margin is not None:
            by_cluster[cluster_id]["margin"].append(margin)
        if normalized_margin is not None:
            by_cluster[cluster_id]["normalized_margin"].append(normalized_margin)
        if reward_margin is not None:
            by_cluster[cluster_id]["reward_margin"].append(reward_margin)
        if normalized_reward_margin is not None:
            by_cluster[cluster_id]["normalized_reward_margin"].append(normalized_reward_margin)
        if eval_logratio is not None:
            by_cluster[cluster_id]["mean_eval_logratio"].append(eval_logratio)
        if token_eval_logratio is not None:
            by_cluster[cluster_id]["mean_token_eval_logratio"].append(token_eval_logratio)
    return {
        "summed_margin": distribution_stats(margins),
        "normalized_margin": distribution_stats(normalized),
        "reward_margin": distribution_stats(reward_margins),
        "normalized_reward_margin": distribution_stats(normalized_reward_margins),
        "mean_eval_logratio": distribution_stats(eval_logratios),
        "mean_token_eval_logratio": distribution_stats(token_eval_logratios),
        # Backward-compatible aliases for older plot/report consumers.
        "sampled_mean_kl": distribution_stats(eval_logratios),
        "normalized_sampled_mean_kl": distribution_stats(token_eval_logratios),
        "chosen_length": distribution_stats(chosen_lengths),
        "rejected_length": distribution_stats(rejected_lengths),
        "by_cluster": {
            cluster_id: {
                "summed_margin": distribution_stats(values.get("margin", [])),
                "normalized_margin": distribution_stats(values.get("normalized_margin", [])),
                "reward_margin": distribution_stats(values.get("reward_margin", [])),
                "normalized_reward_margin": distribution_stats(
                    values.get("normalized_reward_margin", [])
                ),
                "mean_eval_logratio": distribution_stats(
                    values.get("mean_eval_logratio", [])
                ),
                "mean_token_eval_logratio": distribution_stats(
                    values.get("mean_token_eval_logratio", [])
                ),
                "sampled_mean_kl": distribution_stats(
                    values.get("mean_eval_logratio", [])
                ),
                "normalized_sampled_mean_kl": distribution_stats(
                    values.get("mean_token_eval_logratio", [])
                ),
            }
            for cluster_id, values in sorted(by_cluster.items())
        },
    }


def final_from_trainer_state(path: Path) -> dict[str, Any]:
    state = read_json(path)
    if state is None:
        return {"available": False, "path": str(path)}
    log_history = state.get("log_history", [])
    loss_entries = [entry for entry in log_history if "loss" in entry]
    grad_norms = [
        value
        for entry in log_history
        if (value := finite_float(entry.get("grad_norm"))) is not None
    ]
    learning_rates = [
        value
        for entry in log_history
        if (value := finite_float(entry.get("learning_rate"))) is not None
    ]
    reward_margins = [
        value
        for entry in log_history
        if (value := finite_float(entry.get("rewards/margins"))) is not None
    ]
    final_loss_entry = loss_entries[-1] if loss_entries else {}
    losses = [
        value
        for entry in loss_entries
        if (value := finite_float(entry.get("loss"))) is not None
    ]
    return {
        "available": True,
        "path": str(path),
        "global_step": state.get("global_step"),
        "best_metric": state.get("best_metric"),
        "num_log_steps": len(loss_entries),
        "final_step": final_loss_entry.get("step"),
        "final_loss": finite_float(final_loss_entry.get("loss")),
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "max_grad_norm": max(grad_norms) if grad_norms else None,
        "final_learning_rate": learning_rates[-1] if learning_rates else None,
        "final_reward_margin": reward_margins[-1] if reward_margins else None,
        "max_reward_margin": max(reward_margins) if reward_margins else None,
    }


def latest_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    return records[-1] if records else None


def cpo_training_report(directory: Path) -> dict[str, Any]:
    flat_records = read_jsonl(directory / "train_metrics.jsonl")
    grouped_records = read_jsonl(directory / "train_metrics_grouped.jsonl")
    diagnostic_records = read_jsonl(directory / "cpo_diagnostics.jsonl")
    final_diagnostic = next(
        (record for record in reversed(diagnostic_records) if record.get("event") == "final"),
        None,
    )
    sampler_diagnostic = next(
        (record for record in diagnostic_records if record.get("event") == "sampler"),
        None,
    )
    latest_flat = latest_record(flat_records)
    latest_grouped = latest_record(grouped_records)
    losses = [
        value
        for record in flat_records
        if (value := finite_float(record.get("loss"))) is not None
    ]
    grad_norms = [
        value
        for record in flat_records
        if (value := finite_float(record.get("grad_norm"))) is not None
    ]
    return {
        "available": bool(flat_records or grouped_records or diagnostic_records),
        "train_metrics_path": str(directory / "train_metrics.jsonl"),
        "train_metrics_grouped_path": str(directory / "train_metrics_grouped.jsonl"),
        "cpo_diagnostics_path": str(directory / "cpo_diagnostics.jsonl"),
        "num_log_steps": len(flat_records),
        "latest_flat": latest_flat,
        "latest_grouped": latest_grouped,
        "final_diagnostic": final_diagnostic,
        "sampler_diagnostic": sampler_diagnostic,
        "final_loss": finite_float(latest_flat.get("loss")) if latest_flat else None,
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "max_grad_norm": max(grad_norms) if grad_norms else None,
    }


def grouped_training_report(directory: Path) -> dict[str, Any]:
    records = read_jsonl(directory / "train_metrics_grouped.jsonl")
    latest = latest_record(records)
    return {
        "available": bool(records),
        "path": str(directory / "train_metrics_grouped.jsonl"),
        "num_log_steps": len(records),
        "latest": latest,
    }


def non_finite_report(directory: Path) -> dict[str, Any]:
    path = directory / "non_finite_gradients.jsonl"
    records = read_jsonl(path)
    return {
        "path": str(path),
        "exists": path.exists(),
        "count": len(records),
        "latest": latest_record(records),
    }


def judge_report(summary_path: Path, pairwise_path: Path) -> dict[str, Any]:
    summary = read_json(summary_path)
    pairwise_records = read_jsonl(pairwise_path)
    return {
        "available": summary is not None or bool(pairwise_records),
        "summary_path": str(summary_path),
        "pairwise_path": str(pairwise_path),
        "summary": summary,
        "pairwise_record_count": len(pairwise_records),
        "sample_records": pairwise_records[:5],
    }


def unary_reward_report(paths: dict[str, Path]) -> dict[str, Any]:
    return {
        model: {
            "available": (record := read_json(path)) is not None,
            "path": str(path),
            "summary": record,
            "records_path": str(path.with_name(f"{path.stem}_records.jsonl")),
            "records_exist": path.with_name(f"{path.stem}_records.jsonl").exists(),
        }
        for model, path in paths.items()
    }


def diagnostic_flags(report: dict[str, Any]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    for model, files in report["files"].items():
        if not files["winrate_json"]["exists"]:
            flags.append(
                {
                    "level": "error",
                    "name": "missing_winrate_json",
                    "message": f"{model} is missing {files['winrate_json']['path']}",
                }
            )
        if not files["margin_jsonl"]["exists"]:
            flags.append(
                {
                    "level": "warning",
                    "name": "missing_margin_jsonl",
                    "message": f"{model} is missing {files['margin_jsonl']['path']}",
                }
            )
    for model, metrics in report["model_metrics"].items():
        winrate = finite_float(metrics.get("normalized_winrate"))
        if winrate is not None and winrate < 0.5:
            flags.append(
                {
                    "level": "warning",
                    "name": "normalized_winrate_below_half",
                    "message": f"{model} normalized_winrate is {winrate:.6f}",
                }
            )
        chosen = finite_float(metrics.get("mean_chosen_length"))
        rejected = finite_float(metrics.get("mean_rejected_length"))
        if chosen is not None and rejected is not None:
            denom = max(1.0, (chosen + rejected) / 2.0)
            if abs(chosen - rejected) / denom > 0.15:
                flags.append(
                    {
                        "level": "info",
                        "name": "large_length_gap",
                        "message": (
                            f"{model} chosen/rejected mean lengths differ "
                            f"({chosen:.2f} vs {rejected:.2f})"
                        ),
                    }
                )
    cpo = report["training_diagnostics"].get("CPO", {})
    latest_flat = cpo.get("latest_flat") or {}
    final_diag = cpo.get("final_diagnostic") or {}
    sampler = cpo.get("sampler_diagnostic") or {}
    num_pairs = finite_float(latest_flat.get("num_pairs"))
    total_pairs = finite_float(final_diag.get("total_observed_pairs"))
    if num_pairs == 0 or total_pairs == 0:
        flags.append(
            {
                "level": "warning",
                "name": "cpo_no_pairs",
                "message": "CPO logged zero pairwise comparisons.",
            }
        )
    baseline_ready_rate = finite_float(sampler.get("baseline_ready_rate"))
    if baseline_ready_rate is not None and baseline_ready_rate < 0.8:
        flags.append(
            {
                "level": "warning",
                "name": "low_cpo_baseline_ready_rate",
                "message": f"CPO sampler baseline_ready_rate is {baseline_ready_rate:.6f}",
            }
        )
    for model, info in report["non_finite_events"].items():
        if info["count"] > 0:
            flags.append(
                {
                    "level": "warning",
                    "name": "non_finite_events_present",
                    "message": f"{model} has {info['count']} non-finite gradient events.",
                }
            )
    judge = report["judge_metrics"].get("summary") or {}
    judge_models = judge.get("models") if isinstance(judge, dict) else None
    if report["model_metrics"] and judge_models:
        logprob_best = max(
            report["model_metrics"],
            key=lambda model: finite_float(report["model_metrics"][model].get("normalized_winrate")) or -1.0,
        )
        judge_best = max(
            judge_models,
            key=lambda model: finite_float(judge_models[model].get("judge_score")) or -1.0,
        )
        if logprob_best != judge_best:
            flags.append(
                {
                    "level": "info",
                    "name": "judge_logprob_disagreement",
                    "message": f"logprob best is {logprob_best}, judge best is {judge_best}",
                }
            )
    for model, info in report.get("unary_reward_metrics", {}).items():
        if not info.get("available"):
            flags.append(
                {
                    "level": "info",
                    "name": "missing_unary_reward_eval",
                    "message": f"{model} is missing unary reward separation eval.",
                }
            )
            continue
        summary = info.get("summary") or {}
        separation = finite_float(summary.get("reward_separation"))
        normalized_separation = finite_float(summary.get("normalized_reward_separation"))
        if separation is not None and separation <= 0:
            flags.append(
                {
                    "level": "warning",
                    "name": "non_positive_unary_reward_separation",
                    "message": f"{model} reward_separation is {separation:.6f}",
                }
            )
        if normalized_separation is not None and normalized_separation <= 0:
            flags.append(
                {
                    "level": "warning",
                    "name": "non_positive_normalized_unary_reward_separation",
                    "message": (
                        f"{model} normalized_reward_separation is "
                        f"{normalized_separation:.6f}"
                    ),
                }
            )
    return flags


def build_report(
    *,
    result_paths: dict[str, Path],
    training_dirs: dict[str, Path],
    unary_reward_paths: dict[str, Path],
    judge_summary: Path,
    judge_pairwise: Path,
    output_root: Path,
) -> dict[str, Any]:
    result_records = {
        model: result
        for model, path in result_paths.items()
        if (result := read_json(path)) is not None
    }
    margin_records = {
        model: read_jsonl(companion_margins_path(path))
        for model, path in result_paths.items()
    }
    model_metrics = {model: scalar_metrics(result) for model, result in result_records.items()}
    cluster_metrics: dict[str, dict[str, Any]] = defaultdict(dict)
    for model, result in result_records.items():
        for cluster_id, metrics in result.get("clusters", {}).items():
            cluster_metrics[cluster_id][model] = metrics
    training_diagnostics = {}
    for model, directory in training_dirs.items():
        if model.startswith("CPO"):
            training_diagnostics[model] = cpo_training_report(directory)
        else:
            trainer_state = final_from_trainer_state(directory / "trainer_state.json")
            grouped = grouped_training_report(directory)
            training_diagnostics[model] = {
                **trainer_state,
                "grouped_metrics": grouped,
            }
    report = {
        "metadata": {
            "created_at": datetime.now(UTC).isoformat(),
            "output_root": str(output_root),
            "models": list(result_paths),
        },
        "files": {
            model: {
                "winrate_json": {"path": str(path), "exists": path.exists()},
                "margin_jsonl": {
                    "path": str(companion_margins_path(path)),
                    "exists": companion_margins_path(path).exists(),
                },
                "training_dir": {
                    "path": str(training_dirs.get(model, output_root / model.lower())),
                    "exists": training_dirs.get(model, output_root / model.lower()).exists(),
                },
            }
            for model, path in result_paths.items()
        },
        "model_metrics": model_metrics,
        "cluster_metrics": dict(sorted(cluster_metrics.items())),
        "margin_distribution": {
            model: margin_report(records) for model, records in margin_records.items()
        },
        "judge_metrics": judge_report(judge_summary, judge_pairwise),
        "unary_reward_metrics": unary_reward_report(unary_reward_paths),
        "training_diagnostics": training_diagnostics,
        "non_finite_events": {
            model: non_finite_report(directory) for model, directory in training_dirs.items()
        },
    }
    report["diagnostic_flags"] = diagnostic_flags(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a consolidated evaluation diagnostic report.")
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="Model result as NAME=path/to/winrate.json. Defaults to output/{sft,dpo,kto,cpo_unary,cpo}/winrate.json.",
    )
    parser.add_argument(
        "--training-dir",
        action="append",
        default=[],
        help="Training log directory as NAME=path. Defaults to output/{sft,dpo,kto,cpo_unary,cpo}.",
    )
    parser.add_argument(
        "--unary-result",
        action="append",
        default=[],
        help="Unary reward result as NAME=path. Defaults to output/{kto,cpo}/unary_rewards.json.",
    )
    parser.add_argument("--judge-summary", type=Path)
    parser.add_argument("--judge-pairwise", type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("output/evals/evaluation_report.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_defaults = {
        name: args.output_root / slug / "winrate.json" for name, slug in DEFAULT_MODELS.items()
    }
    training_defaults = {name: args.output_root / slug for name, slug in DEFAULT_MODELS.items()}
    unary_defaults = {
        "KTO": args.output_root / "kto" / "unary_rewards.json",
        "CPO_UNARY": args.output_root / "cpo_unary" / "unary_rewards.json",
        "CPO": args.output_root / "cpo" / "unary_rewards.json",
    }
    judge_summary = args.judge_summary or args.output_root / "judge" / "summary.json"
    judge_pairwise = args.judge_pairwise or args.output_root / "judge" / "pairwise.jsonl"
    report = build_report(
        result_paths=parse_named_paths(args.result, result_defaults),
        training_dirs=parse_named_paths(args.training_dir, training_defaults),
        unary_reward_paths=parse_named_paths(args.unary_result, unary_defaults),
        judge_summary=judge_summary,
        judge_pairwise=judge_pairwise,
        output_root=args.output_root,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": str(args.output_json), "flags": report["diagnostic_flags"]}, indent=2))


if __name__ == "__main__":
    main()
