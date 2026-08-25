from __future__ import annotations

from scripts.audit.correctness_report import build_report


def test_correctness_report_declares_required_e0_checks() -> None:
    report = build_report(run=False)

    assert report["experiment"] == "E0_correctness"
    assert report["all_passed"] is True
    assert {check["name"] for check in report["checks"]} == {
        "alpha_endpoints",
        "pair_construction",
        "ema_state",
        "finite_values",
        "pair_aware_sampler",
    }
    assert all("result" not in check for check in report["checks"])
