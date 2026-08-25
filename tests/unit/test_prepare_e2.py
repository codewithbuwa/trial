from __future__ import annotations

import json

import yaml

from scripts.experiments.prepare_e2_controlled_baselines import RUNS, prepare_e2


def test_prepare_e2_writes_five_controlled_baseline_runs(tmp_path) -> None:
    experiment_dir = tmp_path / "E2"
    manifest_path = prepare_e2(experiment_dir=experiment_dir, output_root=tmp_path / "checkpoints")

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert [record["name"] for record in records] == [run[0] for run in RUNS]
    assert len(records) == 5
    assert any(record["name"] == "cpo_unary" and "scripts/train/train_cpo.py" in record["command"] for record in records)
    cpo_unary_config = yaml.safe_load((experiment_dir / "configs" / "cpo_unary.yaml").read_text(encoding="utf-8"))
    assert cpo_unary_config["alpha"] == 0.0
    assert cpo_unary_config["output_dir"] == str(tmp_path / "checkpoints" / "cpo_unary")
