from __future__ import annotations

import json

import yaml

from scripts.experiments.prepare_e7_robustness import METHODS, SEEDS, prepare_e7


def test_prepare_e7_writes_seeded_runs_for_each_method(tmp_path) -> None:
    manifest_path = prepare_e7(experiment_dir=tmp_path / "E7", output_root=tmp_path / "checkpoints")

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert len(records) == len(SEEDS) * len(METHODS)
    assert {record["seed"] for record in records} == set(SEEDS)
    assert {record["method"] for record in records} == set(METHODS)
    cpo_123 = next(record for record in records if record["method"] == "cpo" and record["seed"] == 123)
    config = yaml.safe_load(open(cpo_123["config_path"], encoding="utf-8"))
    assert config["seed"] == 123
    assert config["output_dir"].endswith("cpo_seed_123")
