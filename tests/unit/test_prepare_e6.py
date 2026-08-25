from __future__ import annotations

import json

import yaml

from scripts.experiments.prepare_e6_mechanism_ablation import prepare_e6


def test_prepare_e6_writes_mechanism_ablation_configs(tmp_path) -> None:
    manifest_path = prepare_e6(experiment_dir=tmp_path / "E6", output_root=tmp_path / "checkpoints")

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert {record["family"] for record in records} == {"reference", "ema", "pairwise", "sampler"}
    assert sum(record["family"] == "reference" for record in records) == 3
    assert sum(record["family"] == "ema" for record in records) == 4
    assert sum(record["family"] == "pairwise" for record in records) == 3
    assert sum(record["family"] == "sampler" for record in records) == 2
    plain = next(record for record in records if record["name"] == "plain_batching")
    plain_config = yaml.safe_load(open(plain["config_path"], encoding="utf-8"))
    assert plain_config["pair_aware_batching"] is False
    pairwise_only = next(record for record in records if record["name"] == "alpha_1p0")
    pairwise_config = yaml.safe_load(open(pairwise_only["config_path"], encoding="utf-8"))
    assert pairwise_config["alpha"] == 1.0
