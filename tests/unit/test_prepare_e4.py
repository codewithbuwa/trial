from __future__ import annotations

import json

import yaml

from scripts.experiments.prepare_e4_alpha_sweep import ALPHAS, prepare_e4


def test_prepare_e4_writes_alpha_sweep_configs(tmp_path) -> None:
    manifest_path = prepare_e4(experiment_dir=tmp_path / "E4", output_root=tmp_path / "checkpoints")

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert [record["alpha"] for record in records] == list(ALPHAS)
    assert len(records) == 6
    for record in records:
        config = yaml.safe_load(open(record["config_path"], encoding="utf-8"))
        assert config["alpha"] == record["alpha"]
        assert "scripts/train/train_cpo.py" in record["command"]
