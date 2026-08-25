from __future__ import annotations

import json

import yaml

from scripts.experiments.prepare_e9_final_tuned import prepare_e9


def test_prepare_e9_applies_validation_selected_hyperparameters(tmp_path) -> None:
    best_path = tmp_path / "best_by_method.json"
    best_path.write_text(
        json.dumps(
            {
                "cpo": {
                    "learning_rate": 5e-6,
                    "beta": 0.05,
                    "alpha": 0.75,
                    "max_grad_norm": 1.0,
                    "z_baseline": "same_completion_logratio",
                }
            }
        ),
        encoding="utf-8",
    )

    manifest_path = prepare_e9(
        experiment_dir=tmp_path / "E9",
        output_root=tmp_path / "checkpoints",
        best_by_method=best_path,
        data_root=tmp_path / "data",
        test_split="test.jsonl",
    )
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert {record["method"] for record in records} == {"sft", "dpo", "kto", "cpo_unary", "cpo"}
    cpo = next(record for record in records if record["method"] == "cpo")
    config = yaml.safe_load(open(cpo["config_path"], encoding="utf-8"))
    assert config["learning_rate"] == 5e-6
    assert config["alpha"] == 0.75
    assert config["z_baseline"] == "same_completion_logratio"
    assert str(tmp_path / "data" / "cpo" / "test.jsonl") in cpo["test_eval_command"]
