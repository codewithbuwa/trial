from __future__ import annotations

import json

from scripts.experiments.prepare_e5_cluster_ablation import prepare_e5


def test_prepare_e5_writes_cluster_ablation_manifest(tmp_path) -> None:
    manifest_path = prepare_e5(
        experiment_dir=tmp_path / "E5",
        data_root=tmp_path / "data",
        output_root=tmp_path / "checkpoints",
        limit=10,
        seed=123,
    )

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    assert {record["condition"] for record in records} == {
        "semantic_4",
        "random_4",
        "random_4_matched",
        "single_cluster",
        "alternative_clusters",
    }
    random_4 = next(record for record in records if record["condition"] == "random_4")
    assert random_4["cluster_mode"] == "random4"
    assert "--limit" in random_4["data_command"]
    single = next(record for record in records if record["condition"] == "single_cluster")
    assert single["cluster_mode"] == "single_cluster"
