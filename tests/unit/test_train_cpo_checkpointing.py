from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "train"))

from train_cpo import prune_cpo_checkpoints


def test_prune_cpo_checkpoints_keeps_latest_numeric_checkpoints(tmp_path: Path) -> None:
    for name in ("checkpoint-100", "checkpoint-50", "checkpoint-500", "checkpoint-final", "logs"):
        (tmp_path / name).mkdir()

    removed = prune_cpo_checkpoints(tmp_path, save_total_limit=1)

    assert [path.name for path in removed] == ["checkpoint-50", "checkpoint-100"]
    assert not (tmp_path / "checkpoint-50").exists()
    assert not (tmp_path / "checkpoint-100").exists()
    assert (tmp_path / "checkpoint-500").is_dir()
    assert (tmp_path / "checkpoint-final").is_dir()
    assert (tmp_path / "logs").is_dir()


def test_prune_cpo_checkpoints_can_be_disabled(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-1").mkdir()
    (tmp_path / "checkpoint-2").mkdir()

    removed = prune_cpo_checkpoints(tmp_path, save_total_limit=0)

    assert removed == []
    assert (tmp_path / "checkpoint-1").is_dir()
    assert (tmp_path / "checkpoint-2").is_dir()
