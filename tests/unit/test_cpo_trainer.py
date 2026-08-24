from __future__ import annotations

import pytest
import torch

from cpo_trl.cpo_trainer import CPOConfig, CPOLossComputer, CPOTrainer
from cpo_trl.losses import kto_unary_loss


def test_cpo_trainer_tracks_cluster_z_and_counts() -> None:
    trainer = CPOTrainer(config=CPOConfig(alpha=0.5, beta=0.1, z_momentum=0.0))
    metrics = trainer.compute_cpo_loss(
        policy_logps=torch.tensor([2.0, 1.0, 3.0]),
        ref_logps=torch.tensor([0.5, 0.2, 1.0]),
        labels=torch.tensor([True, False, True]),
        prompt_ids=["p1", "p1", "p2"],
        cluster_ids=["coding", "coding", "math"],
    )
    assert torch.isfinite(metrics.loss)
    assert metrics.num_pairs == 1
    assert metrics.cluster_counts == {"coding": 2, "math": 1}
    assert metrics.z_k["coding"] == pytest.approx(1.15)
    assert metrics.z_k["math"] == pytest.approx(2.0)


def test_cpo_trainer_can_update_z_from_baseline_values() -> None:
    trainer = CPOTrainer(config=CPOConfig(alpha=0.5, beta=0.1, z_momentum=0.0))
    metrics = trainer.compute_cpo_loss(
        policy_logps=torch.tensor([2.0, 1.0]),
        ref_logps=torch.tensor([0.5, 0.2]),
        labels=torch.tensor([True, False]),
        prompt_ids=["p1", "p1"],
        cluster_ids=["coding", "coding"],
        baseline_values=torch.tensor([4.0, 8.0]),
        baseline_cluster_ids=["coding", "coding"],
    )
    assert torch.isfinite(metrics.loss)
    assert metrics.z_k["coding"] == pytest.approx(6.0)


def test_cpo_trainer_uses_kl_values_as_default_reference() -> None:
    trainer = CPOTrainer(config=CPOConfig(alpha=0.0, beta=0.1, z_momentum=0.0))
    metrics = trainer.compute_cpo_loss(
        policy_logps=torch.tensor([2.0, 1.0]),
        ref_logps=torch.tensor([0.5, 0.2]),
        labels=torch.tensor([True, False]),
        prompt_ids=["p1", "p1"],
        cluster_ids=["coding", "coding"],
        kl_values=torch.tensor([0.2, 0.6]),
    )

    assert torch.isfinite(metrics.loss)
    assert metrics.z_k["coding"] == pytest.approx(0.4)


def test_cpo_trainer_updates_z_before_computing_loss() -> None:
    trainer = CPOTrainer(config=CPOConfig(alpha=0.0, beta=0.5, z_momentum=0.0))
    policy = torch.tensor([3.0, 1.0])
    ref = torch.tensor([0.5, 0.2])
    labels = torch.tensor([True, False])
    metrics = trainer.compute_cpo_loss(
        policy_logps=policy,
        ref_logps=ref,
        labels=labels,
        prompt_ids=["p1", "p1"],
        cluster_ids=["coding", "coding"],
        baseline_values=torch.tensor([2.0, 4.0]),
        baseline_cluster_ids=["coding", "coding"],
    )

    expected = kto_unary_loss(policy, ref, labels, z=torch.tensor([3.0, 3.0]), beta=0.5)
    assert torch.allclose(metrics.unary_loss, expected)
    assert torch.allclose(metrics.loss, expected)
    assert metrics.z_k["coding"] == pytest.approx(3.0)


def test_cpo_loss_computer_state_roundtrip() -> None:
    original = CPOLossComputer(CPOConfig(z_momentum=0.25))
    original.z.values = {"coding": 1.5}
    original.z.counts = {"coding": 4}

    restored = CPOLossComputer(CPOConfig(z_momentum=0.9))
    restored.load_state_dict(original.state_dict())

    assert restored.z.values == {"coding": 1.5}
    assert restored.z.counts == {"coding": 4}
    assert restored.z.momentum == pytest.approx(0.25)
