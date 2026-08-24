from __future__ import annotations

import pytest
import torch

from cpo_trl.finite import NonFiniteError, assert_finite_tensor
from cpo_trl.losses import (
    ClusterReferenceZ,
    cpo_combined_loss,
    cpo_unary_pair_loss,
    derived_pair_indices,
    dpo_pair_loss,
    kto_unary_loss,
    sampled_kl_regularizer,
)


def test_alpha_one_matches_pair_loss() -> None:
    chosen = torch.tensor([3.0, 2.0])
    rejected = torch.tensor([1.0, 2.5])
    loss, metrics = cpo_combined_loss(
        chosen,
        rejected,
        None,
        None,
        chosen,
        None,
        torch.ones_like(chosen, dtype=torch.bool),
        0.0,
        alpha=1.0,
        beta=0.5,
    )
    expected = dpo_pair_loss(chosen, rejected, beta=0.5)
    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["pair_loss"], expected)


def test_alpha_zero_matches_unary_loss() -> None:
    logps = torch.tensor([3.0, -2.0])
    labels = torch.tensor([True, False])
    loss, metrics = cpo_combined_loss(
        logps,
        torch.zeros_like(logps),
        None,
        None,
        logps,
        None,
        labels,
        0.0,
        alpha=0.0,
        beta=0.5,
    )
    expected = kto_unary_loss(logps, None, labels, 0.0, beta=0.5)
    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["unary_loss"], expected)


def test_kto_unary_loss_uses_prospect_value() -> None:
    policy = torch.tensor([3.0, 1.0])
    ref = torch.tensor([0.5, 0.2])
    labels = torch.tensor([True, False])
    beta = 0.5
    loss = kto_unary_loss(
        policy,
        ref,
        labels,
        z=0.1,
        beta=beta,
        lambda_desirable=2.0,
        lambda_undesirable=3.0,
        reduction="none",
    )
    margins = policy - ref - 0.1
    signs = torch.tensor([1.0, -1.0])
    weights = torch.tensor([2.0, 3.0])
    expected = weights * (1.0 - torch.sigmoid(signs * beta * margins))
    assert torch.allclose(loss, expected)


def test_sampled_kl_regularizer_reduces_token_kl_values() -> None:
    kl_values = torch.tensor([0.3, -0.1, 0.7])
    loss = sampled_kl_regularizer(kl_values, reduction="none")

    assert torch.allclose(loss, torch.tensor([0.3, 0.0, 0.7]))


def test_cluster_reference_clamps_after_cluster_mean() -> None:
    reference = ClusterReferenceZ(momentum=0.0)

    reference.update(["coding", "coding"], torch.tensor([-1.0, 3.0]))
    assert reference.values["coding"] == pytest.approx(1.0)

    reference.update(["math", "math"], torch.tensor([-3.0, 1.0]))
    assert reference.values["math"] == pytest.approx(0.0)


def test_cpo_reductions_are_finite() -> None:
    loss, metrics = cpo_combined_loss(
        torch.tensor([2.0]),
        torch.tensor([1.0]),
        torch.tensor([0.5]),
        torch.tensor([0.2]),
        torch.tensor([2.0]),
        torch.tensor([0.5]),
        torch.tensor([True]),
        torch.tensor([0.1]),
        alpha=0.25,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["unary_loss"])
    assert torch.isfinite(metrics["pair_loss"])


def test_derived_pair_indices_group_by_prompt_and_cluster() -> None:
    labels = torch.tensor([True, False, False, True])
    pos, neg = derived_pair_indices(
        prompt_ids=["p1", "p1", "p1", "p1"],
        cluster_ids=["coding", "coding", "math", "coding"],
        labels=labels,
    )
    assert pos.tolist() == [0, 3]
    assert neg.tolist() == [1, 1]


def test_unary_native_alpha_one_matches_derived_pair_loss() -> None:
    policy = torch.tensor([3.0, 1.0, 5.0])
    ref = torch.tensor([0.5, 0.2, 1.0])
    labels = torch.tensor([True, False, True])
    loss, metrics = cpo_unary_pair_loss(
        policy,
        ref,
        labels,
        prompt_ids=["p1", "p1", "p2"],
        cluster_ids=["coding", "coding", "coding"],
        z=0.0,
        alpha=1.0,
        beta=0.5,
    )
    expected = dpo_pair_loss(policy[:1], policy[1:2], ref[:1], ref[1:2], beta=0.5)
    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["pair_loss"], expected)
    assert int(metrics["num_pairs"].item()) == 1


def test_unary_native_uses_explicit_pairs_instead_of_cartesian_product() -> None:
    policy = torch.tensor([4.0, 1.0, 6.0, 2.0])
    ref = torch.zeros_like(policy)
    labels = torch.tensor([True, False, True, False])
    loss, metrics = cpo_unary_pair_loss(
        policy,
        ref,
        labels,
        prompt_ids=["p1", "p1", "p1", "p1"],
        cluster_ids=["coding", "coding", "coding", "coding"],
        z=0.0,
        alpha=1.0,
        beta=0.5,
        pair_indices=torch.tensor([[0, 1], [2, 3]]),
    )
    expected = dpo_pair_loss(
        torch.tensor([4.0, 6.0]),
        torch.tensor([1.0, 2.0]),
        torch.zeros(2),
        torch.zeros(2),
        beta=0.5,
    )

    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["pair_loss"], expected)
    assert int(metrics["num_pairs"].item()) == 2


def test_unary_native_alpha_zero_matches_unary_loss() -> None:
    policy = torch.tensor([3.0, 1.0])
    ref = torch.tensor([0.5, 0.2])
    labels = torch.tensor([True, False])
    loss, metrics = cpo_unary_pair_loss(
        policy,
        ref,
        labels,
        prompt_ids=["p1", "p1"],
        cluster_ids=["coding", "coding"],
        z=0.0,
        alpha=0.0,
        beta=0.5,
    )
    expected = kto_unary_loss(policy, ref, labels, 0.0, beta=0.5)
    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["unary_loss"], expected)


def test_unary_native_no_pairs_still_respects_alpha() -> None:
    policy = torch.tensor([3.0, 1.0])
    ref = torch.tensor([0.5, 0.2])
    labels = torch.tensor([True, False])
    loss, metrics = cpo_unary_pair_loss(
        policy,
        ref,
        labels,
        prompt_ids=["p1", "p2"],
        cluster_ids=["coding", "coding"],
        z=0.0,
        alpha=1.0,
        beta=0.5,
    )
    assert metrics["num_pairs"].item() == 0
    assert torch.allclose(loss, torch.tensor(0.0))


def test_unary_native_threads_lambda_weights() -> None:
    policy = torch.tensor([3.0, 1.0])
    ref = torch.tensor([0.5, 0.2])
    labels = torch.tensor([True, False])
    loss, metrics = cpo_unary_pair_loss(
        policy,
        ref,
        labels,
        prompt_ids=["p1", "p1"],
        cluster_ids=["coding", "coding"],
        z=0.0,
        alpha=0.0,
        beta=0.5,
        lambda_desirable=2.0,
        lambda_undesirable=3.0,
    )
    expected = kto_unary_loss(
        policy,
        ref,
        labels,
        0.0,
        beta=0.5,
        lambda_desirable=2.0,
        lambda_undesirable=3.0,
    )
    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["unary_loss"], expected)


def test_unary_native_adds_optional_kl_regularizer() -> None:
    policy = torch.tensor([3.0, 1.0])
    ref = torch.tensor([0.5, 0.2])
    labels = torch.tensor([True, False])
    common = {
        "prompt_ids": ["p1", "p1"],
        "cluster_ids": ["coding", "coding"],
        "z": torch.tensor([1.0, 0.5]),
        "alpha": 0.0,
        "beta": 0.5,
    }
    base_loss, _ = cpo_unary_pair_loss(policy, ref, labels, kl_coef=0.0, **common)
    kl_values = torch.tensor([0.3, 0.7])
    regularized_loss, metrics = cpo_unary_pair_loss(
        policy,
        ref,
        labels,
        kl_coef=0.2,
        kl_values=kl_values,
        **common,
    )
    kl = sampled_kl_regularizer(kl_values)
    assert torch.allclose(regularized_loss, base_loss + 0.2 * kl)
    assert torch.allclose(metrics["kl_loss"], kl)


def test_non_finite_checks_fail_fast() -> None:
    with pytest.raises(NonFiniteError):
        assert_finite_tensor(torch.tensor([float("nan")]), "example")
