"""CPO loss primitives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cpo_trl.references.ema import ClusterReferenceZ
from cpo_trl.utils.finite import assert_finite_tensor


def reduce_loss(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"unsupported reduction: {reduction}")


def dpo_pair_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor | None = None,
    ref_rejected_logps: torch.Tensor | None = None,
    *,
    beta: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """DPO-style pair loss over chosen/rejected sequence log probabilities."""

    ref_chosen = torch.zeros_like(chosen_logps) if ref_chosen_logps is None else ref_chosen_logps
    ref_rejected = (
        torch.zeros_like(rejected_logps) if ref_rejected_logps is None else ref_rejected_logps
    )
    logits = beta * ((chosen_logps - rejected_logps) - (ref_chosen - ref_rejected))
    losses = -F.logsigmoid(logits)
    assert_finite_tensor(losses, "pair_loss")
    return reduce_loss(losses, reduction)


def kto_unary_loss(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor | None,
    desirable: torch.Tensor,
    z: torch.Tensor | float = 0.0,
    *,
    beta: float = 0.1,
    lambda_desirable: float = 1.0,
    lambda_undesirable: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """KTO prospect-style unary loss with a per-row reference ``z`` offset."""

    ref = torch.zeros_like(policy_logps) if ref_logps is None else ref_logps
    desirable_mask = desirable.bool()
    signs = torch.where(desirable_mask, torch.ones_like(policy_logps), -torch.ones_like(policy_logps))
    weights = torch.where(
        desirable_mask,
        torch.full_like(policy_logps, lambda_desirable),
        torch.full_like(policy_logps, lambda_undesirable),
    )
    z_tensor = torch.as_tensor(z, dtype=policy_logps.dtype, device=policy_logps.device)
    logits = signs * beta * (policy_logps - ref - z_tensor)
    losses = weights * (1.0 - torch.sigmoid(logits))
    assert_finite_tensor(losses, "unary_loss")
    return reduce_loss(losses, reduction)


def sampled_kl_regularizer(
    kl_values: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Reduce nonnegative sampled KL(policy || reference) values."""

    losses = kl_values.clamp_min(0.0)
    assert_finite_tensor(losses, "kl_regularizer")
    return reduce_loss(losses, reduction)


def derived_pair_indices(
    prompt_ids: list[str],
    cluster_ids: list[str],
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build positive/negative pair indices for identical prompt and cluster."""

    positives: dict[tuple[str, str], list[int]] = {}
    negatives: dict[tuple[str, str], list[int]] = {}
    for index, (prompt_id, cluster_id, label) in enumerate(
        zip(prompt_ids, cluster_ids, labels.detach().cpu().bool().tolist(), strict=True)
    ):
        key = (str(prompt_id), str(cluster_id))
        if label:
            positives.setdefault(key, []).append(index)
        else:
            negatives.setdefault(key, []).append(index)

    chosen_indices: list[int] = []
    rejected_indices: list[int] = []
    for key, pos_indices in positives.items():
        neg_indices = negatives.get(key, [])
        for pos_index in pos_indices:
            for neg_index in neg_indices:
                chosen_indices.append(pos_index)
                rejected_indices.append(neg_index)

    return torch.tensor(chosen_indices, dtype=torch.long), torch.tensor(rejected_indices, dtype=torch.long)


def cpo_unary_pair_loss(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor | None,
    labels: torch.Tensor,
    prompt_ids: list[str],
    cluster_ids: list[str],
    z: torch.Tensor | float,
    *,
    alpha: float = 0.5,
    beta: float = 0.1,
    lambda_desirable: float = 1.0,
    lambda_undesirable: float = 1.0,
    kl_coef: float = 0.0,
    kl_values: torch.Tensor | None = None,
    pair_indices: torch.Tensor | None = None,
    reduction: str = "mean",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Unary-native CPO loss with explicit or in-cluster positive/negative pairs."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    ref = torch.zeros_like(policy_logps) if ref_logps is None else ref_logps
    unary = kto_unary_loss(
        policy_logps,
        ref,
        labels,
        z,
        beta=beta,
        lambda_desirable=lambda_desirable,
        lambda_undesirable=lambda_undesirable,
        reduction="none",
    )
    kl_source = torch.zeros_like(policy_logps) if kl_values is None else kl_values
    if kl_coef == 0.0:
        kl_source = kl_source.detach()
    kl = sampled_kl_regularizer(kl_source, reduction="none")
    if pair_indices is None:
        pos_indices, neg_indices = derived_pair_indices(prompt_ids, cluster_ids, labels)
        pos_indices = pos_indices.to(policy_logps.device)
        neg_indices = neg_indices.to(policy_logps.device)
    else:
        pair_indices = pair_indices.to(policy_logps.device)
        if pair_indices.numel() == 0:
            pos_indices = torch.empty(0, dtype=torch.long, device=policy_logps.device)
            neg_indices = torch.empty(0, dtype=torch.long, device=policy_logps.device)
        elif pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
            raise ValueError("pair_indices must have shape [num_pairs, 2]")
        else:
            pos_indices = pair_indices[:, 0].long()
            neg_indices = pair_indices[:, 1].long()
    if pos_indices.numel() > 0:
        pair = dpo_pair_loss(
            policy_logps[pos_indices],
            policy_logps[neg_indices],
            ref[pos_indices],
            ref[neg_indices],
            beta=beta,
            reduction="none",
        )
        pair_loss = pair.mean()
    else:
        pair_loss = torch.zeros((), dtype=policy_logps.dtype, device=policy_logps.device)

    total = (1.0 - alpha) * unary.mean() + alpha * pair_loss + kl_coef * kl.mean()

    assert_finite_tensor(total, "cpo_loss")
    if reduction == "none":
        reduced = total.expand_as(policy_logps)
    elif reduction == "mean":
        reduced = total
    elif reduction == "sum":
        reduced = total * policy_logps.numel()
    else:
        raise ValueError(f"unsupported reduction: {reduction}")
    metrics = {
        "unary_loss": unary.mean(),
        "pair_loss": pair_loss,
        "kl_loss": kl.mean(),
        "num_pairs": torch.tensor(pos_indices.numel(), device=policy_logps.device),
    }
    return reduced, metrics
