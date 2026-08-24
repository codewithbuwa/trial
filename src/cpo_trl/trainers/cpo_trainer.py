"""Minimal CPO trainer components built on TRL-style log-probability losses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from cpo_trl.losses.cpo_unary import cpo_unary_pair_loss
from cpo_trl.references.ema import ClusterReferenceZ


@dataclass
class CPOConfig:
    beta: float = 0.1
    alpha: float = 0.3
    lambda_desirable: float = 1.0
    lambda_undesirable: float = 1.0
    kl_coef: float = 0.0
    z_momentum: float = 0.9


@dataclass
class CPOBatchMetrics:
    loss: torch.Tensor
    unary_loss: torch.Tensor
    pair_loss: torch.Tensor
    kl_loss: torch.Tensor
    num_pairs: int = 0
    z_k: dict[str, float] = field(default_factory=dict)
    cluster_counts: dict[str, int] = field(default_factory=dict)


class CPOLossComputer:
    """Stateful CPO loss calculator with cluster reference tracking."""

    def __init__(self, config: CPOConfig | None = None) -> None:
        self.config = config or CPOConfig()
        self.z = ClusterReferenceZ(momentum=self.config.z_momentum)

    def state_dict(self) -> dict[str, Any]:
        return {
            "z_values": dict(self.z.values),
            "z_counts": dict(self.z.counts),
            "z_momentum": self.z.momentum,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.z.values = {
            str(cluster_id): float(value)
            for cluster_id, value in dict(state.get("z_values", {})).items()
        }
        self.z.counts = {
            str(cluster_id): int(value)
            for cluster_id, value in dict(state.get("z_counts", {})).items()
        }
        if "z_momentum" in state:
            self.z.momentum = float(state["z_momentum"])

    def __call__(
        self,
        *,
        policy_logps: torch.Tensor,
        ref_logps: torch.Tensor | None,
        labels: torch.Tensor,
        prompt_ids: list[str],
        cluster_ids: list[str],
        baseline_values: torch.Tensor | None = None,
        baseline_cluster_ids: list[str] | None = None,
        kl_values: torch.Tensor | None = None,
        pair_indices: torch.Tensor | None = None,
        update_z: bool = True,
    ) -> CPOBatchMetrics:
        if update_z:
            if baseline_values is None:
                if kl_values is not None:
                    baseline_values = kl_values.detach()
                else:
                    baseline_values = (
                        torch.zeros_like(policy_logps)
                        if ref_logps is None
                        else policy_logps.detach() - ref_logps.detach()
                    )
                baseline_cluster_ids = cluster_ids
            elif baseline_cluster_ids is None:
                baseline_cluster_ids = cluster_ids
            self.z.update(baseline_cluster_ids, baseline_values)
        z_tensor = self.z.tensor_for(
            cluster_ids,
            device=policy_logps.device,
            dtype=policy_logps.dtype,
        )
        loss, metrics = cpo_unary_pair_loss(
            policy_logps,
            ref_logps,
            labels,
            prompt_ids,
            cluster_ids,
            z_tensor,
            alpha=self.config.alpha,
            beta=self.config.beta,
            lambda_desirable=self.config.lambda_desirable,
            lambda_undesirable=self.config.lambda_undesirable,
            kl_coef=self.config.kl_coef,
            kl_values=kl_values,
            pair_indices=pair_indices,
            reduction="mean",
        )
        return CPOBatchMetrics(
            loss=loss,
            unary_loss=metrics["unary_loss"],
            pair_loss=metrics["pair_loss"],
            kl_loss=metrics["kl_loss"],
            num_pairs=int(metrics["num_pairs"].detach().cpu().item()),
            z_k=dict(self.z.values),
            cluster_counts=dict(self.z.counts),
        )


class CPOTrainer:
    """Small adapter for custom CPO experiments.

    Full model training still uses ``transformers.Trainer`` mechanics in the
    script entrypoint. This class keeps the CPO-specific state and metrics
    isolated so the loss behavior can be smoke-tested without a GPU run.
    """

    def __init__(self, *, config: CPOConfig | None = None, **trainer_kwargs: Any) -> None:
        self.config = config or CPOConfig()
        self.loss_computer = CPOLossComputer(self.config)
        self.trainer_kwargs = trainer_kwargs

    def compute_cpo_loss(self, **batch_logps: Any) -> CPOBatchMetrics:
        return self.loss_computer(**batch_logps)
