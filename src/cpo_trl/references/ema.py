from dataclasses import dataclass, field

import torch


@dataclass
class ClusterReferenceZ:
    """EMA state for cluster-level reference offsets."""

    momentum: float = 0.9
    nonnegative: bool = True
    values: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def tensor_for(
        self,
        cluster_ids: list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.tensor(
            [self.values.get(cluster_id, 0.0) for cluster_id in cluster_ids],
            device=device,
            dtype=dtype,
        )

    def update(self, cluster_ids: list[str], reference_values: torch.Tensor) -> None:
        detached = reference_values.detach().float().cpu()
        grouped: dict[str, list[float]] = {}
        for cluster_id, value in zip(cluster_ids, detached.tolist(), strict=True):
            grouped.setdefault(cluster_id, []).append(value)
        for cluster_id, values in grouped.items():
            batch_mean = sum(values) / len(values)
            if self.nonnegative:
                batch_mean = max(0.0, batch_mean)
            previous = self.values.get(cluster_id, batch_mean)
            self.values[cluster_id] = self.momentum * previous + (1.0 - self.momentum) * batch_mean
            self.counts[cluster_id] = self.counts.get(cluster_id, 0) + len(values)

__all__ = ["ClusterReferenceZ"]
