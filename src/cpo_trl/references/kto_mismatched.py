import torch


def mismatched_logratio(policy_logps: torch.Tensor, mismatched_reference_logps: torch.Tensor) -> torch.Tensor:
    return policy_logps - mismatched_reference_logps


__all__ = ["mismatched_logratio"]
