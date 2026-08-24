import torch


def sequence_logratio(policy_logps: torch.Tensor, reference_logps: torch.Tensor) -> torch.Tensor:
    return policy_logps - reference_logps


__all__ = ["sequence_logratio"]
