import math
from typing import Dict, Iterable, Optional

import torch


def total_grad_norm_from_dict(grad_norms: Dict[str, float]) -> float:
    """
    Compute a single total norm from per-parameter gradient norms.
    """
    if not grad_norms:
        return 0.0
    return math.sqrt(sum(v * v for v in grad_norms.values()))


def compute_total_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    """
    Compute the L2 norm over all parameter gradients in a model.
    """
    total_sq = 0.0

    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2).item()
        total_sq += param_norm * param_norm

    return math.sqrt(total_sq)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the learning rate from the first optimizer param group.
    """
    return optimizer.param_groups[0]["lr"]