from typing import Callable, List

import torch


def register_gradient_hooks(
    model: torch.nn.Module,
    on_grad: Callable[[str, torch.Tensor], None],
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Register a hook on each trainable parameter.

    on_grad(name, grad_tensor) is called whenever that parameter's gradient
    is computed in backprop.
    """
    handles = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        def make_hook(param_name: str):
            def hook_fn(grad: torch.Tensor):
                on_grad(param_name, grad)
            return hook_fn

        handle = param.register_hook(make_hook(name))
        handles.append(handle)

    return handles


def remove_hooks(handles: List[torch.utils.hooks.RemovableHandle]) -> None:
    """
    Remove all registered hooks.
    """
    for h in handles:
        h.remove()