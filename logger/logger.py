from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import visdom
import torch

from logger.hooks import register_gradient_hooks, remove_hooks
from utils.grad_utils import total_grad_norm_from_dict


class AutoVisdomLogger:
    def __init__(
        self,
        env: str = "auto-visdom-logger",
        host: str = "localhost",
        port: int = 8097,
        use_incoming_socket: bool = False,
    ):
        self.env = env
        self.vis = visdom.Visdom(
            server=host,
            port=port,
            env=env,
            use_incoming_socket=use_incoming_socket,
        )

        if not self.vis.check_connection():
            raise RuntimeError(
                "Cannot connect to Visdom server. "
                "Start it first with: python -m visdom.server -port 8097"
            )

        self._windows: Dict[str, str] = {}
        self._grad_buffer: Dict[str, float] = {}
        self._hook_handles = []

    def watch(self, model: torch.nn.Module):
        """
        Attach gradient hooks to a model so gradients are captured automatically.
        """
        self._hook_handles = register_gradient_hooks(model, self._on_grad)
        return self

    def close(self):
        """
        Clean up hooks.
        """
        if self._hook_handles:
            remove_hooks(self._hook_handles)
            self._hook_handles = []

    def _on_grad(self, name: str, grad: torch.Tensor):
        """
        Called automatically by hooks during backward pass.
        Store the L2 norm of that parameter's gradient.
        """
        grad_norm = grad.detach().data.norm(2).item()
        self._grad_buffer[name] = grad_norm

    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Generic scalar metric logging.
        Example:
            logger.log_metrics({"train/loss": 0.42, "train/lr": 0.001}, step=10)
        """
        for name, value in metrics.items():
            self._plot_scalar(name, float(value), step)

    def flush_gradients(self, step: int):
        """
        Call this after loss.backward().
        Logs:
        - total gradient norm
        - per-parameter gradient norms
        """
        if not self._grad_buffer:
            return

        total_norm = total_grad_norm_from_dict(self._grad_buffer)

        # total gradient norm
        self._plot_scalar("grad/total_norm", total_norm, step)

        # per-parameter gradient norms
        for param_name, norm_value in self._grad_buffer.items():
            safe_name = f"grad/{param_name}"
            self._plot_scalar(safe_name, norm_value, step)

        self._grad_buffer.clear()

    def _plot_scalar(self, name: str, value: float, step: int):
        """
        Plot a single scalar to Visdom.
        Creates the window the first time, then appends to it.
        """
        x = np.array([step], dtype=np.float32)
        y = np.array([value], dtype=np.float32)

        if name not in self._windows:
            self.vis.line(
                X=x,
                Y=y,
                win=name,
                opts=dict(
                    title=name,
                    xlabel="step",
                    ylabel=name,
                ),
            )
            self._windows[name] = name
        else:
            self.vis.line(
                X=x,
                Y=y,
                win=self._windows[name],
                update="append",
            )