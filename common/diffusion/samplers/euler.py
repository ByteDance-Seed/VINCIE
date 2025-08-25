 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

"""
Euler ODE solver.
"""

from typing import Callable
import torch

from ..utils import expand_dims
from .base import Sampler, SamplerModelArgs


class EulerSampler(Sampler):
    """
    The Euler method is the simplest ODE solver.
    <https://en.wikipedia.org/wiki/Euler_method>
    """

    def sample(
        self,
        x: torch.Tensor,
        f: Callable[[SamplerModelArgs], torch.Tensor],
    ) -> torch.Tensor:
        timesteps = self.timesteps.timesteps
        progress = self.get_progress_bar()
        i = 0
        for t, s in zip(timesteps[:-1], timesteps[1:]):
            pred = f(SamplerModelArgs(x, t, i))
            x = self.step_to(pred, x, t, s)
            i += 1
            progress.update()

        if self.return_endpoint:
            t = timesteps[-1]
            pred = f(SamplerModelArgs(x, t, i))
            x = self.get_endpoint(pred, x, t)
            progress.update()
        return x

    def step(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step to the next timestep.
        """
        return self.step_to(pred, x_t, t, self.get_next_timestep(t))

    def step_to(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        """
        Steps from x_t at timestep t to x_s at timestep s. Returns x_s.
        """
        t = expand_dims(t, x_t.ndim)
        s = expand_dims(s, x_t.ndim)
        T = self.schedule.T
        # Step from x_t to x_s.
        pred_x_0, pred_x_T = self.schedule.convert_from_pred(pred, self.prediction_type, x_t, t)
        pred_x_s = self.schedule.forward(pred_x_0, pred_x_T, s.clamp(0, T))
        # Clamp x_s to x_0 and x_T if s is out of bound.
        pred_x_s = pred_x_s.where(s >= 0, pred_x_0)
        pred_x_s = pred_x_s.where(s <= T, pred_x_T)
        return pred_x_s
