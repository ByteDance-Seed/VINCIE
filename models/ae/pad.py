# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
from typing import Literal, Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from .types import _direction_t, _memory_tensor_t, _pad_t, _tensor_t


def pad(x: _tensor_t, t: int, mode: _pad_t, direction: _direction_t) -> _tensor_t:
    if t == 0:
        return x
    if mode == "constant" or t < 0:
        if direction is None:
            return F.pad(x, (0, 0, 0, 0, t // 2, t // 2))
        if direction == "forward":
            return F.pad(x, (0, 0, 0, 0, t, 0))
        if direction == "backward":
            return F.pad(x, (0, 0, 0, 0, 0, t))
    if mode == "replicate":
        if direction is None:
            return replicate_pad(x, t)
        if direction == "forward":
            return replicate_forward_pad(x, t)
        if direction == "backward":
            return replicate_backward_pad(x, t)
    raise NotImplementedError


def replicate_pad(x: _tensor_t, t: int) -> _tensor_t:
    """
    (ABCD, 2) -> AABCDD  (Faster than the replication pad in F.pad.)
    """
    l = x[:, :, :1, :, :]
    r = x[:, :, -1:, :, :]
    return torch.cat([l] * (t // 2) + [x] + [r] * (t // 2), dim=2)


def replicate_forward_pad(x: _tensor_t, t: int) -> _tensor_t:
    """
    (ABCD, 2) -> AAABCD  (Faster than the replication pad in F.pad.)
    """
    p = x[:, :, :1, :, :]
    return torch.cat([p] * t + [x], dim=2)


def replicate_backward_pad(x: _tensor_t, t: int) -> _tensor_t:
    """
    (ABCD, 2) -> ABCDDD  (Faster than the replication pad in F.pad.)
    """
    p = x[:, :, -1:, :, :]
    return torch.cat([x] + [p] * t, dim=2)


class CausalPad(nn.Module):
    def __init__(
        self,
        size: int,
        mode: _pad_t,
        direction: _direction_t,
    ):
        assert mode in ["constant", "replicate"]
        assert direction in [None, "forward", "backward"]
        super().__init__()
        self.size = size
        self.mode = mode
        self.direction = direction
        self.memory_device = "same"

    def forward(
        self,
        x: _tensor_t,
        m: _memory_tensor_t,
    ) -> Tuple[
        _tensor_t,
        _memory_tensor_t,
    ]:
        # Skip if not 3d.
        if x.ndim < 5:
            return x, None

        # Initialize variables.
        t = self.size
        pt = t

        # Load old memory.
        if m is not None:
            pt -= m.size(2)
            m = m.to(x)
            if self.direction == "forward":
                x = torch.cat([m, x], dim=2)
            if self.direction == "backward":
                x = torch.cat([x, m], dim=2)

        # Pad input.
        x = pad(x, pt, mode=self.mode, direction=self.direction)

        # Save new memory.
        if t and not self.training and self.memory_device:
            if self.direction == "forward":
                m = x[:, :, -t:]
            if self.direction == "backward":
                m = x[:, :, :t]
            if self.memory_device == "cpu":
                m = m.cpu()

        return x, m

    def extra_repr(self):
        return (
            f"size={self.size}, mode='{self.mode}', "
            + f"direction='{self.direction}', device='{self.memory_device}'"
        )

    def set_memory_device(self, device: Optional[Literal["cpu", "same"]]):
        assert device in [None, "cpu", "same"]
        self.memory_device = device
