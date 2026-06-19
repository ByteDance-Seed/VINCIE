from typing import Optional
import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Conv2d, Conv3d

from models.ae.modules.types import MemoryState, _inflation_mode_t, _memory_device_t


def causal_class_wrapper(module: nn.Module):
    original_forward = module.forward

    def forward(x: torch.Tensor):
        if x.ndim <= 4:
            return original_forward(x)
        if x.ndim == 5:
            t = x.size(2)
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = original_forward(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
            return x

    module.forward = forward
    return module


def remove_head(tensor: Tensor, times: int = 1) -> Tensor:
    """
    Remove duplicated first frame features in the up-sampling process.
    """
    if times == 0:
        return tensor
    return torch.cat(tensors=(tensor[:, :, :1], tensor[:, :, times + 1 :]), dim=2)


def extend_head(tensor: Tensor, times: int = 2, memory: Optional[Tensor] = None) -> Tensor:
    """
    When memory is None:
        - Duplicate first frame features in the down-sampling process.
    When memory is not None:
        - Concatenate memory features with the input features to keep temporal consistency.
    """
    if memory is not None:
        return torch.cat((memory.to(tensor), tensor), dim=2)
    assert times >= 0, "Invalid input for function 'extend_head'!"
    if times == 0:
        return tensor
    else:
        tile_repeat = [1] * tensor.ndim
        tile_repeat[2] = times
        return torch.cat([tensor[:, :, :1]] * times + [tensor], dim=2)


class InflatedCausalConv3d(Conv3d):
    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t,
        memory_device: _memory_device_t = "same",
        **kwargs,
    ):
        """
        Initialize a Causal-3D convolution layer.
        Parameters:
            inflation_mode: Listed as below.
                - none: No inflation will be conducted.
                        The loading logic of state dict will fall back to default.
                - tail / replicate: Refer to the definition of `InflatedCausalConv3d`.
        """
        self.inflation_mode = inflation_mode
        self.memory = None
        super().__init__(*args, **kwargs)
        self.temporal_padding = self.padding[0]
        self.memory_device = memory_device
        self.padding = (0, *self.padding[1:])  # Remove temporal pad to keep causal.

    def set_memory_device(self, memory_device: _memory_device_t):
        self.memory_device = memory_device

    def forward(self, input: Tensor, memory_state: MemoryState = MemoryState.UNSET) -> Tensor:
        assert memory_state != MemoryState.UNSET
        mem_size = self.stride[0] - self.kernel_size[0]
        if (self.memory is not None) and (memory_state == MemoryState.ACTIVE):
            # Note: we already have a `memory` tensor to prepend,
            #   so we just set `times` as an invalid one to set it as a sentinel.
            input = extend_head(input, memory=self.memory, times=-1)
        else:
            input = extend_head(input, times=self.temporal_padding * 2)
        memory = (
            input[:, :, mem_size:].detach()
            if (mem_size != 0 and memory_state != MemoryState.DISABLED)
            else None
        )
        if (
            memory_state != MemoryState.DISABLED
            and not self.training
            and (self.memory_device is not None)
        ):
            self.memory = memory
            if self.memory_device == "cpu" and self.memory is not None:
                self.memory = self.memory.to("cpu")
        return super().forward(input)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.inflation_mode != "none":
            state_dict = inflate_state_dict(self, state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            (strict and self.inflation_mode == "none"),
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class UpscaleConv2d(Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.kernel_size[0] % 2 == 1 and self.kernel_size[1] % 2 == 1
        self.init_upscale_parameters()

    def init_upscale_parameters(self):
        identity = torch.eye(self.in_channels).repeat(
            self.out_channels // self.in_channels, 1
        )  # [out_channels, in_channels]
        k_h, k_w = self.kernel_size[0], self.kernel_size[1]

        with torch.no_grad():
            nn.init.zeros_(self.weight)
            self.weight.data[:, :, k_h // 2, k_w // 2].copy_(identity)
            nn.init.zeros_(self.bias)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class UpscaleCausalConv3d(InflatedCausalConv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_upscale_parameters()

    def init_upscale_parameters(self):
        identity = torch.eye(self.in_channels).repeat(
            self.out_channels // self.in_channels, 1
        )  # [out_channels, in_channels]
        k_h, k_w = self.kernel_size[0], self.kernel_size[1]

        with torch.no_grad():
            nn.init.zeros_(self.weight)
            self.weight.data[:, :, -1, k_h // 2, k_w // 2].copy_(identity)
            nn.init.zeros_(self.bias)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            replicate_state_dict(self, state_dict, prefix),
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def inflate_weight(weight_2d: torch.Tensor, weight_3d: torch.Tensor, inflation_mode: str):
    """
    Inflate a 2D convolution weight matrix to a 3D one.
    Parameters:
        weight_2d:      The weight matrix of 2D conv to be inflated.
        weight_3d:      The weight matrix of 3D conv to be initialized.
        inflation_mode: the mode of inflation
    """
    assert inflation_mode in ["tail", "replicate"]
    assert weight_3d.shape[:2] == weight_2d.shape[:2]
    with torch.no_grad():
        if inflation_mode == "replicate":
            depth = weight_3d.size(2)
            weight_3d.copy_(weight_2d.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth)
        else:
            weight_3d.fill_(0.0)
            weight_3d[:, :, -1].copy_(weight_2d)
    return weight_3d


def inflate_state_dict(layer, state_dict, prefix):
    """
    the main function to inflated 2D parameters to 3D.
    """
    weight_name = prefix + "weight"
    if weight_name in state_dict:
        weight_2d = state_dict[weight_name]
        if weight_2d.dim() == 4:
            # Assuming the 2D weights are 4D tensors (out_channels, in_channels, h, w)
            weight_3d = inflate_weight(
                weight_2d=weight_2d,
                weight_3d=layer.weight,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[weight_name] = weight_3d
    return state_dict


def replicate_state_dict(layer, state_dict, prefix):
    """
    replicate spatial upscale conv -> spatial-temporal upscale conv
    """
    weight_name = prefix + "weight"
    bias_name = prefix + "bias"
    if weight_name in state_dict:
        weight_2d = state_dict[weight_name]
        if weight_2d.dim() == 4:
            expanded_weight_2d = weight_2d.repeat(
                layer.weight.size(0) // weight_2d.size(0), 1, 1, 1
            )
            weight_3d = inflate_weight(
                weight_2d=expanded_weight_2d,
                weight_3d=layer.weight,
                inflation_mode=layer.inflation_mode,
            )
            state_dict[weight_name] = weight_3d
        else:
            return state_dict
    if bias_name in state_dict:
        bias_2d = state_dict[bias_name]
        bias_3d = bias_2d.repeat(layer.bias.size(0) // bias_2d.size(0))
        state_dict[bias_name] = bias_3d
    return state_dict


def clean_memory_bank(module: Optional[nn.Module]):
    if module is None:
        return
    if hasattr(module, "memory"):
        module.memory = None
    for child in module.children():
        clean_memory_bank(child)
