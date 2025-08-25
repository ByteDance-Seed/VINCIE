# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import torch

_activation_t = Union[None, Literal["silu"]]
_direction_t = Union[None, Literal["forward"], Literal["backward"]]
_memory_list_t = Optional[List[Any]]
_memory_dict_t = Optional[Dict[str, Any]]
_memory_tensor_t = Optional[torch.Tensor]
_memory_t = Union[_memory_list_t, _memory_dict_t, _memory_tensor_t]
_norm_t = Union[None, Literal["group"]]
_size_3_t = Union[int, Tuple[int, int, int]]
_tensor_t = torch.Tensor
_pad_t = Union[Literal["constant"], Literal["replicate"]]
_inflation_t = Union[Literal["tail"], Literal["random"]]
