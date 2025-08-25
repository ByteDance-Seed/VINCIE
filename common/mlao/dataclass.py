 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


@dataclass
class FlatTensorMeta:
    tensor: Optional[torch.Tensor]
    offsets: Tuple[int, int]
    to_recompute: bool
    need_offload: bool
    extra_ref_cnt: int
    # == indicate from tensor ==
    shape: torch.Size = None
    dtype: torch.dtype = None
    device: torch.device = None
    numel: int = None
    # == set by manager ==
    offload_event: Optional[torch.cuda.Event] = None
    upload_event: Optional[torch.cuda.Event] = None
    ref_cnt: int = 1
    on_device: bool = True

    def __post_init__(self):
        self.shape = self.tensor.shape
        self.dtype = self.tensor.dtype
        self.device = self.tensor.device
        self.numel = self.tensor.numel()
        if self.to_recompute:
            self.tensor = None
            self.on_device = False

    def deallocate(self):
        self.tensor.data = self.tensor.data.new_empty(0)
        self.on_device = False

    def reallocate(self, tensor):
        if self.tensor is None:
            self.tensor = tensor
        else:
            self.tensor.data = tensor

    def finish_offload(self):
        # Use of synchronize instead of wait for non-blocking cpu offload is intentional.
        if self.offload_event is not None:
            self.offload_event.synchronize()
        self.offload_event = None

    def finish_upload(self):
        if self.upload_event is not None:
            self.upload_event.wait()
        self.upload_event = None

    def check_tensor_match(self, x):
        assert x.shape == self.shape, f"shape mismatch: {x.shape=}, {self.shape=}"
        assert x.dtype == self.dtype, f"dtype mismatch: {x.dtype=}, {self.dtype=}"


@dataclass
class MLAOTensorGroup:
    mlao_root_id: str
    tids: List[str] = field(default_factory=list)


@dataclass
class MLAOStateConfig:
    pack_inputs: bool = False
    pack_activations: bool = False
    pack_on_device: bool = False
