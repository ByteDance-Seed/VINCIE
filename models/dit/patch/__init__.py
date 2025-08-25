# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
def get_patch_layers(patch_type="v1"):
    assert patch_type in ["v1", "v2"]
    if patch_type == "v1":
        from .patch_v1 import PatchIn, PatchOut
    elif patch_type == "v2":
        from .patch_v2 import PatchIn, PatchOut
    return PatchIn, PatchOut


def get_na_patch_layers(patch_type="v1"):
    assert patch_type in ["v1", "v2"]
    if patch_type == "v1":
        from .patch_v1 import NaPatchIn, NaPatchOut
    elif patch_type == "v2":
        from .patch_v2 import NaPatchIn, NaPatchOut
    return NaPatchIn, NaPatchOut
