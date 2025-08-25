# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

class PromptEmbedding(nn.Module):
    def __init__(self, 
                 max_num_prompts: int, 
                 embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_num_prompts, embedding_dim)

    def forward(self, prompt_id: torch.LongTensor):
        return self.embedding(prompt_id)
    