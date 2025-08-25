# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates  
# SPDX-License-Identifier: Apache-2.0
"""
Entrypoint for launching train jobs.
The first argument must be a yaml training configuration file path.
The additional arguments support commandline override.
"""

import os
from sys import argv

from common.config import create_object, load_config
from common.distributed.basic import get_local_rank
from common.entrypoint import Entrypoint

# Require this for mp.spawn (async ndtimeline) to work.
if __name__ == "__main__":
    # Load config.
    config = load_config(argv[1], argv[2:])

    # Load trainer.
    entrypoint = create_object(config)
    
    # Start trainer.
    entrypoint.entrypoint()
