 # Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
 # SPDX-License-Identifier: Apache-2.0 

"""
File system operations. Currently supports local and hadoop file systems.
"""

import hashlib
import os
import pickle
import shutil
import subprocess
import tarfile
import tempfile
from typing import List, Optional

from common.distributed import barrier_if_distributed, get_global_rank, get_local_rank
from common.logger import get_logger

logger = get_logger(__name__)

DOWNLOAD_DIR = None



def is_hdfs_path(path: str) -> bool:
    """
    Detects whether a path is an hdfs path.
    A hdfs path must startswith "hdfs://" protocol prefix.
    """
    return path.lower().startswith("hdfs://")



def listdir(path: str) -> List[str]:
    """
    List directory. Returns full path.

    Examples:
        - listdir("hdfs://dir") -> ["hdfs://dir/file1", "hdfs://dir/file2"]
        - listdir("/dir") -> ["/dir/file1", "/dir/file2"]
    """
    files = []

    if is_hdfs_path(path):
        pipe = subprocess.Popen(
            args=["hdfs", "dfs", "-ls", path],
            shell=False,
            stdout=subprocess.PIPE,
        )

        for line in pipe.stdout:
            parts = line.strip().split()

            # drwxr-xr-x   - user group  4 file
            if len(parts) < 5:
                continue

            # Filter out warning texts when listing files on uswest cluster.
            if "Warn" in parts[0].decode("utf8"):
                continue

            files.append(parts[-1].decode("utf8"))

        pipe.stdout.close()
        pipe.wait()

    else:
        files = [os.path.join(path, file) for file in os.listdir(path)]

    return files



def exists(path: str) -> bool:
    """
    Check whether a path exists.
    Returns True if exists, False otherwise.
    """
    if is_hdfs_path(path):
        process = subprocess.run(["hdfs", "dfs", "-test", "-e", path], capture_output=True)
        return process.returncode == 0
    return os.path.exists(path)


def mkdir(path: str):
    """
    Create a directory.
    Create all parent directory if not present. No-op if directory already present.
    """
    if is_hdfs_path(path):
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", path])
    else:
        os.makedirs(path, exist_ok=True)


def copy(src: str, tgt: str, blocking: bool = True):
    """
    Copy a file.
    """
    if src == tgt:
        return

    src_hdfs = is_hdfs_path(src)
    tgt_hdfs = is_hdfs_path(tgt)

    if not src_hdfs and not tgt_hdfs:
        shutil.copy(src, tgt)
        return

    if src_hdfs and tgt_hdfs:
        process = subprocess.Popen(["hdfs", "dfs", "-cp", "-f", src, tgt])
    elif src_hdfs and not tgt_hdfs:
        process = subprocess.Popen(
            ["hdfs", "dfs", "-get", "-c", "128", "-t", "10", "--ct", "32", src, tgt]
        )
    elif not src_hdfs and tgt_hdfs:
        process = subprocess.Popen(
            ["hdfs", "dfs", "-put", "-f", "-c", "128", "-t", "10", "--ct", "32", src, tgt]
        )

    if blocking:
        process.wait()


def move(src: str, tgt: str):
    """
    Move a file.
    """
    if src == tgt:
        return

    src_hdfs = is_hdfs_path(src)
    tgt_hdfs = is_hdfs_path(tgt)

    if src_hdfs and tgt_hdfs:
        subprocess.run(["hdfs", "dfs", "-mv", src, tgt])
    elif not src_hdfs and not tgt_hdfs:
        shutil.move(src, tgt)
    else:
        copy(src, tgt)
        remove(src)


def remove(path: str):
    """
    Remove a file or directory.
    """
    if is_hdfs_path(path):
        subprocess.run(["hdfs", "dfs", "-rm", "-r", path])
    elif os.path.isfile(path):
        os.remove(path)
    else:
        shutil.rmtree(path)
