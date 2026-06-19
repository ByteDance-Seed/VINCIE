"""
Platform utility methods.
"""

import os
from typing import Optional


def get_region() -> Optional[str]:
    """
    Get the region of the current machine.
    Returns: "cn" | "us"
    """
    region = os.environ.get("CLUSTER_REGION")
    return region.lower() if region else None


def get_task_id() -> Optional[str]:
    """
    Get current job/task id.
    """
    return os.environ.get("JOB_ID")
