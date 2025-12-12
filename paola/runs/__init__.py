"""
Active optimization run management.

Provides agent-controlled run tracking that auto-persists to storage.
"""

from .active_run import OptimizationRun
from .manager import RunManager

__all__ = [
    "OptimizationRun",
    "RunManager",
]
