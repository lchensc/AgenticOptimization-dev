"""
Data Platform - Foundation for PAOLA optimization run management.

The platform module is the backbone of PAOLA, responsible for:
- Optimization run lifecycle (create → track → finalize)
- Data persistence and retrieval
- Problem definition and management

This module replaces the previous runs/ + storage/ separation with
a unified, cohesive design using dependency injection.
"""

from .platform import OptimizationPlatform
from .run import Run, RunRecord
from .problem import Problem
from .storage import StorageBackend, FileStorage

__all__ = [
    "OptimizationPlatform",
    "Run",
    "RunRecord",
    "Problem",
    "StorageBackend",
    "FileStorage",
]
