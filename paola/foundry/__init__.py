"""
Data Foundry - Foundation for PAOLA optimization data.

The foundry provides a single source of truth for optimization data,
managing problems, runs, results with versioning and lineage.

Inspired by Palantir Foundry's ontology layer: creates organized,
trustworthy data that the agent and knowledge base can build upon.

The foundry module is responsible for:
- Optimization run lifecycle (create → track → finalize)
- Data persistence and retrieval with lineage tracking
- Problem definition and management
- Query interfaces for analysis and knowledge extraction

This is the data foundation layer upon which intelligence (agent),
learning (knowledge base), and analysis are built.
"""

from .foundry import OptimizationFoundry
from .run import Run, RunRecord
from .problem import Problem
from .storage import StorageBackend, FileStorage

__all__ = [
    "OptimizationFoundry",
    "Run",
    "RunRecord",
    "Problem",
    "StorageBackend",
    "FileStorage",
]
