"""
Paola Recording Infrastructure.

This module provides automatic recording of optimization evaluations,
enabling crash-safe persistence and cross-graph learning.

Core components:
- RecordingObjective: Wraps evaluator, captures all f(x) calls
- EvaluationCache: Per-graph cache with array hashing
- ArrayHasher: Stable hashing for numpy arrays
"""

from paola.recording.objective import RecordingObjective
from paola.recording.cache import ArrayHasher, EvaluationCache

__all__ = [
    "RecordingObjective",
    "ArrayHasher",
    "EvaluationCache",
]
