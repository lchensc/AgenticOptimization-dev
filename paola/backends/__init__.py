"""
Evaluation backends for optimization problems.

Provides:
- Analytical test functions (Rosenbrock, Sphere, etc.)
- User-provided functions (most common use case)
- Future: CFD/FEA workflow integration
"""

from paola.backends.analytical import (
    AnalyticalFunction,
    Rosenbrock,
    Sphere,
    ConstrainedRosenbrock,
    get_analytical_function,
)
from paola.backends.base import EvaluationBackend, EvaluationResult
from paola.backends.user_function import UserFunctionBackend, create_user_backend

__all__ = [
    # Analytical functions
    "AnalyticalFunction",
    "Rosenbrock",
    "Sphere",
    "ConstrainedRosenbrock",
    "get_analytical_function",
    # Base classes
    "EvaluationBackend",
    "EvaluationResult",
    # User functions (most common)
    "UserFunctionBackend",
    "create_user_backend",
]
