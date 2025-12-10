"""
Evaluation backends for optimization problems.

Provides:
- Analytical test functions (Rosenbrock, Sphere, etc.)
- Future: CFD/FEA workflow integration
"""

from aopt.backends.analytical import (
    AnalyticalFunction,
    Rosenbrock,
    Sphere,
    ConstrainedRosenbrock,
    get_analytical_function,
)

__all__ = [
    "AnalyticalFunction",
    "Rosenbrock",
    "Sphere",
    "ConstrainedRosenbrock",
    "get_analytical_function",
]
