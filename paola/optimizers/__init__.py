"""
Optimizer backends for the agentic optimization platform.

Provides a modular architecture for optimization backends:
- base.py: OptimizerBackend abstract base class
- result.py: OptimizationResult (single + multi-objective)
- wrapper.py: ObjectiveWrapper, GradientWrapper utilities
- registry.py: Backend registration and lookup
- backends/: Individual backend implementations

Usage:
    from paola.optimizers import get_backend, list_backends

    backend = get_backend("scipy")
    result = backend.optimize(objective, bounds, x0, config)
"""

# Core abstractions
from paola.optimizers.base import OptimizerBackend
from paola.optimizers.result import OptimizationResult
from paola.optimizers.wrapper import (
    ObjectiveWrapper,
    GradientWrapper,
    MultiObjectiveWrapper,
)

# Registry functions
from paola.optimizers.registry import (
    get_backend,
    list_backends,
    get_available_backends,
    register_backend,
    get_registry,
    BackendRegistry,
)

# Backend implementations
from paola.optimizers.backends import (
    SciPyBackend,
    IPOPTBackend,
    OptunaBackend,
    PymooBackend,
)

__all__ = [
    # Core abstractions
    "OptimizerBackend",
    "OptimizationResult",
    "ObjectiveWrapper",
    "GradientWrapper",
    "MultiObjectiveWrapper",
    # Registry
    "get_backend",
    "list_backends",
    "get_available_backends",
    "register_backend",
    "get_registry",
    "BackendRegistry",
    # Backends
    "SciPyBackend",
    "IPOPTBackend",
    "OptunaBackend",
    "PymooBackend",
]
