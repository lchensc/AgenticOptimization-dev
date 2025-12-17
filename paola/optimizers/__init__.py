"""
Optimizer wrappers for the agentic optimization platform.

Provides run-to-completion backends: SciPyBackend, IPOPTBackend, OptunaBackend
"""

# Run-to-completion backends (LLM-driven architecture)
from paola.optimizers.backends import (
    OptimizerBackend,
    OptimizationResult,
    SciPyBackend,
    IPOPTBackend,
    OptunaBackend,
    get_backend,
    list_backends,
    get_available_backends,
)

__all__ = [
    # Backends (LLM-driven)
    "OptimizerBackend",
    "OptimizationResult",
    "SciPyBackend",
    "IPOPTBackend",
    "OptunaBackend",
    "get_backend",
    "list_backends",
    "get_available_backends",
]
