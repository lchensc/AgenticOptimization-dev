"""
Optimizer wrappers for the agentic optimization platform.

Provides:
- Run-to-completion backends: SciPyBackend, IPOPTBackend, OptunaBackend
- Step-by-step optimizer: BaseOptimizer, ScipyOptimizer
- OptimizationGate for iteration-level agent control
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

# Step-by-step optimizers (for fine-grained control)
from paola.optimizers.base import BaseOptimizer, OptimizerState
from paola.optimizers.scipy_optimizer import ScipyOptimizer, create_scipy_optimizer
from paola.optimizers.gate import (
    OptimizationGate,
    GateAction,
    GateSignal,
    StopOptimizationSignal,
    RestartOptimizationSignal,
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
    # Step-by-step optimizers
    "BaseOptimizer",
    "OptimizerState",
    "ScipyOptimizer",
    "create_scipy_optimizer",
    # Gate control
    "OptimizationGate",
    "GateAction",
    "GateSignal",
    "StopOptimizationSignal",
    "RestartOptimizationSignal",
]
