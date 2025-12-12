"""
Optimizer wrappers for the agentic optimization platform.

Provides:
- BaseOptimizer abstract interface
- ScipyOptimizer wrapper for scipy.optimize algorithms
- OptimizationGate for iteration-level agent control
- Future: Pymoo, custom optimizers
"""

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
    "BaseOptimizer",
    "OptimizerState",
    "ScipyOptimizer",
    "create_scipy_optimizer",
    "OptimizationGate",
    "GateAction",
    "GateSignal",
    "StopOptimizationSignal",
    "RestartOptimizationSignal",
]
