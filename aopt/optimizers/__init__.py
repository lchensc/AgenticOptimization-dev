"""
Optimizer wrappers for the agentic optimization platform.

Provides:
- BaseOptimizer abstract interface
- ScipyOptimizer wrapper for scipy.optimize algorithms
- OptimizationGate for iteration-level agent control
- Future: Pymoo, custom optimizers
"""

from aopt.optimizers.base import BaseOptimizer, OptimizerState
from aopt.optimizers.scipy_optimizer import ScipyOptimizer, create_scipy_optimizer
from aopt.optimizers.gate import (
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
