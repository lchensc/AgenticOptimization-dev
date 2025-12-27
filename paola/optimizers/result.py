"""
Optimization result types for single and multi-objective optimization.

Provides a unified result structure that works across all optimizer backends,
with optional Pareto front support for multi-objective optimization.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
import numpy as np


@dataclass
class OptimizationResult:
    """
    Result from any optimizer - single or multi-objective.

    For single-objective: best_x and best_f contain the optimal solution.
    For multi-objective: pareto_set and pareto_front contain the Pareto-optimal solutions,
    with best_x/best_f containing a representative solution (first in Pareto set).
    """

    # Status
    success: bool
    message: str

    # Best solution (always present)
    best_x: np.ndarray
    best_f: float

    # Multi-objective (optional)
    pareto_set: Optional[np.ndarray] = None    # Shape: (n_solutions, n_vars)
    pareto_front: Optional[np.ndarray] = None  # Shape: (n_solutions, n_objectives)
    hypervolume: Optional[float] = None

    # Statistics
    n_iterations: int = 0
    n_function_evals: int = 0
    n_gradient_evals: int = 0

    # History (iteration-by-iteration tracking)
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Raw result from underlying optimizer (for advanced use)
    raw_result: Any = None

    @property
    def is_multiobjective(self) -> bool:
        """Check if this is a multi-objective result."""
        return self.pareto_front is not None

    @property
    def n_pareto_solutions(self) -> int:
        """Number of solutions in Pareto front."""
        if self.pareto_front is None:
            return 0
        return len(self.pareto_front)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for tool response.

        Truncates history to last 20 entries to manage context size.
        For multi-objective, includes summary but not full Pareto data.
        """
        result = {
            "success": self.success,
            "message": self.message,
            "best_x": (
                self.best_x.tolist()
                if isinstance(self.best_x, np.ndarray)
                else list(self.best_x)
            ),
            "best_f": float(self.best_f),
            "n_iterations": self.n_iterations,
            "n_function_evals": self.n_function_evals,
            "n_gradient_evals": self.n_gradient_evals,
            "history": self.history[-20:] if len(self.history) > 20 else self.history,
        }

        if self.is_multiobjective:
            result["is_multiobjective"] = True
            result["n_pareto_solutions"] = self.n_pareto_solutions
            result["hypervolume"] = self.hypervolume
            # Full Pareto set stored separately in GraphDetail

        return result

    @classmethod
    def from_failure(
        cls,
        message: str,
        x0: np.ndarray,
        n_evals: int = 0,
        history: Optional[List[Dict]] = None,
    ) -> "OptimizationResult":
        """Create a failure result."""
        return cls(
            success=False,
            message=message,
            best_x=x0,
            best_f=float("inf"),
            n_iterations=0,
            n_function_evals=n_evals,
            n_gradient_evals=0,
            history=history or [],
        )
