"""
Abstract base class for optimizer backends.

All optimizer backends (SciPy, IPOPT, Optuna, pymoo, etc.) implement this interface,
enabling a consistent API for the optimization tool layer.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
import numpy as np

from .result import OptimizationResult


class OptimizerBackend(ABC):
    """
    Abstract base class for optimizer backends.

    Each backend wraps an optimization library and provides:
    - Availability check (is the library installed?)
    - Method listing (what algorithms are available?)
    - Capability info (constraints? gradients? multi-objective?)
    - Optimize method (run optimization to completion)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Backend identifier (e.g., 'scipy', 'ipopt', 'pymoo').

        Used for backend selection in run_optimization(optimizer="scipy:SLSQP").
        """
        pass

    @property
    @abstractmethod
    def family(self) -> str:
        """
        Component family for graph node schemas.

        Determines which Progress/Result components are used:
        - 'gradient': GradientProgress, GradientResult
        - 'bayesian': BayesianProgress, BayesianResult
        - 'evolutionary': EvolutionaryProgress, EvolutionaryResult
        - 'multiobjective': MultiObjectiveProgress, MultiObjectiveResult
        """
        pass

    @property
    def supports_multiobjective(self) -> bool:
        """Whether this backend supports multi-objective optimization."""
        return False

    @property
    def supports_constraints(self) -> bool:
        """Whether this backend supports constraints."""
        return True

    @property
    def supports_gradients(self) -> bool:
        """Whether this backend can use gradient information."""
        return False

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this backend's dependencies are installed.

        Returns:
            True if the backend can be used, False otherwise.
        """
        pass

    @abstractmethod
    def get_methods(self) -> List[str]:
        """
        List available methods/algorithms for this backend.

        Returns:
            List of method names (e.g., ["SLSQP", "L-BFGS-B"] for scipy).
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get backend capabilities for LLM tool response.

        Returns:
            Dict with:
                - name: Display name
                - methods: List of available methods (if applicable)
                - skill: Skill name for detailed documentation
                - available: Whether backend is installed
        """
        pass

    @abstractmethod
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: List[List[float]],
        x0: np.ndarray,
        config: Dict[str, Any],
        constraints: Optional[List[Dict[str, Any]]] = None,
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> OptimizationResult:
        """
        Run optimization to completion.

        Args:
            objective: Function f(x) -> float to minimize
            bounds: Variable bounds [[lb, ub], ...]
            x0: Initial design point
            config: Optimizer configuration (method, options, etc.)
            constraints: Optional constraint specifications
            gradient: Optional gradient function grad_f(x) -> ndarray

        Returns:
            OptimizationResult with solution, statistics, and history.
        """
        pass

    def get_method_info(self, method: str) -> Dict[str, Any]:
        """
        Get information about a specific method.

        Override in subclasses to provide method-specific capabilities.

        Args:
            method: Method name (e.g., "SLSQP")

        Returns:
            Dict with method capabilities (gradient, constraints, bounds, etc.)
        """
        return {
            "name": method,
            "gradient": self.supports_gradients,
            "constraints": self.supports_constraints,
            "bounds": True,
        }
