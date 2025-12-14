"""
User-provided function backend.

This is the most common case: user brings their own evaluator function.

Examples:
- Python function: def my_eval(x): return objective, constraints
- CFD script wrapper
- ML training function
- Any callable that returns objective and constraint values
"""

from typing import Callable, Dict, Any, Optional, Tuple
import numpy as np

from .base import EvaluationBackend, EvaluationResult


class UserFunctionBackend(EvaluationBackend):
    """
    Wrapper for user-provided evaluator functions.

    This enables easy 3-line setup:
    1. Define problem formulation
    2. Provide your evaluator function (you already have this!)
    3. Let PAOLA connect and solve

    Example:
        def my_cfd_evaluator(design):
            drag, lift = run_my_cfd(design)
            return {"drag": drag}, {"lift": lift}

        backend = UserFunctionBackend(my_cfd_evaluator)
        result = backend.evaluate(design)
    """

    def __init__(
        self,
        user_function: Callable,
        has_gradients: bool = False,
        gradient_function: Optional[Callable] = None,
        cost_per_eval: float = 1.0,
        name: str = "user_function"
    ):
        """
        Initialize user function backend.

        Args:
            user_function: Callable that evaluates objective and constraints
                          Signature: (design) -> (objectives_dict, constraints_dict)
                          or: (design) -> objectives_dict (if no constraints)
            has_gradients: Whether gradient_function is provided
            gradient_function: Optional gradient evaluator
                              Signature: (design) -> gradient_array
            cost_per_eval: Computational cost (CPU hours, or arbitrary units)
            name: Name for this evaluator
        """
        self.user_function = user_function
        self._has_gradients = has_gradients
        self.gradient_function = gradient_function
        self._cost_per_eval = cost_per_eval
        self._name = name

    def evaluate(self, design: np.ndarray) -> EvaluationResult:
        """
        Evaluate user function.

        Args:
            design: Design vector

        Returns:
            EvaluationResult with objectives, constraints, cost
        """
        try:
            # Call user function
            result = self.user_function(design)

            # Parse result (flexible formats)
            if isinstance(result, tuple):
                # Format: (objectives_dict, constraints_dict)
                objectives, constraints = result
            elif isinstance(result, dict):
                # Format: objectives_dict only (no constraints)
                objectives = result
                constraints = {}
            elif isinstance(result, (int, float)):
                # Format: single objective value
                objectives = {"objective": float(result)}
                constraints = {}
            else:
                raise ValueError(
                    f"Unsupported return type from user function: {type(result)}"
                )

            # Ensure dicts
            if not isinstance(objectives, dict):
                raise ValueError(
                    f"Objectives must be dict, got {type(objectives)}"
                )
            if not isinstance(constraints, dict):
                constraints = {}

            return EvaluationResult(
                objectives=objectives,
                constraints=constraints,
                cost=self._cost_per_eval,
                metadata={
                    "evaluator": self._name,
                    "design": design.tolist()
                }
            )

        except Exception as e:
            # Wrap exceptions with context
            raise RuntimeError(
                f"User function evaluation failed: {e}\n"
                f"Design: {design}\n"
                f"Function: {self._name}"
            ) from e

    def compute_gradient(
        self,
        design: np.ndarray,
        method: str = "auto"
    ) -> np.ndarray:
        """
        Compute gradient.

        Args:
            design: Design vector
            method: "auto", "user_provided", or "finite_difference"

        Returns:
            Gradient array
        """
        if method == "auto":
            if self._has_gradients:
                method = "user_provided"
            else:
                method = "finite_difference"

        if method == "user_provided":
            if not self._has_gradients or self.gradient_function is None:
                raise ValueError(
                    "Gradient function not provided. "
                    "Set has_gradients=True and provide gradient_function, "
                    "or use method='finite_difference'"
                )
            return self.gradient_function(design)

        elif method == "finite_difference":
            return self._finite_difference_gradient(design)

        else:
            raise ValueError(f"Unknown gradient method: {method}")

    def _finite_difference_gradient(
        self,
        design: np.ndarray,
        step_size: float = 1e-6
    ) -> np.ndarray:
        """
        Compute gradient using finite differences.

        Args:
            design: Design point
            step_size: Step size for finite differences

        Returns:
            Gradient array (same shape as design)
        """
        n = len(design)
        gradient = np.zeros(n)

        # Evaluate at base point
        base_result = self.evaluate(design)
        base_obj = list(base_result.objectives.values())[0]  # First objective

        # Finite difference for each variable
        for i in range(n):
            design_plus = design.copy()
            design_plus[i] += step_size

            result_plus = self.evaluate(design_plus)
            obj_plus = list(result_plus.objectives.values())[0]

            gradient[i] = (obj_plus - base_obj) / step_size

        return gradient

    @property
    def supports_gradients(self) -> bool:
        """Whether gradients are available."""
        return self._has_gradients or True  # FD always available

    @property
    def cost_per_evaluation(self) -> float:
        """Computational cost per evaluation."""
        return self._cost_per_eval

    @property
    def domain(self) -> str:
        """Problem domain."""
        return "user_defined"


def create_user_backend(
    function: Callable,
    **kwargs
) -> UserFunctionBackend:
    """
    Convenience function to create user backend.

    Args:
        function: User's evaluator function
        **kwargs: Additional arguments for UserFunctionBackend

    Returns:
        UserFunctionBackend instance

    Example:
        backend = create_user_backend(my_function, cost_per_eval=2.0)
    """
    return UserFunctionBackend(function, **kwargs)
