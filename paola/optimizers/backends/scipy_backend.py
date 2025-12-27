"""
SciPy optimization backend.

Wraps scipy.optimize.minimize with support for multiple methods:
SLSQP, L-BFGS-B, trust-constr, COBYLA, Nelder-Mead, Powell, BFGS, CG.
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
import logging

from ..base import OptimizerBackend
from ..result import OptimizationResult
from ..wrapper import ObjectiveWrapper, GradientWrapper

logger = logging.getLogger(__name__)


class SciPyBackend(OptimizerBackend):
    """
    SciPy optimization backend.

    Supports 8 methods with varying capabilities:
    - SLSQP: Gradient + constraints + bounds (recommended for constrained)
    - L-BFGS-B: Gradient + bounds (recommended for large-scale unconstrained)
    - trust-constr: Gradient + constraints + bounds
    - COBYLA: Constraints + bounds (derivative-free)
    - Nelder-Mead, Powell: Bounds only (derivative-free)
    - BFGS, CG: Gradient only (no bounds)
    """

    # Method capabilities based on SciPy 1.7+ documentation
    METHODS = {
        "SLSQP": {"gradient": True, "constraints": True, "bounds": True},
        "L-BFGS-B": {"gradient": True, "constraints": False, "bounds": True},
        "trust-constr": {"gradient": True, "constraints": True, "bounds": True},
        "COBYLA": {"gradient": False, "constraints": True, "bounds": True},
        "Nelder-Mead": {"gradient": False, "constraints": False, "bounds": True},
        "Powell": {"gradient": False, "constraints": False, "bounds": True},
        "BFGS": {"gradient": True, "constraints": False, "bounds": False},
        "CG": {"gradient": True, "constraints": False, "bounds": False},
    }

    @property
    def name(self) -> str:
        return "scipy"

    @property
    def family(self) -> str:
        return "gradient"

    @property
    def supports_gradients(self) -> bool:
        return True  # Some methods support gradients

    def is_available(self) -> bool:
        try:
            import scipy.optimize
            return True
        except ImportError:
            return False

    def get_methods(self) -> List[str]:
        return list(self.METHODS.keys())

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "SciPy",
            "methods": self.get_methods(),
            "skill": "scipy",  # Use load_skill("scipy") for detailed options
        }

    def get_method_info(self, method: str) -> Dict[str, Any]:
        """Get capabilities for a specific method."""
        caps = self.METHODS.get(method, {})
        return {
            "name": method,
            "gradient": caps.get("gradient", False),
            "constraints": caps.get("constraints", False),
            "bounds": caps.get("bounds", True),
        }

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
        Run SciPy optimization.

        Args:
            objective: Function f(x) -> float to minimize
            bounds: Variable bounds [[lb, ub], ...]
            x0: Initial design point
            config: Must include 'method', may include scipy options
            constraints: Optional scipy constraint dicts
            gradient: Optional gradient function

        Returns:
            OptimizationResult with solution and statistics
        """
        from scipy.optimize import minimize

        method = config.get("method", "SLSQP")
        method_caps = self.METHODS.get(method, {})

        # Filter out non-option parameters
        excluded_keys = {"method", "max_iterations"}
        options = {k: v for k, v in config.items() if k not in excluded_keys}

        # Ensure maxiter is set
        if "maxiter" not in options:
            options["maxiter"] = config.get("max_iterations", 100)

        # Wrap objective for tracking
        obj_wrapper = ObjectiveWrapper(objective)

        # Wrap gradient if provided and method supports it
        grad_wrapper = None
        jac = None
        if gradient and method_caps.get("gradient", False):
            grad_wrapper = GradientWrapper(gradient)
            jac = grad_wrapper

        # Prepare bounds
        scipy_bounds = None
        if bounds and method_caps.get("bounds", False):
            scipy_bounds = [(b[0], b[1]) for b in bounds]

        # Prepare constraints
        scipy_constraints = None
        if constraints and method_caps.get("constraints", False):
            scipy_constraints = constraints

        try:
            result = minimize(
                fun=obj_wrapper,
                x0=x0,
                method=method,
                jac=jac,
                bounds=scipy_bounds,
                constraints=scipy_constraints,
                options=options,
            )

            return OptimizationResult(
                success=result.success,
                message=str(result.message) if hasattr(result, "message") else "completed",
                best_x=result.x,
                best_f=float(result.fun),
                n_iterations=int(result.nit) if hasattr(result, "nit") else obj_wrapper.n_evals,
                n_function_evals=obj_wrapper.n_evals,
                n_gradient_evals=grad_wrapper.n_evals if grad_wrapper else 0,
                history=obj_wrapper.history,
                raw_result=result,
            )

        except Exception as e:
            logger.error(f"SciPy optimization failed: {e}")
            return OptimizationResult.from_failure(
                message=f"Optimization failed: {str(e)}",
                x0=x0,
                n_evals=obj_wrapper.n_evals,
                history=obj_wrapper.history,
            )
