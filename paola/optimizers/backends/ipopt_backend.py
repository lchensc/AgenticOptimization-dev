"""
IPOPT optimization backend.

Wraps cyipopt for large-scale nonlinear optimization with full support
for constraints, gradients, and warm-starting.
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
import logging

from ..base import OptimizerBackend
from ..result import OptimizationResult
from ..wrapper import ObjectiveWrapper, GradientWrapper

logger = logging.getLogger(__name__)


class IPOPTBackend(OptimizerBackend):
    """
    IPOPT interior-point optimizer backend.

    IPOPT is recommended for:
    - Large-scale constrained optimization
    - Problems with many constraints
    - When gradients are available
    - Warm-starting from previous solutions

    Supports ~250 configuration options via the IPOPT skill.
    """

    @property
    def name(self) -> str:
        return "ipopt"

    @property
    def family(self) -> str:
        return "gradient"

    @property
    def supports_gradients(self) -> bool:
        return True

    @property
    def supports_constraints(self) -> bool:
        return True

    def is_available(self) -> bool:
        try:
            import cyipopt
            return True
        except ImportError:
            return False

    def get_methods(self) -> List[str]:
        return ["IPOPT"]  # Single method

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "IPOPT",
            "skill": "ipopt",  # Use load_skill("ipopt") for detailed options
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
        Run IPOPT optimization.

        Args:
            objective: Function f(x) -> float to minimize
            bounds: Variable bounds [[lb, ub], ...]
            x0: Initial design point
            config: IPOPT options (full passthrough)
            constraints: Optional scipy-style constraint dicts
            gradient: Optional gradient function

        Returns:
            OptimizationResult with solution and statistics
        """
        try:
            from cyipopt import minimize_ipopt
        except ImportError:
            return OptimizationResult.from_failure(
                message="IPOPT not available. Install with: pip install cyipopt",
                x0=x0,
            )

        # Wrap objective for tracking
        obj_wrapper = ObjectiveWrapper(objective)

        # Wrap gradient if provided
        grad_wrapper = None
        jac = None
        if gradient:
            grad_wrapper = GradientWrapper(gradient)
            jac = grad_wrapper

        # Prepare bounds
        ipopt_bounds = [(b[0], b[1]) for b in bounds]

        # Prepare IPOPT options - FULL PASSTHROUGH
        options = {}

        # Convenience mappings for common alternative names
        key_mappings = {
            "maxiter": "max_iter",
            "max_iterations": "max_iter",
        }

        # Keys to exclude (not IPOPT options)
        excluded_keys = {"method", "solver", "backend"}

        # Pass through all options
        for key, value in config.items():
            if key in excluded_keys:
                continue
            # Apply convenience mapping if exists
            ipopt_key = key_mappings.get(key, key)
            options[ipopt_key] = value

        # Set defaults only if not provided
        if "max_iter" not in options:
            options["max_iter"] = 3000  # IPOPT default
        if "print_level" not in options:
            options["print_level"] = 0  # Quiet by default for Paola

        try:
            result = minimize_ipopt(
                fun=obj_wrapper,
                x0=x0,
                jac=jac,
                bounds=ipopt_bounds,
                constraints=constraints,
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
            logger.error(f"IPOPT optimization failed: {e}")
            return OptimizationResult.from_failure(
                message=f"IPOPT optimization failed: {str(e)}",
                x0=x0,
                n_evals=obj_wrapper.n_evals,
                history=obj_wrapper.history,
            )
