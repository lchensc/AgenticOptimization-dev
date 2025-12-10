"""
Evaluator tools for the agentic optimization platform.

Provides LangChain @tool decorated functions for function evaluation:
- evaluate_function: Evaluate objective and constraints (with automatic caching)
- compute_gradient: Compute gradients (analytical or finite-difference)
"""

from typing import Optional, Dict, Any, List
import numpy as np
from langchain_core.tools import tool
import time

from aopt.tools.cache_tools import cache_get, cache_store
from aopt.backends.analytical import get_analytical_function


# Global problem registry
_PROBLEM_REGISTRY: Dict[str, Any] = {}


def register_problem(problem_id: str, problem: Any):
    """Register a problem for evaluation."""
    _PROBLEM_REGISTRY[problem_id] = problem


def _get_problem(problem_id: str) -> Any:
    """Get problem from registry."""
    if problem_id not in _PROBLEM_REGISTRY:
        raise ValueError(
            f"Problem '{problem_id}' not registered. "
            f"Available: {list(_PROBLEM_REGISTRY.keys())}"
        )
    return _PROBLEM_REGISTRY[problem_id]


@tool
def evaluate_function(
    problem_id: str,
    design: List[float],
    use_cache: bool = True,
    compute_constraints: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate objective function (and optionally constraints) at a design point.

    This tool automatically checks the evaluation cache first. If the design
    has been evaluated before (within tolerance), returns cached result instantly.
    Otherwise, performs evaluation and stores result in cache.

    IMPORTANT: Engineering simulations are 10,000× more expensive than optimizer
    iterations (~$500 vs $0.05). Always use caching unless you have a specific
    reason not to.

    Args:
        problem_id: Problem identifier from formulate_problem
        design: Design vector to evaluate
        use_cache: If True, check cache before evaluation (default: True)
        compute_constraints: If True, also evaluate constraints (default: False)

    Returns:
        Dict with:
            - success: bool
            - objective: float - objective function value
            - constraints: Optional[Dict[str, float]] - constraint values (if requested)
            - cost: float - computational cost (0 if cache hit)
            - cache_hit: bool - whether result came from cache
            - evaluation_time: float - time spent evaluating (seconds)
            - message: str

    Example:
        result = evaluate_function(
            problem_id="rosenbrock_2d",
            design=[-1.0, 1.0],
            use_cache=True
        )
        objective = result["objective"]
        was_cached = result["cache_hit"]
    """
    try:
        # Check cache first
        if use_cache:
            cached = cache_get(design=design, problem_id=problem_id)
            if cached is not None:
                return {
                    "success": True,
                    "objective": cached["objectives"][0],  # Single objective for now
                    "constraints": cached.get("constraints"),
                    "cost": 0.0,
                    "cache_hit": True,
                    "evaluation_time": 0.0,
                    "message": f"Cache hit! Retrieved result from {cached['timestamp']}",
                }

        # Get problem
        problem = _get_problem(problem_id)
        design_array = np.array(design)

        # Evaluate
        start_time = time.time()

        if hasattr(problem, "evaluate"):
            # Analytical function
            objective = float(problem.evaluate(design_array))
        else:
            raise ValueError(f"Problem '{problem_id}' doesn't have evaluate() method")

        # Evaluate constraints if requested
        constraints_dict = None
        if compute_constraints and hasattr(problem, "evaluate_constraint"):
            # For now, assume single constraint
            c_val = float(problem.evaluate_constraint(design_array))
            constraints_dict = {"c1": c_val}

        eval_time = time.time() - start_time

        # Determine cost (for analytical functions, use symbolic cost)
        # In real engineering problems, this would be actual computational cost
        cost = 1.0  # 1 unit for analytical evaluation

        # Store in cache
        if use_cache:
            cache_store(
                design=design,
                problem_id=problem_id,
                objectives=[objective],
                gradient=None,  # Stored separately by compute_gradient
                constraints=constraints_dict,
                cost=cost,
            )

        return {
            "success": True,
            "objective": objective,
            "constraints": constraints_dict,
            "cost": cost,
            "cache_hit": False,
            "evaluation_time": eval_time,
            "message": f"Evaluated objective = {objective:.6f} (cost: {cost:.1f})",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error evaluating function: {str(e)}",
        }


@tool
def compute_gradient(
    problem_id: str,
    design: List[float],
    method: str = "analytical",
    use_cache: bool = True,
    fd_step: float = 1e-6,
) -> Dict[str, Any]:
    """
    Compute gradient of objective function.

    Supports multiple gradient computation methods:
    - "analytical": Use analytical derivatives (fastest, most accurate if available)
    - "finite-difference": Use finite-difference approximation (slower, more robust)

    Like evaluate_function, this tool uses caching to avoid recomputing expensive
    gradients (especially important for adjoint methods in CFD/FEA).

    Args:
        problem_id: Problem identifier
        design: Design vector
        method: Gradient method - "analytical" or "finite-difference"
        use_cache: If True, check cache before computation (default: True)
        fd_step: Step size for finite-difference (default: 1e-6)

    Returns:
        Dict with:
            - success: bool
            - gradient: List[float] - gradient vector
            - gradient_norm: float - L2 norm of gradient
            - cost: float - computational cost
            - cache_hit: bool
            - method_used: str
            - evaluation_time: float
            - message: str

    Example:
        result = compute_gradient(
            problem_id="rosenbrock_2d",
            design=[-1.0, 1.0],
            method="analytical"
        )
        gradient = result["gradient"]
        grad_norm = result["gradient_norm"]
    """
    try:
        # Check cache first
        if use_cache:
            cached = cache_get(design=design, problem_id=problem_id)
            if cached is not None and cached.get("gradient") is not None:
                gradient = cached["gradient"]
                gradient_norm = float(np.linalg.norm(gradient))
                return {
                    "success": True,
                    "gradient": gradient.tolist() if isinstance(gradient, np.ndarray) else gradient,
                    "gradient_norm": gradient_norm,
                    "cost": 0.0,
                    "cache_hit": True,
                    "method_used": "cached",
                    "evaluation_time": 0.0,
                    "message": f"Cache hit! Gradient norm: {gradient_norm:.6e}",
                }

        # Get problem
        problem = _get_problem(problem_id)
        design_array = np.array(design)

        start_time = time.time()

        # Compute gradient
        if method == "analytical":
            if not hasattr(problem, "gradient"):
                return {
                    "success": False,
                    "message": f"Problem '{problem_id}' doesn't support analytical gradients. Use method='finite-difference'.",
                }
            gradient = problem.gradient(design_array)

        elif method == "finite-difference":
            # Central finite-difference
            gradient = np.zeros_like(design_array)
            f0 = problem.evaluate(design_array)

            for i in range(len(design_array)):
                design_plus = design_array.copy()
                design_plus[i] += fd_step
                f_plus = problem.evaluate(design_plus)

                design_minus = design_array.copy()
                design_minus[i] -= fd_step
                f_minus = problem.evaluate(design_minus)

                gradient[i] = (f_plus - f_minus) / (2 * fd_step)

        else:
            return {
                "success": False,
                "message": f"Unknown gradient method: {method}. Use 'analytical' or 'finite-difference'.",
            }

        eval_time = time.time() - start_time
        gradient_norm = float(np.linalg.norm(gradient))

        # Determine cost
        if method == "analytical":
            cost = 1.5  # Analytical gradient costs ~1.5× function evaluation for analytical functions
        else:  # finite-difference
            cost = 2 * len(design_array)  # 2n function evaluations

        # Update cache with gradient
        if use_cache:
            # Get cached objective or compute it
            cached = cache_get(design=design, problem_id=problem_id)
            if cached is not None:
                objective = cached["objectives"][0]
                constraints = cached.get("constraints")
            else:
                objective = float(problem.evaluate(design_array))
                constraints = None
                cost += 1.0  # Add cost of objective evaluation

            # Store with gradient
            cache_store(
                design=design,
                problem_id=problem_id,
                objectives=[objective],
                gradient=gradient.tolist(),
                constraints=constraints,
                cost=cost,
            )

        return {
            "success": True,
            "gradient": gradient.tolist(),
            "gradient_norm": gradient_norm,
            "cost": cost,
            "cache_hit": False,
            "method_used": method,
            "evaluation_time": eval_time,
            "message": f"Computed gradient using {method}. Norm: {gradient_norm:.6e} (cost: {cost:.1f})",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error computing gradient: {str(e)}",
        }


# Utility functions
def clear_problem_registry():
    """Clear all problems from registry."""
    _PROBLEM_REGISTRY.clear()


def get_problem_by_id(problem_id: str) -> Optional[Any]:
    """Get problem instance by ID."""
    return _PROBLEM_REGISTRY.get(problem_id)
