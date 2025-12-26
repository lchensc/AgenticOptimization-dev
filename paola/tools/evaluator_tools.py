"""
Evaluator tools for function evaluation.

Tools for function evaluation:
- evaluate_function: Evaluate objective and constraints (with automatic caching)
- compute_gradient: Compute gradients (analytical or finite-difference)
- create_benchmark_problem: Create built-in benchmark problems

Internal utilities:
- _get_problem: Get problem evaluator from Foundry
- register_problem: Register problem evaluator (deprecated)
- clear_problem_registry: Clear problem cache
- get_problem_by_id: Get problem by ID
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np
from langchain_core.tools import tool
import time

from paola.tools.cache_tools import cache_get, cache_store
from paola.tools.schemas import (
    normalize_problem_id,
    ProblemIdType,
    EvaluateFunctionArgs,
    ComputeGradientArgs,
)
from paola.backends.analytical import get_analytical_function


def _get_problem(problem_id: ProblemIdType) -> Any:
    """Get problem evaluator from Foundry (single source of truth).

    Args:
        problem_id: Problem ID (str or int, normalized to int)

    Returns:
        Problem evaluator (NLPEvaluator or similar)

    Raises:
        ValueError: If problem not found in storage
    """
    from paola.tools.graph_tools import get_foundry

    key = normalize_problem_id(problem_id)
    foundry = get_foundry()

    if foundry is None:
        raise ValueError(
            "Foundry not initialized. CLI should call set_foundry() at startup."
        )

    evaluator = foundry.get_problem_evaluator(key)
    if evaluator is None:
        # Get available IDs from cache for error message
        cached_ids = foundry.get_cached_problem_ids()
        raise ValueError(
            f"Problem {key} not found in storage. "
            f"Cached: {cached_ids}"
        )
    return evaluator


def register_problem(problem_id: ProblemIdType, problem: Any) -> None:
    """Register a problem evaluator in Foundry's cache.

    DEPRECATED: Use foundry.register_problem_evaluator() directly.
    This function is kept for backward compatibility.

    Args:
        problem_id: Problem ID (str or int, normalized to int)
        problem: Problem evaluator (NLPEvaluator or similar)
    """
    from paola.tools.graph_tools import get_foundry

    key = normalize_problem_id(problem_id)
    foundry = get_foundry()

    if foundry is not None:
        # Add to Foundry's cache directly (for backward compatibility)
        foundry._problem_cache[key] = problem


@tool(args_schema=EvaluateFunctionArgs)
def evaluate_function(
    problem_id: ProblemIdType,
    design: List[float],
    use_cache: bool = True,
    compute_constraints: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate objective function at a design point.

    Args:
        problem_id: Problem identifier (int or str)
        design: Design vector to evaluate
        use_cache: Check cache before evaluation (default: True)
        compute_constraints: Also evaluate constraints (default: False)

    Returns:
        success, objective, constraints, cost, cache_hit, evaluation_time

    Example:
        evaluate_function(problem_id=1, design=[-1.0, 1.0])
    """
    try:
        # Normalize problem_id (handles str/int from LLM)
        problem_id = normalize_problem_id(problem_id)

        # Check cache first
        if use_cache:
            cached = cache_get(design=design, problem_id=problem_id)
            if cached is not None:
                return {
                    "success": True,
                    "objective": cached["objectives"][0],
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
            objective = float(problem.evaluate(design_array))
        else:
            raise ValueError(f"Problem '{problem_id}' doesn't have evaluate() method")

        # Evaluate constraints if requested
        constraints_dict = None
        if compute_constraints and hasattr(problem, "evaluate_constraint"):
            c_val = float(problem.evaluate_constraint(design_array))
            constraints_dict = {"c1": c_val}

        eval_time = time.time() - start_time
        cost = 1.0  # 1 unit for analytical evaluation

        # Store in cache
        if use_cache:
            cache_store(
                design=design,
                problem_id=problem_id,
                objectives=[objective],
                gradient=None,
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


@tool(args_schema=ComputeGradientArgs)
def compute_gradient(
    problem_id: ProblemIdType,
    design: List[float],
    method: str = "analytical",
    use_cache: bool = True,
    fd_step: float = 1e-6,
) -> Dict[str, Any]:
    """
    Compute gradient of objective function.

    Args:
        problem_id: Problem identifier (int or str)
        design: Design vector
        method: "analytical" or "finite-difference"
        use_cache: Check cache before computation (default: True)
        fd_step: Step size for finite-difference (default: 1e-6)

    Returns:
        success, gradient, gradient_norm, cost, cache_hit, method_used

    Example:
        compute_gradient(problem_id=1, design=[-1.0, 1.0], method="analytical")
    """
    try:
        # Normalize problem_id (handles str/int from LLM)
        problem_id = normalize_problem_id(problem_id)

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
            cost = 1.5
        else:  # finite-difference
            cost = 2 * len(design_array)

        # Update cache with gradient
        if use_cache:
            cached = cache_get(design=design, problem_id=problem_id)
            if cached is not None:
                objective = cached["objectives"][0]
                constraints = cached.get("constraints")
            else:
                objective = float(problem.evaluate(design_array))
                constraints = None
                cost += 1.0

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


@tool
def create_benchmark_problem(
    problem_id: str,
    function_name: str,
    dimension: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create built-in analytical benchmark problem.

    Args:
        problem_id: Unique identifier (e.g., "rosenbrock_10d")
        function_name: "rosenbrock", "sphere", or "constrained_rosenbrock"
        dimension: Problem dimensionality (default: 2)

    Returns:
        success, problem_id, function_name, dimension, global_optimum

    Example:
        create_benchmark_problem("rosenbrock_10d", "rosenbrock", 10)
    """
    try:
        from paola.tools.graph_tools import get_foundry

        foundry = get_foundry()
        if foundry is None:
            return {
                "success": False,
                "message": "Foundry not initialized. CLI should call set_foundry() at startup.",
            }

        if problem_id in foundry._problem_cache:
            return {
                "success": False,
                "problem_id": problem_id,
                "message": f"Problem '{problem_id}' already registered. Use a different ID or clear registry first.",
            }

        # Create analytical function
        problem = get_analytical_function(name=function_name, dimension=dimension)

        # Register in Foundry's cache
        foundry._problem_cache[problem_id] = problem

        # Get optimum info
        x_opt, f_opt = problem.get_optimum()

        return {
            "success": True,
            "problem_id": problem_id,
            "function_name": function_name,
            "dimension": problem.dimension,
            "global_optimum": {
                "x_opt": x_opt.tolist() if hasattr(x_opt, 'tolist') else list(x_opt),
                "f_opt": float(f_opt),
            },
            "message": f"Created {problem.dimension}D {function_name} problem with ID '{problem_id}'. Global optimum: f* = {f_opt:.6f}",
        }

    except ValueError as e:
        return {
            "success": False,
            "message": str(e),
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error creating benchmark problem: {str(e)}",
        }


# Utility functions
def clear_problem_registry():
    """Clear problem evaluator cache in Foundry."""
    from paola.tools.graph_tools import get_foundry

    foundry = get_foundry()
    if foundry is not None:
        foundry.clear_problem_cache()


def get_problem_by_id(problem_id: ProblemIdType) -> Optional[Any]:
    """Get problem evaluator by ID from Foundry.

    Args:
        problem_id: Problem ID (str or int, normalized to int)

    Returns:
        Problem evaluator or None if not found
    """
    from paola.tools.graph_tools import get_foundry

    key = normalize_problem_id(problem_id)
    foundry = get_foundry()
    if foundry is None:
        return None
    return foundry.get_problem_evaluator(key)
