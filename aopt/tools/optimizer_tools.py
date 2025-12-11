"""
Optimizer tools for the agentic optimization platform.

Provides LangChain @tool decorated functions for optimizer operations:
- optimizer_create: Create optimizer instance
- optimizer_propose: Propose next design
- optimizer_update: Update with evaluation results
- optimizer_restart: Strategic restart from design
"""

from typing import Optional, Dict, Any, List
import numpy as np
from langchain_core.tools import tool
import json

from aopt.optimizers.base import BaseOptimizer
from aopt.optimizers.scipy_optimizer import create_scipy_optimizer


# Global optimizer registry
_OPTIMIZER_REGISTRY: Dict[str, BaseOptimizer] = {}


def _get_optimizer(optimizer_id: str) -> BaseOptimizer:
    """Get optimizer from registry."""
    if optimizer_id not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Optimizer '{optimizer_id}' not found. "
            f"Available: {list(_OPTIMIZER_REGISTRY.keys())}"
        )
    return _OPTIMIZER_REGISTRY[optimizer_id]


@tool
def optimizer_create(
    optimizer_id: str,
    problem_id: str,
    algorithm: str,
    bounds: List[List[float]],
    initial_design: Optional[List[float]] = None,
    options: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new optimizer instance.

    Use this tool to initialize an optimizer for a problem. The optimizer
    will be stored in the registry and can be referenced by optimizer_id
    in subsequent calls to optimizer_propose, optimizer_update, etc.

    Args:
        optimizer_id: Unique identifier for this optimizer (e.g., "opt_1")
        problem_id: Problem identifier from formulate_problem
        algorithm: Algorithm name - one of:
            - "SLSQP": Sequential Least Squares Programming (gradient-based, constrained)
            - "L-BFGS-B": Limited-memory BFGS (gradient-based, box constraints only)
            - "COBYLA": Constrained Optimization (derivative-free)
        bounds: Variable bounds as [[lower1, upper1], [lower2, upper2], ...]
        initial_design: Optional starting design [x1, x2, ...]. If not provided,
                       a random point within bounds will be generated
        options: Optional JSON string with algorithm-specific options, e.g.:
                '{"maxiter": 50, "ftol": 1e-6}'

    Returns:
        Dict with:
            - success: bool
            - optimizer_id: str
            - algorithm: str
            - n_variables: int
            - initial_design: List[float]
            - message: str

    Example:
        result = optimizer_create(
            optimizer_id="opt_1",
            problem_id="rosenbrock_2d",
            algorithm="SLSQP",
            bounds=[[-5.0, 10.0], [-5.0, 10.0]],
            initial_design=[-1.0, 1.0]
        )
    """
    try:
        # Check if optimizer_id already exists
        if optimizer_id in _OPTIMIZER_REGISTRY:
            return {
                "success": False,
                "optimizer_id": optimizer_id,
                "message": f"Optimizer '{optimizer_id}' already exists. Use a different ID or call optimizer_restart.",
            }

        # Parse bounds
        bounds_array = np.array(bounds)
        if bounds_array.ndim != 2 or bounds_array.shape[1] != 2:
            return {
                "success": False,
                "message": f"Invalid bounds format. Expected [[lower, upper], ...], got shape {bounds_array.shape}",
            }

        lower_bounds = bounds_array[:, 0]
        upper_bounds = bounds_array[:, 1]
        bounds_tuple = (lower_bounds, upper_bounds)

        # Parse initial design
        if initial_design is not None:
            initial_array = np.array(initial_design)
            if len(initial_array) != len(lower_bounds):
                return {
                    "success": False,
                    "message": f"Initial design dimension {len(initial_array)} doesn't match bounds {len(lower_bounds)}",
                }
        else:
            # Generate random initial design
            initial_array = lower_bounds + np.random.rand(len(lower_bounds)) * (upper_bounds - lower_bounds)

        # Parse options
        options_dict = None
        if options:
            try:
                options_dict = json.loads(options)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "message": f"Invalid options JSON: {e}",
                }

        # Create optimizer
        optimizer = create_scipy_optimizer(
            problem_id=problem_id,
            algorithm=algorithm,
            bounds=bounds_tuple,
            initial_design=initial_array,
            options=options_dict,
        )

        # Register
        _OPTIMIZER_REGISTRY[optimizer_id] = optimizer

        return {
            "success": True,
            "optimizer_id": optimizer_id,
            "algorithm": algorithm,
            "n_variables": len(lower_bounds),
            "initial_design": initial_array.tolist(),
            "message": f"Created {algorithm} optimizer '{optimizer_id}' with {len(lower_bounds)} variables",
        }

    except Exception as e:
        return {
            "success": False,
            "optimizer_id": optimizer_id,
            "message": f"Error creating optimizer: {str(e)}",
        }


@tool
def optimizer_propose(optimizer_id: str) -> Dict[str, Any]:
    """
    Propose next design to evaluate.

    Use this tool to get the next design point from the optimizer.
    After receiving the design, evaluate it using evaluate_function,
    then update the optimizer using optimizer_update.

    Args:
        optimizer_id: Optimizer identifier from optimizer_create

    Returns:
        Dict with:
            - success: bool
            - design: List[float] - proposed design to evaluate
            - iteration: int - current iteration number
            - converged: bool - whether optimizer has converged
            - message: str

    Example:
        proposal = optimizer_propose("opt_1")
        if proposal["success"]:
            design = proposal["design"]
            # Now evaluate this design using evaluate_function
    """
    try:
        optimizer = _get_optimizer(optimizer_id)

        # Check if already converged
        if optimizer.is_converged():
            info = optimizer.get_convergence_info()
            return {
                "success": False,
                "converged": True,
                "message": f"Optimizer converged: {info['reason']}",
                "convergence_info": info,
            }

        # Propose design
        design = optimizer.propose_design()

        return {
            "success": True,
            "design": design.tolist(),
            "iteration": optimizer.state.iteration,
            "converged": False,
            "message": f"Proposed design at iteration {optimizer.state.iteration}",
        }

    except StopIteration as e:
        return {
            "success": False,
            "converged": True,
            "message": str(e),
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error proposing design: {str(e)}",
        }


@tool
def optimizer_update(
    optimizer_id: str,
    design: List[float],
    objective: float,
    gradient: Optional[List[float]] = None,
    constraints: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update optimizer with evaluation results.

    Use this tool after evaluating a proposed design to update the
    optimizer's internal state. The optimizer will use this information
    to propose the next design.

    Args:
        optimizer_id: Optimizer identifier
        design: Evaluated design vector
        objective: Objective function value
        gradient: Optional gradient vector (required for gradient-based algorithms)
        constraints: Optional JSON string with constraint values, e.g.:
                    '{"c1": 0.5, "c2": -0.1}' (negative = satisfied)

    Returns:
        Dict with:
            - success: bool
            - iteration: int
            - converged: bool
            - reason: Optional[str] - convergence reason
            - improvement: float - objective improvement from previous best
            - gradient_norm: Optional[float]
            - best_objective: float
            - message: str

    Example:
        update_info = optimizer_update(
            optimizer_id="opt_1",
            design=[-1.0, 1.0],
            objective=4.0,
            gradient=[-2.0, 0.0]
        )
        if update_info["converged"]:
            print(f"Converged: {update_info['reason']}")
    """
    try:
        optimizer = _get_optimizer(optimizer_id)

        # Parse inputs
        design_array = np.array(design)
        gradient_array = np.array(gradient) if gradient is not None else None

        constraints_dict = None
        if constraints:
            try:
                constraints_dict = json.loads(constraints)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "message": f"Invalid constraints JSON: {e}",
                }

        # Update optimizer
        update_info = optimizer.update(
            design=design_array,
            objective=objective,
            gradient=gradient_array,
            constraints=constraints_dict,
        )

        # Get best so far
        _, best_obj = optimizer.get_best()

        return {
            "success": True,
            "iteration": update_info["iteration"],
            "converged": update_info["converged"],
            "reason": update_info["reason"],
            "improvement": update_info["improvement"],
            "gradient_norm": update_info["gradient_norm"],
            "best_objective": best_obj,
            "message": (
                f"Updated iteration {update_info['iteration']}. "
                f"Best objective: {best_obj:.6f}"
            ),
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error updating optimizer: {str(e)}",
        }


@tool
def optimizer_restart(
    optimizer_id: str,
    restart_from: str = "best",
    reuse_cache: bool = True,
    checkpoint_old: bool = True,
    new_options: Optional[str] = None,
    new_initial_design: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Restart optimizer from a specific design with optionally modified settings.

    Use this tool for strategic restarts after detecting numerical issues,
    constraint violations, or when you want to try different optimizer settings.

    IMPORTANT: This is a strategic restart, not arbitrary experimentation.
    Restart from a known good position (usually best design) with informed
    changes to settings. The evaluation cache is reused to preserve expensive work.

    Args:
        optimizer_id: Optimizer identifier
        restart_from: Where to restart from:
            - "best": Restart from best design found so far (default, safest)
            - "current": Restart from current design
            - "custom": Restart from new_initial_design (must provide)
        reuse_cache: If True, cached evaluations remain valid (default: True).
                    Set False only if problem formulation changed.
        checkpoint_old: If True, save checkpoint before restart (default: True).
                       Enables rollback if restart doesn't improve.
        new_options: Optional JSON string with modified optimizer options, e.g.:
                    '{"ftol": 1e-8}' to tighten tolerance
        new_initial_design: Design to restart from (only if restart_from="custom")

    Returns:
        Dict with:
            - success: bool
            - restart_design: List[float]
            - restart_objective: float
            - old_checkpoint: Optional[str] - checkpoint ID if checkpoint_old=True
            - message: str

    Example:
        # Restart from best design with tighter tolerance
        result = optimizer_restart(
            optimizer_id="opt_1",
            restart_from="best",
            new_options='{"ftol": 1e-8}'
        )
    """
    try:
        optimizer = _get_optimizer(optimizer_id)

        # Create checkpoint if requested
        checkpoint_id = None
        if checkpoint_old:
            checkpoint_id = f"{optimizer_id}_checkpoint_{optimizer.state.iteration}"
            checkpoint = optimizer.checkpoint()
            # Store checkpoint (in real implementation, save to file/database)
            # For now, just store in memory
            _OPTIMIZER_REGISTRY[checkpoint_id] = checkpoint

        # Determine restart design
        if restart_from == "best":
            restart_design, restart_obj = optimizer.get_best()
        elif restart_from == "current":
            restart_design = optimizer.state.current_design
            restart_obj = optimizer.state.current_objective
        elif restart_from == "custom":
            if new_initial_design is None:
                return {
                    "success": False,
                    "message": "new_initial_design required when restart_from='custom'",
                }
            restart_design = np.array(new_initial_design)
            # Need to evaluate at this design (or get from cache)
            restart_obj = None  # Will need evaluation
        else:
            return {
                "success": False,
                "message": f"Invalid restart_from: {restart_from}. Use 'best', 'current', or 'custom'.",
            }

        # Parse new options
        new_options_dict = None
        if new_options:
            try:
                new_options_dict = json.loads(new_options)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "message": f"Invalid new_options JSON: {e}",
                }

        # Get gradient at restart point if available from history
        gradient_at_restart = None
        for record in reversed(optimizer.history):
            if np.allclose(record["design"], restart_design, rtol=1e-9):
                gradient_at_restart = record["gradient"]
                break

        # Restart optimizer
        optimizer.restart_from_design(
            design=restart_design,
            objective=restart_obj if restart_obj is not None else float("inf"),
            gradient=gradient_at_restart,
            new_options=new_options_dict,
        )

        return {
            "success": True,
            "restart_design": restart_design.tolist(),
            "restart_objective": restart_obj if restart_obj is not None else None,
            "old_checkpoint": checkpoint_id,
            "cache_reused": reuse_cache,
            "message": (
                f"Restarted from {restart_from} design. "
                f"Objective: {restart_obj if restart_obj is not None else 'unknown':.6f}. "
                f"Cache {'preserved' if reuse_cache else 'cleared'}."
            ),
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error restarting optimizer: {str(e)}",
        }


@tool
def run_scipy_optimization(
    problem_id: str,
    algorithm: str,
    bounds: List[List[float]],
    initial_design: Optional[List[float]] = None,
    options: Optional[str] = None,
    use_gradient: bool = True,
) -> Dict[str, Any]:
    """
    Run scipy optimization to completion in a single call.

    This tool runs a full scipy.optimize.minimize optimization from start to finish.
    Use this when you want to run an optimization without step-by-step interception.
    The optimization runs autonomously and returns final results.

    For step-by-step control, use optimizer_create/propose/update instead.

    Args:
        problem_id: Problem identifier (must be registered via register_problem)
        algorithm: Scipy algorithm - one of:
            - "SLSQP": Sequential Least Squares Programming (gradient-based, constrained)
            - "L-BFGS-B": Limited-memory BFGS (gradient-based, box constraints)
            - "COBYLA": Constrained Optimization BY Linear Approximation (derivative-free)
            - "Nelder-Mead": Simplex algorithm (derivative-free)
            - "Powell": Powell's method (derivative-free)
        bounds: Variable bounds as [[lower1, upper1], [lower2, upper2], ...]
        initial_design: Starting design [x1, x2, ...]. Random if not provided.
        options: JSON string with algorithm options, e.g.:
                '{"maxiter": 200, "ftol": 1e-9}'
        use_gradient: If True (default), use analytical gradient for gradient-based methods.

    Returns:
        Dict with:
            - success: bool - scipy optimization success flag
            - message: str - scipy termination message
            - final_design: List[float] - optimal design found
            - final_objective: float - final objective value
            - n_iterations: int - number of iterations
            - n_function_evals: int - number of function evaluations
            - n_gradient_evals: int - number of gradient evaluations (if used)
            - optimization_history: List[Dict] - history of iterations
            - convergence_info: Dict - convergence analysis

    Example:
        result = run_scipy_optimization(
            problem_id="rosenbrock_2d",
            algorithm="SLSQP",
            bounds=[[-5.0, 10.0], [-5.0, 10.0]],
            initial_design=[0.0, 0.0],
            options='{"maxiter": 200, "ftol": 1e-9}'
        )
    """
    from scipy.optimize import minimize
    from aopt.tools.evaluator_tools import _get_problem

    try:
        # Get problem
        problem = _get_problem(problem_id)

        # Parse bounds
        bounds_array = np.array(bounds)
        if bounds_array.ndim != 2 or bounds_array.shape[1] != 2:
            return {
                "success": False,
                "message": f"Invalid bounds format. Expected [[lower, upper], ...], got shape {bounds_array.shape}",
            }

        lower_bounds = bounds_array[:, 0]
        upper_bounds = bounds_array[:, 1]
        n_vars = len(lower_bounds)

        # Parse initial design
        if initial_design is not None:
            x0 = np.array(initial_design)
            if len(x0) != n_vars:
                return {
                    "success": False,
                    "message": f"Initial design dimension {len(x0)} doesn't match bounds {n_vars}",
                }
        else:
            # Random initial design within bounds
            x0 = lower_bounds + np.random.rand(n_vars) * (upper_bounds - lower_bounds)

        # Parse options
        options_dict = {"maxiter": 200, "disp": False}
        if options:
            try:
                user_options = json.loads(options)
                options_dict.update(user_options)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "message": f"Invalid options JSON: {e}",
                }

        # Prepare history tracking
        history = []
        n_func_evals = [0]
        n_grad_evals = [0]

        def objective_with_history(x):
            obj = float(problem.evaluate(x))
            n_func_evals[0] += 1
            history.append({
                "iteration": len(history),
                "objective": obj,
                "design": x.tolist(),
            })
            return obj

        def gradient_with_count(x):
            grad = problem.gradient(x)
            n_grad_evals[0] += 1
            return grad

        # Determine if gradient-based
        gradient_methods = {"SLSQP", "L-BFGS-B", "BFGS", "CG", "Newton-CG", "TNC", "trust-constr"}
        is_gradient_method = algorithm.upper() in [m.upper() for m in gradient_methods]

        # Set up scipy bounds
        scipy_bounds = [(lb, ub) for lb, ub in zip(lower_bounds, upper_bounds)]

        # Run optimization
        import time
        start_time = time.time()

        if is_gradient_method and use_gradient and hasattr(problem, "gradient"):
            result = minimize(
                fun=objective_with_history,
                x0=x0,
                method=algorithm,
                jac=gradient_with_count,
                bounds=scipy_bounds,
                options=options_dict,
            )
        else:
            result = minimize(
                fun=objective_with_history,
                x0=x0,
                method=algorithm,
                bounds=scipy_bounds,
                options=options_dict,
            )

        elapsed = time.time() - start_time

        # Analyze convergence from history
        objectives = [h["objective"] for h in history]
        if len(objectives) >= 2:
            total_improvement = objectives[0] - objectives[-1]
            avg_improvement = total_improvement / len(objectives) if objectives else 0
            convergence_info = {
                "initial_objective": objectives[0],
                "final_objective": objectives[-1],
                "total_improvement": total_improvement,
                "improvement_rate": avg_improvement,
                "converged": result.success,
            }
        else:
            convergence_info = {
                "converged": result.success,
            }

        return {
            "success": result.success,
            "message": result.message if hasattr(result, "message") else str(result.get("message", "completed")),
            "final_design": result.x.tolist(),
            "final_objective": float(result.fun),
            "n_iterations": int(result.nit) if hasattr(result, "nit") else len(history),
            "n_function_evals": n_func_evals[0],
            "n_gradient_evals": n_grad_evals[0],
            "elapsed_time": elapsed,
            "optimization_history": history[-20:] if len(history) > 20 else history,  # Last 20 entries
            "convergence_info": convergence_info,
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error running optimization: {str(e)}",
            "traceback": traceback.format_exc(),
        }


# Utility function to clear registry (for testing)
def clear_optimizer_registry():
    """Clear all optimizers from registry."""
    _OPTIMIZER_REGISTRY.clear()


# Get optimizer by ID (for testing/internal use)
def get_optimizer_by_id(optimizer_id: str) -> Optional[BaseOptimizer]:
    """Get optimizer instance by ID."""
    return _OPTIMIZER_REGISTRY.get(optimizer_id)
