"""
Intent-based optimization tools - LLM-driven architecture.

The Paola Principle: "Optimization complexity is agent intelligence, not user burden."

This module provides tools where:
- Information tools provide data TO the LLM for reasoning
- Execution tools execute decisions FROM the LLM
- The LLM's trained knowledge (IPOPT docs, optimization theory, etc.) IS the intelligence

Architecture:
    LLM reasons → selects optimizer → constructs config → calls run_optimization
    (Intelligence)                                        (Execution)

Example workflow:
    1. LLM calls get_problem_info("rosenbrock_10d")
    2. LLM reasons: "10D unconstrained, gradient available → L-BFGS-B is ideal"
    3. LLM calls run_optimization(
           problem_id="rosenbrock_10d",
           optimizer="scipy:L-BFGS-B",
           config='{"ftol": 1e-8}'
       )
"""

from typing import Optional, Dict, Any, List
import numpy as np
from langchain_core.tools import tool
import time
import logging
import json

from ..optimizers.backends import (
    get_backend,
    list_backends,
    get_available_backends,
    OptimizationResult,
)
from ..foundry.nlp_schema import NLPProblem

logger = logging.getLogger(__name__)


@tool
def run_optimization(
    problem_id: str,
    optimizer: str,
    config: Optional[str] = None,
    max_iterations: int = 100,
    run_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute optimization with LLM-specified backend and configuration.

    The LLM decides which optimizer and configuration to use based on:
    - Problem characteristics (from get_problem_info)
    - Available backends (from list_available_optimizers)
    - Paola's learned knowledge of optimization algorithms

    Args:
        problem_id: Problem ID (from create_nlp_problem)
        optimizer: Backend specification:
            - "scipy" or "scipy:SLSQP" - SciPy with method
            - "scipy:L-BFGS-B" - SciPy L-BFGS-B
            - "scipy:trust-constr" - SciPy trust-constr
            - "ipopt" - IPOPT interior-point optimizer
            - "optuna" or "optuna:TPE" - Optuna with sampler
            - "optuna:CMA-ES" - Optuna CMA-ES sampler
        config: JSON string with optimizer-specific options.
            LLM constructs this based on its knowledge.
            SciPy: '{"ftol": 1e-6, "gtol": 1e-5}'
            IPOPT: '{"tol": 1e-6, "mu_strategy": "adaptive"}'
            Optuna: '{"n_trials": 100, "seed": 42}'
        max_iterations: Maximum iterations (default: 100)
        run_id: Optional run ID for recording results

    Returns:
        Dict with:
            success: bool
            message: str
            final_design: List[float]
            final_objective: float
            optimizer_used: str
            n_iterations: int
            n_function_evals: int
            n_gradient_evals: int
            convergence_info: Dict

    Example:
        # LLM decides to use L-BFGS-B for unconstrained problem
        result = run_optimization(
            problem_id="rosenbrock_10d",
            optimizer="scipy:L-BFGS-B",
            config='{"ftol": 1e-8}',
            max_iterations=200
        )

        # LLM decides IPOPT for constrained problem
        result = run_optimization(
            problem_id="constrained_nlp",
            optimizer="ipopt",
            config='{"tol": 1e-6, "mu_strategy": "adaptive"}'
        )

        # LLM decides Optuna for multi-modal problem
        result = run_optimization(
            problem_id="ackley_10d",
            optimizer="optuna:TPE",
            config='{"n_trials": 500}'
        )
    """
    from .evaluator_tools import _get_problem

    try:
        # Get problem
        problem = _get_problem(problem_id)

        # Parse optimizer specification (e.g., "scipy:SLSQP" → backend="scipy", method="SLSQP")
        parts = optimizer.split(":")
        backend_name = parts[0].lower()
        method = parts[1] if len(parts) > 1 else None

        # Get backend
        backend = get_backend(backend_name)
        if backend is None:
            available = get_available_backends()
            return {
                "success": False,
                "message": f"Unknown optimizer backend '{backend_name}'. Available: {available}",
            }

        if not backend.is_available():
            return {
                "success": False,
                "message": f"Backend '{backend_name}' is not installed. Check package requirements.",
            }

        # Parse config
        config_dict = {}
        if config:
            try:
                config_dict = json.loads(config)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "message": f"Invalid config JSON: {e}",
                }

        # Add method to config if specified
        if method:
            config_dict["method"] = method
        config_dict["max_iterations"] = max_iterations

        # Get bounds and constraints from problem
        nlp_problem = None
        if hasattr(problem, "problem"):
            nlp_problem = problem.problem

        if nlp_problem:
            bounds = nlp_problem.bounds
            x0 = np.array(nlp_problem.get_bounds_center())
        elif hasattr(problem, "bounds"):
            bounds = problem.bounds
            x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
        elif hasattr(problem, "get_bounds"):
            # AnalyticalFunction with get_bounds() method
            lb, ub = problem.get_bounds()
            bounds = [[float(lb[i]), float(ub[i])] for i in range(len(lb))]
            x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
        else:
            return {
                "success": False,
                "message": "Problem must have bounds defined",
            }

        # Prepare objective function
        def objective(x):
            return float(problem.evaluate(x))

        # Prepare gradient if available
        gradient = None
        if hasattr(problem, "gradient"):
            gradient = problem.gradient

        # Get constraints if available
        constraints = None
        if hasattr(problem, "get_scipy_constraints"):
            constraints = problem.get_scipy_constraints()

        # Run optimization
        start_time = time.time()

        result = backend.optimize(
            objective=objective,
            bounds=bounds,
            x0=x0,
            config=config_dict,
            constraints=constraints,
            gradient=gradient,
        )

        elapsed = time.time() - start_time

        # Record to run if run_id provided
        if run_id is not None:
            try:
                from .run_tools import _FOUNDRY
                if _FOUNDRY is not None:
                    run = _FOUNDRY.get_run(run_id)
                    if run:
                        for h in result.history:
                            if "design" in h:
                                run.record_iteration(
                                    design=np.array(h["design"]),
                                    objective=h["objective"]
                                )
                        # Create a standardized result object for finalize
                        # This ensures compatibility with all backends (SciPy, IPOPT, Optuna)
                        from types import SimpleNamespace
                        std_result = SimpleNamespace(
                            fun=result.final_objective,
                            x=result.final_design,
                            success=result.success,
                            nfev=result.n_function_evals,
                            nit=result.n_iterations,
                            message=result.message
                        )
                        run.finalize(std_result, metadata={"elapsed": elapsed, "optimizer": optimizer})
            except Exception as e:
                logger.warning(f"Failed to record run: {e}")

        # Return result
        return {
            "success": result.success,
            "message": result.message,
            "final_design": result.final_design.tolist() if isinstance(result.final_design, np.ndarray) else list(result.final_design),
            "final_objective": float(result.final_objective),
            "optimizer_used": optimizer,
            "n_iterations": result.n_iterations,
            "n_function_evals": result.n_function_evals,
            "n_gradient_evals": result.n_gradient_evals,
            "elapsed_time": elapsed,
            "optimization_history": result.history[-20:] if len(result.history) > 20 else result.history,
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error running optimization: {str(e)}",
            "traceback": traceback.format_exc(),
        }


@tool
def get_problem_info(problem_id: str) -> Dict[str, Any]:
    """
    Get problem characteristics for LLM reasoning.

    Use this tool to understand a problem before deciding on optimizer
    selection and configuration. Returns information the LLM needs
    to make intelligent optimization decisions.

    Args:
        problem_id: Problem ID (from create_nlp_problem or create_benchmark_problem)

    Returns:
        Dict with:
            success: bool
            problem_id: str
            dimension: int - number of design variables
            bounds: List[List[float]] - variable bounds [[lb, ub], ...]
            bounds_summary: str - compact summary of bounds
            bounds_center: List[float] - center of bounds (typical starting point)
            bounds_width: List[float] - width of bounds per variable
            num_inequality_constraints: int
            num_equality_constraints: int
            has_gradient: bool - whether analytical gradient is available
            domain_hint: Optional[str] - e.g., "shape_optimization"
            description: str

    Example:
        # LLM queries problem info to decide optimizer
        info = get_problem_info("rosenbrock_10d")
        # Returns dimension=10, no constraints, description mentions valley

        # LLM reasons: "Unconstrained, 10D with narrow valley → L-BFGS-B is good"
    """
    from .evaluator_tools import _get_problem

    try:
        problem = _get_problem(problem_id)

        # Get NLPProblem if available
        nlp_problem = None
        if hasattr(problem, "problem"):
            nlp_problem = problem.problem

        if nlp_problem:
            bounds = nlp_problem.bounds
            n_vars = nlp_problem.dimension
            bounds_center = nlp_problem.get_bounds_center()
            bounds_width = nlp_problem.get_bounds_width()

            # Count constraints
            n_ineq = len(nlp_problem.inequality_constraints) if nlp_problem.inequality_constraints else 0
            n_eq = len(nlp_problem.equality_constraints) if nlp_problem.equality_constraints else 0

            # Summarize bounds
            if all(b[0] == bounds[0][0] and b[1] == bounds[0][1] for b in bounds):
                bounds_summary = f"uniform [{bounds[0][0]}, {bounds[0][1]}] for all {n_vars} variables"
            else:
                lb_min, lb_max = min(b[0] for b in bounds), max(b[0] for b in bounds)
                ub_min, ub_max = min(b[1] for b in bounds), max(b[1] for b in bounds)
                bounds_summary = f"lower: [{lb_min}, {lb_max}], upper: [{ub_min}, {ub_max}]"

            return {
                "success": True,
                "problem_id": problem_id,
                "dimension": n_vars,
                "bounds": bounds[:10] if n_vars > 10 else bounds,  # First 10 for large problems
                "bounds_truncated": n_vars > 10,
                "bounds_summary": bounds_summary,
                "bounds_center": bounds_center[:10] if n_vars > 10 else bounds_center,
                "bounds_width": bounds_width[:10] if n_vars > 10 else bounds_width,
                "num_inequality_constraints": n_ineq,
                "num_equality_constraints": n_eq,
                "is_constrained": n_ineq > 0 or n_eq > 0,
                "has_gradient": hasattr(problem, "gradient"),
                "domain_hint": nlp_problem.domain_hint,
                "description": nlp_problem.description or "No description",
            }

        elif hasattr(problem, "bounds"):
            bounds = problem.bounds
            n_vars = len(bounds)

            return {
                "success": True,
                "problem_id": problem_id,
                "dimension": n_vars,
                "bounds": bounds[:10] if n_vars > 10 else bounds,
                "bounds_truncated": n_vars > 10,
                "bounds_summary": f"{n_vars} variables",
                "bounds_center": [(b[0] + b[1]) / 2 for b in bounds[:10]],
                "bounds_width": [b[1] - b[0] for b in bounds[:10]],
                "num_inequality_constraints": 0,
                "num_equality_constraints": 0,
                "is_constrained": False,
                "has_gradient": hasattr(problem, "gradient"),
                "domain_hint": None,
                "description": getattr(problem, "description", "No description"),
            }

        elif hasattr(problem, "get_bounds"):
            # AnalyticalFunction with get_bounds() method
            lb, ub = problem.get_bounds()
            bounds = [[float(lb[i]), float(ub[i])] for i in range(len(lb))]
            n_vars = len(bounds)

            return {
                "success": True,
                "problem_id": problem_id,
                "dimension": n_vars,
                "bounds": bounds[:10] if n_vars > 10 else bounds,
                "bounds_truncated": n_vars > 10,
                "bounds_summary": f"{n_vars} variables, bounds: [{lb[0]}, {ub[0]}]" if all(lb[i] == lb[0] and ub[i] == ub[0] for i in range(n_vars)) else f"{n_vars} variables",
                "bounds_center": [(b[0] + b[1]) / 2 for b in bounds[:10]],
                "bounds_width": [b[1] - b[0] for b in bounds[:10]],
                "num_inequality_constraints": 0,
                "num_equality_constraints": 0,
                "is_constrained": hasattr(problem, "constraints"),
                "has_gradient": hasattr(problem, "gradient"),
                "domain_hint": None,
                "description": getattr(problem, "name", "Analytical function"),
            }

        else:
            return {
                "success": False,
                "message": "Problem does not have bounds defined",
            }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting problem info: {str(e)}",
        }


@tool
def list_available_optimizers() -> Dict[str, Any]:
    """
    List available optimizer backends and their capabilities.

    Use this tool to discover what optimizers are installed and
    their characteristics. The LLM uses this to decide which
    optimizer to use for a given problem.

    Returns:
        Dict with backend information:
            scipy: {available, methods, supports_constraints, ...}
            ipopt: {available, supports_constraints, key_options, ...}
            optuna: {available, samplers, ...}

    Example:
        # LLM checks available optimizers
        optimizers = list_available_optimizers()

        # If IPOPT available and problem is constrained:
        # LLM reasons: "IPOPT is ideal for large constrained NLPs"
    """
    backends = list_backends()

    result = {
        "success": True,
        "available_backends": get_available_backends(),
        "backends": backends,
        "recommendation": (
            "Choose optimizer based on problem characteristics:\n"
            "- scipy:SLSQP/L-BFGS-B: General purpose, gradient-based\n"
            "- ipopt: Large-scale constrained NLP (if available)\n"
            "- optuna:TPE: Black-box, multi-modal, no gradients needed"
        ),
    }

    return result


@tool
def get_optimizer_options(optimizer: str) -> Dict[str, Any]:
    """
    Get detailed options for a specific optimizer.

    Use this tool to understand what configuration options are
    available for a specific optimizer backend.

    Args:
        optimizer: Optimizer name ("scipy", "ipopt", "optuna")

    Returns:
        Dict with:
            name: str
            available: bool
            methods/samplers: List[str]
            key_options: List[str] - important configuration options
            option_descriptions: Dict[str, str]

    Example:
        # LLM wants to configure IPOPT
        options = get_optimizer_options("ipopt")
        # Returns key options like tol, mu_strategy, linear_solver
    """
    backend = get_backend(optimizer)

    if backend is None:
        return {
            "success": False,
            "message": f"Unknown optimizer '{optimizer}'. Use list_available_optimizers() to see available options.",
        }

    info = backend.get_info()
    info["success"] = True
    info["available"] = backend.is_available()

    # Add detailed option descriptions
    if optimizer.lower() == "scipy":
        info["option_descriptions"] = {
            "method": "Optimization algorithm (SLSQP, L-BFGS-B, trust-constr, etc.)",
            "maxiter": "Maximum iterations",
            "ftol": "Function tolerance for convergence",
            "gtol": "Gradient tolerance for convergence",
            "disp": "Display optimization progress",
        }
    elif optimizer.lower() == "ipopt":
        info["option_descriptions"] = {
            "tol": "Convergence tolerance (default 1e-8)",
            "max_iter": "Maximum iterations (default 3000)",
            "mu_strategy": "Barrier parameter strategy ('monotone' or 'adaptive')",
            "mu_init": "Initial barrier parameter",
            "linear_solver": "Linear solver ('mumps', 'ma27', 'ma57', 'pardiso')",
            "print_level": "Output verbosity (0-12)",
        }
    elif optimizer.lower() == "optuna":
        info["option_descriptions"] = {
            "sampler": "Sampling strategy (TPE, CMA-ES, Random)",
            "n_trials": "Number of trials to run",
            "seed": "Random seed for reproducibility",
        }

    return info
