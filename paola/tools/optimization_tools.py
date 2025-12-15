"""
Intent-based optimization tools - LLM-driven architecture.

The Paola Principle: "Optimization complexity is agent intelligence, not user burden."

This module provides tools where:
- Information tools provide data TO the LLM for reasoning
- Execution tools execute decisions FROM the LLM
- The LLM's trained knowledge (IPOPT docs, optimization theory, etc.) IS the intelligence

v0.2.0: Session-based architecture
- run_optimization requires session_id
- Records runs with polymorphic components per optimizer family

Architecture:
    LLM reasons → selects optimizer → constructs config → calls run_optimization
    (Intelligence)                                        (Execution)
"""

from typing import Optional, Dict, Any
import numpy as np
from langchain_core.tools import tool
import time
import logging
import json

from ..optimizers.backends import (
    get_backend,
    list_backends,
    get_available_backends,
)
from ..foundry import (
    COMPONENT_REGISTRY,
    GradientInitialization,
    GradientProgress,
    GradientResult,
    BayesianInitialization,
    BayesianProgress,
    BayesianResult,
)

logger = logging.getLogger(__name__)


@tool
def run_optimization(
    session_id: int,
    optimizer: str,
    config: Optional[str] = None,
    max_iterations: int = 100,
    init_strategy: str = "center",
) -> Dict[str, Any]:
    """
    Execute optimization within a session.

    The LLM decides which optimizer and configuration to use based on:
    - Problem characteristics (from get_problem_info)
    - Available backends (from list_available_optimizers)
    - Paola's learned knowledge of optimization algorithms

    Args:
        session_id: Session ID from start_session (required)
        optimizer: Backend specification:
            - "scipy" or "scipy:SLSQP" - SciPy with method
            - "scipy:L-BFGS-B" - SciPy L-BFGS-B
            - "scipy:trust-constr" - SciPy trust-constr
            - "ipopt" - IPOPT interior-point optimizer
            - "optuna" or "optuna:TPE" - Optuna with sampler
        config: JSON string with optimizer-specific options.
            LLM constructs this based on its knowledge.
            SciPy: '{"ftol": 1e-6, "gtol": 1e-5}'
            IPOPT: '{"tol": 1e-6, "mu_strategy": "adaptive"}'
            Optuna: '{"n_trials": 100, "seed": 42}'
        max_iterations: Maximum iterations (default: 100)
        init_strategy: Initialization strategy:
            - "center" - center of bounds (default)
            - "random" - random point within bounds
            - "warm_start" - warm-start from previous run in session

    Returns:
        Dict with:
            success: bool
            message: str
            run_id: int - ID of this run within the session
            final_design: List[float]
            final_objective: float
            optimizer_used: str
            n_iterations: int
            n_function_evals: int
            n_gradient_evals: int

    Example:
        # First start a session
        session = start_session(problem_id="rosenbrock_10d")
        # session_id = 1

        # Then run optimization
        result = run_optimization(
            session_id=1,
            optimizer="scipy:L-BFGS-B",
            config='{"ftol": 1e-8}',
            max_iterations=200
        )
    """
    from .evaluator_tools import _get_problem
    from .session_tools import _FOUNDRY

    try:
        # Check foundry
        if _FOUNDRY is None:
            return {
                "success": False,
                "message": "Foundry not initialized. Call set_foundry() first.",
            }

        # Get session
        session = _FOUNDRY.get_session(session_id)
        if session is None:
            return {
                "success": False,
                "message": f"Session {session_id} not found or already finalized. Use start_session first.",
            }

        # Get problem
        problem = _get_problem(session.problem_id)

        # Parse optimizer specification
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
                "message": f"Backend '{backend_name}' is not installed.",
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

        # Add method to config
        if method:
            config_dict["method"] = method
        config_dict["max_iterations"] = max_iterations

        # Get bounds from problem
        nlp_problem = None
        if hasattr(problem, "problem"):
            nlp_problem = problem.problem

        if nlp_problem:
            bounds = nlp_problem.bounds
            bounds_center = nlp_problem.get_bounds_center()
        elif hasattr(problem, "bounds"):
            bounds = problem.bounds
            bounds_center = [(b[0] + b[1]) / 2 for b in bounds]
        elif hasattr(problem, "get_bounds"):
            lb, ub = problem.get_bounds()
            bounds = [[float(lb[i]), float(ub[i])] for i in range(len(lb))]
            bounds_center = [(b[0] + b[1]) / 2 for b in bounds]
        else:
            return {
                "success": False,
                "message": "Problem must have bounds defined",
            }

        # Determine initialization
        family = COMPONENT_REGISTRY.get_family(optimizer)
        warm_start_from = None

        if init_strategy == "warm_start" and session.runs:
            # Use best design from previous run
            best_run = session.get_best_run()
            if best_run:
                x0 = np.array(best_run.best_design)
                warm_start_from = best_run.run_id
            else:
                x0 = np.array(bounds_center)
        elif init_strategy == "random":
            # Random point within bounds
            lower = np.array([b[0] for b in bounds])
            upper = np.array([b[1] for b in bounds])
            x0 = lower + np.random.random(len(bounds)) * (upper - lower)
        else:
            # Default: center of bounds
            x0 = np.array(bounds_center)

        # Create initialization component based on family
        if family == "gradient":
            initialization = GradientInitialization(
                specification={"type": init_strategy, "source_run": warm_start_from},
                x0=x0.tolist(),
            )
        elif family == "bayesian":
            initialization = BayesianInitialization(
                specification={"type": init_strategy},
                warm_start_trials=None,
                n_initial_random=config_dict.get("n_startup_trials", 10),
            )
        else:
            # Default to gradient-style for unknown families
            initialization = GradientInitialization(
                specification={"type": init_strategy},
                x0=x0.tolist(),
            )

        # Start run within session
        active_run = session.start_run(
            optimizer=optimizer,
            initialization=initialization,
            warm_start_from=warm_start_from,
        )

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

        # Record iterations to active run
        for h in result.history:
            active_run.record_iteration(h)

        # Create progress and result components based on family
        if family == "gradient":
            progress = GradientProgress()
            for h in result.history:
                progress.add_iteration(
                    iteration=h.get("iteration", 0),
                    objective=h.get("objective", 0.0),
                    design=h.get("design", []),
                    gradient_norm=h.get("gradient_norm"),
                    step_size=h.get("step_size"),
                    constraint_violation=h.get("constraint_violation"),
                )
            result_component = GradientResult(
                termination_reason=result.message,
                final_gradient_norm=None,
                final_constraint_violation=None,
            )
        elif family == "bayesian":
            progress = BayesianProgress()
            for i, h in enumerate(result.history):
                progress.add_trial(
                    trial_number=h.get("trial", i + 1),
                    design=h.get("design", []),
                    objective=h.get("objective", 0.0),
                    state="complete",
                )
            result_component = BayesianResult(
                termination_reason=result.message,
                best_trial_number=len(result.history),
                n_complete_trials=len(result.history),
                n_pruned_trials=0,
            )
        else:
            # Default to gradient-style
            progress = GradientProgress()
            for h in result.history:
                progress.add_iteration(
                    iteration=h.get("iteration", 0),
                    objective=h.get("objective", 0.0),
                    design=h.get("design", []),
                )
            result_component = GradientResult(
                termination_reason=result.message,
            )

        # Complete run
        final_design = (
            result.final_design.tolist()
            if isinstance(result.final_design, np.ndarray)
            else list(result.final_design)
        )

        completed_run = session.complete_run(
            progress=progress,
            result=result_component,
            best_objective=result.final_objective,
            best_design=final_design,
            success=result.success,
        )

        # Return result
        return {
            "success": result.success,
            "message": result.message,
            "run_id": completed_run.run_id,
            "final_design": final_design,
            "final_objective": float(result.final_objective),
            "optimizer_used": optimizer,
            "optimizer_family": family,
            "n_iterations": result.n_iterations,
            "n_function_evals": result.n_function_evals,
            "n_gradient_evals": result.n_gradient_evals,
            "elapsed_time": elapsed,
            "warm_started_from": warm_start_from,
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
    selection and configuration.

    Args:
        problem_id: Problem ID (from create_nlp_problem)

    Returns:
        Dict with problem characteristics
    """
    from .evaluator_tools import _get_problem

    try:
        problem = _get_problem(problem_id)

        nlp_problem = None
        if hasattr(problem, "problem"):
            nlp_problem = problem.problem

        if nlp_problem:
            bounds = nlp_problem.bounds
            n_vars = nlp_problem.dimension
            bounds_center = nlp_problem.get_bounds_center()
            bounds_width = nlp_problem.get_bounds_width()

            n_ineq = len(nlp_problem.inequality_constraints) if nlp_problem.inequality_constraints else 0
            n_eq = len(nlp_problem.equality_constraints) if nlp_problem.equality_constraints else 0

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
                "bounds": bounds[:10] if n_vars > 10 else bounds,
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
            lb, ub = problem.get_bounds()
            bounds = [[float(lb[i]), float(ub[i])] for i in range(len(lb))]
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

    Use this tool to discover what optimizers are installed.

    Returns:
        Dict with backend information
    """
    backends = list_backends()

    return {
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


@tool
def get_optimizer_options(optimizer: str) -> Dict[str, Any]:
    """
    Get detailed options for a specific optimizer.

    Args:
        optimizer: Optimizer name ("scipy", "ipopt", "optuna")

    Returns:
        Dict with options and descriptions
    """
    backend = get_backend(optimizer)

    if backend is None:
        return {
            "success": False,
            "message": f"Unknown optimizer '{optimizer}'.",
        }

    info = backend.get_info()
    info["success"] = True
    info["available"] = backend.is_available()

    if optimizer.lower() == "scipy":
        info["option_descriptions"] = {
            "method": "Optimization algorithm (SLSQP, L-BFGS-B, trust-constr)",
            "maxiter": "Maximum iterations",
            "ftol": "Function tolerance for convergence",
            "gtol": "Gradient tolerance for convergence",
        }
    elif optimizer.lower() == "ipopt":
        info["option_descriptions"] = {
            "tol": "Convergence tolerance (default 1e-8)",
            "max_iter": "Maximum iterations (default 3000)",
            "mu_strategy": "Barrier parameter strategy",
            "linear_solver": "Linear solver",
        }
    elif optimizer.lower() == "optuna":
        info["option_descriptions"] = {
            "sampler": "Sampling strategy (TPE, CMA-ES, Random)",
            "n_trials": "Number of trials to run",
            "seed": "Random seed for reproducibility",
        }

    return info
