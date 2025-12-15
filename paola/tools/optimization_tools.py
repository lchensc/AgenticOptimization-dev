"""
Intent-based optimization tools.

The Paola Principle: "Optimization complexity is Paola intelligence, not user burden."

This module provides intent-based tools where the agent specifies WHAT they want
(optimizer="auto", priority="robustness") and Paola handles the HOW:
- Algorithm selection based on problem characteristics
- Initial point computation based on domain and history
- Configuration based on priority and problem type

Example:
    # Instead of:
    run_scipy_optimization(
        problem_id="wing",
        algorithm="SLSQP",
        bounds=[[...] * 100],  # Large explicit bounds
        initial_design=[...],  # User must know this
        options='{"maxiter": 200, "ftol": 1e-6}'  # User must know options
    )

    # Use:
    run_optimization(
        problem_id="wing",
        optimizer="auto",     # Paola selects
        priority="robustness" # Intent, not options
    )
"""

from typing import Optional, Dict, Any, List
import numpy as np
from langchain_core.tools import tool
import time
import logging

from ..agent.initialization import InitializationManager
from ..agent.configuration import ConfigurationManager
from ..foundry.nlp_schema import NLPProblem

logger = logging.getLogger(__name__)


# Global managers (could be injected in a more sophisticated setup)
_init_manager = InitializationManager()
_config_manager = ConfigurationManager()


@tool
def run_optimization(
    problem_id: str,
    optimizer: str = "auto",
    priority: str = "balanced",
    max_iterations: int = 100,
    run_id: Optional[int] = None,
    config: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run optimization with Paola intelligence.

    The Paola Principle: Agent specifies INTENT, Paola handles complexity.

    This tool automatically:
    - Selects appropriate algorithm (if optimizer="auto")
    - Computes initial point based on domain and history
    - Configures algorithm based on priority

    Args:
        problem_id: Problem ID (created via create_nlp_problem)
        optimizer: Algorithm selection:
            - "auto": Paola selects based on problem (default, recommended)
            - "scipy": Use SciPy optimizer (Paola selects specific method)
            - "SLSQP": Specific SciPy method
            - "L-BFGS-B": Specific SciPy method
            - "trust-constr": Specific SciPy method
        priority: Optimization priority:
            - "balanced": Middle ground (default)
            - "robustness": Conservative settings, reliable convergence
            - "speed": Fast results, relaxed tolerances
            - "accuracy": Tight tolerances, high precision
        max_iterations: Maximum iterations (default: 100)
        run_id: Optional run ID from start_optimization_run to record results
        config: Optional JSON string with expert config override (from config_* tools)

    Returns:
        Dict with:
            - success: bool
            - message: str
            - final_design: List[float]
            - final_objective: float
            - algorithm_used: str
            - n_iterations: int
            - n_function_evals: int
            - convergence_info: Dict

    Example:
        # Auto-select algorithm with robustness priority
        result = run_optimization(
            problem_id="wing_design",
            optimizer="auto",
            priority="robustness"
        )

        # Explicit algorithm with speed priority
        result = run_optimization(
            problem_id="wing_design",
            optimizer="SLSQP",
            priority="speed",
            max_iterations=50
        )
    """
    from scipy.optimize import minimize
    from .evaluator_tools import _get_problem
    import json

    try:
        # Get problem
        problem = _get_problem(problem_id)

        # Get NLPProblem if available (for domain_hint and bounds)
        nlp_problem = None
        if hasattr(problem, "problem"):
            nlp_problem = problem.problem

        # Step 1: Select algorithm
        if optimizer == "auto" or optimizer == "scipy":
            if nlp_problem:
                algorithm = _config_manager.select_algorithm(
                    nlp_problem, priority=priority
                )
            else:
                # Default to SLSQP for constrained, L-BFGS-B for unconstrained
                algorithm = "SLSQP"
            logger.info(f"Auto-selected algorithm: {algorithm}")
        else:
            algorithm = optimizer

        # Step 2: Get configuration
        config_dict = None
        if config:
            # Expert override
            try:
                config_dict = json.loads(config)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "message": f"Invalid config JSON: {e}"
                }
        elif nlp_problem:
            config_dict = _config_manager.configure_algorithm(
                algorithm, nlp_problem, priority=priority, max_iterations=max_iterations
            )
        else:
            config_dict = {"options": {"maxiter": max_iterations, "disp": False}}

        # Step 3: Compute initial point (Paola intelligence)
        if nlp_problem:
            x0 = _init_manager.compute_initial_point(
                nlp_problem, algorithm, run_history=None
            )
            if x0 is None:
                # Bayesian or sampler-handled
                x0 = np.array(nlp_problem.get_bounds_center())
            bounds = nlp_problem.bounds
        else:
            # Fallback: get bounds from problem if available
            if hasattr(problem, "bounds"):
                bounds = problem.bounds
                n_vars = len(bounds)
                x0 = np.array([(b[0] + b[1]) / 2 for b in bounds])
            else:
                return {
                    "success": False,
                    "message": "Problem must have bounds defined"
                }

        n_vars = len(bounds)
        logger.info(f"Initial point: {x0[:3]}... (first 3 of {n_vars})")

        # Step 4: Prepare for optimization
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

        # Set up scipy bounds
        scipy_bounds = [(b[0], b[1]) for b in bounds]

        # Get constraints if available
        scipy_constraints = None
        if hasattr(problem, "get_scipy_constraints"):
            scipy_constraints = problem.get_scipy_constraints()

        # Determine if gradient-based
        gradient_methods = {"SLSQP", "L-BFGS-B", "BFGS", "CG", "Newton-CG", "TNC", "trust-constr"}
        is_gradient_method = algorithm.upper() in [m.upper() for m in gradient_methods]
        use_gradient = is_gradient_method and hasattr(problem, "gradient")

        # Get options from config
        options = config_dict.get("options", {"maxiter": max_iterations, "disp": False})

        # Step 5: Run optimization
        start_time = time.time()

        if use_gradient:
            result = minimize(
                fun=objective_with_history,
                x0=x0,
                method=algorithm,
                jac=gradient_with_count,
                bounds=scipy_bounds,
                constraints=scipy_constraints,
                options=options,
            )
        else:
            result = minimize(
                fun=objective_with_history,
                x0=x0,
                method=algorithm,
                bounds=scipy_bounds,
                constraints=scipy_constraints,
                options=options,
            )

        elapsed = time.time() - start_time

        # Step 6: Analyze results
        objectives = [h["objective"] for h in history]
        if len(objectives) >= 2:
            convergence_info = {
                "initial_objective": objectives[0],
                "final_objective": objectives[-1],
                "total_improvement": objectives[0] - objectives[-1],
                "improvement_rate": (objectives[0] - objectives[-1]) / len(objectives),
                "converged": result.success,
            }
        else:
            convergence_info = {"converged": result.success}

        # Step 7: Record to run if run_id provided
        if run_id is not None:
            try:
                from .run_tools import _FOUNDRY
                if _FOUNDRY is not None:
                    run = _FOUNDRY.get_run(run_id)
                    if run:
                        for h in history:
                            run.record_iteration(
                                design=np.array(h["design"]),
                                objective=h["objective"]
                            )
                        run.finalize(result, metadata={"convergence_info": convergence_info})
            except Exception as e:
                logger.warning(f"Failed to record run: {e}")

        return {
            "success": result.success,
            "message": result.message if hasattr(result, "message") else "completed",
            "final_design": result.x.tolist(),
            "final_objective": float(result.fun),
            "algorithm_used": algorithm,
            "priority_used": priority,
            "n_iterations": int(result.nit) if hasattr(result, "nit") else len(history),
            "n_function_evals": n_func_evals[0],
            "n_gradient_evals": n_grad_evals[0],
            "elapsed_time": elapsed,
            "optimization_history": history[-20:] if len(history) > 20 else history,
            "convergence_info": convergence_info,
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error running optimization: {str(e)}",
            "traceback": traceback.format_exc(),
        }


@tool
def get_optimization_strategy(
    problem_id: str,
    priority: str = "balanced"
) -> Dict[str, Any]:
    """
    Preview what optimization strategy Paola would use.

    Use this tool to understand what algorithm and configuration Paola
    would select for a problem before running optimization.

    Args:
        problem_id: Problem ID (created via create_nlp_problem)
        priority: Optimization priority ("balanced", "robustness", "speed", "accuracy")

    Returns:
        Dict with:
            - algorithm: str - selected algorithm
            - algorithm_info: Dict - information about the algorithm
            - initialization_strategy: Dict - how initial point will be computed
            - configuration: Dict - algorithm configuration
            - priority_description: str - what the priority means

    Example:
        strategy = get_optimization_strategy(
            problem_id="wing_design",
            priority="robustness"
        )
        # Returns what Paola would do, without running optimization
    """
    from .evaluator_tools import _get_problem

    try:
        problem = _get_problem(problem_id)

        nlp_problem = None
        if hasattr(problem, "problem"):
            nlp_problem = problem.problem

        if nlp_problem:
            # Select algorithm
            algorithm = _config_manager.select_algorithm(nlp_problem, priority=priority)

            # Get configuration
            config = _config_manager.configure_algorithm(
                algorithm, nlp_problem, priority=priority
            )

            # Get initialization strategy
            init_strategy = _init_manager.get_initialization_strategy(
                algorithm, nlp_problem.domain_hint
            )

            # Get algorithm info
            algo_info = _config_manager.get_algorithm_info(algorithm)

            return {
                "success": True,
                "problem_id": problem_id,
                "dimension": nlp_problem.dimension,
                "is_constrained": nlp_problem.is_constrained,
                "domain_hint": nlp_problem.domain_hint,
                "algorithm": algorithm,
                "algorithm_info": algo_info,
                "initialization_strategy": init_strategy,
                "configuration": config,
                "priority": priority,
                "priority_description": _config_manager.get_priority_description(priority),
            }
        else:
            return {
                "success": False,
                "message": "Problem must be an NLPProblem for strategy preview"
            }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting strategy: {str(e)}"
        }


@tool
def list_available_algorithms() -> Dict[str, Any]:
    """
    List available optimization algorithms.

    Returns information about all algorithms Paola can use,
    their capabilities, and what problems they're best for.

    Returns:
        Dict with:
            - algorithms: List[Dict] - list of algorithm information
            - recommendation: str - general recommendation
    """
    algorithms = [
        {
            "name": "SLSQP",
            "type": "gradient-based",
            "supports_bounds": True,
            "supports_constraints": True,
            "best_for": "Small to medium constrained problems",
            "default_for": ["constrained", "balanced"]
        },
        {
            "name": "L-BFGS-B",
            "type": "gradient-based",
            "supports_bounds": True,
            "supports_constraints": False,
            "best_for": "Large unconstrained or bound-constrained problems",
            "default_for": ["unconstrained", "speed"]
        },
        {
            "name": "trust-constr",
            "type": "gradient-based",
            "supports_bounds": True,
            "supports_constraints": True,
            "best_for": "High-accuracy constrained optimization",
            "default_for": ["accuracy"]
        },
        {
            "name": "COBYLA",
            "type": "derivative-free",
            "supports_bounds": False,
            "supports_constraints": True,
            "best_for": "Noisy or non-differentiable constrained problems",
            "default_for": ["noisy"]
        },
        {
            "name": "Nelder-Mead",
            "type": "derivative-free",
            "supports_bounds": False,
            "supports_constraints": False,
            "best_for": "Simple unconstrained problems without gradients",
            "default_for": []
        }
    ]

    return {
        "success": True,
        "algorithms": algorithms,
        "recommendation": (
            "Use optimizer='auto' to let Paola select the best algorithm. "
            "Paola considers problem dimension, constraints, and your priority "
            "to make the optimal choice."
        ),
        "priorities": {
            "balanced": "Middle ground between speed and accuracy (default)",
            "robustness": "Conservative settings that prioritize convergence reliability",
            "speed": "Relaxed tolerances and early stopping for faster results",
            "accuracy": "Tight tolerances for high-precision solutions"
        }
    }
