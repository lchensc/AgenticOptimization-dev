"""
Expert configuration tools (escape hatch).

The Paola Principle: "Optimization complexity is Paola intelligence, not user burden."

While Paola handles configuration automatically via priority-based settings,
some expert users may need direct control over optimizer options. These tools
provide an escape hatch for advanced use cases.

Usage:
    # Normal use (Paola handles configuration)
    run_optimization(problem_id="wing", optimizer="auto", priority="robustness")

    # Expert use (direct configuration)
    config = config_scipy(method="SLSQP", maxiter=500, ftol=1e-9)
    run_optimization(problem_id="wing", config=config)

These tools output JSON configuration strings that can be passed directly
to run_optimization's config parameter.
"""

from typing import Optional, Dict, Any
from langchain_core.tools import tool
import json


@tool
def config_scipy(
    method: str = "SLSQP",
    maxiter: int = 100,
    ftol: float = 1e-6,
    gtol: Optional[float] = None,
    eps: Optional[float] = None,
    disp: bool = False,
    options: Optional[str] = None,
) -> str:
    """
    Configure SciPy optimizer (expert use only).

    This tool creates a configuration for run_optimization. Use only if you need
    direct control over optimizer options. For most cases, use priority-based
    configuration via run_optimization directly.

    Args:
        method: SciPy method name (default: "SLSQP")
            - "SLSQP": Sequential Least Squares Programming (constrained)
            - "L-BFGS-B": Limited-memory BFGS (bound-constrained)
            - "trust-constr": Trust region (constrained, high accuracy)
            - "COBYLA": Derivative-free (constrained)
            - "Nelder-Mead": Derivative-free simplex
        maxiter: Maximum iterations (default: 100)
        ftol: Function tolerance (default: 1e-6)
        gtol: Gradient tolerance (optional, method-dependent)
        eps: Finite difference step size (optional)
        disp: Display convergence messages (default: False)
        options: Additional options as JSON string (merged with above)

    Returns:
        JSON configuration string for run_optimization

    Example:
        # Create custom config
        config = config_scipy(
            method="SLSQP",
            maxiter=500,
            ftol=1e-9
        )

        # Use with run_optimization
        run_optimization(
            problem_id="wing_design",
            config=config
        )
    """
    config_dict = {
        "method": method,
        "options": {
            "maxiter": maxiter,
            "ftol": ftol,
            "disp": disp
        }
    }

    if gtol is not None:
        config_dict["options"]["gtol"] = gtol

    if eps is not None:
        config_dict["options"]["eps"] = eps

    # Merge additional options if provided
    if options:
        try:
            extra_opts = json.loads(options)
            config_dict["options"].update(extra_opts)
        except json.JSONDecodeError:
            pass  # Ignore invalid JSON

    return json.dumps(config_dict)


@tool
def config_ipopt(
    max_iter: int = 1000,
    tol: float = 1e-6,
    acceptable_tol: float = 1e-5,
    mu_strategy: str = "adaptive",
    mu_init: float = 0.1,
    print_level: int = 0,
    linear_solver: str = "mumps",
    options: Optional[str] = None,
) -> str:
    """
    Configure IPOPT optimizer (expert use only).

    IPOPT (Interior Point Optimizer) is a large-scale nonlinear optimizer.
    This tool creates an IPOPT configuration for future IPOPT integration.

    Note: IPOPT backend is not yet implemented. This tool is provided for
    future-proofing and documentation of common IPOPT options.

    Args:
        max_iter: Maximum iterations (default: 1000)
        tol: Overall tolerance (default: 1e-6)
        acceptable_tol: Acceptable tolerance for termination (default: 1e-5)
        mu_strategy: Barrier parameter strategy (default: "adaptive")
            - "adaptive": Update adaptively
            - "monotone": Decrease monotonically
        mu_init: Initial barrier parameter (default: 0.1)
        print_level: Output verbosity 0-12 (default: 0)
        linear_solver: Linear solver to use (default: "mumps")
            - "mumps": MUMPS sparse solver
            - "ma57": HSL MA57 solver
            - "pardiso": Intel PARDISO solver
        options: Additional options as JSON string

    Returns:
        JSON configuration string for run_optimization (IPOPT backend)

    Example:
        # High-accuracy IPOPT config
        config = config_ipopt(
            max_iter=2000,
            tol=1e-9,
            mu_strategy="adaptive"
        )
    """
    config_dict = {
        "solver": "ipopt",
        "options": {
            "max_iter": max_iter,
            "tol": tol,
            "acceptable_tol": acceptable_tol,
            "mu_strategy": mu_strategy,
            "mu_init": mu_init,
            "print_level": print_level,
            "linear_solver": linear_solver
        }
    }

    # Merge additional options if provided
    if options:
        try:
            extra_opts = json.loads(options)
            config_dict["options"].update(extra_opts)
        except json.JSONDecodeError:
            pass

    return json.dumps(config_dict)


@tool
def config_nlopt(
    algorithm: str = "LD_SLSQP",
    maxeval: int = 1000,
    ftol_rel: float = 1e-6,
    ftol_abs: float = 1e-8,
    xtol_rel: float = 1e-6,
    xtol_abs: float = 1e-8,
    options: Optional[str] = None,
) -> str:
    """
    Configure NLopt optimizer (expert use only).

    NLopt provides a common interface to many optimization algorithms.
    This tool creates an NLopt configuration for future NLopt integration.

    Note: NLopt backend is not yet implemented. This tool is provided for
    future-proofing and documentation of common NLopt options.

    Args:
        algorithm: NLopt algorithm name (default: "LD_SLSQP")
            Gradient-based (LD_*):
            - "LD_SLSQP": Sequential Least Squares
            - "LD_LBFGS": Limited-memory BFGS
            - "LD_MMA": Method of Moving Asymptotes
            - "LD_CCSAQ": Conservative Convex Separable Approximation

            Derivative-free (LN_*):
            - "LN_COBYLA": Constrained Optimization BY Linear Approx
            - "LN_BOBYQA": Bound Optimization BY Quadratic Approx
            - "LN_NELDERMEAD": Nelder-Mead simplex

            Global (G*):
            - "GN_DIRECT": DIviding RECTangles
            - "GN_CRS2_LM": Controlled Random Search
        maxeval: Maximum function evaluations (default: 1000)
        ftol_rel: Relative function tolerance (default: 1e-6)
        ftol_abs: Absolute function tolerance (default: 1e-8)
        xtol_rel: Relative parameter tolerance (default: 1e-6)
        xtol_abs: Absolute parameter tolerance (default: 1e-8)
        options: Additional options as JSON string

    Returns:
        JSON configuration string for run_optimization (NLopt backend)

    Example:
        # Derivative-free config for noisy problems
        config = config_nlopt(
            algorithm="LN_COBYLA",
            maxeval=2000,
            ftol_rel=1e-4
        )
    """
    config_dict = {
        "solver": "nlopt",
        "algorithm": algorithm,
        "options": {
            "maxeval": maxeval,
            "ftol_rel": ftol_rel,
            "ftol_abs": ftol_abs,
            "xtol_rel": xtol_rel,
            "xtol_abs": xtol_abs
        }
    }

    # Merge additional options if provided
    if options:
        try:
            extra_opts = json.loads(options)
            config_dict["options"].update(extra_opts)
        except json.JSONDecodeError:
            pass

    return json.dumps(config_dict)


@tool
def config_optuna(
    n_trials: int = 100,
    sampler: str = "TPE",
    pruner: str = "median",
    seed: Optional[int] = None,
    options: Optional[str] = None,
) -> str:
    """
    Configure Optuna optimizer (expert use only).

    Optuna is a hyperparameter optimization framework using Bayesian optimization.
    This tool creates an Optuna configuration for future Optuna integration.

    Note: Optuna backend is not yet implemented. This tool is provided for
    future-proofing and documentation of common Optuna options.

    Args:
        n_trials: Number of trials to run (default: 100)
        sampler: Sampling algorithm (default: "TPE")
            - "TPE": Tree-structured Parzen Estimator (default)
            - "CmaEs": CMA-ES algorithm
            - "Random": Random sampling
            - "Grid": Grid search (requires search space discretization)
        pruner: Trial pruning strategy (default: "median")
            - "median": Prune if below median
            - "percentile": Prune if below percentile
            - "none": No pruning
        seed: Random seed for reproducibility (optional)
        options: Additional options as JSON string

    Returns:
        JSON configuration string for run_optimization (Optuna backend)

    Example:
        # Bayesian optimization config
        config = config_optuna(
            n_trials=200,
            sampler="TPE",
            seed=42
        )
    """
    config_dict = {
        "solver": "optuna",
        "options": {
            "n_trials": n_trials,
            "sampler": sampler,
            "pruner": pruner
        }
    }

    if seed is not None:
        config_dict["options"]["seed"] = seed

    # Merge additional options if provided
    if options:
        try:
            extra_opts = json.loads(options)
            config_dict["options"].update(extra_opts)
        except json.JSONDecodeError:
            pass

    return json.dumps(config_dict)


@tool
def explain_config_option(
    solver: str,
    option_name: str
) -> Dict[str, Any]:
    """
    Explain what a configuration option does.

    Use this tool to understand what a specific optimizer option controls
    and what values are appropriate.

    Args:
        solver: Solver name ("scipy", "ipopt", "nlopt", "optuna")
        option_name: Name of the option to explain

    Returns:
        Dict with:
            - option: str - option name
            - description: str - what it does
            - typical_values: Dict - typical value ranges
            - recommendation: str - when to adjust this option

    Example:
        explain_config_option("scipy", "ftol")
    """
    # Knowledge base of optimizer options
    options_db = {
        "scipy": {
            "ftol": {
                "description": "Function tolerance - optimization stops when function change is below this",
                "typical_values": {"loose": 1e-4, "standard": 1e-6, "tight": 1e-9},
                "recommendation": "Tighten (1e-9) for high-accuracy; loosen (1e-4) for speed"
            },
            "gtol": {
                "description": "Gradient tolerance - optimization stops when gradient norm is below this",
                "typical_values": {"loose": 1e-4, "standard": 1e-5, "tight": 1e-8},
                "recommendation": "Tighten for smooth problems; loosen for noisy gradients"
            },
            "maxiter": {
                "description": "Maximum number of iterations allowed",
                "typical_values": {"small": 50, "standard": 100, "large": 500},
                "recommendation": "Increase for large problems; decrease for quick exploration"
            },
            "eps": {
                "description": "Step size for finite difference gradient approximation",
                "typical_values": {"standard": 1e-8, "noisy": 1e-5},
                "recommendation": "Increase if gradients seem noisy"
            }
        },
        "ipopt": {
            "tol": {
                "description": "Overall convergence tolerance",
                "typical_values": {"loose": 1e-4, "standard": 1e-6, "tight": 1e-9},
                "recommendation": "Standard 1e-6 works for most problems"
            },
            "mu_strategy": {
                "description": "Barrier parameter update strategy",
                "typical_values": {"options": ["adaptive", "monotone"]},
                "recommendation": "'adaptive' (default) is more robust; 'monotone' may be faster"
            },
            "max_iter": {
                "description": "Maximum iterations",
                "typical_values": {"small": 500, "standard": 1000, "large": 3000},
                "recommendation": "IPOPT often needs more iterations than SciPy"
            }
        }
    }

    solver_opts = options_db.get(solver.lower(), {})
    opt_info = solver_opts.get(option_name.lower())

    if opt_info:
        return {
            "success": True,
            "solver": solver,
            "option": option_name,
            **opt_info
        }
    else:
        return {
            "success": False,
            "message": f"Option '{option_name}' not found for solver '{solver}'",
            "available_options": list(solver_opts.keys()) if solver_opts else f"Solver '{solver}' not in knowledge base"
        }
