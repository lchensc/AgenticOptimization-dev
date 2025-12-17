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

from paola.tools.cache_tools import cache_get, cache_store
from paola.backends.analytical import get_analytical_function


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


@tool
def create_benchmark_problem(
    problem_id: str,
    function_name: str,
    dimension: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Create and register a built-in analytical benchmark problem.

    IMPORTANT: Only use this if no suitable evaluator is registered in Foundry.
    Always check foundry_list_evaluators first and prefer create_nlp_problem
    with registered evaluators.

    Available benchmark functions:
    - "rosenbrock": Classic Rosenbrock function with narrow curved valley
      Global minimum: f(1,1,...,1) = 0
      Recommended bounds: [-5, 10] for each dimension
    - "sphere": Simple quadratic bowl, easy to optimize
      Global minimum: f(0,0,...,0) = 0
      Recommended bounds: [-5, 5] for each dimension
    - "constrained_rosenbrock": 2D Rosenbrock with circular constraint
      Constrained minimum: f(0.786, 0.618) ≈ 0.046
      Fixed at 2D, dimension parameter ignored

    Args:
        problem_id: Unique identifier for this problem (e.g., "rosenbrock_10d")
        function_name: Benchmark function name (case-insensitive)
        dimension: Problem dimensionality (default: 2). Ignored for constrained_rosenbrock.

    Returns:
        Dict with:
            - success: bool
            - problem_id: str - identifier to use in other tools
            - function_name: str
            - dimension: int
            - global_optimum: Dict with x_opt and f_opt
            - message: str

    Example:
        # Agent receives: "Solve 10D Rosenbrock problem"
        # Agent calls:
        result = create_benchmark_problem(
            problem_id="rosenbrock_10d",
            function_name="rosenbrock",
            dimension=10
        )
        # Returns: {"success": True, "problem_id": "rosenbrock_10d", ...}
        # Now agent can use "rosenbrock_10d" with run_scipy_optimization
    """
    try:
        # Check if problem_id already exists
        if problem_id in _PROBLEM_REGISTRY:
            return {
                "success": False,
                "problem_id": problem_id,
                "message": f"Problem '{problem_id}' already registered. Use a different ID or clear registry first.",
            }

        # Create analytical function
        problem = get_analytical_function(name=function_name, dimension=dimension)

        # Register it
        register_problem(problem_id, problem)

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
        # Invalid function name
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
    """Clear all problems from registry."""
    _PROBLEM_REGISTRY.clear()


def get_problem_by_id(problem_id: str) -> Optional[Any]:
    """Get problem instance by ID."""
    return _PROBLEM_REGISTRY.get(problem_id)


@tool
def create_nlp_problem(
    name: str,
    objective_evaluator_id: str,
    bounds: Any,  # Accept both explicit list OR compact BoundsSpec dict
    objective_sense: str = "minimize",
    inequality_constraints: Optional[List[Dict[str, Any]]] = None,
    equality_constraints: Optional[List[Dict[str, Any]]] = None,
    domain_hint: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create Nonlinear Programming (NLP) problem from registered Foundry evaluators.

    IMPORTANT - BOUNDS SPECIFICATION:
    For problems with more than 5 variables, use COMPACT bounds format (not explicit arrays):

    Compact format (REQUIRED for large problems):
        {"type": "uniform", "lower": -5, "upper": 10, "dimension": 50}

    Explicit format (only for small problems, <=5 variables):
        [[-5, 10], [-5, 10]]

    DO NOT use Python syntax like "[[-5, 10] for _ in range(50)]" - this is invalid JSON!

    Args:
        name: Human-readable problem name (e.g., "Wing Design Optimization")
        objective_evaluator_id: Evaluator ID for objective function f(x)
        bounds: Variable bounds - use ONE of these formats:
            - Compact (RECOMMENDED): {"type": "uniform", "lower": -5, "upper": 10, "dimension": 50}
            - Grouped: {"type": "grouped", "groups": {"x": {"lower": 0, "upper": 1, "count": 30}, "y": {"lower": -1, "upper": 1, "count": 20}}}
            - Explicit (small problems only): [[-5, 10], [-5, 10], [-5, 10]]
        objective_sense: "minimize" or "maximize" (default: "minimize")
        inequality_constraints: List of constraint specs:
            [{"name": "c1", "evaluator_id": "eval_id", "type": "<=", "value": 100}]
        equality_constraints: List of equality constraint specs:
            [{"name": "eq1", "evaluator_id": "eval_id", "value": 0.0}]
        domain_hint: Hint for initialization strategy:
            - "shape_optimization": Initialize at zero (baseline geometry)
            - "general": Initialize at center of bounds (default)
        description: Human-readable problem description

    Returns:
        Dict with success, problem_id, dimension, etc.

    Examples:
        # 50-dimensional problem - use compact bounds
        create_nlp_problem(
            problem_id="high_dim_problem",
            objective_evaluator_id="rosenbrock_eval",
            bounds={"type": "uniform", "lower": -5, "upper": 10, "dimension": 50}
        )

        # Small 2D problem - explicit bounds OK
        create_nlp_problem(
            problem_id="rosenbrock_2d",
            objective_evaluator_id="rosenbrock_eval",
            bounds=[[-5, 10], [-5, 10]]
        )

        # Grouped bounds for mixed variable types
        create_nlp_problem(
            problem_id="wing_design",
            objective_evaluator_id="drag_eval",
            bounds={"type": "grouped", "groups": {
                "shape": {"lower": -0.05, "upper": 0.05, "count": 40},
                "twist": {"lower": -15, "upper": 15, "count": 10}
            }},
            domain_hint="shape_optimization"
        )
    """
    try:
        from paola.foundry import (
            OptimizationFoundry,
            FileStorage,
            NLPProblem,
            InequalityConstraint,
            EqualityConstraint,
            NLPEvaluator,
            SolverSelector,
            Problem
        )
        from datetime import datetime

        # Unified storage: all data in .paola_foundry
        storage = FileStorage()
        foundry = OptimizationFoundry(storage=storage)

        # Get next numeric problem_id (v0.4.3)
        problem_id = storage.get_next_problem_id()

        # Verify objective evaluator exists
        try:
            foundry.get_evaluator_config(objective_evaluator_id)
        except KeyError:
            available = foundry.list_evaluators()
            return {
                "success": False,
                "message": (
                    f"Objective evaluator '{objective_evaluator_id}' not found in Foundry.\n"
                    f"Available evaluators: {[e['evaluator_id'] for e in available]}\n"
                    f"Use foundry_list_evaluators to see all registered evaluators."
                )
            }

        # Parse inequality constraints
        ineq_constraints_objs = []
        if inequality_constraints:
            for cons_dict in inequality_constraints:
                # Verify constraint evaluator exists
                cons_eval_id = cons_dict["evaluator_id"]
                try:
                    foundry.get_evaluator_config(cons_eval_id)
                except KeyError:
                    return {
                        "success": False,
                        "message": f"Constraint evaluator '{cons_eval_id}' not found in Foundry."
                    }

                ineq_constraints_objs.append(InequalityConstraint(
                    name=cons_dict["name"],
                    evaluator_id=cons_dict["evaluator_id"],
                    constraint_type=cons_dict["type"],
                    value=float(cons_dict["value"])
                ))

        # Parse equality constraints
        eq_constraints_objs = []
        if equality_constraints:
            for cons_dict in equality_constraints:
                # Verify constraint evaluator exists
                cons_eval_id = cons_dict["evaluator_id"]
                try:
                    foundry.get_evaluator_config(cons_eval_id)
                except KeyError:
                    return {
                        "success": False,
                        "message": f"Constraint evaluator '{cons_eval_id}' not found in Foundry."
                    }

                eq_constraints_objs.append(EqualityConstraint(
                    name=cons_dict["name"],
                    evaluator_id=cons_dict["evaluator_id"],
                    value=float(cons_dict["value"]),
                    tolerance=cons_dict.get("tolerance", 1e-6)
                ))

        # Parse bounds - support both compact format (dict) and explicit list
        from paola.foundry.bounds_spec import parse_bounds_input

        if isinstance(bounds, dict):
            # Compact format: {"type": "uniform", "lower": -5, "upper": 10, "dimension": 50}
            bounds_spec = parse_bounds_input(bounds)
            explicit_bounds = bounds_spec.expand()
            dimension = bounds_spec.get_dimension()
        elif isinstance(bounds, list):
            # Explicit format: [[-5, 10], [-5, 10], ...]
            explicit_bounds = bounds
            dimension = len(bounds)
        else:
            return {
                "success": False,
                "message": (
                    f"Invalid bounds format. Expected dict (compact) or list (explicit).\n"
                    f"Compact format: {{'type': 'uniform', 'lower': -5, 'upper': 10, 'dimension': 50}}\n"
                    f"Explicit format: [[-5, 10], [-5, 10], ...]"
                )
            }

        # Create NLPProblem specification (v0.4.3 - uses numeric IDs)
        # Note: initial_point is NOT specified - Paola computes it automatically
        # based on domain_hint, algorithm, and run history (The Paola Principle)
        nlp_problem = NLPProblem(
            problem_id=problem_id,  # Numeric ID
            name=name,  # Human-readable name
            n_variables=dimension,
            n_constraints=0,  # Will be computed in __post_init__
            objective_evaluator_id=objective_evaluator_id,
            objective_sense=objective_sense,
            bounds=explicit_bounds,
            inequality_constraints=ineq_constraints_objs,
            equality_constraints=eq_constraints_objs,
            description=description or name,
            domain_hint=domain_hint,
            bounds_spec=bounds if isinstance(bounds, dict) else None
        )

        # Create NLPEvaluator (composite evaluator)
        nlp_evaluator = NLPEvaluator.from_problem(nlp_problem, foundry)

        # Register in problem registry (for runtime use) - use numeric ID
        register_problem(problem_id, nlp_evaluator)

        # Store problem in storage with index
        storage.save_problem(nlp_problem)

        # Recommend solvers
        has_constraints = nlp_problem.is_constrained
        recommended_solvers = SolverSelector.recommend_solver(
            problem_type="NLP",
            gradient_available=True,  # FoundryEvaluator always supports gradients
            has_constraints=has_constraints
        )

        # Build message with domain hint if provided
        hint_msg = f"  Domain hint: {domain_hint}\n" if domain_hint else ""

        return {
            "success": True,
            "problem_id": problem_id,  # Numeric ID
            "name": name,  # Human-readable name
            "problem_type": "NLP",
            "dimension": dimension,
            "num_inequality_constraints": nlp_problem.num_inequality_constraints,
            "num_equality_constraints": nlp_problem.num_equality_constraints,
            "evaluators_used": nlp_problem.get_all_evaluator_ids(),
            "recommended_solvers": recommended_solvers,
            "domain_hint": domain_hint,
            "message": (
                f"Created NLP problem #{problem_id} '{name}':\n"
                f"  Objective: {objective_sense} {objective_evaluator_id}\n"
                f"  Dimension: {dimension}\n"
                f"{hint_msg}"
                f"  Inequality constraints: {nlp_problem.num_inequality_constraints}\n"
                f"  Equality constraints: {nlp_problem.num_equality_constraints}\n"
                f"  Recommended solvers: {', '.join(recommended_solvers[:2])}\n"
                f"  Note: Initial point will be computed by Paola based on domain and algorithm"
            )
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error creating NLP problem: {str(e)}\n{traceback.format_exc()}",
        }


@tool
def derive_problem(
    parent_problem_id: int,
    derivation_type: str,
    modifications: str,  # JSON string
    new_name: Optional[str] = None,
    reason: Optional[str] = None,
    source_graph_id: Optional[int] = None,
    source_node_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Derive a new problem from an existing one with modifications.

    Use this to adapt search space based on optimization results.
    The derived problem maintains lineage to the parent.

    Args:
        parent_problem_id: Numeric ID of the problem to derive from
        derivation_type: Type of derivation:
            - "narrow_bounds" - Shrink bounds around a region
            - "widen_bounds" - Expand search space
        modifications: JSON string with derivation-specific parameters:
            narrow_bounds: {"center": [...], "width_factor": 0.3}
            widen_bounds: {"width_factor": 1.5}
        new_name: Optional name for derived problem (auto-generated if not provided)
        reason: Why this derivation was needed
        source_graph_id: Graph that motivated this derivation
        source_node_id: Node that motivated this derivation

    Returns:
        Dict with:
            - success: bool
            - problem_id: int (new derived problem ID)
            - parent_problem_id: int
            - derivation_type: str
            - n_variables: int
            - message: str

    Example (narrow bounds after global search):
        # After TPE finds promising region at x=[1.2, 3.4, 5.6]
        derive_problem(
            parent_problem_id=1,
            derivation_type="narrow_bounds",
            modifications='{"center": [1.2, 3.4, 5.6], "width_factor": 0.3}',
            reason="Focus on region found by TPE in graph #42",
            source_graph_id=42,
            source_node_id="n1"
        )
    """
    try:
        import json as json_module
        from paola.foundry import FileStorage
        from paola.foundry.nlp_schema import NLPProblem
        from paola.foundry.nlp_evaluator import NLPEvaluator
        from paola.foundry import OptimizationFoundry

        # Parse modifications
        try:
            mods = json_module.loads(modifications)
        except json_module.JSONDecodeError as e:
            return {
                "success": False,
                "message": f"Invalid JSON in modifications: {e}"
            }

        # Get storage and load parent problem
        storage = FileStorage()  # unified .paola_foundry
        parent = storage.load_problem(parent_problem_id)

        if parent is None:
            return {
                "success": False,
                "message": f"Parent problem #{parent_problem_id} not found in storage."
            }

        # Check that it's an NLPProblem
        if not isinstance(parent, NLPProblem):
            return {
                "success": False,
                "message": f"Parent problem #{parent_problem_id} is not an NLPProblem. "
                           f"Derivation only supported for NLP problems currently."
            }

        # Get next numeric ID for derived problem (v0.4.3)
        new_problem_id = storage.get_next_problem_id()

        # Generate name if not provided
        derived_name = new_name or f"{parent.name} ({derivation_type})"

        # Apply derivation
        if derivation_type == "narrow_bounds":
            center = mods.get("center")
            if center is None:
                return {
                    "success": False,
                    "message": "narrow_bounds derivation requires 'center' in modifications"
                }
            width_factor = mods.get("width_factor", 0.3)

            derived = parent.derive_narrow_bounds(
                new_problem_id=new_problem_id,
                new_name=derived_name,
                center=center,
                width_factor=width_factor,
                reason=reason,
                source_graph_id=source_graph_id,
                source_node_id=source_node_id,
            )

        elif derivation_type == "widen_bounds":
            width_factor = mods.get("width_factor", 1.5)

            derived = parent.derive_widen_bounds(
                new_problem_id=new_problem_id,
                new_name=derived_name,
                width_factor=width_factor,
                reason=reason,
            )

        else:
            return {
                "success": False,
                "message": f"Unknown derivation_type: {derivation_type}. "
                           f"Supported: narrow_bounds, widen_bounds"
            }

        # Save derived problem to storage
        storage.save_problem(derived)

        # Create NLPEvaluator and register for runtime use
        foundry = OptimizationFoundry(storage=FileStorage())  # unified .paola_foundry
        nlp_evaluator = NLPEvaluator.from_problem(derived, foundry)
        register_problem(derived.problem_id, nlp_evaluator)

        return {
            "success": True,
            "problem_id": derived.problem_id,
            "name": derived.name,
            "parent_problem_id": derived.parent_problem_id,
            "derivation_type": derived.derivation_type,
            "n_variables": derived.n_variables,
            "version": derived.version,
            "message": (
                f"Derived problem #{derived.problem_id} '{derived.name}' from #{parent_problem_id}:\n"
                f"  Derivation: {derivation_type}\n"
                f"  Variables: {derived.n_variables}\n"
                f"  Version: {derived.version}\n"
                f"  Reason: {reason or 'Not specified'}"
            )
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error deriving problem: {str(e)}\n{traceback.format_exc()}",
        }


@tool
def list_problems(
    problem_type: Optional[str] = None,
    show_derived: bool = True,
) -> Dict[str, Any]:
    """
    List all registered optimization problems.

    Shows problems stored in Paola with their metadata and lineage info.

    Args:
        problem_type: Filter by type ("NLP", "LP", etc.). None for all.
        show_derived: Include derived problems (default: True)

    Returns:
        Dict with:
            - success: bool
            - problems: List of problem summaries
            - count: int

    Example:
        # List all NLP problems
        result = list_problems(problem_type="NLP")

        # List only root problems (no derived)
        result = list_problems(show_derived=False)
    """
    try:
        from paola.foundry import FileStorage

        storage = FileStorage()  # unified .paola_foundry
        problems = storage.list_problems(
            problem_type=problem_type,
            show_derived=show_derived
        )

        return {
            "success": True,
            "problems": problems,
            "count": len(problems),
            "message": f"Found {len(problems)} problem(s)"
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error listing problems: {str(e)}\n{traceback.format_exc()}",
        }


@tool
def get_problem_lineage(problem_id: int) -> Dict[str, Any]:
    """
    Get the derivation lineage of a problem.

    Shows the chain of derivations from root problem to this one,
    including which graphs used each version.

    Args:
        problem_id: Numeric problem ID to trace lineage for

    Returns:
        Dict with:
            - success: bool
            - problem_id: int
            - lineage: List[Dict] - Chain from root to this problem
            - children: List[int] - Direct children of this problem

    Example:
        # Trace how problem #3 was derived
        result = get_problem_lineage(3)
        for p in result["lineage"]:
            print(f"#{p['problem_id']} {p['name']} ({p.get('derivation_type', 'root')})")
    """
    try:
        from paola.foundry import FileStorage

        storage = FileStorage()  # unified .paola_foundry

        # Get lineage
        lineage = storage.get_problem_lineage(problem_id)
        if not lineage:
            return {
                "success": False,
                "message": f"Problem #{problem_id} not found in storage."
            }

        # Get children
        children = storage.get_problem_children(problem_id)

        # Build chain string
        chain_parts = [f"#{p['problem_id']}" for p in lineage]
        chain_str = ' -> '.join(chain_parts)

        return {
            "success": True,
            "problem_id": problem_id,
            "lineage": lineage,
            "children": children,
            "message": (
                f"Lineage for problem #{problem_id}:\n"
                f"  Chain: {chain_str}\n"
                f"  Children: {children if children else 'None'}"
            )
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error getting lineage: {str(e)}\n{traceback.format_exc()}",
        }
