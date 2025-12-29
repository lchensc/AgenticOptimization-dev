"""
Intent-based optimization tools - LLM-driven architecture.

The Paola Principle: "Optimization complexity is agent intelligence, not user burden."

This module provides tools where:
- Information tools provide data TO the LLM for reasoning
- Execution tools execute decisions FROM the LLM
- The LLM's trained knowledge (IPOPT docs, optimization theory, etc.) IS the intelligence

v0.3.0: Graph-based architecture
- run_optimization requires graph_id
- Agent explicitly specifies parent_node and edge_type
- Records nodes with polymorphic components per optimizer family

v0.4.5: Pydantic validation for problem_id (handles str/int coercion)

v0.5.0 (v0.2.0 redesign): run_optimization is DEPRECATED
- LLM now writes Python code directly using paola.objective(), scipy, optuna, etc.
- Use paola.objective(problem_id) to get RecordingObjective callable
- See paola.api module for Recording API

Architecture:
    LLM reasons → selects optimizer → constructs config → calls run_optimization
    (Intelligence)                                        (Execution)
"""
import warnings

from typing import Optional, Dict, Any
import numpy as np
from langchain_core.tools import tool
import time
import logging
import json

from ..optimizers import (
    get_backend,
    list_backends,
    get_available_backends,
)
from ..foundry import (
    COMPONENT_REGISTRY,
    EdgeType,
    # Gradient family
    GradientInitialization,
    GradientProgress,
    GradientResult,
    # Bayesian family
    BayesianInitialization,
    BayesianProgress,
    BayesianResult,
    # Evolutionary family (pymoo SOO)
    EvolutionaryInitialization,
    EvolutionaryProgress,
    EvolutionaryResult,
    # Multi-objective family (pymoo MOO)
    MultiObjectiveInitialization,
    MultiObjectiveProgress,
    MultiObjectiveResult,
    # Population family (legacy)
    PopulationInitialization,
    PopulationProgress,
    PopulationResult,
    # CMA-ES family
    CMAESInitialization,
    CMAESProgress,
    CMAESResult,
)
from .schemas import (
    normalize_problem_id,
    ProblemIdType,
    RunOptimizationArgs,
    GetProblemInfoArgs,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Graph-Based Optimization (v0.3.0+)
# =============================================================================

@tool(args_schema=RunOptimizationArgs)
def run_optimization(
    graph_id: int,
    optimizer: str,
    optimizer_config: Optional[str] = None,
    max_iterations: int = 100,
    init_strategy: str = "center",
    parent_node: Optional[str] = None,
    edge_type: Optional[str] = None,
    problem_id: Optional[ProblemIdType] = None,
) -> Dict[str, Any]:
    """
    Run optimization, creating a new node in the graph.

    Args:
        graph_id: Graph ID from start_graph
        optimizer: Backend:method specification. Supported:
            - Gradient: "scipy:SLSQP", "scipy:L-BFGS-B", "ipopt"
            - Bayesian: "optuna:TPE", "optuna:CMA-ES"
            - Evolutionary: "pymoo:GA", "pymoo:DE", "pymoo:PSO", "pymoo:CMA-ES"
            - Multi-objective: "pymoo:NSGA-II", "pymoo:NSGA-III", "pymoo:MOEA/D"
        optimizer_config: JSON string with optimizer options (e.g., '{"n_obj": 2, "pop_size": 100}')
        max_iterations: Maximum iterations (default: 100)
        init_strategy: "center", "random", or "warm_start"
        parent_node: Node ID to continue from (e.g., "n1")
        edge_type: "warm_start", "refine", "branch", "explore"
        problem_id: Override graph's problem (for derived problems)

    Returns:
        success, node_id, best_x, best_objective, n_evaluations, elapsed_time
        For multi-objective: also includes is_multiobjective, n_pareto_solutions, hypervolume

    Example:
        run_optimization(graph_id=1, optimizer="scipy:SLSQP")
        run_optimization(graph_id=1, optimizer="scipy:L-BFGS-B", parent_node="n1", edge_type="warm_start")
        run_optimization(graph_id=1, optimizer="pymoo:NSGA-II", optimizer_config='{"n_obj": 2, "pop_size": 100}')

    .. deprecated:: 0.5.0
        Use paola.objective() with scipy/optuna directly instead.
        LLM now writes Python optimization code directly.
    """
    warnings.warn(
        "run_optimization is deprecated in v0.5.0. "
        "Use paola.objective(problem_id) and call scipy/optuna directly. "
        "See paola.api for the Recording API.",
        DeprecationWarning,
        stacklevel=2,
    )

    from .evaluation import _get_problem
    from .graph import _FOUNDRY

    try:
        # Check foundry
        if _FOUNDRY is None:
            return {
                "success": False,
                "message": "Foundry not initialized. Call set_foundry() first.",
            }

        # Get graph
        graph = _FOUNDRY.get_graph(graph_id)
        if graph is None:
            return {
                "success": False,
                "message": f"Graph {graph_id} not found or already finalized. Use start_graph first.",
            }

        # Determine which problem to use (v0.4.1 - support problem_id override)
        # Normalize problem_id (handles str/int from LLM) - v0.4.5
        if problem_id is not None:
            actual_problem_id = normalize_problem_id(problem_id)
        else:
            actual_problem_id = graph.problem_id

        # Get problem
        problem = _get_problem(actual_problem_id)

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

        # Parse optimizer_config
        config_dict = {}
        if optimizer_config:
            try:
                config_dict = json.loads(optimizer_config)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "message": f"Invalid optimizer_config JSON: {e}",
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

        # Validate parent_node and edge_type
        if init_strategy == "warm_start":
            if parent_node is None:
                return {
                    "success": False,
                    "message": "parent_node is required when init_strategy='warm_start'",
                }
            if edge_type is None:
                edge_type = EdgeType.WARM_START  # Default edge type

        if parent_node is not None and edge_type is None:
            return {
                "success": False,
                "message": "edge_type is required when parent_node is specified",
            }

        # Get x0 based on init_strategy
        if init_strategy == "warm_start" and parent_node:
            # Get parent node's best solution
            parent = graph.get_node(parent_node)
            if parent is None:
                return {
                    "success": False,
                    "message": f"Parent node '{parent_node}' not found in graph",
                }
            if parent.best_x is None:
                return {
                    "success": False,
                    "message": f"Parent node '{parent_node}' has no solution to warm-start from",
                }
            x0 = np.array(parent.best_x)
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
                specification={"type": init_strategy, "parent_node": parent_node},
                x0=x0.tolist(),
            )
        elif family == "bayesian":
            initialization = BayesianInitialization(
                specification={"type": init_strategy, "parent_node": parent_node},
                warm_start_trials=None,
                n_initial_random=config_dict.get("n_startup_trials", 10),
            )
        elif family == "evolutionary":
            initialization = EvolutionaryInitialization(
                specification={"type": init_strategy, "parent_node": parent_node},
                pop_size=config_dict.get("pop_size", 100),
                seed=config_dict.get("seed"),
                initial_population=None,
            )
        elif family == "multiobjective":
            initialization = MultiObjectiveInitialization(
                specification={"type": init_strategy, "parent_node": parent_node},
                pop_size=config_dict.get("pop_size", 100),
                n_objectives=config_dict.get("n_objectives", 2),
                seed=config_dict.get("seed"),
            )
        elif family == "cmaes":
            initialization = CMAESInitialization(
                specification={"type": init_strategy, "parent_node": parent_node},
                x0=x0.tolist(),
                sigma0=config_dict.get("sigma", 0.5),
            )
        elif family == "population":
            initialization = PopulationInitialization(
                specification={"type": init_strategy, "parent_node": parent_node},
                pop_size=config_dict.get("pop_size", 50),
                seed=config_dict.get("seed"),
            )
        else:
            # Default to gradient-style for unknown families
            initialization = GradientInitialization(
                specification={"type": init_strategy, "parent_node": parent_node},
                x0=x0.tolist(),
            )

        # Start node within graph
        active_node = graph.start_node(
            optimizer=optimizer,
            config=config_dict,
            initialization=initialization,
            parent_node=parent_node,
            edge_type=edge_type,
        )

        # Prepare objective function
        # Agent specifies n_obj in config for MOO (no auto-detection)
        n_obj = config_dict.get("n_obj", 1)
        if n_obj > 1:
            # MOO: return array of objectives
            # Agent must ensure evaluator is registered with matching n_outputs
            def objective(x):
                result = problem.objective_eval.evaluate(np.atleast_1d(x))
                # Extract objectives in order: f0, f1, f2, ... or by output_names
                values = []
                for i in range(n_obj):
                    key = f"f{i}"
                    if key in result.objectives:
                        values.append(result.objectives[key])
                    elif i == 0 and "objective" in result.objectives:
                        values.append(result.objectives["objective"])
                return np.array(values)
        else:
            # SOO: return single float
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

        # Record iterations to active node
        for h in result.history:
            active_node.record_iteration(h)

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
        elif family == "evolutionary":
            progress = EvolutionaryProgress()
            for i, h in enumerate(result.history):
                progress.add_generation(
                    generation=i + 1,
                    best_fitness=h.get("best_f", h.get("f", 0.0)),
                    mean_fitness=h.get("mean_f", 0.0),
                    best_individual=h.get("x", []),
                    diversity=h.get("diversity"),
                    constraint_violation=h.get("constraint_violation"),
                )
            result_component = EvolutionaryResult(
                termination_reason=result.message,
                n_generations=result.n_iterations,
                final_pop_size=config_dict.get("pop_size", 100),
                final_diversity=None,
            )
        elif family == "multiobjective":
            progress = MultiObjectiveProgress()
            for i, h in enumerate(result.history):
                progress.add_generation(
                    generation=i + 1,
                    n_nondominated=h.get("n_nondominated", 0),
                    hypervolume=h.get("hypervolume"),
                    igd=h.get("igd"),
                    spread=h.get("spread"),
                )
            result_component = MultiObjectiveResult(
                termination_reason=result.message,
                n_generations=result.n_iterations,
                n_pareto_solutions=result.n_pareto_solutions if hasattr(result, 'n_pareto_solutions') else 0,
                final_hypervolume=result.hypervolume if hasattr(result, 'hypervolume') else None,
                pareto_ref=None,  # Will be set below if applicable
            )
        elif family == "cmaes":
            progress = CMAESProgress()
            for i, h in enumerate(result.history):
                progress.add_generation(
                    generation=i + 1,
                    best_fitness=h.get("best_f", h.get("f", 0.0)),
                    mean_fitness=h.get("mean_f", 0.0),
                    sigma=h.get("sigma"),
                    condition_number=h.get("condition_number"),
                )
            result_component = CMAESResult(
                termination_reason=result.message,
                n_generations=result.n_iterations,
                final_sigma=None,
                final_condition_number=None,
            )
        elif family == "population":
            progress = PopulationProgress()
            for i, h in enumerate(result.history):
                progress.add_generation(
                    generation=i + 1,
                    best_fitness=h.get("best_f", h.get("f", 0.0)),
                    mean_fitness=h.get("mean_f", 0.0),
                    best_individual=h.get("x", []),
                    diversity=h.get("diversity"),
                )
            result_component = PopulationResult(
                termination_reason=result.message,
                n_generations=result.n_iterations,
                final_pop_size=config_dict.get("pop_size", 50),
                final_diversity=None,
            )
        else:
            # Default to gradient-style for unknown families
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

        # Store Pareto front to ParetoStorage for any MOO result (v0.4.9)
        pareto_ref = None
        if result.is_multiobjective and result.pareto_set is not None and result.pareto_front is not None:
            try:
                from ..foundry.schema.pareto import ParetoFront
                from ..foundry.storage.pareto_storage import ParetoStorage

                # Get objective names from evaluator if available
                objective_names = [f"f{i}" for i in range(result.pareto_front.shape[1])]
                if hasattr(problem, "objective_eval") and hasattr(problem.objective_eval, "output_names"):
                    if problem.objective_eval.output_names:
                        objective_names = problem.objective_eval.output_names

                # Create ParetoFront
                pareto_front_obj = ParetoFront.from_arrays(
                    pareto_set=result.pareto_set,
                    pareto_front=result.pareto_front,
                    objective_names=objective_names,
                    hypervolume=result.hypervolume,
                    graph_id=graph_id,
                    node_id=active_node.node_id,
                    algorithm=optimizer,
                    n_generations=result.n_iterations,
                )

                # Store to ParetoStorage
                # FileStorage uses base_dir, not base_path
                storage_path = str(_FOUNDRY.storage.base_dir)
                pareto_storage = ParetoStorage(storage_path)
                pareto_ref = pareto_storage.store(pareto_front_obj)
                logger.info(f"Stored Pareto front: {pareto_ref}")

                # Update result_component if it has pareto_ref field
                if hasattr(result_component, 'pareto_ref'):
                    result_component.pareto_ref = pareto_ref
            except Exception as e:
                logger.warning(f"Could not store Pareto front: {e}")

        # Complete node
        best_x = (
            result.best_x.tolist()
            if isinstance(result.best_x, np.ndarray)
            else list(result.best_x)
        )

        # Prepare MOO fields for node completion (v0.4.9)
        moo_kwargs = {}
        if result.is_multiobjective:
            moo_kwargs = {
                "is_multiobjective": True,
                "n_pareto_solutions": result.n_pareto_solutions,
                "hypervolume": result.hypervolume,
                "pareto_ref": pareto_ref,  # Set earlier if Pareto storage succeeded
            }

        completed_node = graph.complete_node(
            progress=progress,
            result=result_component,
            best_objective=result.best_f,
            best_x=best_x,
            success=result.success,
            **moo_kwargs,
        )

        # Return result
        response = {
            "success": result.success,
            "message": result.message,
            "node_id": completed_node.node_id,
            "best_x": best_x,
            "best_objective": float(result.best_f),
            "optimizer_used": optimizer,
            "optimizer_family": family,
            "n_iterations": result.n_iterations,
            "n_evaluations": result.n_function_evals,
            "n_gradient_evals": result.n_gradient_evals,
            "elapsed_time": elapsed,
            "parent_node": parent_node,
            "problem_id": actual_problem_id,
        }

        # Add multi-objective fields if applicable
        if hasattr(result, 'is_multiobjective') and result.is_multiobjective:
            response["is_multiobjective"] = True
            response["n_pareto_solutions"] = result.n_pareto_solutions
            response["hypervolume"] = result.hypervolume
            # Add pareto_ref if stored (v0.4.9)
            if pareto_ref:
                response["pareto_ref"] = pareto_ref

        return response

    except Exception as e:
        import traceback
        return {
            "success": False,
            "message": f"Error running optimization: {str(e)}",
            "traceback": traceback.format_exc(),
        }


# =============================================================================
# Information Tools
# =============================================================================

@tool(args_schema=GetProblemInfoArgs)
def get_problem_info(problem_id: ProblemIdType) -> Dict[str, Any]:
    """
    Get problem characteristics for optimizer selection.

    Use this to understand a problem before deciding on optimizer
    selection and configuration.

    Args:
        problem_id: Problem ID (int or str, auto-normalized)

    Returns:
        Dict with:
            success: bool
            problem_id: int
            dimension: int - Number of variables
            bounds: List - Variable bounds (truncated if >10)
            bounds_summary: str - Human-readable bounds description
            is_constrained: bool - Has constraints
            has_gradient: bool - Gradient available
            description: str - Problem description
    """
    from .evaluation import _get_problem

    try:
        # Normalize problem_id (handles str/int from LLM) - v0.4.5
        problem_id = normalize_problem_id(problem_id)
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
def list_optimizers() -> Dict[str, Any]:
    """
    List available optimizer backends and their capabilities.

    Use this to discover what optimizers are installed.
    For detailed configuration options, use load_skill(backend_name).

    Returns:
        Dict with:
            success: bool
            available_backends: List[str] - Installed backends
            backends: Dict - Capabilities (each has 'skill' field for options)
    """
    backends = list_backends()

    return {
        "success": True,
        "available_backends": get_available_backends(),
        "backends": backends,
    }


# Backward compatibility alias
list_available_optimizers = list_optimizers
