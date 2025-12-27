"""
Agent tools for Paola.

Organized into logical modules:
- problem.py: Problem formulation (create_nlp_problem, derive_problem, list_problems)
- evaluator.py: Evaluator registration (foundry_store_evaluator, foundry_list_evaluators)
- evaluation.py: Function evaluation (evaluate_function, compute_gradient)
- optimizer.py: Optimization execution (run_optimization, get_problem_info)
- graph.py: Graph management (start_graph, get_graph_state, finalize_graph)
- analysis.py: Analysis tools (analyze_convergence, detect_pattern)
- file_tools.py: File operations (read_file, write_file, execute_python)
- cache.py: Cache operations (cache_get, cache_store, cache_clear)
"""

# Pydantic schemas for tool validation
from paola.tools.schemas import (
    normalize_problem_id,
    ProblemIdType,
)

# Cache tools
from paola.tools.cache import (
    cache_get,
    cache_store,
    cache_clear,
    cache_stats,
    run_db_log,
    run_db_query,
)

# Optimization tools
from paola.tools.optimizer import (
    run_optimization,
    get_problem_info,
    list_available_optimizers,
)

# Function evaluation tools
from paola.tools.evaluation import (
    evaluate_function,
    compute_gradient,
    create_benchmark_problem,
    register_problem,
    clear_problem_registry,
    get_problem_by_id,
)

# Problem formulation tools
from paola.tools.problem import (
    create_nlp_problem,
    derive_problem,
    list_problems,
    get_problem_lineage,
)

# Observation tools
from paola.tools.analysis import (
    analyze_convergence,
    detect_pattern,
    check_feasibility,
    get_gradient_quality,
    compute_improvement_statistics,
)

# Evaluator registration tools
from paola.tools.evaluator import (
    foundry_store_evaluator,
    foundry_list_evaluators,
    foundry_get_evaluator,
)

# File operation tools
from paola.tools.file_tools import (
    read_file,
    write_file,
    execute_python,
)

# Backward compatibility - re-export from registration_tools
from paola.tools.registration_tools import (
    ALL_REGISTRATION_TOOLS,
)

# Graph management tools
from paola.tools.graph import (
    start_graph,
    get_graph_state,
    finalize_graph,
    query_past_graphs,
    get_past_graph,
    set_foundry,
    get_foundry,
)

# Pareto front tools
from paola.tools.pareto import (
    query_pareto,
    compare_pareto_fronts,
    list_pareto_fronts,
    set_pareto_storage,
    get_pareto_storage,
    PARETO_TOOLS,
)

__all__ = [
    # Pydantic schemas
    "normalize_problem_id",
    "ProblemIdType",
    # Cache tools
    "cache_get",
    "cache_store",
    "cache_clear",
    "cache_stats",
    "run_db_log",
    "run_db_query",
    # Optimization tools
    "run_optimization",
    "get_problem_info",
    "list_available_optimizers",
    # Function evaluation tools
    "evaluate_function",
    "compute_gradient",
    "create_benchmark_problem",
    "register_problem",
    "clear_problem_registry",
    "get_problem_by_id",
    # Problem formulation tools
    "create_nlp_problem",
    "derive_problem",
    "list_problems",
    "get_problem_lineage",
    # Observation tools
    "analyze_convergence",
    "detect_pattern",
    "check_feasibility",
    "get_gradient_quality",
    "compute_improvement_statistics",
    # Evaluator registration tools
    "foundry_store_evaluator",
    "foundry_list_evaluators",
    "foundry_get_evaluator",
    # File operation tools
    "read_file",
    "write_file",
    "execute_python",
    # Backward compatibility
    "ALL_REGISTRATION_TOOLS",
    # Graph management tools
    "start_graph",
    "get_graph_state",
    "finalize_graph",
    "query_past_graphs",
    "get_past_graph",
    "set_foundry",
    "get_foundry",
    # Pareto front tools
    "query_pareto",
    "compare_pareto_fronts",
    "list_pareto_fronts",
    "set_pareto_storage",
    "get_pareto_storage",
    "PARETO_TOOLS",
]
