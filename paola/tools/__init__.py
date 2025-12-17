"""
Agent tools for Paola.

Provides LangChain @tool decorated functions for:
- Graph management (start_graph, get_graph_state, finalize_graph, query_past_graphs)
- Optimization execution (run_optimization, get_problem_info, list_available_optimizers)
- Analysis (analyze_convergence, analyze_efficiency, get_all_metrics, analyze_run_with_ai)
- Evaluator management (foundry_list_evaluators, foundry_get_evaluator)
- Cache operations (cache_get, cache_store, cache_clear, cache_stats)

Note: Expert configuration is now handled via Skills infrastructure (paola.skills).
"""

from paola.tools.cache_tools import (
    cache_get,
    cache_store,
    run_db_log,
)

# LangChain-wrapped cache tools for agent use
from paola.tools.cache_agent_tools import (
    cache_clear,
    cache_stats,
    run_db_query,
)

# Optimization tools (LLM-driven architecture)
from paola.tools.optimization_tools import (
    run_optimization,
    get_problem_info,
    list_available_optimizers,
)

from paola.tools.evaluator_tools import (
    evaluate_function,
    compute_gradient,
    create_benchmark_problem,
    register_problem,
    clear_problem_registry,
    get_problem_by_id,
    # Problem management (v0.4.3+)
    create_nlp_problem,
    derive_problem,
    list_problems,
    get_problem_lineage,
)

from paola.tools.observation_tools import (
    analyze_convergence,
    detect_pattern,
    check_feasibility,
    get_gradient_quality,
    compute_improvement_statistics,
)

from paola.tools.registration_tools import (
    read_file,
    execute_python,
    foundry_store_evaluator,
    foundry_list_evaluators,
    foundry_get_evaluator,
    ALL_REGISTRATION_TOOLS,
)

# Graph management tools (v0.3.x)
from paola.tools.graph_tools import (
    start_graph,
    get_graph_state,
    finalize_graph,
    query_past_graphs,
    get_past_graph,
    set_foundry,
    get_foundry,
)

__all__ = [
    # Cache tools
    "cache_get",
    "cache_store",
    "cache_clear",
    "cache_stats",
    "run_db_log",
    "run_db_query",
    # Optimization tools (LLM-driven architecture)
    "run_optimization",
    "get_problem_info",
    "list_available_optimizers",
    # Evaluator tools
    "evaluate_function",
    "compute_gradient",
    "create_benchmark_problem",
    "register_problem",
    "clear_problem_registry",
    "get_problem_by_id",
    # Problem management (v0.4.3+)
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
    # Registration tools
    "read_file",
    "execute_python",
    "foundry_store_evaluator",
    "foundry_list_evaluators",
    "foundry_get_evaluator",
    "ALL_REGISTRATION_TOOLS",
    # Graph management tools (v0.3.x)
    "start_graph",
    "get_graph_state",
    "finalize_graph",
    "query_past_graphs",
    "get_past_graph",
    "set_foundry",
    "get_foundry",
]
