"""
Agent tools for Paola.

Provides LangChain @tool decorated functions for:
- Graph management (start_graph, get_graph_state, finalize_graph, query_past_graphs)
- Optimization execution (run_optimization, get_problem_info, list_available_optimizers)
- Expert configuration (config_scipy, config_ipopt, config_nlopt, config_optuna)
- Analysis (analyze_convergence, analyze_efficiency, get_all_metrics, analyze_run_with_ai)
- Evaluator management (foundry_list_evaluators, foundry_get_evaluator)
- Knowledge (store_optimization_insight, retrieve_optimization_knowledge)
- Cache operations (cache_get, cache_store, cache_clear, cache_stats)
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

from paola.tools.optimizer_tools import (
    optimizer_create,
    optimizer_propose,
    optimizer_update,
    optimizer_restart,
    run_scipy_optimization,
    clear_optimizer_registry,
    get_optimizer_by_id,
)

# Intent-based optimization tools (LLM-driven architecture)
from paola.tools.optimization_tools import (
    run_optimization,
    get_problem_info,
    list_available_optimizers,
)

# Expert configuration tools (escape hatch)
from paola.tools.config_tools import (
    config_scipy,
    config_ipopt,
    config_nlopt,
    config_optuna,
    explain_config_option,
)

from paola.tools.evaluator_tools import (
    evaluate_function,
    compute_gradient,
    create_benchmark_problem,
    register_problem,
    clear_problem_registry,
    get_problem_by_id,
)

from paola.tools.gate_control_tools import (
    gate_continue,
    gate_stop,
    gate_restart_from,
    gate_get_history,
    gate_get_statistics,
    register_gate,
    clear_gate_registry,
    get_gate_by_id,
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
    # Optimizer tools (legacy)
    "optimizer_create",
    "optimizer_propose",
    "optimizer_update",
    "optimizer_restart",
    "run_scipy_optimization",  # Deprecated - use run_optimization
    "clear_optimizer_registry",
    "get_optimizer_by_id",
    # Intent-based optimization tools (LLM-driven architecture)
    "run_optimization",
    "get_problem_info",
    "list_available_optimizers",
    # Expert configuration tools (escape hatch)
    "config_scipy",
    "config_ipopt",
    "config_nlopt",
    "config_optuna",
    "explain_config_option",
    # Evaluator tools
    "evaluate_function",
    "compute_gradient",
    "create_benchmark_problem",
    "register_problem",
    "clear_problem_registry",
    "get_problem_by_id",
    # Gate control tools
    "gate_continue",
    "gate_stop",
    "gate_restart_from",
    "gate_get_history",
    "gate_get_statistics",
    "register_gate",
    "clear_gate_registry",
    "get_gate_by_id",
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
