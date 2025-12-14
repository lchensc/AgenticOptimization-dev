"""
Agent tools for the agentic optimization platform.

Provides LangChain @tool decorated functions for:
- Cache operations (cache_get, cache_store, cache_clear, cache_stats)
- Database operations (run_db_log, run_db_query)
- Optimizer operations (optimizer_create, optimizer_propose, optimizer_update, optimizer_restart)
- Evaluator operations (evaluate_function, compute_gradient)
- Gate control operations (gate_continue, gate_stop, gate_restart_from, gate_get_history)
- Observation operations (analyze_convergence, detect_pattern, check_feasibility, get_gradient_quality)
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

__all__ = [
    # Cache tools
    "cache_get",
    "cache_store",
    "cache_clear",
    "cache_stats",
    "run_db_log",
    "run_db_query",
    # Optimizer tools
    "optimizer_create",
    "optimizer_propose",
    "optimizer_update",
    "optimizer_restart",
    "run_scipy_optimization",
    "clear_optimizer_registry",
    "get_optimizer_by_id",
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
]
