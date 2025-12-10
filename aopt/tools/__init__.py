"""
Agent tools for the agentic optimization platform.

Provides LangChain @tool decorated functions for:
- Cache operations (cache_get, cache_store, cache_clear, cache_stats)
- Database operations (run_db_log, run_db_query)
- Optimizer operations (optimizer_create, optimizer_propose, optimizer_update, optimizer_restart)
- Evaluator operations (evaluate_function, compute_gradient)
"""

from aopt.tools.cache_tools import (
    cache_get,
    cache_store,
    cache_clear,
    cache_stats,
    run_db_log,
    run_db_query,
)

from aopt.tools.optimizer_tools import (
    optimizer_create,
    optimizer_propose,
    optimizer_update,
    optimizer_restart,
    clear_optimizer_registry,
    get_optimizer_by_id,
)

from aopt.tools.evaluator_tools import (
    evaluate_function,
    compute_gradient,
    register_problem,
    clear_problem_registry,
    get_problem_by_id,
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
    "clear_optimizer_registry",
    "get_optimizer_by_id",
    # Evaluator tools
    "evaluate_function",
    "compute_gradient",
    "register_problem",
    "clear_problem_registry",
    "get_problem_by_id",
]
