"""
LangChain-wrapped cache tools for agent use.

These wrap the core cache_tools.py functions with @tool decorators
to make them usable by the LangChain agent.

Purpose: Evaluation caching (simulation results), not LLM prompt caching!
These are for caching expensive CFD/FEA simulations to avoid re-evaluation.
"""

from langchain_core.tools import tool
from typing import Optional
from .cache_tools import (
    cache_stats as _cache_stats,
    cache_clear as _cache_clear,
    run_db_query as _run_db_query,
)


@tool
def cache_stats() -> dict:
    """
    Get evaluation cache statistics.

    Shows how many simulation results are cached and estimated cost savings.
    Use this to monitor cache effectiveness during optimization.

    Returns:
        {
            "total_entries": 145,           # Number of cached evaluations
            "total_cost_saved": 72.5,       # Estimated CPU hours saved
            "hit_rate": 0.0                 # Cache hit rate (TODO: implement tracking)
        }

    Example:
        Check cache status:
        cache_stats()
    """
    return _cache_stats()


@tool
def cache_clear(problem_id: Optional[str] = None) -> dict:
    """
    Clear evaluation cache.

    Use when:
    - Problem formulation changes (invalidates cached results)
    - Starting completely new optimization
    - Cache grows too large

    Args:
        problem_id: If specified, only clear cache for this problem.
                   If None, clear entire cache.

    Returns:
        {"cleared": True, "entries_removed": 145}

    Example:
        Clear entire cache:
        cache_clear()

        Clear for specific problem:
        cache_clear(problem_id="prob_001")
    """
    return _cache_clear(problem_id)


@tool
def run_db_query(optimizer_id: str, limit: Optional[int] = None) -> list[dict]:
    """
    Query optimization run database.

    Retrieves history of evaluations, adaptations, and decisions
    for a specific optimization run.

    Use when:
    - Analyzing optimization trajectory
    - Debugging convergence issues
    - Understanding past decisions

    Args:
        optimizer_id: Optimizer instance ID to query
        limit: Maximum number of entries to return (most recent first)

    Returns:
        List of log entries with design, objectives, and reasoning.
        Each entry: {
            "id": 12,
            "optimizer_id": "opt_001",
            "iteration": 10,
            "design": [1.0, 2.0, ...],
            "objectives": [0.0245],
            "action": "evaluate",
            "reasoning": "Proposed by SLSQP line search",
            "metadata": {...},
            "timestamp": "2025-12-10T10:30:00"
        }

    Example:
        Get last 10 entries for current optimizer:
        run_db_query(optimizer_id="opt_001", limit=10)
    """
    return _run_db_query(optimizer_id, limit)
