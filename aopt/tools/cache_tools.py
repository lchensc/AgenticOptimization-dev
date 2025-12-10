"""
Cache and provenance tools for evaluation efficiency.

Critical for preventing re-evaluation during line searches and population duplicates.
Engineering simulations: 10,000× more expensive than optimizer iterations.
"""

from typing import Optional
import numpy as np
import sqlite3
from pathlib import Path
import json
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# Global cache storage (in-memory for Milestone 1)
_EVALUATION_CACHE: dict[str, dict] = {}
_RUN_DATABASE: Optional[sqlite3.Connection] = None


def _design_to_key(design: list[float], problem_id: str, tolerance: float = 1e-9) -> str:
    """
    Convert design to cache key.

    Uses hashing for exact matching or numpy for tolerance-based matching.

    Args:
        design: Design variables
        problem_id: Problem identifier
        tolerance: Tolerance for design matching

    Returns:
        Cache key string
    """
    # Round design to tolerance precision
    rounded = [round(x / tolerance) * tolerance for x in design]
    # Create deterministic hash
    design_str = json.dumps(rounded, sort_keys=True)
    key_hash = hashlib.md5(f"{problem_id}:{design_str}".encode()).hexdigest()
    return key_hash


def cache_get(
    design: list[float],
    problem_id: str,
    tolerance: float = 1e-9
) -> Optional[dict]:
    """
    Retrieve cached evaluation result for design.

    Prevents re-evaluation during line searches and population duplicates.

    Args:
        design: Design variables
        problem_id: Problem identifier (for cache isolation)
        tolerance: Similarity tolerance for design matching

    Returns:
        {
            "objectives": [0.0245],
            "gradient": [...],  # If available
            "constraints": {...},  # If available
            "cost": 0.5,  # CPU hours
            "timestamp": "2025-12-10T10:30:00",
            "hit": True
        }

        Returns None if not in cache.

    Example:
        >>> cached = cache_get([1.0, 2.0], "prob_001")
        >>> if cached and cached["hit"]:
        ...     print(f"Cache hit! Saved {cached['cost']:.2f} CPU hours")
    """
    key = _design_to_key(design, problem_id, tolerance)

    if key in _EVALUATION_CACHE:
        result = _EVALUATION_CACHE[key].copy()
        result["hit"] = True
        logger.debug(f"Cache hit for design {design[:3]}... (key={key[:8]})")
        return result

    logger.debug(f"Cache miss for design {design[:3]}... (key={key[:8]})")
    return None


def cache_store(
    design: list[float],
    problem_id: str,
    objectives: list[float],
    gradient: Optional[list[float]] = None,
    constraints: Optional[dict] = None,
    cost: float = 0.0,
    metadata: Optional[dict] = None
) -> dict:
    """
    Store evaluation result in cache.

    Args:
        design: Design variables
        problem_id: Problem identifier
        objectives: Objective values
        gradient: Gradient (optional)
        constraints: Constraint values (optional)
        cost: Evaluation cost in CPU hours
        metadata: Additional metadata (optional)

    Returns:
        {
            "stored": True,
            "cache_size": 245,
            "duplicate": False  # True if design already cached
        }

    Example:
        >>> result = cache_store(
        ...     design=[1.0, 2.0],
        ...     problem_id="prob_001",
        ...     objectives=[0.0245],
        ...     gradient=[0.1, 0.2],
        ...     cost=0.5
        ... )
        >>> print(f"Cache size: {result['cache_size']}")
    """
    key = _design_to_key(design, problem_id)

    # Check if already in cache
    duplicate = key in _EVALUATION_CACHE

    # Store in cache
    _EVALUATION_CACHE[key] = {
        "design": design,
        "objectives": objectives,
        "gradient": gradient,
        "constraints": constraints,
        "cost": cost,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata or {}
    }

    logger.debug(
        f"Cached evaluation for design {design[:3]}... "
        f"(key={key[:8]}, duplicate={duplicate})"
    )

    return {
        "stored": True,
        "cache_size": len(_EVALUATION_CACHE),
        "duplicate": duplicate
    }


def cache_clear(problem_id: Optional[str] = None) -> dict:
    """
    Clear evaluation cache.

    Args:
        problem_id: If specified, only clear cache for this problem.
                   If None, clear entire cache.

    Returns:
        {"cleared": True, "entries_removed": N}
    """
    global _EVALUATION_CACHE

    if problem_id is None:
        # Clear entire cache
        count = len(_EVALUATION_CACHE)
        _EVALUATION_CACHE.clear()
        logger.info(f"Cleared entire cache ({count} entries)")
        return {"cleared": True, "entries_removed": count}
    else:
        # Clear only for specific problem
        to_remove = [
            key for key in _EVALUATION_CACHE
            if _EVALUATION_CACHE[key].get("metadata", {}).get("problem_id") == problem_id
        ]
        for key in to_remove:
            del _EVALUATION_CACHE[key]

        logger.info(f"Cleared cache for problem {problem_id} ({len(to_remove)} entries)")
        return {"cleared": True, "entries_removed": len(to_remove)}


def cache_stats() -> dict:
    """
    Get cache statistics.

    Returns:
        {
            "total_entries": N,
            "total_cost_saved": X,  # Estimate
            "hit_rate": 0.0-1.0  # If tracking
        }
    """
    total_entries = len(_EVALUATION_CACHE)

    # Estimate cost saved (sum of all cached costs)
    total_cost = sum(
        entry.get("cost", 0) for entry in _EVALUATION_CACHE.values()
    )

    return {
        "total_entries": total_entries,
        "total_cost_saved": total_cost,
        "hit_rate": 0.0  # TODO: Track hits/misses for accurate rate
    }


def run_db_log(
    optimizer_id: str,
    iteration: int,
    design: list[float],
    objectives: list[float],
    action: str,  # "evaluate", "adapt", "restart"
    reasoning: str,
    metadata: Optional[dict] = None
) -> dict:
    """
    Log optimization run for provenance and knowledge accumulation.

    Stores:
    - Every evaluation (design → objectives)
    - Every adaptation (reasoning + old/new problem)
    - Every agent decision (action + reasoning)

    This enables:
    - Run replay and debugging
    - Pattern detection across runs
    - Knowledge base accumulation

    Args:
        optimizer_id: Optimizer instance ID
        iteration: Current iteration number
        design: Design variables
        objectives: Objective values
        action: Action type
        reasoning: Agent's reasoning
        metadata: Additional metadata

    Returns:
        {"logged": True, "run_id": "run_001", "entry_id": 12}

    Example:
        >>> run_db_log(
        ...     optimizer_id="opt_001",
        ...     iteration=10,
        ...     design=[1.0, 2.0],
        ...     objectives=[0.0245],
        ...     action="evaluate",
        ...     reasoning="Proposed by SLSQP line search"
        ... )
    """
    global _RUN_DATABASE

    # Initialize database if not already done
    if _RUN_DATABASE is None:
        _init_run_database()

    # Create entry
    entry = {
        "optimizer_id": optimizer_id,
        "iteration": iteration,
        "design": json.dumps(design),
        "objectives": json.dumps(objectives),
        "action": action,
        "reasoning": reasoning,
        "metadata": json.dumps(metadata or {}),
        "timestamp": datetime.now().isoformat()
    }

    # Insert into database
    cursor = _RUN_DATABASE.cursor()
    cursor.execute("""
        INSERT INTO run_log (optimizer_id, iteration, design, objectives, action, reasoning, metadata, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        entry["optimizer_id"],
        entry["iteration"],
        entry["design"],
        entry["objectives"],
        entry["action"],
        entry["reasoning"],
        entry["metadata"],
        entry["timestamp"]
    ))
    _RUN_DATABASE.commit()

    entry_id = cursor.lastrowid

    logger.debug(f"Logged entry {entry_id} for optimizer {optimizer_id}")

    return {
        "logged": True,
        "run_id": optimizer_id,
        "entry_id": entry_id
    }


def _init_run_database(db_path: str = ":memory:") -> None:
    """
    Initialize run database (SQLite).

    Args:
        db_path: Path to database file (":memory:" for in-memory)
    """
    global _RUN_DATABASE

    _RUN_DATABASE = sqlite3.connect(db_path)

    # Create table
    _RUN_DATABASE.execute("""
        CREATE TABLE IF NOT EXISTS run_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            optimizer_id TEXT NOT NULL,
            iteration INTEGER NOT NULL,
            design TEXT NOT NULL,
            objectives TEXT NOT NULL,
            action TEXT NOT NULL,
            reasoning TEXT,
            metadata TEXT,
            timestamp TEXT NOT NULL
        )
    """)

    # Create index for fast queries
    _RUN_DATABASE.execute("""
        CREATE INDEX IF NOT EXISTS idx_optimizer_iteration
        ON run_log(optimizer_id, iteration)
    """)

    _RUN_DATABASE.commit()

    logger.info(f"Initialized run database: {db_path}")


def run_db_query(optimizer_id: str, limit: Optional[int] = None) -> list[dict]:
    """
    Query run database.

    Args:
        optimizer_id: Optimizer ID to query
        limit: Limit number of results

    Returns:
        List of log entries
    """
    global _RUN_DATABASE

    if _RUN_DATABASE is None:
        return []

    cursor = _RUN_DATABASE.cursor()

    if limit:
        cursor.execute("""
            SELECT * FROM run_log
            WHERE optimizer_id = ?
            ORDER BY iteration DESC
            LIMIT ?
        """, (optimizer_id, limit))
    else:
        cursor.execute("""
            SELECT * FROM run_log
            WHERE optimizer_id = ?
            ORDER BY iteration DESC
        """, (optimizer_id,))

    rows = cursor.fetchall()

    # Convert to dict
    entries = []
    for row in rows:
        entries.append({
            "id": row[0],
            "optimizer_id": row[1],
            "iteration": row[2],
            "design": json.loads(row[3]),
            "objectives": json.loads(row[4]),
            "action": row[5],
            "reasoning": row[6],
            "metadata": json.loads(row[7]) if row[7] else {},
            "timestamp": row[8]
        })

    return entries
