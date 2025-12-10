"""
Tests for cache tools.
"""

import pytest
from aopt.tools.cache_tools import (
    cache_get,
    cache_store,
    cache_clear,
    cache_stats,
    run_db_log,
    run_db_query
)


def test_cache_store_and_get():
    """Test basic cache store and retrieval."""
    # Clear cache first
    cache_clear()

    design = [1.0, 2.0, 3.0]
    problem_id = "test_prob"

    # Store evaluation
    result = cache_store(
        design=design,
        problem_id=problem_id,
        objectives=[0.0245],
        gradient=[0.1, 0.2, 0.3],
        cost=0.5
    )

    assert result["stored"]
    assert result["cache_size"] == 1
    assert not result["duplicate"]

    # Retrieve from cache
    cached = cache_get(design=design, problem_id=problem_id)

    assert cached is not None
    assert cached["hit"]
    assert cached["objectives"] == [0.0245]
    assert cached["gradient"] == [0.1, 0.2, 0.3]
    assert cached["cost"] == 0.5


def test_cache_miss():
    """Test cache miss."""
    cache_clear()

    # Should miss
    cached = cache_get(design=[1.0, 2.0], problem_id="test")

    assert cached is None


def test_cache_duplicate_detection():
    """Test that storing same design twice is detected."""
    cache_clear()

    design = [1.0, 2.0]
    problem_id = "test"

    # First store
    result1 = cache_store(design=design, problem_id=problem_id, objectives=[1.0])
    assert not result1["duplicate"]

    # Second store (same design)
    result2 = cache_store(design=design, problem_id=problem_id, objectives=[2.0])
    assert result2["duplicate"]

    # Cache size should still be 1 (overwritten)
    assert result2["cache_size"] == 1


def test_cache_tolerance():
    """Test cache matching with tolerance."""
    cache_clear()

    # Store design
    cache_store(
        design=[1.0, 2.0],
        problem_id="test",
        objectives=[1.0]
    )

    # Should match with small difference (within tolerance)
    cached = cache_get(
        design=[1.0 + 1e-10, 2.0 + 1e-10],
        problem_id="test",
        tolerance=1e-9
    )

    assert cached is not None
    assert cached["hit"]


def test_cache_isolation_by_problem():
    """Test that different problems have isolated caches."""
    cache_clear()

    design = [1.0, 2.0]

    # Store for problem A
    cache_store(design=design, problem_id="prob_a", objectives=[1.0])

    # Store for problem B (same design, different problem)
    cache_store(design=design, problem_id="prob_b", objectives=[2.0])

    # Should get different results
    cached_a = cache_get(design=design, problem_id="prob_a")
    cached_b = cache_get(design=design, problem_id="prob_b")

    assert cached_a["objectives"] == [1.0]
    assert cached_b["objectives"] == [2.0]


def test_cache_clear():
    """Test cache clearing."""
    # Add some entries
    cache_store(design=[1.0], problem_id="test", objectives=[1.0])
    cache_store(design=[2.0], problem_id="test", objectives=[2.0])

    stats_before = cache_stats()
    assert stats_before["total_entries"] > 0

    # Clear
    result = cache_clear()
    assert result["cleared"]
    assert result["entries_removed"] > 0

    # Should be empty
    stats_after = cache_stats()
    assert stats_after["total_entries"] == 0


def test_cache_stats():
    """Test cache statistics."""
    cache_clear()

    # Add some entries with costs
    cache_store(design=[1.0], problem_id="test", objectives=[1.0], cost=0.5)
    cache_store(design=[2.0], problem_id="test", objectives=[2.0], cost=1.0)
    cache_store(design=[3.0], problem_id="test", objectives=[3.0], cost=1.5)

    stats = cache_stats()

    assert stats["total_entries"] == 3
    assert stats["total_cost_saved"] == 3.0  # 0.5 + 1.0 + 1.5


def test_run_db_log():
    """Test run database logging."""
    result = run_db_log(
        optimizer_id="opt_001",
        iteration=5,
        design=[1.0, 2.0],
        objectives=[0.5],
        action="evaluate",
        reasoning="Testing database logging"
    )

    assert result["logged"]
    assert result["run_id"] == "opt_001"
    assert result["entry_id"] is not None


def test_run_db_query():
    """Test querying run database."""
    optimizer_id = "opt_test_query"

    # Log some entries
    for i in range(5):
        run_db_log(
            optimizer_id=optimizer_id,
            iteration=i,
            design=[float(i)],
            objectives=[float(i) ** 2],
            action="evaluate",
            reasoning=f"Iteration {i}"
        )

    # Query all entries
    entries = run_db_query(optimizer_id=optimizer_id)

    assert len(entries) == 5
    # Should be in reverse order (most recent first)
    assert entries[0]["iteration"] == 4
    assert entries[4]["iteration"] == 0

    # Query with limit
    limited = run_db_query(optimizer_id=optimizer_id, limit=2)
    assert len(limited) == 2


def test_cache_prevents_reevaluation():
    """
    Test that cache actually prevents re-evaluation.

    This is the critical use case for expensive engineering simulations.
    """
    cache_clear()

    design = [1.5, 2.5, 3.5]
    problem_id = "expensive_problem"

    # First evaluation - cache miss
    cached1 = cache_get(design=design, problem_id=problem_id)
    assert cached1 is None  # Miss

    # "Evaluate" (expensive operation)
    objectives = [0.0245]
    gradient = [0.1, 0.2, 0.3]
    evaluation_cost = 10.0  # 10 CPU hours!

    # Store result
    cache_store(
        design=design,
        problem_id=problem_id,
        objectives=objectives,
        gradient=gradient,
        cost=evaluation_cost
    )

    # Second evaluation - cache hit (saves 10 CPU hours!)
    cached2 = cache_get(design=design, problem_id=problem_id)
    assert cached2 is not None
    assert cached2["hit"]
    assert cached2["objectives"] == objectives
    assert cached2["gradient"] == gradient
    assert cached2["cost"] == evaluation_cost  # Original cost stored

    # Verify cache stats show savings
    stats = cache_stats()
    assert stats["total_cost_saved"] >= evaluation_cost
