"""
Tests for Paola Recording Cache functionality.

Tests EvaluationCache behavior including:
- Cache hit/miss behavior
- Tolerance-based matching
- Persistence to disk
- Per-graph isolation
- Cache sharing across nodes in same graph
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
import numpy as np

import paola
from paola.api import _checkpoint_summaries, _active_objectives
from paola.recording import ArrayHasher, EvaluationCache, RecordingObjective
from paola.foundry.storage import FileStorage
from paola.foundry.problem import OptimizationProblem, Variable, Objective


# Test functions with counter to track actual calls
class CountingFunction:
    """Function that counts how many times it's actually called."""

    def __init__(self, func):
        self.func = func
        self.call_count = 0

    def __call__(self, x):
        self.call_count += 1
        return self.func(x)


def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


@pytest.fixture
def temp_cache_dir():
    """Create temp directory for cache testing."""
    temp_dir = tempfile.mkdtemp(prefix="paola_cache_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_test_dir():
    """Create isolated temp directory and change to it."""
    original_cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp(prefix="paola_test_")
    os.chdir(temp_dir)
    yield temp_dir
    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)


@pytest.fixture
def api_foundry(temp_test_dir):
    """Set up temp foundry for API tests."""
    foundry_dir = Path(temp_test_dir) / ".paola_test_foundry"

    original_migrate = FileStorage._migrate_legacy_data
    FileStorage._migrate_legacy_data = lambda self: None

    try:
        paola.set_foundry_dir(str(foundry_dir))
        paola.api._foundry = None
        _active_objectives.clear()
        _checkpoint_summaries.clear()

        foundry = paola.get_foundry()

        problem = OptimizationProblem(
            problem_id=1,
            name="Rosenbrock 2D",
            variables=[
                Variable(name="x0", type="continuous", lower=-5.0, upper=5.0),
                Variable(name="x1", type="continuous", lower=-5.0, upper=5.0),
            ],
            objectives=[
                Objective(name="rosenbrock", evaluator_id="rosenbrock_inline"),
            ],
        )

        foundry.register_problem_evaluator(problem, rosenbrock)
        yield foundry_dir

    finally:
        FileStorage._migrate_legacy_data = original_migrate
        paola.api._foundry = None
        _active_objectives.clear()
        _checkpoint_summaries.clear()


class TestArrayHasher:
    """Tests for ArrayHasher stable hashing."""

    def test_same_array_same_hash(self):
        """Identical arrays produce same hash."""
        hasher = ArrayHasher()

        x = np.array([1.0, 2.0, 3.0])
        h1 = hasher.hash(x)
        h2 = hasher.hash(x.copy())

        assert h1 == h2

    def test_different_arrays_different_hash(self):
        """Different arrays produce different hashes."""
        hasher = ArrayHasher()

        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([1.0, 2.0, 4.0])

        assert hasher.hash(x1) != hasher.hash(x2)

    def test_tolerance_matching(self):
        """Arrays within tolerance produce same hash."""
        hasher = ArrayHasher(tolerance=1e-8)

        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([1.0 + 1e-10, 2.0, 3.0])  # Within tolerance

        assert hasher.hash(x1) == hasher.hash(x2)

    def test_beyond_tolerance_different(self):
        """Arrays beyond tolerance produce different hashes."""
        hasher = ArrayHasher(tolerance=1e-10)

        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([1.0 + 1e-8, 2.0, 3.0])  # Beyond tolerance

        assert hasher.hash(x1) != hasher.hash(x2)

    def test_arrays_equal_method(self):
        """arrays_equal() uses hash-based comparison."""
        hasher = ArrayHasher(tolerance=1e-8)

        x1 = np.array([1.0, 2.0])
        x2 = np.array([1.0 + 1e-10, 2.0])

        assert hasher.arrays_equal(x1, x2)


class TestEvaluationCache:
    """Tests for EvaluationCache."""

    def test_cache_get_miss(self, temp_cache_dir):
        """Cache returns None for uncached values."""
        cache = EvaluationCache(cache_dir=temp_cache_dir)

        result = cache.get(np.array([1.0, 2.0]))

        assert result is None

    def test_cache_put_get(self, temp_cache_dir):
        """Cache stores and retrieves values."""
        cache = EvaluationCache(cache_dir=temp_cache_dir)

        x = np.array([1.0, 2.0])
        cache.put(x, 42.0)

        result = cache.get(x)

        assert result is not None
        f, grad = result
        assert f == 42.0
        assert grad is None

    def test_cache_hit_rate(self, temp_cache_dir):
        """Cache tracks hit rate correctly."""
        cache = EvaluationCache(cache_dir=temp_cache_dir)

        x = np.array([1.0, 2.0])
        cache.put(x, 42.0)

        # Miss
        cache.get(np.array([3.0, 4.0]))
        # Hit
        cache.get(x)
        # Hit
        cache.get(x)

        assert cache.hit_rate == pytest.approx(2/3)

    def test_cache_persists_to_disk(self, temp_cache_dir):
        """Cache survives recreation from disk."""
        x = np.array([1.0, 2.0])

        # Create and populate cache
        cache1 = EvaluationCache(cache_dir=temp_cache_dir)
        cache1.put(x, 42.0)

        # Create new cache from same directory
        cache2 = EvaluationCache(cache_dir=temp_cache_dir, load_existing=True)

        result = cache2.get(x)
        assert result is not None
        assert result[0] == 42.0

    def test_cache_size(self, temp_cache_dir):
        """Cache reports correct size."""
        cache = EvaluationCache(cache_dir=temp_cache_dir)

        cache.put(np.array([1.0]), 1.0)
        cache.put(np.array([2.0]), 2.0)
        cache.put(np.array([3.0]), 3.0)

        assert cache.size == 3

    def test_cache_contains(self, temp_cache_dir):
        """Cache contains() method works."""
        cache = EvaluationCache(cache_dir=temp_cache_dir)

        x = np.array([1.0, 2.0])
        cache.put(x, 42.0)

        assert cache.contains(x)
        assert not cache.contains(np.array([3.0, 4.0]))


class TestRecordingObjectiveCache:
    """Tests for cache integration in RecordingObjective."""

    def test_cache_prevents_duplicate_evals(self, temp_cache_dir):
        """Same x returns cached value without re-evaluation."""
        counter = CountingFunction(rosenbrock)

        f = RecordingObjective(
            evaluator=counter,
            graph_id=1,
            node_id="n1",
            cache_dir=temp_cache_dir,
        )

        x = np.array([0.5, 0.5])

        # First call - actual evaluation
        result1 = f(x)
        assert counter.call_count == 1

        # Second call - should be cached
        result2 = f(x)
        assert counter.call_count == 1  # No new call
        assert result1 == result2

    def test_cache_hit_count(self, temp_cache_dir):
        """RecordingObjective tracks cache hits."""
        f = RecordingObjective(
            evaluator=rosenbrock,
            graph_id=1,
            node_id="n1",
            cache_dir=temp_cache_dir,
        )

        x = np.array([0.5, 0.5])

        f(x)  # Miss
        f(x)  # Hit
        f(x)  # Hit

        assert f.n_cache_hits == 2
        assert f.n_calls == 3
        assert f.n_actual_evals == 1

    def test_cache_hit_rate(self, temp_cache_dir):
        """RecordingObjective reports cache hit rate."""
        f = RecordingObjective(
            evaluator=rosenbrock,
            graph_id=1,
            node_id="n1",
            cache_dir=temp_cache_dir,
        )

        f(np.array([0.5, 0.5]))  # Miss
        f(np.array([0.5, 0.5]))  # Hit
        f(np.array([1.0, 1.0]))  # Miss
        f(np.array([1.0, 1.0]))  # Hit

        assert f.cache_hit_rate == pytest.approx(0.5)

    def test_cache_disabled(self, temp_cache_dir):
        """Cache can be disabled."""
        counter = CountingFunction(rosenbrock)

        f = RecordingObjective(
            evaluator=counter,
            graph_id=1,
            node_id="n1",
            cache_dir=temp_cache_dir,
            use_cache=False,
        )

        x = np.array([0.5, 0.5])

        f(x)
        f(x)

        # Without cache, evaluator called twice
        assert counter.call_count == 2


class TestCacheAcrossNodes:
    """Tests for cache sharing across nodes in same graph."""

    def test_cache_shared_within_graph(self, api_foundry):
        """Nodes in same graph share cache."""
        # Turn 1
        f1 = paola.objective(problem_id=1)
        f1(np.array([0.5, 0.5]))  # Evaluated
        s1 = paola.checkpoint(f1, script="# t1")

        # Turn 2 - same graph
        f2 = paola.continue_graph(s1["graph_id"], "n1", "warm_start")
        f2(np.array([0.5, 0.5]))  # Should be cached!

        s2 = paola.checkpoint(f2, script="# t2")

        # n2 should have cache hit from n1
        assert s2["n_cache_hits"] >= 1
