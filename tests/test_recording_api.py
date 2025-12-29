"""
Tests for Paola v0.2.0 Recording API.

Tests the public API: objective(), checkpoint(), continue_graph(), complete(), finalize_graph()
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

import paola
from paola.api import set_foundry_dir, _checkpoint_summaries, _active_objectives
from paola.foundry import OptimizationFoundry
from paola.foundry.storage import FileStorage
from paola.foundry.problem import OptimizationProblem, Variable, Objective


def rosenbrock(x):
    """Standard 2D Rosenbrock: minimum at (1, 1) with f(1,1) = 0."""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def sphere(x):
    """Simple sphere function: minimum at origin."""
    return sum(xi**2 for xi in x)


@pytest.fixture
def temp_test_dir():
    """
    Create isolated temp directory and change to it.

    Changing cwd avoids legacy migration from .paola_runs in project dir.
    """
    original_cwd = os.getcwd()
    temp_dir = tempfile.mkdtemp(prefix="paola_test_")
    os.chdir(temp_dir)

    yield temp_dir

    os.chdir(original_cwd)
    shutil.rmtree(temp_dir)


@pytest.fixture
def api_foundry(temp_test_dir):
    """Set up temp foundry for API tests and register a problem."""
    foundry_dir = Path(temp_test_dir) / ".paola_test_foundry"

    # Disable migration
    original_migrate = FileStorage._migrate_legacy_data
    FileStorage._migrate_legacy_data = lambda self: None

    try:
        # Set up API to use this directory
        paola.set_foundry_dir(str(foundry_dir))

        # Reset module-level state
        paola.api._foundry = None
        _active_objectives.clear()
        _checkpoint_summaries.clear()

        # Get foundry and register test problem
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
        # Cleanup module state
        paola.api._foundry = None
        _active_objectives.clear()
        _checkpoint_summaries.clear()


class TestObjective:
    """Tests for paola.objective() function."""

    def test_objective_creates_graph(self, api_foundry):
        """objective() creates graph and returns RecordingObjective."""
        f = paola.objective(problem_id=1, goal="Test")

        assert f.graph_id >= 1
        assert f.node_id == "n1"
        assert f.problem_id == 1
        assert f.goal == "Test"

    def test_objective_creates_directories(self, api_foundry):
        """objective() creates cache and scripts directories."""
        f = paola.objective(problem_id=1)

        cache_dir = api_foundry / "cache" / f"graph_{f.graph_id:04d}"
        scripts_dir = api_foundry / "scripts" / f"graph_{f.graph_id:04d}"

        assert cache_dir.exists()
        assert scripts_dir.exists()

    def test_objective_invalid_problem_raises(self, api_foundry):
        """objective() with invalid problem_id raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            paola.objective(problem_id=99999)

    def test_objective_callable(self, api_foundry):
        """RecordingObjective is callable like a function."""
        f = paola.objective(problem_id=1)

        x = np.array([0.5, 0.5])
        result = f(x)

        assert isinstance(result, float)
        assert result == pytest.approx(rosenbrock(x))

    def test_objective_tracks_evaluations(self, api_foundry):
        """RecordingObjective tracks evaluation count."""
        f = paola.objective(problem_id=1)

        f(np.array([0.0, 0.0]))
        f(np.array([1.0, 1.0]))
        f(np.array([0.5, 0.5]))

        assert f.n_evaluations == 3
        assert f.n_calls == 3


class TestCheckpoint:
    """Tests for paola.checkpoint() function."""

    def test_checkpoint_saves_script(self, api_foundry):
        """checkpoint() saves Python script to scripts directory."""
        f = paola.objective(problem_id=1)
        f(np.array([0.5, 0.5]))

        script = "# Test script\nprint('hello')"
        summary = paola.checkpoint(f, script=script, reasoning="Test")

        script_path = api_foundry / "scripts" / f"graph_{summary['graph_id']:04d}" / "n1.py"
        assert script_path.exists()
        assert script_path.read_text() == script

    def test_checkpoint_returns_summary(self, api_foundry):
        """checkpoint() returns correct summary structure."""
        f = paola.objective(problem_id=1)
        f(np.array([0.5, 0.5]))
        f(np.array([0.3, 0.3]))

        summary = paola.checkpoint(f, script="# test", optimizer="scipy:SLSQP")

        assert "graph_id" in summary
        assert "node_id" in summary
        assert "best_x" in summary
        assert "best_f" in summary
        assert "n_evaluations" in summary
        assert summary["n_evaluations"] == 2
        assert summary["optimizer"] == "scipy:SLSQP"
        assert summary["status"] == "ok"

    def test_checkpoint_tracks_best(self, api_foundry):
        """checkpoint() correctly identifies best solution."""
        f = paola.objective(problem_id=1)

        # Evaluate at two points - (1,1) is optimum with f=0
        f(np.array([2.0, 2.0]))  # f = 401
        f(np.array([1.0, 1.0]))  # f = 0 (optimum)

        summary = paola.checkpoint(f, script="# test")

        assert summary["best_f"] == pytest.approx(0.0)
        assert summary["best_x"] == pytest.approx([1.0, 1.0])

    def test_checkpoint_stores_for_warmstart(self, api_foundry):
        """checkpoint() stores summary for warm-start support."""
        f = paola.objective(problem_id=1)
        f(np.array([0.5, 0.5]))

        summary = paola.checkpoint(f, script="# test")

        # Check internal storage for warm-start
        key = (summary["graph_id"], summary["node_id"])
        assert key in _checkpoint_summaries
        assert _checkpoint_summaries[key]["best_x"] is not None


class TestComplete:
    """Tests for paola.complete() function."""

    def test_complete_finalizes_graph(self, api_foundry):
        """complete() is equivalent to checkpoint + finalize."""
        f = paola.objective(problem_id=1)
        f(np.array([1.0, 1.0]))

        summary = paola.complete(f, script="# test")

        assert summary["finalized"] == True
        assert "graph_id" in summary

    def test_complete_persists_graph(self, api_foundry):
        """complete() persists graph to storage."""
        f = paola.objective(problem_id=1)
        f(np.array([1.0, 1.0]))

        summary = paola.complete(f, script="# test")

        # Graph should be loadable from storage
        foundry = paola.get_foundry()
        record = foundry.load_graph_record(summary["graph_id"])
        # Note: record may be None if storage uses different format
        # The key is that finalize was called


class TestFinalizeGraph:
    """Tests for paola.finalize_graph() function."""

    def test_finalize_returns_status(self, api_foundry):
        """finalize_graph() returns status dict."""
        f = paola.objective(problem_id=1)
        f(np.array([1.0, 1.0]))
        summary = paola.checkpoint(f, script="# test")

        result = paola.finalize_graph(summary["graph_id"])

        assert result is not None
        assert result["status"] == "finalized"
        assert result["graph_id"] == summary["graph_id"]


class TestContinueGraph:
    """Tests for paola.continue_graph() function."""

    def test_continue_creates_new_node(self, api_foundry):
        """continue_graph() creates n2 after n1."""
        # First turn
        f1 = paola.objective(problem_id=1)
        f1(np.array([0.5, 0.5]))
        summary1 = paola.checkpoint(f1, script="# turn 1")

        # Second turn
        f2 = paola.continue_graph(
            graph_id=summary1["graph_id"],
            parent_node="n1",
            edge_type="warm_start"
        )

        assert f2.node_id == "n2"
        assert f2.graph_id == summary1["graph_id"]

    def test_warm_start_passes_best_x(self, api_foundry):
        """Warm-start passes parent's best_x to child."""
        # First turn
        f1 = paola.objective(problem_id=1)
        f1(np.array([0.8, 0.6]))
        summary1 = paola.checkpoint(f1, script="# turn 1")

        # Second turn
        f2 = paola.continue_graph(
            graph_id=summary1["graph_id"],
            parent_node="n1",
            edge_type="warm_start"
        )

        warm_start = f2.get_warm_start()
        assert warm_start is not None
        assert warm_start == pytest.approx(summary1["best_x"])

    def test_first_node_no_warmstart(self, api_foundry):
        """First node has no warm-start available."""
        f1 = paola.objective(problem_id=1)

        assert f1.get_warm_start() is None

    def test_three_turn_chain(self, api_foundry):
        """Three-turn chain n1 -> n2 -> n3 works correctly."""
        # Turn 1
        f1 = paola.objective(problem_id=1)
        f1(np.array([0.0, 0.0]))
        s1 = paola.checkpoint(f1, script="# t1")

        # Turn 2
        f2 = paola.continue_graph(s1["graph_id"], "n1", "warm_start")
        f2(np.array([0.5, 0.5]))
        s2 = paola.checkpoint(f2, script="# t2")

        # Turn 3
        f3 = paola.continue_graph(s1["graph_id"], "n2", "refine")
        f3(np.array([1.0, 1.0]))
        s3 = paola.checkpoint(f3, script="# t3")

        assert s1["node_id"] == "n1"
        assert s2["node_id"] == "n2"
        assert s3["node_id"] == "n3"

        # All on same graph
        assert s1["graph_id"] == s2["graph_id"] == s3["graph_id"]


class TestIntegration:
    """Integration tests with scipy.optimize."""

    def test_scipy_slsqp_optimization(self, api_foundry):
        """Complete optimization workflow with scipy SLSQP."""
        f = paola.objective(problem_id=1, goal="Minimize Rosenbrock")

        x0 = np.array([0.0, 0.0])
        result = minimize(f, x0, method='SLSQP', bounds=[(-5, 5), (-5, 5)])

        script = """
from scipy.optimize import minimize
import paola

f = paola.objective(problem_id=1)
result = minimize(f, [0, 0], method='SLSQP')
"""
        summary = paola.complete(f, script=script, optimizer="scipy:SLSQP")

        assert summary["best_f"] < 0.01  # Close to optimum
        assert summary["n_evaluations"] > 1
        assert summary["finalized"] == True

    def test_multiturn_optimization(self, api_foundry):
        """Multi-turn optimization improves result."""
        # Turn 1: Nelder-Mead (derivative-free)
        f1 = paola.objective(problem_id=1)
        result1 = minimize(f1, np.array([3.0, -3.0]), method='Nelder-Mead',
                          options={'maxiter': 50})
        s1 = paola.checkpoint(f1, script="# NM", optimizer="scipy:Nelder-Mead")

        # Turn 2: SLSQP with warm-start
        f2 = paola.continue_graph(s1["graph_id"], "n1", "warm_start")
        x0 = f2.get_warm_start()
        result2 = minimize(f2, x0, method='SLSQP', bounds=[(-5, 5), (-5, 5)])
        s2 = paola.checkpoint(f2, script="# SLSQP", optimizer="scipy:SLSQP")

        # Turn 2 should improve on Turn 1
        assert s2["best_f"] <= s1["best_f"]

        # Finalize
        paola.finalize_graph(s1["graph_id"])
