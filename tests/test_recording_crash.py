"""
Tests for Paola Recording crash recovery and error handling.

Tests:
- Crash logging with error info
- JSONL persistence survives crashes
- has_crashes flag in summary
- Pending -> crash status transition
"""

import pytest
import tempfile
import shutil
import os
import json
from pathlib import Path
import numpy as np

import paola
from paola.api import _checkpoint_summaries, _active_objectives
from paola.recording import RecordingObjective
from paola.foundry.storage import FileStorage
from paola.foundry.problem import OptimizationProblem, Variable, Objective


def crashing_function(x):
    """Function that crashes for certain inputs."""
    if x[0] < 0:
        raise ValueError("Negative input not allowed")
    return sum(xi**2 for xi in x)


def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


@pytest.fixture
def temp_cache_dir():
    """Create temp directory for cache testing."""
    temp_dir = tempfile.mkdtemp(prefix="paola_crash_test_")
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

        # Register crashing function as problem evaluator
        problem = OptimizationProblem(
            problem_id=1,
            name="Crashing Function",
            variables=[
                Variable(name="x0", type="continuous", lower=-5.0, upper=5.0),
                Variable(name="x1", type="continuous", lower=-5.0, upper=5.0),
            ],
            objectives=[
                Objective(name="sphere", evaluator_id="crashing"),
            ],
        )

        foundry.register_problem_evaluator(problem, crashing_function)
        yield foundry_dir

    finally:
        FileStorage._migrate_legacy_data = original_migrate
        paola.api._foundry = None
        _active_objectives.clear()
        _checkpoint_summaries.clear()


class TestCrashLogging:
    """Tests for crash/error logging in RecordingObjective."""

    def test_crash_logged_with_error(self, temp_cache_dir):
        """Crash is logged with error message and type."""
        f = RecordingObjective(
            evaluator=crashing_function,
            graph_id=1,
            node_id="n1",
            cache_dir=temp_cache_dir,
        )

        # First call succeeds
        f(np.array([1.0, 1.0]))

        # Second call crashes
        with pytest.raises(ValueError):
            f(np.array([-1.0, 1.0]))

        # Check crash info
        crash_info = f.get_crash_info()
        assert len(crash_info) == 1
        assert crash_info[0]["error_type"] == "ValueError"
        assert "Negative" in crash_info[0]["error"]

    def test_crash_includes_x_value(self, temp_cache_dir):
        """Crash record includes the design point that caused it."""
        f = RecordingObjective(
            evaluator=crashing_function,
            graph_id=1,
            node_id="n1",
            cache_dir=temp_cache_dir,
        )

        with pytest.raises(ValueError):
            f(np.array([-1.0, 2.0]))

        crash_info = f.get_crash_info()
        assert crash_info[0]["x"] == pytest.approx([-1.0, 2.0])

    def test_has_crashes_flag(self, temp_cache_dir):
        """Summary reports has_crashes correctly."""
        f = RecordingObjective(
            evaluator=crashing_function,
            graph_id=1,
            node_id="n1",
            cache_dir=temp_cache_dir,
        )

        # No crashes yet
        f(np.array([1.0, 1.0]))
        summary1 = f.summary()
        assert summary1["has_crashes"] == False

        # Now crash
        with pytest.raises(ValueError):
            f(np.array([-1.0, 1.0]))

        summary2 = f.summary()
        assert summary2["has_crashes"] == True

    def test_status_counts(self, temp_cache_dir):
        """Summary includes status counts."""
        f = RecordingObjective(
            evaluator=crashing_function,
            graph_id=1,
            node_id="n1",
            cache_dir=temp_cache_dir,
        )

        # 2 successful
        f(np.array([1.0, 1.0]))
        f(np.array([2.0, 2.0]))

        # 1 crash
        with pytest.raises(ValueError):
            f(np.array([-1.0, 1.0]))

        summary = f.summary()
        assert summary["status_counts"]["ok"] == 2
        assert summary["status_counts"]["crash"] == 1


class TestJSONLPersistence:
    """Tests for JSONL log file persistence."""

    def test_jsonl_created(self, temp_cache_dir):
        """JSONL log file is created."""
        f = RecordingObjective(
            evaluator=crashing_function,
            graph_id=1,
            node_id="n1",
            cache_dir=temp_cache_dir,
        )

        f(np.array([1.0, 1.0]))

        log_file = temp_cache_dir / "evaluations.jsonl"
        assert log_file.exists()

    def test_jsonl_contains_records(self, temp_cache_dir):
        """JSONL contains evaluation records."""
        f = RecordingObjective(
            evaluator=crashing_function,
            graph_id=1,
            node_id="n1",
            cache_dir=temp_cache_dir,
        )

        f(np.array([1.0, 1.0]))
        f(np.array([2.0, 2.0]))

        log_file = temp_cache_dir / "evaluations.jsonl"
        lines = log_file.read_text().strip().split('\n')

        # Each eval has pending then ok, so 4 lines
        assert len(lines) >= 2

        # Parse and verify
        records = [json.loads(line) for line in lines]
        # Find 'ok' records
        ok_records = [r for r in records if r.get("status") == "ok"]
        assert len(ok_records) == 2

    def test_jsonl_survives_crash(self, temp_cache_dir):
        """JSONL file is intact after exception."""
        f = RecordingObjective(
            evaluator=crashing_function,
            graph_id=1,
            node_id="n1",
            cache_dir=temp_cache_dir,
        )

        # Successful eval
        f(np.array([1.0, 1.0]))

        # Crash
        with pytest.raises(ValueError):
            f(np.array([-1.0, 1.0]))

        # File should still be readable
        log_file = temp_cache_dir / "evaluations.jsonl"
        lines = log_file.read_text().strip().split('\n')

        # Should have records for both attempts
        records = [json.loads(line) for line in lines]
        assert any(r.get("status") == "ok" for r in records)
        assert any(r.get("status") == "crash" for r in records)

    def test_pending_to_crash_transition(self, temp_cache_dir):
        """JSONL shows pending entry before crash."""
        f = RecordingObjective(
            evaluator=crashing_function,
            graph_id=1,
            node_id="n1",
            cache_dir=temp_cache_dir,
        )

        with pytest.raises(ValueError):
            f(np.array([-1.0, 1.0]))

        log_file = temp_cache_dir / "evaluations.jsonl"
        lines = log_file.read_text().strip().split('\n')
        records = [json.loads(line) for line in lines]

        # Should have pending, then crash
        statuses = [r.get("status") for r in records]
        assert "pending" in statuses
        assert "crash" in statuses

        # Pending should come before crash
        pending_idx = statuses.index("pending")
        crash_idx = statuses.index("crash")
        assert pending_idx < crash_idx


class TestAPIWithCrashes:
    """Tests for Recording API handling of crashes."""

    def test_checkpoint_reports_crashes(self, api_foundry):
        """checkpoint() summary includes crash info."""
        f = paola.objective(problem_id=1)

        # Successful eval
        f(np.array([1.0, 1.0]))

        # Crash
        with pytest.raises(ValueError):
            f(np.array([-1.0, 1.0]))

        summary = paola.checkpoint(f, script="# test")

        assert summary["has_crashes"] == True
        assert summary["status"] == "has_crashes"
        assert "last_crash" in summary
        assert summary["last_crash"]["error_type"] == "ValueError"

    def test_best_from_successful_only(self, api_foundry):
        """Best solution only considers successful evaluations."""
        f = paola.objective(problem_id=1)

        # Crash first (would have f=0 if it worked)
        with pytest.raises(ValueError):
            f(np.array([-1.0, 1.0]))

        # Then successful
        f(np.array([2.0, 2.0]))  # f = 8

        summary = paola.checkpoint(f, script="# test")

        # Best should be from successful eval only
        assert summary["best_f"] is not None
        assert summary["best_f"] > 0  # Not from the crash
