"""
Tests for v0.2.0 Session-based architecture.

Tests cover:
- SessionRecord and OptimizationRun schemas
- Polymorphic component serialization/deserialization
- ComponentRegistry
- ActiveSession and ActiveRun lifecycle
- Session tools integration
"""

import pytest
import tempfile
import shutil
from pathlib import Path


class TestSchemaComponents:
    """Test polymorphic schema components."""

    def test_gradient_initialization(self):
        """Test GradientInitialization component."""
        from paola.foundry.schema import GradientInitialization

        init = GradientInitialization(
            specification={"type": "center"},
            x0=[0.0, 0.0, 0.0],
        )

        # Test serialization
        data = init.to_dict()
        assert data["family"] == "gradient"
        assert data["x0"] == [0.0, 0.0, 0.0]

        # Test deserialization
        restored = GradientInitialization.from_dict(data)
        assert restored.x0 == [0.0, 0.0, 0.0]

    def test_gradient_progress(self):
        """Test GradientProgress with iterations."""
        from paola.foundry.schema import GradientProgress

        progress = GradientProgress()
        progress.add_iteration(
            iteration=1,
            objective=10.0,
            design=[1.0, 2.0],
            gradient_norm=1.5,
        )
        progress.add_iteration(
            iteration=2,
            objective=5.0,
            design=[0.5, 1.0],
            gradient_norm=0.5,
        )

        assert len(progress.iterations) == 2
        assert progress.iterations[0].objective == 10.0
        assert progress.iterations[1].objective == 5.0

        # Test serialization
        data = progress.to_dict()
        assert data["family"] == "gradient"
        assert len(data["iterations"]) == 2

        # Test deserialization
        restored = GradientProgress.from_dict(data)
        assert len(restored.iterations) == 2
        assert restored.iterations[0].objective == 10.0

    def test_gradient_result(self):
        """Test GradientResult component."""
        from paola.foundry.schema import GradientResult

        result = GradientResult(
            termination_reason="Optimization converged",
            final_gradient_norm=1e-8,
            final_constraint_violation=None,
        )

        data = result.to_dict()
        assert data["termination_reason"] == "Optimization converged"

        restored = GradientResult.from_dict(data)
        assert restored.termination_reason == "Optimization converged"

    def test_bayesian_progress(self):
        """Test BayesianProgress with trials."""
        from paola.foundry.schema import BayesianProgress

        progress = BayesianProgress()
        progress.add_trial(
            trial_number=1,
            design=[1.0, 2.0],
            objective=10.0,
            state="complete",
        )
        progress.add_trial(
            trial_number=2,
            design=[0.5, 1.0],
            objective=5.0,
            state="complete",
        )

        assert len(progress.trials) == 2

        data = progress.to_dict()
        restored = BayesianProgress.from_dict(data)
        assert len(restored.trials) == 2


class TestComponentRegistry:
    """Test ComponentRegistry functionality."""

    def test_list_families(self):
        """Test listing registered families."""
        from paola.foundry.schema import COMPONENT_REGISTRY

        families = COMPONENT_REGISTRY.list_families()
        assert "gradient" in families
        assert "bayesian" in families
        assert "population" in families
        assert "cmaes" in families

    def test_get_family_from_optimizer(self):
        """Test getting family from optimizer string."""
        from paola.foundry.schema import COMPONENT_REGISTRY

        assert COMPONENT_REGISTRY.get_family("scipy") == "gradient"
        assert COMPONENT_REGISTRY.get_family("scipy:SLSQP") == "gradient"
        assert COMPONENT_REGISTRY.get_family("ipopt") == "gradient"
        assert COMPONENT_REGISTRY.get_family("optuna") == "bayesian"
        assert COMPONENT_REGISTRY.get_family("optuna:TPE") == "bayesian"

    def test_get_family_unknown(self):
        """Test getting family for unknown optimizer."""
        from paola.foundry.schema import COMPONENT_REGISTRY

        # Unknown optimizer defaults to "gradient"
        assert COMPONENT_REGISTRY.get_family("unknown_optimizer") == "gradient"

    def test_deserialize_components(self):
        """Test deserializing components from dict."""
        from paola.foundry.schema import (
            COMPONENT_REGISTRY,
            GradientInitialization,
            GradientProgress,
            GradientResult,
        )

        init_data = {
            "family": "gradient",
            "specification": {"type": "center"},
            "x0": [0.0, 0.0],
        }
        progress_data = {
            "family": "gradient",
            "iterations": [],
        }
        result_data = {
            "family": "gradient",
            "termination_reason": "Converged",
            "final_gradient_norm": None,
            "final_constraint_violation": None,
        }

        init, progress, result = COMPONENT_REGISTRY.deserialize_components(
            family="gradient",
            init_data=init_data,
            progress_data=progress_data,
            result_data=result_data,
        )

        assert isinstance(init, GradientInitialization)
        assert isinstance(progress, GradientProgress)
        assert isinstance(result, GradientResult)


class TestOptimizationRun:
    """Test OptimizationRun schema."""

    def test_optimization_run_creation(self):
        """Test creating OptimizationRun."""
        from paola.foundry.schema import (
            OptimizationRun,
            GradientInitialization,
            GradientProgress,
            GradientResult,
        )

        init = GradientInitialization(
            specification={"type": "center"},
            x0=[0.0, 0.0],
        )
        progress = GradientProgress()
        result = GradientResult(termination_reason="Converged")

        run = OptimizationRun(
            run_id=1,
            optimizer="scipy:SLSQP",
            optimizer_family="gradient",
            warm_start_from=None,
            n_evaluations=100,
            wall_time=1.5,
            run_success=True,
            best_objective=0.001,
            best_design=[1.0, 1.0],
            initialization=init,
            progress=progress,
            result=result,
        )

        assert run.run_id == 1
        assert run.optimizer == "scipy:SLSQP"
        assert run.optimizer_family == "gradient"

    def test_optimization_run_serialization(self):
        """Test OptimizationRun serialization."""
        from paola.foundry.schema import (
            OptimizationRun,
            GradientInitialization,
            GradientProgress,
            GradientResult,
        )

        init = GradientInitialization(
            specification={"type": "center"},
            x0=[0.0, 0.0],
        )
        progress = GradientProgress()
        result = GradientResult(termination_reason="Converged")

        run = OptimizationRun(
            run_id=1,
            optimizer="scipy:SLSQP",
            optimizer_family="gradient",
            warm_start_from=None,
            n_evaluations=100,
            wall_time=1.5,
            run_success=True,
            best_objective=0.001,
            best_design=[1.0, 1.0],
            initialization=init,
            progress=progress,
            result=result,
        )

        data = run.to_dict()
        assert data["run_id"] == 1
        assert data["optimizer"] == "scipy:SLSQP"
        assert "initialization" in data
        assert data["initialization"]["family"] == "gradient"

        restored = OptimizationRun.from_dict(data)
        assert restored.run_id == 1
        assert restored.optimizer == "scipy:SLSQP"


class TestSessionRecord:
    """Test SessionRecord schema."""

    def test_session_record_creation(self):
        """Test creating SessionRecord."""
        from paola.foundry.schema import (
            SessionRecord,
            OptimizationRun,
            GradientInitialization,
            GradientProgress,
            GradientResult,
        )

        init = GradientInitialization(specification={"type": "center"}, x0=[0.0])
        progress = GradientProgress()
        result = GradientResult(termination_reason="Converged")

        run = OptimizationRun(
            run_id=1,
            optimizer="scipy:SLSQP",
            optimizer_family="gradient",
            warm_start_from=None,
            n_evaluations=50,
            wall_time=1.0,
            run_success=True,
            best_objective=0.001,
            best_design=[1.0],
            initialization=init,
            progress=progress,
            result=result,
        )

        session = SessionRecord(
            session_id=1,
            problem_id="test_problem",
            created_at="2024-01-01T00:00:00",
            config={"goal": "minimize"},
            runs=[run],
            success=True,
            final_objective=0.001,
            final_design=[1.0],
            total_evaluations=50,
            total_wall_time=1.0,
            decisions=[],
        )

        assert session.session_id == 1
        assert len(session.runs) == 1
        assert session.final_objective == 0.001

    def test_session_record_json_serialization(self):
        """Test SessionRecord JSON serialization."""
        from paola.foundry.schema import (
            SessionRecord,
            OptimizationRun,
            GradientInitialization,
            GradientProgress,
            GradientResult,
        )

        init = GradientInitialization(specification={"type": "center"}, x0=[0.0])
        progress = GradientProgress()
        result = GradientResult(termination_reason="Converged")

        run = OptimizationRun(
            run_id=1,
            optimizer="scipy:SLSQP",
            optimizer_family="gradient",
            warm_start_from=None,
            n_evaluations=50,
            wall_time=1.0,
            run_success=True,
            best_objective=0.001,
            best_design=[1.0],
            initialization=init,
            progress=progress,
            result=result,
        )

        session = SessionRecord(
            session_id=1,
            problem_id="test_problem",
            created_at="2024-01-01T00:00:00",
            config={},
            runs=[run],
            success=True,
            final_objective=0.001,
            final_design=[1.0],
            total_evaluations=50,
            total_wall_time=1.0,
            decisions=[],
        )

        # Serialize to JSON
        json_str = session.to_json()

        # Deserialize from JSON
        restored = SessionRecord.from_json(json_str)

        assert restored.session_id == 1
        assert len(restored.runs) == 1
        assert restored.runs[0].optimizer == "scipy:SLSQP"


class TestActiveSession:
    """Test ActiveSession lifecycle."""

    def test_active_session_creation(self):
        """Test creating ActiveSession."""
        from paola.foundry.active_session import ActiveSession

        session = ActiveSession(
            session_id=1,
            problem_id="test_problem",
            config={"goal": "minimize"},
        )

        assert session.session_id == 1
        assert session.problem_id == "test_problem"
        assert len(session.runs) == 0
        assert session.current_run is None

    def test_active_session_run_lifecycle(self):
        """Test starting and completing runs in session."""
        from paola.foundry.active_session import ActiveSession
        from paola.foundry.schema import (
            GradientInitialization,
            GradientProgress,
            GradientResult,
        )

        session = ActiveSession(
            session_id=1,
            problem_id="test_problem",
        )

        # Start a run
        init = GradientInitialization(specification={"type": "center"}, x0=[0.0])
        active_run = session.start_run(
            optimizer="scipy:SLSQP",
            initialization=init,
        )

        assert session.current_run is not None
        assert active_run.optimizer == "scipy:SLSQP"

        # Record iterations
        active_run.record_iteration({"iteration": 1, "objective": 10.0, "design": [0.0]})
        active_run.record_iteration({"iteration": 2, "objective": 5.0, "design": [0.5]})

        assert len(active_run.raw_iterations) == 2

        # Complete run
        progress = GradientProgress()
        progress.add_iteration(1, 10.0, [0.0])
        progress.add_iteration(2, 5.0, [0.5])

        result = GradientResult(termination_reason="Converged")

        completed_run = session.complete_run(
            progress=progress,
            result=result,
            best_objective=5.0,
            best_design=[0.5],
            success=True,
        )

        assert len(session.runs) == 1
        assert session.current_run is None
        assert completed_run.best_objective == 5.0

    def test_active_session_finalize(self):
        """Test finalizing session to record."""
        from paola.foundry.active_session import ActiveSession
        from paola.foundry.schema import (
            GradientInitialization,
            GradientProgress,
            GradientResult,
            SessionRecord,
        )

        session = ActiveSession(session_id=1, problem_id="test")

        # Complete one run
        init = GradientInitialization(specification={"type": "center"}, x0=[0.0])
        session.start_run("scipy:SLSQP", init)

        progress = GradientProgress()
        progress.add_iteration(1, 1.0, [1.0])
        result = GradientResult(termination_reason="Done")

        session.complete_run(progress, result, 1.0, [1.0], True)

        # Finalize
        record = session.finalize(success=True)

        assert isinstance(record, SessionRecord)
        assert record.session_id == 1
        assert record.success is True
        assert len(record.runs) == 1


class TestFileStorage:
    """Test FileStorage for sessions."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_session_persistence(self, temp_storage):
        """Test saving and loading sessions."""
        from paola.foundry.storage import FileStorage
        from paola.foundry.schema import (
            SessionRecord,
            OptimizationRun,
            GradientInitialization,
            GradientProgress,
            GradientResult,
        )

        storage = FileStorage(base_dir=temp_storage)

        # Create session record
        init = GradientInitialization(specification={"type": "center"}, x0=[0.0])
        progress = GradientProgress()
        result = GradientResult(termination_reason="Converged")

        run = OptimizationRun(
            run_id=1,
            optimizer="scipy:SLSQP",
            optimizer_family="gradient",
            warm_start_from=None,
            n_evaluations=50,
            wall_time=1.0,
            run_success=True,
            best_objective=0.001,
            best_design=[1.0],
            initialization=init,
            progress=progress,
            result=result,
        )

        session = SessionRecord(
            session_id=1,
            problem_id="test_problem",
            created_at="2024-01-01T00:00:00",
            config={},
            runs=[run],
            success=True,
            final_objective=0.001,
            final_design=[1.0],
            total_evaluations=50,
            total_wall_time=1.0,
            decisions=[],
        )

        # Save
        storage.save_session(session)

        # Load
        loaded = storage.load_session(1)

        assert loaded is not None
        assert loaded.session_id == 1
        assert loaded.problem_id == "test_problem"
        assert len(loaded.runs) == 1


class TestFoundrySessionManagement:
    """Test OptimizationFoundry session management."""

    @pytest.fixture
    def foundry(self):
        """Create foundry with temp storage."""
        temp_dir = tempfile.mkdtemp()
        from paola.foundry import OptimizationFoundry, FileStorage

        storage = FileStorage(base_dir=temp_dir)
        foundry = OptimizationFoundry(storage=storage)
        yield foundry
        shutil.rmtree(temp_dir)

    def test_create_session(self, foundry):
        """Test creating session through foundry."""
        session = foundry.create_session(
            problem_id="test_problem",
            config={"goal": "minimize"},
        )

        assert session.session_id >= 1
        assert session.problem_id == "test_problem"

    def test_get_active_session(self, foundry):
        """Test getting active session."""
        session = foundry.create_session(problem_id="test")

        retrieved = foundry.get_session(session.session_id)
        assert retrieved is session

    def test_finalize_and_load_session(self, foundry):
        """Test finalizing and loading session."""
        from paola.foundry.schema import (
            GradientInitialization,
            GradientProgress,
            GradientResult,
        )

        # Create session
        session = foundry.create_session(problem_id="test")
        session_id = session.session_id

        # Add a run
        init = GradientInitialization(specification={"type": "center"}, x0=[0.0])
        session.start_run("scipy:SLSQP", init)

        progress = GradientProgress()
        progress.add_iteration(1, 1.0, [1.0])
        result = GradientResult(termination_reason="Done")
        session.complete_run(progress, result, 1.0, [1.0], True)

        # Finalize
        record = foundry.finalize_session(session_id, success=True)
        assert record is not None

        # Session should no longer be active
        assert foundry.get_session(session_id) is None

        # Load from storage
        loaded = foundry.load_session(session_id)
        assert loaded is not None
        assert loaded.session_id == session_id


class TestSessionTools:
    """Test session management tools."""

    @pytest.fixture
    def setup_foundry(self):
        """Setup foundry for tools."""
        temp_dir = tempfile.mkdtemp()
        from paola.foundry import OptimizationFoundry, FileStorage
        from paola.tools.session_tools import set_foundry

        storage = FileStorage(base_dir=temp_dir)
        foundry = OptimizationFoundry(storage=storage)
        set_foundry(foundry)

        yield foundry

        shutil.rmtree(temp_dir)

    def test_start_session_tool(self, setup_foundry):
        """Test start_session tool."""
        from paola.tools.session_tools import start_session

        result = start_session.invoke({
            "problem_id": "test_problem",
            "goal": "Minimize test function",
        })

        assert result["success"] is True
        assert "session_id" in result
        assert result["problem_id"] == "test_problem"

    def test_get_active_sessions_tool(self, setup_foundry):
        """Test get_active_sessions tool."""
        from paola.tools.session_tools import start_session, get_active_sessions

        # Start a session
        start_session.invoke({"problem_id": "test1"})
        start_session.invoke({"problem_id": "test2"})

        result = get_active_sessions.invoke({})

        assert result["success"] is True
        assert result["count"] == 2

    def test_finalize_session_tool(self, setup_foundry):
        """Test finalize_session tool."""
        from paola.tools.session_tools import start_session, finalize_session

        # Start a session
        start_result = start_session.invoke({"problem_id": "test"})
        session_id = start_result["session_id"]

        # Finalize
        result = finalize_session.invoke({
            "session_id": session_id,
            "success": True,
            "notes": "Test completed",
        })

        assert result["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
