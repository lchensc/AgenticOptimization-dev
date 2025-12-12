"""Tests for gate control tools."""

import pytest
import numpy as np

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
from paola.optimizers.gate import OptimizationGate


class TestGateControlTools:
    """Tests for gate control tools."""

    def setup_method(self):
        """Setup for each test."""
        clear_gate_registry()

    def test_gate_registration(self):
        """Test gate registration and retrieval."""
        gate = OptimizationGate(problem_id="test_prob", blocking=True)
        register_gate("gate_1", gate)

        retrieved = get_gate_by_id("gate_1")
        assert retrieved is not None
        assert retrieved.problem_id == "test_prob"

    def test_gate_not_found(self):
        """Test error when gate not found."""
        result = gate_continue.invoke({"gate_id": "nonexistent"})
        assert not result["success"]
        assert "not found" in result["message"]

    def test_gate_continue_non_blocking(self):
        """Test that gate_continue warns for non-blocking gates."""
        gate = OptimizationGate(problem_id="test_prob", blocking=False)
        register_gate("gate_nb", gate)

        result = gate_continue.invoke({"gate_id": "gate_nb"})
        assert not result["success"]
        assert "non-blocking" in result["message"]

    def test_gate_stop(self):
        """Test gate_stop tool."""
        gate = OptimizationGate(problem_id="test_prob", blocking=True)

        # Add some history
        gate.history.append({
            "iteration": 0,
            "objective": 10.0,
            "design": np.array([1.0, 2.0]),
        })
        gate.history.append({
            "iteration": 1,
            "objective": 5.0,
            "design": np.array([1.2, 1.8]),
        })
        gate.iteration = 2

        register_gate("gate_stop", gate)

        result = gate_stop.invoke({
            "gate_id": "gate_stop",
            "reason": "Test stop reason"
        })

        assert result["success"]
        assert result["final_iteration"] == 2
        assert result["best_objective"] == 5.0
        assert result["reason"] == "Test stop reason"

    def test_gate_restart_from(self):
        """Test gate_restart_from tool."""
        gate = OptimizationGate(problem_id="test_prob", blocking=True)
        gate.iteration = 10
        register_gate("gate_restart", gate)

        result = gate_restart_from.invoke({
            "gate_id": "gate_restart",
            "restart_design": [1.5, 2.5],
            "reason": "Constraint violation detected",
            "new_options": '{"ftol": 1e-8}',
        })

        assert result["success"]
        assert result["restart_from"] == [1.5, 2.5]
        assert result["iteration_at_restart"] == 10
        assert result["reason"] == "Constraint violation detected"

    def test_gate_restart_invalid_json(self):
        """Test gate_restart_from with invalid JSON options."""
        gate = OptimizationGate(problem_id="test_prob", blocking=True)
        register_gate("gate_bad_json", gate)

        result = gate_restart_from.invoke({
            "gate_id": "gate_bad_json",
            "restart_design": [1.0, 2.0],
            "reason": "Test",
            "new_options": "not valid json",
        })

        assert not result["success"]
        assert "Invalid" in result["message"]

    def test_gate_get_history(self):
        """Test gate_get_history tool."""
        gate = OptimizationGate(problem_id="test_prob", blocking=False)

        # Add history
        for i in range(10):
            gate.history.append({
                "iteration": i,
                "objective": 10.0 - i,
                "design": np.array([float(i), float(i + 1)]),
            })
        gate.iteration = 10

        register_gate("gate_hist", gate)

        # Get all history
        result = gate_get_history.invoke({"gate_id": "gate_hist"})
        assert result["success"]
        assert len(result["history"]) == 10
        assert result["total_iterations"] == 10
        assert result["best_objective"] == 1.0

        # Get last 3
        result = gate_get_history.invoke({
            "gate_id": "gate_hist",
            "last_n": 3,
        })
        assert result["success"]
        assert len(result["history"]) == 3

    def test_gate_get_statistics(self):
        """Test gate_get_statistics tool."""
        gate = OptimizationGate(problem_id="test_prob", blocking=False)
        gate.iteration = 25
        gate.n_pauses = 0
        gate.total_wait_time = 0.0

        register_gate("gate_stats", gate)

        result = gate_get_statistics.invoke({"gate_id": "gate_stats"})
        assert result["success"]
        assert result["total_iterations"] == 25
        assert result["blocking_mode"] is False
        assert result["n_pauses"] == 0


class TestGateIntegration:
    """Integration tests for gate tools with actual optimization."""

    def setup_method(self):
        """Setup for each test."""
        clear_gate_registry()

    def test_gate_with_scipy_optimization(self):
        """Test gate tools with actual scipy optimization."""
        import numpy as np
        from scipy.optimize import minimize
        from paola.backends.analytical import Sphere

        # Setup
        problem = Sphere(dimension=2)
        gate = OptimizationGate(problem_id="sphere_test", blocking=False)
        register_gate("gate_scipy", gate)

        # Run optimization through gate
        result = minimize(
            fun=gate.wrap_objective(problem.evaluate),
            x0=np.array([3.0, 4.0]),
            method='BFGS',
            jac=gate.wrap_gradient(problem.gradient),
        )

        # Use tools to analyze
        history_result = gate_get_history.invoke({"gate_id": "gate_scipy"})
        assert history_result["success"]
        assert len(history_result["history"]) > 0
        assert history_result["best_objective"] < 0.01  # Should be near 0

        stats_result = gate_get_statistics.invoke({"gate_id": "gate_scipy"})
        assert stats_result["success"]
        assert stats_result["total_iterations"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
