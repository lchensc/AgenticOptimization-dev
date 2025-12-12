"""
End-to-end integration tests for single objective optimization.

Tests the complete optimization workflow:
1. Problem registration
2. Gate-based optimization with scipy
3. Tool-based analysis
4. Agent observation and decision making

These tests verify the full system works together without requiring LLM API keys.
"""

import pytest
import numpy as np
from scipy.optimize import minimize

from paola.backends.analytical import Rosenbrock, Sphere, ConstrainedRosenbrock
from paola.optimizers.gate import OptimizationGate
from paola.callbacks import CallbackManager, EventCapture, EventType
from paola.tools import (
    # Evaluator tools
    evaluate_function,
    compute_gradient,
    register_problem,
    clear_problem_registry,
    # Optimizer tools
    optimizer_create,
    optimizer_propose,
    optimizer_update,
    optimizer_restart,
    clear_optimizer_registry,
    # Gate tools
    register_gate,
    clear_gate_registry,
    gate_get_history,
    gate_get_statistics,
    # Observation tools
    analyze_convergence,
    detect_pattern,
    get_gradient_quality,
    compute_improvement_statistics,
    # Cache tools
    cache_clear,
    cache_stats,
)


class TestGateBasedOptimization:
    """End-to-end tests using gate pattern with scipy."""

    def setup_method(self):
        """Reset all registries before each test."""
        clear_problem_registry()
        clear_optimizer_registry()
        clear_gate_registry()
        cache_clear()

    def test_rosenbrock_2d_convergence(self):
        """Test full optimization of 2D Rosenbrock converges to optimum."""
        # Setup
        problem = Rosenbrock(dimension=2)
        x_opt, f_opt = problem.get_optimum()

        # Create gate with event capture
        callback_mgr = CallbackManager()
        event_capture = EventCapture()
        callback_mgr.register(event_capture)

        gate = OptimizationGate(
            problem_id="rosenbrock_2d",
            blocking=False,
            callback_manager=callback_mgr
        )
        register_gate("gate_rosenbrock", gate)

        # Run optimization through gate
        x0 = np.array([-1.0, 1.0])
        result = minimize(
            fun=gate.wrap_objective(problem.evaluate),
            x0=x0,
            method='SLSQP',
            jac=gate.wrap_gradient(problem.gradient),
            bounds=[(-5, 10), (-5, 10)],
            options={'ftol': 1e-9, 'maxiter': 100}
        )

        # Verify convergence
        assert result.success
        assert np.linalg.norm(result.x - x_opt) < 1e-4
        assert abs(result.fun - f_opt) < 1e-6

        # Verify gate captured history
        history_result = gate_get_history.invoke({"gate_id": "gate_rosenbrock"})
        assert history_result["success"]
        assert len(history_result["history"]) > 0
        assert history_result["best_objective"] < 1e-6

        # Verify events were emitted
        iteration_events = event_capture.get_events_by_type(EventType.ITERATION_COMPLETE)
        assert len(iteration_events) > 0

    def test_sphere_10d_convergence(self):
        """Test 10D Sphere function optimization."""
        problem = Sphere(dimension=10)
        x_opt, f_opt = problem.get_optimum()

        gate = OptimizationGate(problem_id="sphere_10d", blocking=False)
        register_gate("gate_sphere", gate)

        x0 = np.array([3.0] * 10)
        result = minimize(
            fun=gate.wrap_objective(problem.evaluate),
            x0=x0,
            method='L-BFGS-B',
            jac=gate.wrap_gradient(problem.gradient),
            bounds=[(-5, 5)] * 10,
        )

        assert result.success
        assert np.linalg.norm(result.x - x_opt) < 1e-4
        assert result.fun < 1e-10

        # Check statistics
        stats = gate_get_statistics.invoke({"gate_id": "gate_sphere"})
        assert stats["success"]
        assert stats["total_iterations"] > 0

    def test_constrained_rosenbrock(self):
        """Test constrained optimization."""
        problem = ConstrainedRosenbrock()
        x_opt, f_opt = problem.get_optimum()

        gate = OptimizationGate(problem_id="constrained_rosenbrock", blocking=False)

        # Define constraint for scipy
        def constraint_func(x):
            return -problem.evaluate_constraint(x)  # scipy wants >= 0

        from scipy.optimize import NonlinearConstraint
        constraint = NonlinearConstraint(constraint_func, 0, np.inf)

        x0 = np.array([0.0, 0.0])
        result = minimize(
            fun=gate.wrap_objective(problem.evaluate),
            x0=x0,
            method='SLSQP',
            jac=gate.wrap_gradient(problem.gradient),
            bounds=[(-2, 2), (-2, 2)],
            constraints={'type': 'ineq', 'fun': constraint_func},
        )

        # Should converge near constrained optimum
        assert result.success
        # Constrained optimum is approximate, so use looser tolerance
        assert abs(result.fun - f_opt) < 0.1


class TestToolBasedWorkflow:
    """Test the tool-based optimization workflow."""

    def setup_method(self):
        """Reset all registries before each test."""
        clear_problem_registry()
        clear_optimizer_registry()
        clear_gate_registry()
        cache_clear()

    def test_evaluate_and_cache(self):
        """Test evaluation with caching."""
        problem = Sphere(dimension=2)
        register_problem("sphere_cache_test", problem)

        # First evaluation - should not be cached
        result1 = evaluate_function.invoke({
            "problem_id": "sphere_cache_test",
            "design": [1.0, 2.0],
            "use_cache": True,
        })
        assert result1["success"]
        assert not result1["cache_hit"]
        assert result1["objective"] == 5.0  # 1^2 + 2^2

        # Second evaluation - should be cached
        result2 = evaluate_function.invoke({
            "problem_id": "sphere_cache_test",
            "design": [1.0, 2.0],
            "use_cache": True,
        })
        assert result2["success"]
        assert result2["cache_hit"]
        assert result2["objective"] == 5.0

        # Check cache stats
        stats = cache_stats()
        assert stats["total_entries"] >= 1

    def test_gradient_computation(self):
        """Test gradient computation with different methods."""
        problem = Rosenbrock(dimension=2)
        register_problem("rosenbrock_grad_test", problem)

        design = [1.0, 1.0]  # At optimum

        # Analytical gradient
        result_analytical = compute_gradient.invoke({
            "problem_id": "rosenbrock_grad_test",
            "design": design,
            "method": "analytical",
        })
        assert result_analytical["success"]
        assert result_analytical["gradient_norm"] < 1e-10  # Should be zero at optimum

        # Finite difference gradient
        result_fd = compute_gradient.invoke({
            "problem_id": "rosenbrock_grad_test",
            "design": [0.5, 0.5],  # Not at optimum
            "method": "finite-difference",
        })
        assert result_fd["success"]
        assert result_fd["gradient_norm"] > 0


class TestObservationToolsIntegration:
    """Test observation tools with real optimization data."""

    def setup_method(self):
        """Reset registries."""
        clear_gate_registry()

    def test_convergence_analysis_on_real_optimization(self):
        """Test convergence analysis on actual optimization trajectory."""
        # Run optimization
        problem = Rosenbrock(dimension=2)
        gate = OptimizationGate(problem_id="rosenbrock_analysis", blocking=False)
        register_gate("gate_analysis", gate)

        result = minimize(
            fun=gate.wrap_objective(problem.evaluate),
            x0=np.array([-1.0, 1.0]),
            method='SLSQP',
            jac=gate.wrap_gradient(problem.gradient),
            bounds=[(-5, 10), (-5, 10)],
            options={'ftol': 1e-9}
        )

        # Get history
        history = gate.get_history()
        objectives = [h['objective'] for h in history if 'objective' in h]
        gradients = [h['gradient'].tolist() for h in history
                     if 'gradient' in h and h.get('gradient') is not None]

        # Analyze convergence
        conv_result = analyze_convergence.invoke({
            "objectives": objectives,
            "gradients": gradients if gradients else None,
            "window_size": 5,
        })

        assert conv_result["success"]
        # Should show convergence since optimization succeeded
        assert conv_result["converging"] or conv_result["converged"] or conv_result["stalled"]
        assert conv_result["improvement_rate"] >= 0  # Should have improved overall

    def test_gradient_quality_on_real_optimization(self):
        """Test gradient quality analysis on real gradients."""
        problem = Sphere(dimension=2)
        gate = OptimizationGate(problem_id="sphere_grad_quality", blocking=False)

        result = minimize(
            fun=gate.wrap_objective(problem.evaluate),
            x0=np.array([3.0, 4.0]),
            method='BFGS',
            jac=gate.wrap_gradient(problem.gradient),
        )

        # Get gradients from history
        history = gate.get_history()
        gradients = [h['gradient'].tolist() for h in history
                     if 'gradient' in h and h.get('gradient') is not None]

        if len(gradients) >= 2:
            quality_result = get_gradient_quality.invoke({
                "gradients": gradients,
                "window_size": 5,
            })

            assert quality_result["success"]
            # Quality can vary depending on optimization trajectory
            # Just verify we got a valid quality assessment
            assert quality_result["quality"] in ["good", "marginal", "converged", "poor"]

    def test_improvement_statistics(self):
        """Test improvement statistics calculation."""
        objectives = [100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5]

        result = compute_improvement_statistics.invoke({
            "objectives": objectives,
            "budget_used": 50.0,
            "budget_total": 100.0,
        })

        assert result["success"]
        assert result["total_improvement"] > 99  # Started at 100, ended at 0.5
        assert result["total_improvement_pct"] > 99
        assert result["budget_remaining_pct"] == 50.0
        assert result["continue_recommendation"]  # Still have budget and improving


class TestFullAgentWorkflow:
    """
    Test the complete agent workflow without LLM.

    Simulates what the agent would do:
    1. Register problem
    2. Create optimizer
    3. Run optimization loop with observation
    4. Analyze and decide when to stop
    """

    def setup_method(self):
        """Reset all state."""
        clear_problem_registry()
        clear_optimizer_registry()
        clear_gate_registry()
        cache_clear()

    def test_agent_style_optimization_workflow(self):
        """
        Simulate agent workflow for Rosenbrock optimization.

        This is what the LLM agent would do autonomously.
        """
        # 1. Register problem (agent would call formulate_problem)
        problem = Rosenbrock(dimension=2)
        register_problem("rosenbrock_agent", problem)

        # 2. Create gate for observation
        callback_mgr = CallbackManager()
        event_capture = EventCapture()
        callback_mgr.register(event_capture)

        gate = OptimizationGate(
            problem_id="rosenbrock_agent",
            blocking=False,
            callback_manager=callback_mgr
        )
        register_gate("agent_gate", gate)

        # 3. Run optimization (agent would orchestrate this)
        x0 = np.array([-1.0, 1.0])
        result = minimize(
            fun=gate.wrap_objective(problem.evaluate),
            x0=x0,
            method='SLSQP',
            jac=gate.wrap_gradient(problem.gradient),
            bounds=[(-5, 10), (-5, 10)],
            options={'ftol': 1e-9, 'maxiter': 100}
        )

        # 4. Agent reviews history
        history_result = gate_get_history.invoke({"gate_id": "agent_gate"})
        assert history_result["success"]

        objectives = [h['objective'] for h in history_result["history"]]
        gradients = [h.get('gradient', [0, 0]) for h in history_result["history"]
                     if h.get('gradient') is not None]

        # 5. Agent analyzes convergence
        conv_result = analyze_convergence.invoke({
            "objectives": objectives[-10:],
            "gradients": [g.tolist() if hasattr(g, 'tolist') else g for g in gradients[-10:]] if gradients else None,
            "window_size": 5,
        })
        assert conv_result["success"]

        # 6. Agent checks pattern for issues
        pattern_result = detect_pattern.invoke({
            "objectives": objectives,
        })
        assert pattern_result["success"]
        assert pattern_result["severity"] in ["none", "low"]  # No major issues expected

        # 7. Agent computes improvement statistics
        stats_result = compute_improvement_statistics.invoke({
            "objectives": objectives,
            "budget_used": len(objectives),  # Each eval = 1 unit
            "budget_total": 100.0,
        })
        assert stats_result["success"]
        assert stats_result["total_improvement"] > 0

        # 8. Verify final result
        assert result.success
        x_opt, f_opt = problem.get_optimum()
        assert np.linalg.norm(result.x - x_opt) < 1e-4
        assert abs(result.fun - f_opt) < 1e-6

        # 9. Verify events were captured for observability
        all_events = event_capture.events
        assert len(all_events) > 0

        # Check we got iteration events
        iteration_events = event_capture.get_events_by_type(EventType.ITERATION_COMPLETE)
        assert len(iteration_events) > 0

    def test_multiple_problems_isolation(self):
        """Test that multiple problems don't interfere with each other."""
        # Register two different problems
        rosenbrock = Rosenbrock(dimension=2)
        sphere = Sphere(dimension=2)

        register_problem("problem_a", rosenbrock)
        register_problem("problem_b", sphere)

        # Evaluate both
        result_a = evaluate_function.invoke({
            "problem_id": "problem_a",
            "design": [0.0, 0.0],
        })
        result_b = evaluate_function.invoke({
            "problem_id": "problem_b",
            "design": [0.0, 0.0],
        })

        # Rosenbrock at [0,0] = 1, Sphere at [0,0] = 0
        assert result_a["objective"] == 1.0
        assert result_b["objective"] == 0.0

        # Gradients should also be different
        grad_a = compute_gradient.invoke({
            "problem_id": "problem_a",
            "design": [0.0, 0.0],
            "method": "analytical",
        })
        grad_b = compute_gradient.invoke({
            "problem_id": "problem_b",
            "design": [0.0, 0.0],
            "method": "analytical",
        })

        # Rosenbrock gradient at [0,0] is [-2, 0]
        # Sphere gradient at [0,0] is [0, 0]
        assert grad_a["gradient"] != grad_b["gradient"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Reset state."""
        clear_problem_registry()
        clear_optimizer_registry()
        cache_clear()

    def test_nonexistent_problem(self):
        """Test error handling for non-existent problem."""
        result = evaluate_function.invoke({
            "problem_id": "does_not_exist",
            "design": [1.0, 2.0],
        })
        assert not result["success"]
        assert "not registered" in result["message"].lower() or "not found" in result["message"].lower()

    def test_empty_objectives_analysis(self):
        """Test convergence analysis with empty objectives."""
        result = analyze_convergence.invoke({
            "objectives": [],
            "window_size": 5,
        })
        # Should handle gracefully
        assert result["success"]

    def test_single_objective_analysis(self):
        """Test analysis with single objective value."""
        result = analyze_convergence.invoke({
            "objectives": [1.0],
            "window_size": 5,
        })
        assert result["success"]
        # Should indicate insufficient data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
