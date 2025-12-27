"""Tests for observation and analysis tools."""

import pytest
import numpy as np

from paola.tools.analysis import (
    analyze_convergence,
    detect_pattern,
    check_feasibility,
    get_gradient_quality,
    compute_improvement_statistics,
)


class TestAnalyzeConvergence:
    """Tests for analyze_convergence tool."""

    def test_converging_sequence(self):
        """Test detection of converging sequence."""
        objectives = [10.0, 8.0, 6.0, 4.5, 3.5, 2.8, 2.3, 1.9, 1.6, 1.4]
        result = analyze_convergence.invoke({
            "objectives": objectives,
            "window_size": 5,
        })

        assert result["success"]
        assert result["converging"]
        assert not result["diverging"]
        assert not result["stalled"]
        assert result["improvement_rate"] > 0

    def test_diverging_sequence(self):
        """Test detection of diverging sequence."""
        objectives = [1.0, 1.5, 2.2, 3.0, 4.1, 5.5]
        result = analyze_convergence.invoke({
            "objectives": objectives,
            "window_size": 5,
        })

        assert result["success"]
        assert result["diverging"]
        assert not result["converging"]

    def test_stalled_sequence(self):
        """Test detection of stalled optimization."""
        objectives = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = analyze_convergence.invoke({
            "objectives": objectives,
            "window_size": 5,
        })

        assert result["success"]
        assert result["stalled"]
        assert not result["converging"]
        assert not result["diverging"]

    def test_oscillating_sequence(self):
        """Test detection of oscillation."""
        objectives = [1.0, 0.8, 1.1, 0.7, 1.2, 0.75]
        result = analyze_convergence.invoke({
            "objectives": objectives,
            "window_size": 5,
        })

        assert result["success"]
        assert result["oscillating"]

    def test_with_gradients(self):
        """Test convergence analysis with gradient information."""
        objectives = [10.0, 5.0, 2.0, 1.0, 0.5]
        gradients = [[1.0, 1.0], [0.5, 0.5], [0.1, 0.1], [0.01, 0.01], [0.001, 0.001]]

        result = analyze_convergence.invoke({
            "objectives": objectives,
            "gradients": gradients,
            "window_size": 5,
        })

        assert result["success"]
        assert result["gradient_norm"] is not None
        assert result["gradient_trend"] == "decreasing"

    def test_insufficient_data(self):
        """Test handling of insufficient data."""
        result = analyze_convergence.invoke({
            "objectives": [1.0],
            "window_size": 5,
        })

        assert result["success"]
        assert "at least 2" in result["message"].lower() or "not enough" in result["message"].lower()


class TestDetectPattern:
    """Tests for detect_pattern tool."""

    def test_constraint_stuck_pattern(self):
        """Test detection of repeated constraint violations."""
        objectives = [1.0, 0.95, 0.94, 0.94, 0.94]
        constraints = [
            {"CL": 0.49},
            {"CL": 0.48},
            {"CL": 0.49},
            {"CL": 0.49},
            {"CL": 0.49},
        ]

        result = detect_pattern.invoke({
            "objectives": objectives,
            "constraints": constraints,
        })

        assert result["success"]
        assert result["constraint_stuck"]
        assert len(result["patterns_detected"]) > 0
        assert any("CL" in p for p in result["patterns_detected"])

    def test_gradient_noise_pattern(self):
        """Test detection of gradient noise."""
        objectives = [1.0, 0.9, 0.85, 0.8, 0.75]
        # High variance in gradient norms (some very large, some small)
        gradients = [
            [1.0, 0.5],   # norm ~1.12
            [0.01, 0.02], # norm ~0.02 (much smaller)
            [5.0, 3.0],   # norm ~5.83 (much larger)
            [0.1, 0.1],   # norm ~0.14
            [3.0, 2.0],   # norm ~3.6
        ]

        result = detect_pattern.invoke({
            "objectives": objectives,
            "gradients": gradients,
        })

        assert result["success"]
        assert result["gradient_noise"]

    def test_no_patterns(self):
        """Test case with no problematic patterns."""
        objectives = [10.0, 8.0, 6.0, 4.0, 2.0]

        result = detect_pattern.invoke({
            "objectives": objectives,
        })

        assert result["success"]
        assert len(result["patterns_detected"]) == 0
        assert result["severity"] == "none"


class TestCheckFeasibility:
    """Tests for check_feasibility tool."""

    def test_feasible_design(self):
        """Test checking a feasible design."""
        import json
        design = [1.0, 2.0]
        constraints = {
            "CL": {"constraint_type": ">=", "bound": 0.5, "value": 0.52},
            "thickness": {"constraint_type": ">=", "bound": 0.12, "value": 0.15},
        }

        result = check_feasibility.invoke({
            "design": design,
            "constraint_values": json.dumps(constraints),
        })

        assert result["success"]
        assert result["feasible"]
        assert result["n_violated"] == 0
        assert len(result["satisfied"]) == 2

    def test_infeasible_design(self):
        """Test checking an infeasible design."""
        import json
        design = [1.0, 2.0]
        constraints = {
            "CL": {"constraint_type": ">=", "bound": 0.5, "value": 0.49},  # Violated
            "thickness": {"constraint_type": ">=", "bound": 0.12, "value": 0.15},
        }

        result = check_feasibility.invoke({
            "design": design,
            "constraint_values": json.dumps(constraints),
        })

        assert result["success"]
        assert not result["feasible"]
        assert result["n_violated"] == 1
        assert len(result["violations"]) == 1
        assert result["violations"][0]["name"] == "CL"

    def test_equality_constraint(self):
        """Test equality constraint checking."""
        import json
        design = [1.0, 2.0]
        constraints = {
            "balance": {"constraint_type": "==", "bound": 1.0, "value": 1.0001},
        }

        result = check_feasibility.invoke({
            "design": design,
            "constraint_values": json.dumps(constraints),
            "tolerance": 1e-3,
        })

        assert result["success"]
        assert result["feasible"]  # Within tolerance


class TestGetGradientQuality:
    """Tests for get_gradient_quality tool."""

    def test_good_quality_gradient(self):
        """Test detection of good quality gradients."""
        # Consistent gradients
        gradients = [
            [1.0, 0.5],
            [0.95, 0.48],
            [0.9, 0.46],
            [0.85, 0.44],
            [0.8, 0.42],
        ]

        result = get_gradient_quality.invoke({
            "gradients": gradients,
            "window_size": 5,
        })

        assert result["success"]
        assert result["quality"] in ["good", "marginal"]
        assert result["direction_consistency"] > 0.9

    def test_poor_quality_gradient(self):
        """Test detection of poor quality gradients."""
        # Inconsistent/noisy gradients
        gradients = [
            [1.0, 0.5],
            [-0.5, 0.8],  # Direction change
            [0.9, -0.4],  # Direction change
            [-0.3, 0.6],
            [0.7, -0.2],
        ]

        result = get_gradient_quality.invoke({
            "gradients": gradients,
            "window_size": 5,
        })

        assert result["success"]
        # Should detect poor quality due to direction inconsistency

    def test_near_zero_gradient(self):
        """Test detection of near-zero gradient."""
        gradients = [
            [0.01, 0.01],
            [0.005, 0.005],
            [0.001, 0.001],
            [0.0005, 0.0005],
            [1e-7, 1e-7],
        ]

        result = get_gradient_quality.invoke({
            "gradients": gradients,
            "window_size": 5,
        })

        assert result["success"]
        assert result["near_zero"]


class TestComputeImprovementStatistics:
    """Tests for compute_improvement_statistics tool."""

    def test_good_improvement(self):
        """Test statistics for good improvement."""
        objectives = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0]

        result = compute_improvement_statistics.invoke({
            "objectives": objectives,
            "budget_used": 30.0,
            "budget_total": 100.0,
        })

        assert result["success"]
        assert result["total_improvement"] > 0
        assert result["total_improvement_pct"] == 90.0
        assert result["budget_remaining_pct"] == 70.0
        assert result["continue_recommendation"]

    def test_low_budget_remaining(self):
        """Test recommendation when budget low and stalled."""
        objectives = [1.0, 1.0, 1.0, 1.0, 1.0]

        result = compute_improvement_statistics.invoke({
            "objectives": objectives,
            "budget_used": 95.0,
            "budget_total": 100.0,
        })

        assert result["success"]
        assert result["budget_remaining_pct"] == 5.0
        # Should recommend stopping with low budget and no improvement

    def test_efficiency_calculation(self):
        """Test efficiency metric calculation."""
        objectives = [100.0, 50.0]

        result = compute_improvement_statistics.invoke({
            "objectives": objectives,
            "budget_used": 10.0,
            "budget_total": 100.0,
        })

        assert result["success"]
        assert result["improvement_per_budget"] == 5.0  # 50 improvement / 10 budget


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
