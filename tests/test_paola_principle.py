"""
Tests for Paola Principle implementation.

Tests cover:
- BoundsSpec: Compact bounds specification
- InitializationManager: Agent intelligence for initialization
- ConfigurationManager: Agent intelligence for algorithm selection
- Intent-based run_optimization tool
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestBoundsSpec:
    """Test BoundsSpec for compact bounds specification."""

    def test_uniform_bounds_creation(self):
        """Test creating uniform bounds."""
        from paola.foundry.bounds_spec import BoundsSpec

        spec = BoundsSpec.uniform(lower=-0.05, upper=0.05, dimension=100)

        assert spec.spec_type == "uniform"
        assert spec.lower == -0.05
        assert spec.upper == 0.05
        assert spec.dimension == 100
        assert spec.get_dimension() == 100

    def test_uniform_bounds_expansion(self):
        """Test expanding uniform bounds to explicit list."""
        from paola.foundry.bounds_spec import BoundsSpec

        spec = BoundsSpec.uniform(lower=-0.05, upper=0.05, dimension=3)
        bounds = spec.expand()

        assert len(bounds) == 3
        assert bounds[0] == [-0.05, 0.05]
        assert bounds[1] == [-0.05, 0.05]
        assert bounds[2] == [-0.05, 0.05]

    def test_grouped_bounds(self):
        """Test grouped bounds specification."""
        from paola.foundry.bounds_spec import BoundsSpec, BoundsGroup

        spec = BoundsSpec(
            spec_type="grouped",
            groups={
                "control_points": BoundsGroup(-0.05, 0.05, 80),
                "angles": BoundsGroup(-15.0, 15.0, 20)
            }
        )

        assert spec.get_dimension() == 100
        bounds = spec.expand()
        assert len(bounds) == 100
        assert bounds[0] == [-0.05, 0.05]  # First control point
        assert bounds[79] == [-0.05, 0.05]  # Last control point
        assert bounds[80] == [-15.0, 15.0]  # First angle
        assert bounds[99] == [-15.0, 15.0]  # Last angle

    def test_explicit_bounds(self):
        """Test explicit bounds passthrough."""
        from paola.foundry.bounds_spec import BoundsSpec

        explicit = [[0, 1], [0, 2], [-1, 1]]
        spec = BoundsSpec.from_explicit(explicit)

        assert spec.spec_type == "explicit"
        assert spec.expand() == explicit
        assert spec.get_dimension() == 3

    def test_parse_bounds_input_list(self):
        """Test parsing list input."""
        from paola.foundry.bounds_spec import parse_bounds_input

        bounds = [[0, 1], [0, 2]]
        spec = parse_bounds_input(bounds)

        assert spec.spec_type == "explicit"
        assert spec.expand() == bounds

    def test_parse_bounds_input_dict(self):
        """Test parsing dict input."""
        from paola.foundry.bounds_spec import parse_bounds_input

        bounds_dict = {"type": "uniform", "lower": -1, "upper": 1, "dimension": 50}
        spec = parse_bounds_input(bounds_dict)

        assert spec.spec_type == "uniform"
        assert spec.get_dimension() == 50

    def test_bounds_center(self):
        """Test getting center of bounds."""
        from paola.foundry.bounds_spec import BoundsSpec

        spec = BoundsSpec.uniform(lower=0, upper=10, dimension=3)
        center = spec.get_center()

        np.testing.assert_array_equal(center, [5, 5, 5])

    def test_bounds_width(self):
        """Test getting width of bounds."""
        from paola.foundry.bounds_spec import BoundsSpec

        spec = BoundsSpec.uniform(lower=0, upper=10, dimension=3)
        width = spec.get_width()

        np.testing.assert_array_equal(width, [10, 10, 10])

    def test_bounds_validation(self):
        """Test bounds validation."""
        from paola.foundry.bounds_spec import BoundsSpec

        # Lower >= upper should fail
        with pytest.raises(ValueError):
            BoundsSpec.uniform(lower=10, upper=0, dimension=5)

        # Zero dimension should fail
        with pytest.raises(ValueError):
            BoundsSpec.uniform(lower=0, upper=1, dimension=0)


class TestNLPProblemSchema:
    """Test updated NLPProblem schema with domain_hint."""

    def test_nlp_problem_without_initial_point(self):
        """Test that NLPProblem no longer requires initial_point."""
        from paola.foundry.nlp_schema import NLPProblem

        problem = NLPProblem(
            problem_id="test",
            objective_evaluator_id="obj_eval",
            dimension=2,
            bounds=[[0, 1], [0, 1]]
        )

        # Should not have initial_point attribute
        assert not hasattr(problem, "initial_point")
        assert problem.dimension == 2

    def test_nlp_problem_with_domain_hint(self):
        """Test NLPProblem with domain_hint."""
        from paola.foundry.nlp_schema import NLPProblem

        problem = NLPProblem(
            problem_id="wing",
            objective_evaluator_id="drag_eval",
            dimension=2,
            bounds=[[0, 1], [0, 1]],
            domain_hint="shape_optimization"
        )

        assert problem.domain_hint == "shape_optimization"

    def test_nlp_problem_from_bounds_spec(self):
        """Test creating NLPProblem from BoundsSpec."""
        from paola.foundry.nlp_schema import NLPProblem

        problem = NLPProblem.from_bounds_spec(
            problem_id="ffd_wing",
            objective_evaluator_id="drag_eval",
            bounds_spec={"type": "uniform", "lower": -0.05, "upper": 0.05, "dimension": 100},
            domain_hint="shape_optimization"
        )

        assert problem.dimension == 100
        assert problem.domain_hint == "shape_optimization"
        assert len(problem.bounds) == 100
        assert problem.bounds[0] == [-0.05, 0.05]

    def test_nlp_problem_get_bounds_center(self):
        """Test get_bounds_center method."""
        from paola.foundry.nlp_schema import NLPProblem

        problem = NLPProblem(
            problem_id="test",
            objective_evaluator_id="obj",
            dimension=3,
            bounds=[[0, 10], [0, 20], [-5, 5]]
        )

        center = problem.get_bounds_center()
        assert center == [5, 10, 0]

    def test_nlp_problem_backward_compatibility(self):
        """Test backward compatibility with initial_point in dict."""
        from paola.foundry.nlp_schema import NLPProblem

        # Old format with initial_point
        data = {
            "problem_id": "test",
            "objective_evaluator_id": "obj",
            "dimension": 2,
            "bounds": [[0, 1], [0, 1]],
            "initial_point": [0.5, 0.5]  # Should be ignored
        }

        problem = NLPProblem.from_dict(data)
        assert problem.problem_id == "test"
        # initial_point should be stripped out


class TestInitializationManager:
    """Test InitializationManager for agent initialization decisions."""

    def test_gradient_based_center_init(self):
        """Test gradient-based algorithms use center of bounds."""
        from paola.agent.initialization import InitializationManager
        from paola.foundry.nlp_schema import NLPProblem

        manager = InitializationManager()
        problem = NLPProblem(
            problem_id="test",
            objective_evaluator_id="obj",
            dimension=3,
            bounds=[[0, 10], [0, 20], [-5, 5]]
        )

        x0 = manager.compute_initial_point(problem, "SLSQP")

        np.testing.assert_array_equal(x0, [5, 10, 0])

    def test_shape_optimization_zero_init(self):
        """Test shape optimization domain uses zero initialization."""
        from paola.agent.initialization import InitializationManager
        from paola.foundry.nlp_schema import NLPProblem

        manager = InitializationManager()
        problem = NLPProblem(
            problem_id="wing_ffd",
            objective_evaluator_id="drag",
            dimension=100,
            bounds=[[-0.05, 0.05]] * 100,
            domain_hint="shape_optimization"
        )

        x0 = manager.compute_initial_point(problem, "SLSQP")

        np.testing.assert_array_equal(x0, np.zeros(100))

    def test_bayesian_returns_none(self):
        """Test Bayesian algorithms return None (sampler handles)."""
        from paola.agent.initialization import InitializationManager
        from paola.foundry.nlp_schema import NLPProblem

        manager = InitializationManager()
        problem = NLPProblem(
            problem_id="test",
            objective_evaluator_id="obj",
            dimension=3,
            bounds=[[0, 10], [0, 20], [-5, 5]]
        )

        x0 = manager.compute_initial_point(problem, "TPE")

        assert x0 is None

    def test_cmaes_params(self):
        """Test CMA-ES parameter computation."""
        from paola.agent.initialization import InitializationManager
        from paola.foundry.nlp_schema import NLPProblem

        manager = InitializationManager()
        problem = NLPProblem(
            problem_id="test",
            objective_evaluator_id="obj",
            dimension=3,
            bounds=[[0, 10], [0, 20], [-5, 5]]  # widths: 10, 20, 10
        )

        mean, sigma = manager.compute_cmaes_params(problem)

        np.testing.assert_array_equal(mean, [5, 10, 0])
        # sigma ≈ 0.25 × mean(widths) = 0.25 × (10+20+10)/3 ≈ 3.33
        assert 3.0 < sigma < 4.0

    def test_population_generation_lhs(self):
        """Test Latin Hypercube Sampling for population."""
        from paola.agent.initialization import InitializationManager
        from paola.foundry.nlp_schema import NLPProblem

        manager = InitializationManager()
        problem = NLPProblem(
            problem_id="test",
            objective_evaluator_id="obj",
            dimension=3,
            bounds=[[0, 10], [0, 10], [0, 10]]
        )

        population = manager.generate_population(problem, size=50, method="lhs")

        assert population.shape == (50, 3)
        # All values should be within bounds
        assert np.all(population >= 0)
        assert np.all(population <= 10)

    def test_force_strategy(self):
        """Test forcing a specific initialization strategy."""
        from paola.agent.initialization import InitializationManager
        from paola.foundry.nlp_schema import NLPProblem

        manager = InitializationManager()
        problem = NLPProblem(
            problem_id="test",
            objective_evaluator_id="obj",
            dimension=3,
            bounds=[[0, 10], [0, 10], [0, 10]]
        )

        x0 = manager.compute_initial_point(problem, "SLSQP", force_strategy="zero")

        np.testing.assert_array_equal(x0, np.zeros(3))


class TestConfigurationManager:
    """Test ConfigurationManager for algorithm selection and configuration."""

    def test_algorithm_selection_constrained(self):
        """Test algorithm selection for constrained problems."""
        from paola.agent.configuration import ConfigurationManager
        from paola.foundry.nlp_schema import NLPProblem, InequalityConstraint

        manager = ConfigurationManager()
        problem = NLPProblem(
            problem_id="test",
            objective_evaluator_id="obj",
            dimension=3,
            bounds=[[0, 10], [0, 10], [0, 10]],
            inequality_constraints=[
                InequalityConstraint("c1", "cons_eval", "<=", 0)
            ]
        )

        algorithm = manager.select_algorithm(problem, priority="balanced")

        assert algorithm == "SLSQP"  # Default for constrained

    def test_algorithm_selection_unconstrained(self):
        """Test algorithm selection for unconstrained problems."""
        from paola.agent.configuration import ConfigurationManager
        from paola.foundry.nlp_schema import NLPProblem

        manager = ConfigurationManager()
        problem = NLPProblem(
            problem_id="test",
            objective_evaluator_id="obj",
            dimension=3,
            bounds=[[0, 10], [0, 10], [0, 10]]
        )

        algorithm = manager.select_algorithm(problem, priority="balanced")

        assert algorithm == "L-BFGS-B"  # Default for unconstrained

    def test_configuration_by_priority(self):
        """Test configuration varies by priority."""
        from paola.agent.configuration import ConfigurationManager
        from paola.foundry.nlp_schema import NLPProblem

        manager = ConfigurationManager()
        problem = NLPProblem(
            problem_id="test",
            objective_evaluator_id="obj",
            dimension=3,
            bounds=[[0, 10], [0, 10], [0, 10]]
        )

        config_robust = manager.configure_algorithm("SLSQP", problem, "robustness")
        config_speed = manager.configure_algorithm("SLSQP", problem, "speed")

        # Robustness should have more iterations
        assert config_robust["options"]["maxiter"] > config_speed["options"]["maxiter"]
        # Speed should have looser tolerance
        assert config_speed["options"]["ftol"] > config_robust["options"]["ftol"]

    def test_max_iterations_override(self):
        """Test max iterations override."""
        from paola.agent.configuration import ConfigurationManager
        from paola.foundry.nlp_schema import NLPProblem

        manager = ConfigurationManager()
        problem = NLPProblem(
            problem_id="test",
            objective_evaluator_id="obj",
            dimension=3,
            bounds=[[0, 10], [0, 10], [0, 10]]
        )

        config = manager.configure_algorithm(
            "SLSQP", problem, "balanced", max_iterations=500
        )

        assert config["options"]["maxiter"] == 500

    def test_custom_options_merge(self):
        """Test custom options merge."""
        from paola.agent.configuration import ConfigurationManager
        from paola.foundry.nlp_schema import NLPProblem

        manager = ConfigurationManager()
        problem = NLPProblem(
            problem_id="test",
            objective_evaluator_id="obj",
            dimension=3,
            bounds=[[0, 10], [0, 10], [0, 10]]
        )

        config = manager.configure_algorithm(
            "SLSQP", problem, "balanced",
            custom_options={"options": {"eps": 1e-10}}
        )

        assert config["options"]["eps"] == 1e-10


class TestConfigTools:
    """Test expert configuration tools (escape hatch)."""

    def test_config_scipy_json_output(self):
        """Test config_scipy produces valid JSON."""
        from paola.tools.config_tools import config_scipy
        import json

        config_str = config_scipy.invoke({
            "method": "SLSQP",
            "maxiter": 500,
            "ftol": 1e-9
        })

        config = json.loads(config_str)
        assert config["method"] == "SLSQP"
        assert config["options"]["maxiter"] == 500
        assert config["options"]["ftol"] == 1e-9

    def test_config_ipopt_json_output(self):
        """Test config_ipopt produces valid JSON."""
        from paola.tools.config_tools import config_ipopt
        import json

        config_str = config_ipopt.invoke({
            "max_iter": 2000,
            "tol": 1e-8
        })

        config = json.loads(config_str)
        assert config["solver"] == "ipopt"
        assert config["options"]["max_iter"] == 2000

    def test_explain_config_option(self):
        """Test explain_config_option provides info."""
        from paola.tools.config_tools import explain_config_option

        result = explain_config_option.invoke({
            "solver": "scipy",
            "option_name": "ftol"
        })

        assert result["success"] == True
        assert "description" in result
        assert "typical_values" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
