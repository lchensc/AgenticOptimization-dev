"""
Tests for Paola Principle implementation.

Tests cover:
- BoundsSpec: Compact bounds specification
- NLPProblem schema: Updated schema without initial_point
- Optimizer backends: SciPy, IPOPT, Optuna backends
- Config tools: Expert escape hatch
- create_nlp_problem: Compact bounds parsing

Note: InitializationManager and ConfigurationManager tests were removed
in the LLM-driven architecture refactor. The intelligence is now in the LLM,
not in hardcoded Python classes.
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


class TestOptimizerBackends:
    """Test optimizer backends (LLM-driven architecture)."""

    def test_scipy_backend_available(self):
        """Test SciPy backend is available."""
        from paola.optimizers.backends import SciPyBackend

        backend = SciPyBackend()
        assert backend.is_available() is True
        assert backend.name == "scipy"

    def test_scipy_backend_info(self):
        """Test SciPy backend info."""
        from paola.optimizers.backends import SciPyBackend

        backend = SciPyBackend()
        info = backend.get_info()

        assert "methods" in info
        assert "SLSQP" in info["methods"]
        assert "L-BFGS-B" in info["methods"]

    def test_scipy_backend_optimize_rosenbrock(self):
        """Test SciPy backend on Rosenbrock function."""
        from paola.optimizers.backends import SciPyBackend

        backend = SciPyBackend()

        # Rosenbrock function
        def rosenbrock(x):
            return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

        bounds = [[-5, 10], [-5, 10]]
        x0 = np.array([0.0, 0.0])
        config = {"method": "L-BFGS-B", "max_iterations": 200}

        result = backend.optimize(
            objective=rosenbrock,
            bounds=bounds,
            x0=x0,
            config=config,
            constraints=None,
            gradient=None
        )

        assert result.success is True
        # Should converge near (1, 1)
        assert abs(result.final_design[0] - 1.0) < 0.1
        assert abs(result.final_design[1] - 1.0) < 0.1
        assert result.final_objective < 0.01

    def test_list_backends(self):
        """Test list_backends function."""
        from paola.optimizers.backends import list_backends

        backends = list_backends()

        assert "scipy" in backends
        assert "ipopt" in backends
        assert "optuna" in backends

    def test_get_available_backends(self):
        """Test get_available_backends function."""
        from paola.optimizers.backends import get_available_backends

        available = get_available_backends()

        assert "scipy" in available  # SciPy should always be available

    def test_get_backend(self):
        """Test get_backend function."""
        from paola.optimizers.backends import get_backend

        backend = get_backend("scipy")
        assert backend is not None
        assert backend.name == "scipy"

        backend = get_backend("scipy:SLSQP")  # Should handle method suffix
        assert backend is not None
        assert backend.name == "scipy"


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


class TestCreateNLPProblemCompactBounds:
    """Test create_nlp_problem with compact bounds specification."""

    @pytest.fixture(autouse=True)
    def setup_evaluator(self, tmp_path):
        """Setup a test evaluator."""
        import os
        from paola.tools.registration_tools import write_file, foundry_store_evaluator
        from paola.tools.evaluator_tools import clear_problem_registry

        # Clear any existing problems
        clear_problem_registry()

        # Create a simple evaluator
        evaluator_code = '''
import numpy as np

def evaluate(x):
    """Sphere function."""
    x = np.atleast_1d(x)
    return float(np.sum(x**2))
'''
        eval_file = str(tmp_path / "sphere.py")
        write_file.invoke({"file_path": eval_file, "content": evaluator_code})

        foundry_store_evaluator.invoke({
            "evaluator_id": "test_sphere",
            "name": "Test Sphere",
            "file_path": eval_file,
            "callable_name": "evaluate",
            "description": "Test sphere function"
        })

        yield

        # Cleanup
        clear_problem_registry()

    def test_uniform_compact_bounds(self):
        """Test create_nlp_problem with uniform compact bounds."""
        from paola.tools.evaluator_tools import create_nlp_problem

        result = create_nlp_problem.invoke({
            "problem_id": "uniform_test",
            "objective_evaluator_id": "test_sphere",
            "bounds": {"type": "uniform", "lower": -10, "upper": 10, "dimension": 50}
        })

        assert result["success"] is True
        assert result["dimension"] == 50

    def test_grouped_compact_bounds(self):
        """Test create_nlp_problem with grouped compact bounds."""
        from paola.tools.evaluator_tools import create_nlp_problem, clear_problem_registry

        clear_problem_registry()

        result = create_nlp_problem.invoke({
            "problem_id": "grouped_test",
            "objective_evaluator_id": "test_sphere",
            "bounds": {
                "type": "grouped",
                "groups": {
                    "x": {"lower": -0.05, "upper": 0.05, "count": 40},
                    "y": {"lower": -15, "upper": 15, "count": 10}
                }
            },
            "domain_hint": "shape_optimization"
        })

        assert result["success"] is True
        assert result["dimension"] == 50
        assert result["domain_hint"] == "shape_optimization"

    def test_explicit_bounds_backward_compatible(self):
        """Test create_nlp_problem still accepts explicit bounds."""
        from paola.tools.evaluator_tools import create_nlp_problem, clear_problem_registry

        clear_problem_registry()

        result = create_nlp_problem.invoke({
            "problem_id": "explicit_test",
            "objective_evaluator_id": "test_sphere",
            "bounds": [[-5, 5], [-10, 10], [0, 1]]
        })

        assert result["success"] is True
        assert result["dimension"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
