"""
Test Phase 6: Problem Modeling and Parsing

Tests for:
- Natural language parsing
- Code parsing
- Structured parsing
- User function backend
- Problem validation
"""

import pytest
import numpy as np

from paola.modeling.parsers import ProblemParser, parse_problem
from paola.modeling.validation import validate_problem, check_initial_feasibility
from paola.backends.user_function import UserFunctionBackend, create_user_backend


class TestProblemParsing:
    """Test problem parsing from various formats."""

    def test_natural_language_simple(self):
        """Test simple mathematical expression parsing."""
        parser = ProblemParser()

        # Simple minimize x^2 + 3x
        problem = parser.from_natural_language("minimize x^2 + 3x subject to x > 1")

        assert problem is not None
        assert len(problem.objectives) == 1
        assert problem.objectives[0].sense == "minimize"
        assert len(problem.variables) > 0

    def test_natural_language_engineering(self):
        """Test engineering problem description (simplified for Phase 6 Week 1)."""
        parser = ProblemParser()

        # For now, engineering descriptions need variables mentioned
        # Full LLM parsing will come later
        problem = parser.from_natural_language(
            "minimize x subject to x >= 0.5"
        )

        assert problem is not None
        assert problem.objectives[0].sense == "minimize"

    def test_code_parsing(self):
        """Test parsing Python code."""
        parser = ProblemParser()

        code = """
def objective(x):
    return x**2 + 3*x
"""

        problem = parser.from_code(code)

        assert problem is not None
        assert len(problem.objectives) == 1

    def test_structured_parsing(self):
        """Test parsing from dict."""
        parser = ProblemParser()

        data = {
            "objectives": [{"name": "f", "sense": "minimize"}],
            "variables": [
                {"name": "x", "bounds": [0, 10], "initial": 5},
                {"name": "y", "bounds": [-5, 5], "initial": 0}
            ],
            "constraints": [
                {"name": "c1", "type": "inequality", "expression": "x + y >= 1"}
            ]
        }

        problem = parser.from_structured(data)

        assert problem is not None
        assert len(problem.variables) == 2
        assert len(problem.objectives) == 1
        assert len(problem.constraints) == 1

    def test_scipy_format(self):
        """Test SciPy format conversion."""
        parser = ProblemParser()

        bounds = [(0, 10), (-5, 5), (0, 1)]
        problem = parser.from_scipy_format(bounds)

        assert len(problem.variables) == 3
        assert problem.variables[0].bounds == (0, 10)
        assert problem.variables[1].bounds == (-5, 5)

    def test_simple_dict(self):
        """Test simple problem creation."""
        parser = ProblemParser()

        problem = parser.from_dict_simple(
            n_variables=5,
            bounds=[(-10, 10)] * 5,
            n_constraints=2
        )

        assert len(problem.variables) == 5
        assert len(problem.constraints) == 2

    def test_auto_format_detection(self):
        """Test automatic format detection."""
        # Dict format
        problem1 = parse_problem({
            "objectives": [{"name": "f", "sense": "minimize"}],
            "variables": [{"name": "x", "bounds": [0, 10], "initial": 5}]
        })
        assert problem1 is not None

        # Natural language
        problem2 = parse_problem("minimize x^2")
        assert problem2 is not None

        # Code
        problem3 = parse_problem("def objective(x): return x**2")
        assert problem3 is not None


class TestProblemValidation:
    """Test problem validation."""

    def test_valid_problem(self):
        """Test validation of valid problem."""
        parser = ProblemParser()
        problem = parser.from_dict_simple(n_variables=3)

        validation = validate_problem(problem)

        assert validation["valid"]
        assert len(validation["errors"]) == 0

    def test_invalid_bounds(self):
        """Test detection of invalid bounds."""
        parser = ProblemParser()

        data = {
            "objectives": [{"name": "f", "sense": "minimize"}],
            "variables": [
                {"name": "x", "bounds": [10, 0], "initial": 5}  # Invalid: lower > upper
            ]
        }

        problem = parser.from_structured(data)
        validation = validate_problem(problem)

        assert not validation["valid"]
        assert len(validation["errors"]) > 0

    def test_initial_feasibility(self):
        """Test initial point feasibility check."""
        parser = ProblemParser()

        data = {
            "objectives": [{"name": "f", "sense": "minimize"}],
            "variables": [
                {"name": "x", "bounds": [0, 10], "initial": 15}  # Outside bounds
            ]
        }

        problem = parser.from_structured(data)
        feasibility = check_initial_feasibility(problem)

        assert not feasibility["feasible"]
        assert feasibility["n_violations"] > 0


class TestUserFunctionBackend:
    """Test user-provided function backend."""

    def test_simple_function(self):
        """Test simple objective function."""
        def my_function(x):
            return {"objective": np.sum(x**2)}

        backend = UserFunctionBackend(my_function)

        design = np.array([1.0, 2.0, 3.0])
        result = backend.evaluate(design)

        assert result.objectives["objective"] == pytest.approx(14.0)  # 1 + 4 + 9

    def test_function_with_constraints(self):
        """Test function returning objectives and constraints."""
        def my_function(x):
            obj = {"drag": x[0]**2 + x[1]**2}
            cons = {"lift": x[0] + x[1] - 1.0}
            return obj, cons

        backend = UserFunctionBackend(my_function)

        design = np.array([1.0, 2.0])
        result = backend.evaluate(design)

        assert "drag" in result.objectives
        assert "lift" in result.constraints
        assert result.constraints["lift"] == pytest.approx(2.0)  # 1 + 2 - 1

    def test_single_value_return(self):
        """Test function returning single value."""
        def my_function(x):
            return np.sum(x**2)

        backend = UserFunctionBackend(my_function)

        design = np.array([1.0, 2.0])
        result = backend.evaluate(design)

        assert "objective" in result.objectives
        assert result.objectives["objective"] == pytest.approx(5.0)

    def test_finite_difference_gradient(self):
        """Test finite difference gradient computation."""
        def my_function(x):
            return {"objective": x[0]**2 + 2*x[1]**2}

        backend = UserFunctionBackend(my_function)

        design = np.array([1.0, 2.0])
        gradient = backend.compute_gradient(design, method="finite_difference")

        # Analytical gradient: [2*x0, 4*x1] = [2, 8]
        assert gradient[0] == pytest.approx(2.0, abs=1e-4)
        assert gradient[1] == pytest.approx(8.0, abs=1e-4)

    def test_user_provided_gradient(self):
        """Test user-provided gradient function."""
        def my_function(x):
            return {"objective": x[0]**2 + 2*x[1]**2}

        def my_gradient(x):
            return np.array([2*x[0], 4*x[1]])

        backend = UserFunctionBackend(
            my_function,
            has_gradients=True,
            gradient_function=my_gradient
        )

        design = np.array([1.0, 2.0])
        gradient = backend.compute_gradient(design, method="user_provided")

        assert gradient[0] == pytest.approx(2.0)
        assert gradient[1] == pytest.approx(8.0)

    def test_create_user_backend(self):
        """Test convenience function."""
        def my_function(x):
            return np.sum(x**2)

        backend = create_user_backend(my_function, cost_per_eval=2.5)

        assert backend.cost_per_evaluation == 2.5
        assert backend.domain == "user_defined"

    def test_error_handling(self):
        """Test error handling in user function."""
        def bad_function(x):
            raise ValueError("Simulation failed!")

        backend = UserFunctionBackend(bad_function)

        design = np.array([1.0])

        with pytest.raises(RuntimeError, match="User function evaluation failed"):
            backend.evaluate(design)


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("PHASE 6 MODELING TESTS")
    print("=" * 60)

    # Test Problem Parsing
    print("\n[1/3] Testing Problem Parsing...")
    test_parsing = TestProblemParsing()

    test_parsing.test_natural_language_simple()
    print("  ✓ Natural language (simple)")

    test_parsing.test_natural_language_engineering()
    print("  ✓ Natural language (engineering)")

    test_parsing.test_code_parsing()
    print("  ✓ Code parsing")

    test_parsing.test_structured_parsing()
    print("  ✓ Structured parsing")

    test_parsing.test_scipy_format()
    print("  ✓ SciPy format")

    test_parsing.test_simple_dict()
    print("  ✓ Simple dict")

    test_parsing.test_auto_format_detection()
    print("  ✓ Auto format detection")

    # Test Problem Validation
    print("\n[2/3] Testing Problem Validation...")
    test_validation = TestProblemValidation()

    test_validation.test_valid_problem()
    print("  ✓ Valid problem")

    test_validation.test_invalid_bounds()
    print("  ✓ Invalid bounds detection")

    test_validation.test_initial_feasibility()
    print("  ✓ Initial feasibility check")

    # Test User Function Backend
    print("\n[3/3] Testing User Function Backend...")
    test_backend = TestUserFunctionBackend()

    test_backend.test_simple_function()
    print("  ✓ Simple function")

    test_backend.test_function_with_constraints()
    print("  ✓ Function with constraints")

    test_backend.test_single_value_return()
    print("  ✓ Single value return")

    test_backend.test_finite_difference_gradient()
    print("  ✓ Finite difference gradient")

    test_backend.test_user_provided_gradient()
    print("  ✓ User-provided gradient")

    test_backend.test_create_user_backend()
    print("  ✓ Convenience function")

    test_backend.test_error_handling()
    print("  ✓ Error handling")

    print("\n" + "=" * 60)
    print("✅ ALL PHASE 6 MODELING TESTS PASSED!")
    print("=" * 60)

    print("\nComponents implemented:")
    print("  ✓ Problem parsing (natural language, code, structured)")
    print("  ✓ Problem validation")
    print("  ✓ User function backend (most common use case)")
    print("  ✓ Finite difference gradients")
    print("  ✓ User-provided gradients")

    print("\nReady for:")
    print("  → Agent integration (modeling tools)")
    print("  → CLI usage with user functions")
    print("  → Phase 6 Week 2 (optimizer integration)")


if __name__ == "__main__":
    run_tests()
