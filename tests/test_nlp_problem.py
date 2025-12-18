"""
Unit tests for NLP problem construction components.

Tests:
- NLP schema classes (NLPProblem, constraints)
- NLPEvaluator composite evaluator
- Problem type detection
- Solver selection
- create_nlp_problem tool
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from paola.foundry.nlp_schema import NLPProblem, InequalityConstraint, EqualityConstraint
from paola.foundry.nlp_evaluator import NLPEvaluator
from paola.foundry.problem_types import ProblemTypeDetector, SolverSelector
from paola.foundry import OptimizationFoundry, FileStorage
from paola.foundry.evaluator_schema import create_python_function_config
from paola.tools.evaluator_tools import create_nlp_problem, clear_problem_registry
from paola.tools.graph_tools import set_foundry


class TestNLPSchema:
    """Test NLP schema classes."""

    def test_inequality_constraint_creation(self):
        """Test creating inequality constraint."""
        cons = InequalityConstraint(
            name="min_lift",
            evaluator_id="lift_eval",
            constraint_type=">=",
            value=1000.0
        )

        assert cons.name == "min_lift"
        assert cons.evaluator_id == "lift_eval"
        assert cons.constraint_type == ">="
        assert cons.value == 1000.0

    def test_inequality_constraint_serialization(self):
        """Test constraint serialization."""
        cons = InequalityConstraint(
            name="max_stress",
            evaluator_id="stress_eval",
            constraint_type="<=",
            value=200.0
        )

        # To dict
        cons_dict = cons.to_dict()
        assert cons_dict["name"] == "max_stress"
        assert cons_dict["constraint_type"] == "<="

        # From dict
        cons_restored = InequalityConstraint.from_dict(cons_dict)
        assert cons_restored.name == cons.name
        assert cons_restored.value == cons.value

    def test_equality_constraint_creation(self):
        """Test creating equality constraint."""
        cons = EqualityConstraint(
            name="moment_balance",
            evaluator_id="moment_eval",
            value=0.0,
            tolerance=1e-6
        )

        assert cons.name == "moment_balance"
        assert cons.value == 0.0
        assert cons.tolerance == 1e-6

    def test_nlp_problem_unconstrained(self):
        """Test creating unconstrained NLP problem."""
        problem = NLPProblem(
            problem_id=1,
            name="Rosenbrock 2D",
            n_variables=2,  # Will be validated against bounds
            n_constraints=0,
            objective_evaluator_id="rosenbrock_eval",
            bounds=[[-5, 10], [-5, 10]],
            objective_sense="minimize"
        )

        assert problem.problem_id == 1
        assert problem.problem_type == "NLP"
        assert problem.dimension == 2  # computed from bounds
        assert problem.is_unconstrained
        assert not problem.is_constrained
        assert problem.num_constraints == 0

    def test_nlp_problem_constrained(self):
        """Test creating constrained NLP problem."""
        ineq_cons = InequalityConstraint(
            name="min_lift",
            evaluator_id="lift_eval",
            constraint_type=">=",
            value=1000.0
        )

        eq_cons = EqualityConstraint(
            name="balance",
            evaluator_id="balance_eval",
            value=0.0
        )

        problem = NLPProblem(
            problem_id=2,
            name="Wing Design",
            n_variables=2,
            n_constraints=2,
            objective_evaluator_id="drag_eval",
            bounds=[[0, 15], [0.1, 0.5]],
            inequality_constraints=[ineq_cons],
            equality_constraints=[eq_cons]
        )

        assert problem.is_constrained
        assert problem.num_inequality_constraints == 1
        assert problem.num_equality_constraints == 1
        assert problem.num_constraints == 2

    def test_nlp_problem_validation(self):
        """Test NLP problem validation."""
        # Invalid bounds (lower >= upper)
        with pytest.raises(ValueError, match="lower >= upper"):
            NLPProblem(
                problem_id=3,
                name="Test",
                n_variables=1,
                n_constraints=0,
                objective_evaluator_id="obj",
                bounds=[[10, 5]]  # Lower > upper
            )

        # Invalid bounds format (not [lower, upper])
        with pytest.raises(ValueError, match="must be"):
            NLPProblem(
                problem_id=4,
                name="Test",
                n_variables=1,
                n_constraints=0,
                objective_evaluator_id="obj",
                bounds=[[1, 2, 3]]  # Three values instead of two
            )

    def test_nlp_problem_serialization(self):
        """Test NLP problem serialization."""
        problem = NLPProblem(
            problem_id=5,
            name="Test Problem",
            n_variables=2,
            n_constraints=1,
            objective_evaluator_id="obj_eval",
            bounds=[[-5, 5], [-5, 5]],
            inequality_constraints=[
                InequalityConstraint("c1", "cons_eval", ">=", 0.0)
            ]
        )

        # To dict
        problem_dict = problem.to_dict()
        assert problem_dict["problem_id"] == 5
        assert len(problem_dict["inequality_constraints"]) == 1

        # From dict
        problem_restored = NLPProblem.from_dict(problem_dict)
        assert problem_restored.problem_id == problem.problem_id
        assert problem_restored.dimension == problem.dimension
        assert len(problem_restored.inequality_constraints) == 1

    def test_nlp_problem_evaluator_ids(self):
        """Test getting all evaluator IDs from problem."""
        problem = NLPProblem(
            problem_id=6,
            name="Test",
            n_variables=2,
            n_constraints=3,
            objective_evaluator_id="obj_eval",
            bounds=[[-5, 5], [-5, 5]],
            inequality_constraints=[
                InequalityConstraint("c1", "cons1_eval", ">=", 0.0),
                InequalityConstraint("c2", "cons2_eval", "<=", 100.0)
            ],
            equality_constraints=[
                EqualityConstraint("c3", "cons3_eval", 0.0)
            ]
        )

        evaluator_ids = problem.get_all_evaluator_ids()
        assert "obj_eval" in evaluator_ids
        assert "cons1_eval" in evaluator_ids
        assert "cons2_eval" in evaluator_ids
        assert "cons3_eval" in evaluator_ids
        assert len(evaluator_ids) == 4


class TestProblemTypeDetector:
    """Test problem type detection."""

    def test_detect_nlp_from_problem(self):
        """Test detecting NLP from problem specification."""
        problem = NLPProblem(
            problem_id=7,
            name="Test",
            n_variables=2,
            n_constraints=0,
            objective_evaluator_id="obj",
            bounds=[[-5, 5], [-5, 5]]
        )

        problem_type = ProblemTypeDetector.detect_from_nlp_problem(problem)
        assert problem_type == "NLP"

    def test_detect_from_specification(self):
        """Test detecting problem type from generic spec."""
        # NLP specification
        nlp_spec = {
            "has_integer_variables": False,
            "is_linear": False,
            "num_objectives": 1
        }
        assert ProblemTypeDetector.detect_from_specification(nlp_spec) == "NLP"

        # LP specification
        lp_spec = {
            "has_integer_variables": False,
            "is_linear": True,
            "num_objectives": 1
        }
        assert ProblemTypeDetector.detect_from_specification(lp_spec) == "LP"

        # MILP specification
        milp_spec = {
            "has_integer_variables": True,
            "has_continuous_variables": True,
            "is_linear": True,
            "num_objectives": 1
        }
        assert ProblemTypeDetector.detect_from_specification(milp_spec) == "MILP"

        # MOO specification
        moo_spec = {
            "num_objectives": 2
        }
        assert ProblemTypeDetector.detect_from_specification(moo_spec) == "MOO"


class TestSolverSelector:
    """Test solver selection."""

    def test_recommend_nlp_constrained_gradient(self):
        """Test recommending solver for constrained NLP with gradients."""
        solvers = SolverSelector.recommend_solver(
            problem_type="NLP",
            gradient_available=True,
            has_constraints=True
        )

        assert "SLSQP" in solvers
        assert len(solvers) > 0

    def test_recommend_nlp_unconstrained_gradient(self):
        """Test recommending solver for unconstrained NLP with gradients."""
        solvers = SolverSelector.recommend_solver(
            problem_type="NLP",
            gradient_available=True,
            has_constraints=False
        )

        assert "L-BFGS-B" in solvers or "BFGS" in solvers

    def test_recommend_nlp_constrained_no_gradient(self):
        """Test recommending solver for constrained NLP without gradients."""
        solvers = SolverSelector.recommend_solver(
            problem_type="NLP",
            gradient_available=False,
            has_constraints=True
        )

        assert "COBYLA" in solvers

    def test_solver_availability(self):
        """Test checking solver availability."""
        assert SolverSelector.is_solver_available("SLSQP")
        assert SolverSelector.is_solver_available("COBYLA")
        assert not SolverSelector.is_solver_available("NonExistentSolver")

    def test_solver_validation(self):
        """Test validating solver for problem."""
        # Valid: SLSQP for constrained NLP
        result = SolverSelector.validate_solver_for_problem(
            solver_name="SLSQP",
            problem_type="NLP",
            has_constraints=True
        )
        assert result["valid"]

        # Invalid: Nelder-Mead for constrained problem
        result = SolverSelector.validate_solver_for_problem(
            solver_name="Nelder-Mead",
            problem_type="NLP",
            has_constraints=True
        )
        assert not result["valid"]
        assert "doesn't support constraints" in result["warning"]


class TestCreateNLPProblemTool:
    """Test create_nlp_problem tool integration."""

    @pytest.fixture
    def temp_foundry(self):
        """Create temporary foundry for testing."""
        temp_dir = tempfile.mkdtemp()
        storage = FileStorage(base_dir=temp_dir)
        foundry = OptimizationFoundry(storage=storage)

        # Set as global foundry for tools
        set_foundry(foundry)

        # Register test evaluator (rosenbrock from evaluators.py)
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        config = create_python_function_config(
            evaluator_id="rosenbrock_eval",
            name="Rosenbrock Function",
            file_path="evaluators.py",
            callable_name="rosenbrock"
        )
        foundry.register_evaluator(config)

        yield foundry

        # Cleanup
        clear_problem_registry()
        shutil.rmtree(temp_dir)

    def test_create_unconstrained_nlp(self, temp_foundry):
        """Test creating unconstrained NLP problem."""
        result = create_nlp_problem.invoke({
            "name": "Rosenbrock 2D",
            "objective_evaluator_id": "rosenbrock_eval",
            "bounds": [[-5, 10], [-5, 10]]
        })

        assert result["success"]
        assert isinstance(result["problem_id"], int)  # Auto-generated int ID
        assert result["problem_type"] == "NLP"
        assert result["dimension"] == 2
        assert result["num_inequality_constraints"] == 0
        assert result["num_equality_constraints"] == 0
        assert "SLSQP" in result["recommended_solvers"] or "L-BFGS-B" in result["recommended_solvers"]

    def test_create_nlp_missing_evaluator(self, temp_foundry):
        """Test creating NLP with missing evaluator."""
        result = create_nlp_problem.invoke({
            "name": "Test Problem",
            "objective_evaluator_id": "nonexistent_eval",
            "bounds": [[-5, 5]]
        })

        assert not result["success"]
        assert "not found" in result["message"].lower() or "not registered" in result["message"].lower()

    def test_create_nlp_same_name(self, temp_foundry):
        """Test creating NLP with same name (allowed - each gets unique ID)."""
        # Create first problem
        result1 = create_nlp_problem.invoke({
            "name": "Test Problem",
            "objective_evaluator_id": "rosenbrock_eval",
            "bounds": [[-5, 5]]
        })
        assert result1["success"]

        # Create second problem with same name (should work - different problem_id)
        result2 = create_nlp_problem.invoke({
            "name": "Test Problem",
            "objective_evaluator_id": "rosenbrock_eval",
            "bounds": [[-5, 5]]
        })

        assert result2["success"]
        assert result2["problem_id"] != result1["problem_id"]  # Different auto-generated IDs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
