"""
Tests for the unified OptimizationProblem schema (v1.0).

Tests:
- Variable, Objective, Constraint dataclasses
- OptimizationProblem creation and validation
- Problem class auto-detection (NLP, MINLP, MOO, MO-MINLP)
- Serialization/deserialization
- Problem derivation
"""

import pytest
import numpy as np

from paola.foundry.problem import (
    Variable,
    Objective,
    Constraint,
    OptimizationProblem,
    DerivationType,
)


class TestVariable:
    """Tests for Variable dataclass."""

    def test_continuous_variable(self):
        v = Variable(name="x", type="continuous", lower=0.0, upper=1.0)
        assert v.name == "x"
        assert v.type == "continuous"
        assert v.lower == 0.0
        assert v.upper == 1.0

    def test_integer_variable(self):
        v = Variable(name="n", type="integer", lower=1, upper=10)
        assert v.type == "integer"
        # Bounds should be floats
        assert v.lower == 1.0
        assert v.upper == 10.0

    def test_binary_variable(self):
        # Binary variables auto-set bounds to [0, 1]
        v = Variable(name="flag", type="binary", lower=0, upper=1)
        assert v.type == "binary"
        assert v.lower == 0.0
        assert v.upper == 1.0

    def test_invalid_bounds_raises(self):
        with pytest.raises(ValueError, match="lower.*must be < upper"):
            Variable(name="x", type="continuous", lower=1.0, upper=0.0)

    def test_serialization(self):
        v = Variable(name="x", type="continuous", lower=0.0, upper=1.0, unit="m")
        d = v.to_dict()
        assert d["name"] == "x"
        assert d["unit"] == "m"

        v2 = Variable.from_dict(d)
        assert v2.name == v.name
        assert v2.unit == v.unit


class TestObjective:
    """Tests for Objective dataclass."""

    def test_minimize_objective(self):
        obj = Objective(name="drag", evaluator_id="cfd_drag")
        assert obj.name == "drag"
        assert obj.evaluator_id == "cfd_drag"
        assert obj.sense == "minimize"  # default
        assert obj.index is None

    def test_maximize_objective(self):
        obj = Objective(name="lift", evaluator_id="cfd_lift", sense="maximize")
        assert obj.sense == "maximize"

    def test_array_objective_with_index(self):
        obj = Objective(name="f1", evaluator_id="zdt1", index=0)
        assert obj.index == 0

    def test_serialization(self):
        obj = Objective(name="drag", evaluator_id="cfd_drag", sense="minimize")
        d = obj.to_dict()
        obj2 = Objective.from_dict(d)
        assert obj2.name == obj.name
        assert obj2.sense == obj.sense


class TestConstraint:
    """Tests for Constraint dataclass."""

    def test_inequality_le(self):
        c = Constraint(name="stress", evaluator_id="stress_eval", type="<=", bound=250)
        assert c.type == "<="
        assert c.bound == 250
        assert not c.is_equality

    def test_inequality_ge(self):
        c = Constraint(name="lift", evaluator_id="lift_eval", type=">=", bound=1000)
        assert c.type == ">="
        assert not c.is_equality

    def test_equality(self):
        c = Constraint(name="volume", evaluator_id="vol_eval", type="==", bound=1.0)
        assert c.type == "=="
        assert c.is_equality
        assert c.tolerance == 1e-6

    def test_serialization(self):
        c = Constraint(name="stress", evaluator_id="stress_eval", type="<=", bound=250)
        d = c.to_dict()
        c2 = Constraint.from_dict(d)
        assert c2.name == c.name
        assert c2.bound == c.bound


class TestOptimizationProblemCreation:
    """Tests for OptimizationProblem creation and validation."""

    def test_simple_nlp(self):
        """Simple NLP: continuous variables, single objective."""
        problem = OptimizationProblem(
            problem_id=1,
            name="Test NLP",
            variables=[
                Variable(name="x1", type="continuous", lower=0, upper=1),
                Variable(name="x2", type="continuous", lower=0, upper=1),
            ],
            objectives=[
                Objective(name="f", evaluator_id="sphere"),
            ],
        )
        assert problem.n_variables == 2
        assert problem.n_objectives == 1
        assert problem.n_constraints == 0
        assert problem.problem_class == "NLP"
        assert problem.problem_family == "continuous"
        assert not problem.is_multiobjective
        assert not problem.has_integers

    def test_minlp(self):
        """MINLP: mixed variables, single objective."""
        problem = OptimizationProblem(
            problem_id=2,
            name="Test MINLP",
            variables=[
                Variable(name="x", type="continuous", lower=0, upper=1),
                Variable(name="n", type="integer", lower=1, upper=10),
            ],
            objectives=[
                Objective(name="f", evaluator_id="obj"),
            ],
        )
        assert problem.problem_class == "MINLP"
        assert problem.problem_family == "mixed"
        assert problem.has_integers

    def test_moo(self):
        """MOO: continuous variables, multiple objectives."""
        problem = OptimizationProblem(
            problem_id=3,
            name="Test MOO",
            variables=[
                Variable(name="x1", type="continuous", lower=0, upper=1),
                Variable(name="x2", type="continuous", lower=0, upper=1),
            ],
            objectives=[
                Objective(name="f1", evaluator_id="obj1"),
                Objective(name="f2", evaluator_id="obj2"),
            ],
        )
        assert problem.problem_class == "MOO"
        assert problem.is_multiobjective
        assert not problem.has_integers

    def test_mo_minlp(self):
        """MO-MINLP: mixed variables, multiple objectives."""
        problem = OptimizationProblem(
            problem_id=4,
            name="Test MO-MINLP",
            variables=[
                Variable(name="x", type="continuous", lower=0, upper=1),
                Variable(name="n", type="integer", lower=1, upper=10),
            ],
            objectives=[
                Objective(name="f1", evaluator_id="obj1"),
                Objective(name="f2", evaluator_id="obj2"),
            ],
        )
        assert problem.problem_class == "MO-MINLP"
        assert problem.is_multiobjective
        assert problem.has_integers

    def test_with_constraints(self):
        """Problem with constraints."""
        problem = OptimizationProblem(
            problem_id=5,
            name="Constrained",
            variables=[
                Variable(name="x", type="continuous", lower=0, upper=1),
            ],
            objectives=[
                Objective(name="f", evaluator_id="obj"),
            ],
            constraints=[
                Constraint(name="g1", evaluator_id="c1", type="<=", bound=0),
                Constraint(name="h1", evaluator_id="c2", type="==", bound=1),
            ],
        )
        assert problem.n_constraints == 2
        assert problem.n_inequality_constraints == 1
        assert problem.n_equality_constraints == 1
        assert problem.is_constrained

    def test_no_variables_raises(self):
        with pytest.raises(ValueError, match="at least one variable"):
            OptimizationProblem(
                problem_id=1,
                name="Empty",
                variables=[],
                objectives=[Objective(name="f", evaluator_id="obj")],
            )

    def test_no_objectives_raises(self):
        with pytest.raises(ValueError, match="at least one objective"):
            OptimizationProblem(
                problem_id=1,
                name="Empty",
                variables=[Variable(name="x", type="continuous", lower=0, upper=1)],
                objectives=[],
            )

    def test_duplicate_variable_names_raises(self):
        with pytest.raises(ValueError, match="Variable names must be unique"):
            OptimizationProblem(
                problem_id=1,
                name="Duplicate",
                variables=[
                    Variable(name="x", type="continuous", lower=0, upper=1),
                    Variable(name="x", type="continuous", lower=0, upper=2),
                ],
                objectives=[Objective(name="f", evaluator_id="obj")],
            )


class TestOptimizationProblemBounds:
    """Tests for bounds extraction methods."""

    @pytest.fixture
    def problem(self):
        return OptimizationProblem(
            problem_id=1,
            name="Test",
            variables=[
                Variable(name="x1", type="continuous", lower=-1, upper=1),
                Variable(name="x2", type="continuous", lower=0, upper=10),
                Variable(name="n", type="integer", lower=1, upper=5),
            ],
            objectives=[Objective(name="f", evaluator_id="obj")],
        )

    def test_get_bounds(self, problem):
        lb, ub = problem.get_bounds()
        np.testing.assert_array_equal(lb, [-1, 0, 1])
        np.testing.assert_array_equal(ub, [1, 10, 5])

    def test_get_bounds_list(self, problem):
        bounds = problem.get_bounds_list()
        assert bounds == [[-1, 1], [0, 10], [1, 5]]

    def test_get_integer_indices(self, problem):
        assert problem.get_integer_indices() == [2]

    def test_get_continuous_indices(self, problem):
        assert problem.get_continuous_indices() == [0, 1]

    def test_get_integer_mask(self, problem):
        mask = problem.get_integer_mask()
        np.testing.assert_array_equal(mask, [False, False, True])

    def test_get_bounds_center(self, problem):
        center = problem.get_bounds_center()
        np.testing.assert_array_equal(center, [0, 5, 3])

    def test_get_bounds_width(self, problem):
        width = problem.get_bounds_width()
        np.testing.assert_array_equal(width, [2, 10, 4])


class TestOptimizationProblemEvaluators:
    """Tests for evaluator-related methods."""

    def test_get_all_evaluator_ids(self):
        problem = OptimizationProblem(
            problem_id=1,
            name="Test",
            variables=[Variable(name="x", type="continuous", lower=0, upper=1)],
            objectives=[
                Objective(name="f1", evaluator_id="obj1"),
                Objective(name="f2", evaluator_id="obj2"),
            ],
            constraints=[
                Constraint(name="g", evaluator_id="con1", type="<=", bound=0),
            ],
        )
        ids = problem.get_all_evaluator_ids()
        assert set(ids) == {"obj1", "obj2", "con1"}

    def test_validate_evaluators_all_present(self):
        problem = OptimizationProblem(
            problem_id=1,
            name="Test",
            variables=[Variable(name="x", type="continuous", lower=0, upper=1)],
            objectives=[Objective(name="f", evaluator_id="obj1")],
        )
        missing = problem.validate_evaluators(["obj1", "obj2", "obj3"])
        assert missing == []

    def test_validate_evaluators_some_missing(self):
        problem = OptimizationProblem(
            problem_id=1,
            name="Test",
            variables=[Variable(name="x", type="continuous", lower=0, upper=1)],
            objectives=[
                Objective(name="f1", evaluator_id="obj1"),
                Objective(name="f2", evaluator_id="obj2"),
            ],
        )
        missing = problem.validate_evaluators(["obj1"])
        assert missing == ["obj2"]


class TestOptimizationProblemSerialization:
    """Tests for serialization/deserialization."""

    @pytest.fixture
    def problem(self):
        return OptimizationProblem(
            problem_id=1,
            name="Test Problem",
            variables=[
                Variable(name="x1", type="continuous", lower=0, upper=1),
                Variable(name="x2", type="integer", lower=1, upper=10),
            ],
            objectives=[
                Objective(name="f1", evaluator_id="obj1"),
                Objective(name="f2", evaluator_id="obj2"),
            ],
            constraints=[
                Constraint(name="g", evaluator_id="con1", type="<=", bound=0),
            ],
            description="A test problem",
            domain_hint="general",
        )

    def test_to_dict(self, problem):
        d = problem.to_dict()
        assert d["problem_id"] == 1
        assert d["name"] == "Test Problem"
        assert len(d["variables"]) == 2
        assert len(d["objectives"]) == 2
        assert len(d["constraints"]) == 1
        # Computed fields should be included
        assert d["problem_class"] == "MO-MINLP"
        assert d["n_variables"] == 2

    def test_from_dict(self, problem):
        d = problem.to_dict()
        p2 = OptimizationProblem.from_dict(d)
        assert p2.problem_id == problem.problem_id
        assert p2.name == problem.name
        assert p2.n_variables == problem.n_variables
        assert p2.problem_class == problem.problem_class

    def test_to_json_from_json(self, problem):
        json_str = problem.to_json()
        p2 = OptimizationProblem.from_json(json_str)
        assert p2.problem_id == problem.problem_id
        assert p2.problem_class == problem.problem_class


class TestOptimizationProblemDerivation:
    """Tests for problem derivation methods."""

    @pytest.fixture
    def base_problem(self):
        return OptimizationProblem(
            problem_id=1,
            name="Base Problem",
            variables=[
                Variable(name="x1", type="continuous", lower=-10, upper=10),
                Variable(name="x2", type="continuous", lower=-10, upper=10),
            ],
            objectives=[
                Objective(name="f", evaluator_id="obj"),
            ],
        )

    def test_derive_narrow_bounds(self, base_problem):
        derived = base_problem.derive_narrow_bounds(
            new_problem_id=2,
            new_name="Narrowed Problem",
            center=[2.0, 3.0],
            width_factor=0.2,
            reason="Focus on promising region",
        )

        assert derived.problem_id == 2
        assert derived.name == "Narrowed Problem"
        assert derived.parent_problem_id == 1
        assert derived.derivation_type == DerivationType.NARROW_BOUNDS
        assert derived.version == 2

        # Check bounds are narrowed
        # Original width = 20, new width = 4 (20 * 0.2)
        # x1: center=2, so bounds = [0, 4]
        # x2: center=3, so bounds = [1, 5]
        assert derived.variables[0].lower == 0.0
        assert derived.variables[0].upper == 4.0
        assert derived.variables[1].lower == 1.0
        assert derived.variables[1].upper == 5.0

    def test_derive_add_objective(self, base_problem):
        new_obj = Objective(name="f2", evaluator_id="obj2")
        derived = base_problem.derive_add_objective(
            new_problem_id=2,
            new_name="Multi-objective Problem",
            new_objective=new_obj,
        )

        assert derived.n_objectives == 2
        assert derived.problem_class == "MOO"
        assert derived.parent_problem_id == 1
        assert derived.derivation_type == DerivationType.ADD_OBJECTIVE


class TestOptimizationProblemSignature:
    """Tests for problem signature generation."""

    def test_get_signature(self):
        problem = OptimizationProblem(
            problem_id=1,
            name="Test",
            variables=[
                Variable(name="x1", type="continuous", lower=-1, upper=1),
                Variable(name="x2", type="continuous", lower=0, upper=10),
            ],
            objectives=[
                Objective(name="f1", evaluator_id="obj1"),
                Objective(name="f2", evaluator_id="obj2"),
            ],
            constraints=[
                Constraint(name="h", evaluator_id="con", type="==", bound=0),
            ],
            domain_hint="aerodynamic",
        )

        sig = problem.get_signature()
        assert sig["n_variables"] == 2
        assert sig["n_objectives"] == 2
        assert sig["n_constraints"] == 1
        assert sig["problem_class"] == "MOO"
        assert sig["problem_family"] == "continuous"
        assert sig["bounds_range"] == (-1.0, 10.0)
        assert sig["has_equality_constraints"] is True
        assert sig["domain_hint"] == "aerodynamic"
