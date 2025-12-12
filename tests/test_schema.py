"""
Tests for optimization problem schema.
"""

import pytest
from paola.formulation.schema import (
    OptimizationProblem,
    Objective,
    Variable,
    Constraint
)


def test_objective_creation():
    """Test creating objectives."""
    obj = Objective(name="drag", sense="minimize")
    assert obj.name == "drag"
    assert obj.sense == "minimize"


def test_variable_creation():
    """Test creating variables."""
    var = Variable(name="x1", bounds=(0, 10))
    assert var.name == "x1"
    assert var.bounds == (0, 10)
    assert var.type == "continuous"  # default


def test_constraint_creation():
    """Test creating constraints."""
    cons = Constraint(
        name="lift",
        type="inequality",
        expression="CL >= 0.8"
    )
    assert cons.name == "lift"
    assert cons.type == "inequality"


def test_single_objective_problem():
    """Test single-objective problem creation."""
    problem = OptimizationProblem(
        problem_type="nonlinear_single",
        objectives=[Objective(name="f", sense="minimize")],
        variables=[
            Variable(name="x1", bounds=(0, 10)),
            Variable(name="x2", bounds=(-5, 5))
        ]
    )

    assert problem.is_single_objective
    assert not problem.is_multi_objective
    assert problem.n_variables == 2
    assert problem.n_objectives == 1
    assert problem.n_constraints == 0
    assert not problem.has_constraints


def test_multi_objective_problem():
    """Test multi-objective problem creation."""
    problem = OptimizationProblem(
        problem_type="nonlinear_multi",
        objectives=[
            Objective(name="drag", sense="minimize"),
            Objective(name="weight", sense="minimize")
        ],
        variables=[Variable(name="x", bounds=(0, 1))]
    )

    assert not problem.is_single_objective
    assert problem.is_multi_objective
    assert problem.n_objectives == 2


def test_constrained_problem():
    """Test problem with constraints."""
    problem = OptimizationProblem(
        problem_type="nonlinear_single",
        objectives=[Objective(name="cost", sense="minimize")],
        variables=[Variable(name="x", bounds=(0, 100))],
        constraints=[
            Constraint(name="stress", type="inequality", expression="stress <= 200")
        ]
    )

    assert problem.has_constraints
    assert problem.n_constraints == 1


def test_get_bounds():
    """Test extracting bounds."""
    problem = OptimizationProblem(
        problem_type="nonlinear_single",
        objectives=[Objective(name="f", sense="minimize")],
        variables=[
            Variable(name="x1", bounds=(0, 10)),
            Variable(name="x2", bounds=(-5, 5)),
            Variable(name="x3", bounds=(-1, 1))
        ]
    )

    lower, upper = problem.get_bounds()
    assert lower == [0, -5, -1]
    assert upper == [10, 5, 1]


def test_get_initial_design():
    """Test extracting initial design."""
    # With initial values
    problem1 = OptimizationProblem(
        problem_type="nonlinear_single",
        objectives=[Objective(name="f", sense="minimize")],
        variables=[
            Variable(name="x1", bounds=(0, 10), initial=5.0),
            Variable(name="x2", bounds=(0, 10), initial=3.0)
        ]
    )

    initial = problem1.get_initial_design()
    assert initial == [5.0, 3.0]

    # Without initial values
    problem2 = OptimizationProblem(
        problem_type="nonlinear_single",
        objectives=[Objective(name="f", sense="minimize")],
        variables=[
            Variable(name="x1", bounds=(0, 10)),
            Variable(name="x2", bounds=(0, 10))
        ]
    )

    initial = problem2.get_initial_design()
    assert initial is None


def test_problem_immutability():
    """Test that components are immutable."""
    obj = Objective(name="f", sense="minimize")

    with pytest.raises(Exception):  # Pydantic frozen models raise error
        obj.name = "g"
