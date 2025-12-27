"""
Tests for MOO evaluator with array support.
"""

import pytest
import numpy as np

from paola.foundry.moo_evaluator import (
    MOOEvaluator,
    ArrayEvaluatorCache,
    create_moo_evaluator,
)
from paola.foundry.problem import (
    OptimizationProblem,
    Variable,
    Objective,
    Constraint,
)


class TestArrayEvaluatorCache:
    """Tests for ArrayEvaluatorCache."""

    def test_cache_store_and_get(self):
        cache = ArrayEvaluatorCache()

        design = np.array([1.0, 2.0])
        result = np.array([0.5, 0.3])

        cache.store("eval1", design, result)
        cached = cache.get("eval1", design)

        np.testing.assert_array_equal(cached, result)

    def test_cache_miss(self):
        cache = ArrayEvaluatorCache()

        design = np.array([1.0, 2.0])
        cached = cache.get("eval1", design)

        assert cached is None

    def test_different_designs(self):
        cache = ArrayEvaluatorCache()

        design1 = np.array([1.0, 2.0])
        design2 = np.array([1.0, 2.1])
        result = np.array([0.5, 0.3])

        cache.store("eval1", design1, result)

        assert cache.get("eval1", design1) is not None
        assert cache.get("eval1", design2) is None

    def test_clear_specific_evaluator(self):
        cache = ArrayEvaluatorCache()

        design = np.array([1.0, 2.0])
        result = np.array([0.5, 0.3])

        cache.store("eval1", design, result)
        cache.store("eval2", design, result)

        cache.clear("eval1")

        assert cache.get("eval1", design) is None
        assert cache.get("eval2", design) is not None

    def test_clear_all(self):
        cache = ArrayEvaluatorCache()

        design = np.array([1.0, 2.0])
        result = np.array([0.5, 0.3])

        cache.store("eval1", design, result)
        cache.store("eval2", design, result)

        cache.clear()

        assert cache.get("eval1", design) is None
        assert cache.get("eval2", design) is None

    def test_cache_max_size(self):
        cache = ArrayEvaluatorCache(max_size=2)

        result = np.array([0.5])

        cache.store("eval1", np.array([1.0]), result)
        cache.store("eval1", np.array([2.0]), result)
        cache.store("eval1", np.array([3.0]), result)

        # First entry should be evicted
        assert cache.get("eval1", np.array([1.0])) is None
        assert cache.get("eval1", np.array([2.0])) is not None
        assert cache.get("eval1", np.array([3.0])) is not None


class TestMOOEvaluator:
    """Tests for MOOEvaluator."""

    @pytest.fixture
    def zdt1_evaluator(self):
        """ZDT1 evaluator returning [f1, f2]."""
        def zdt1(x):
            n = len(x)
            f1 = x[0]
            g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
            f2 = g * (1.0 - np.sqrt(f1 / g))
            return np.array([f1, f2])
        return zdt1

    @pytest.fixture
    def moo_problem(self):
        """Create ZDT1 MOO problem."""
        variables = [
            Variable(name=f"x{i}", type="continuous", lower=0.0, upper=1.0)
            for i in range(5)
        ]
        objectives = [
            Objective(name="f1", evaluator_id="zdt1", index=0, sense="minimize"),
            Objective(name="f2", evaluator_id="zdt1", index=1, sense="minimize"),
        ]
        return OptimizationProblem(
            problem_id=1,
            name="ZDT1 Test",
            variables=variables,
            objectives=objectives,
        )

    def test_evaluate_single_objective(self, moo_problem, zdt1_evaluator):
        registry = {"zdt1": zdt1_evaluator}
        moo_eval = create_moo_evaluator(moo_problem, registry)

        design = np.array([0.5, 0.0, 0.0, 0.0, 0.0])

        f1 = moo_eval.evaluate_objective(moo_problem.objectives[0], design)
        f2 = moo_eval.evaluate_objective(moo_problem.objectives[1], design)

        assert f1 == pytest.approx(0.5, rel=1e-6)
        # g = 1 + 9*0/4 = 1, f2 = 1 * (1 - sqrt(0.5/1)) = 1 - sqrt(0.5)
        assert f2 == pytest.approx(1.0 - np.sqrt(0.5), rel=1e-6)

    def test_evaluate_all(self, moo_problem, zdt1_evaluator):
        registry = {"zdt1": zdt1_evaluator}
        moo_eval = create_moo_evaluator(moo_problem, registry)

        design = np.array([0.25, 0.0, 0.0, 0.0, 0.0])
        result = moo_eval.evaluate_all(design)

        assert len(result) == 2
        assert result[0] == pytest.approx(0.25, rel=1e-6)
        assert result[1] == pytest.approx(1.0 - np.sqrt(0.25), rel=1e-6)

    def test_objective_functions(self, moo_problem, zdt1_evaluator):
        registry = {"zdt1": zdt1_evaluator}
        moo_eval = create_moo_evaluator(moo_problem, registry)

        funcs = moo_eval.get_objective_functions()

        assert len(funcs) == 2

        design = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        assert funcs[0](design) == pytest.approx(0.0, abs=1e-10)
        assert funcs[1](design) == pytest.approx(1.0, rel=1e-6)

    def test_caching_efficiency(self, moo_problem):
        call_count = [0]

        def counting_evaluator(x):
            call_count[0] += 1
            return np.array([x[0], x[1]])

        registry = {"zdt1": counting_evaluator}
        moo_eval = create_moo_evaluator(moo_problem, registry)

        design = np.array([0.5, 0.3, 0.0, 0.0, 0.0])

        # Evaluate both objectives at same design point
        funcs = moo_eval.get_objective_functions()
        funcs[0](design)
        funcs[1](design)

        # Evaluator should only be called once due to caching
        assert call_count[0] == 1

    def test_properties(self, moo_problem, zdt1_evaluator):
        registry = {"zdt1": zdt1_evaluator}
        moo_eval = create_moo_evaluator(moo_problem, registry)

        assert moo_eval.n_objectives == 2
        assert moo_eval.n_constraints == 0
        assert moo_eval.objective_names == ["f1", "f2"]
        assert moo_eval.is_multiobjective is True

    def test_maximize_objective(self):
        variables = [
            Variable(name="x", type="continuous", lower=0.0, upper=1.0)
        ]
        objectives = [
            Objective(name="neg_f", evaluator_id="f", sense="maximize"),
        ]
        problem = OptimizationProblem(
            problem_id=1,
            name="Maximize Test",
            variables=variables,
            objectives=objectives,
        )

        registry = {"f": lambda x: np.array([x[0]])}
        moo_eval = create_moo_evaluator(problem, registry)

        design = np.array([0.5])
        # maximize x -> minimize -x
        result = moo_eval.evaluate_objective(objectives[0], design)
        assert result == pytest.approx(-0.5, rel=1e-6)

    def test_constraint_functions(self):
        variables = [
            Variable(name="x", type="continuous", lower=-10.0, upper=10.0)
        ]
        objectives = [
            Objective(name="f", evaluator_id="obj"),
        ]
        constraints = [
            Constraint(name="c1", evaluator_id="con1", type="<=", bound=1.0),
            Constraint(name="c2", evaluator_id="con2", type=">=", bound=0.0),
        ]
        problem = OptimizationProblem(
            problem_id=1,
            name="Constrained Test",
            variables=variables,
            objectives=objectives,
            constraints=constraints,
        )

        registry = {
            "obj": lambda x: np.array([x[0] ** 2]),
            "con1": lambda x: np.array([x[0]]),
            "con2": lambda x: np.array([x[0] + 5]),
        }
        moo_eval = create_moo_evaluator(problem, registry)

        g_funcs = moo_eval.get_constraint_functions()
        assert len(g_funcs) == 2

        # c1: x <= 1 -> g1 = x - 1 <= 0
        # At x = 0.5: g1 = 0.5 - 1 = -0.5 (feasible)
        design = np.array([0.5])
        assert g_funcs[0](design) == pytest.approx(-0.5, rel=1e-6)

        # c2: x + 5 >= 0 -> g2 = 0 - (x + 5) = -(x+5) <= 0
        # At x = 0.5: g2 = -(0.5 + 5) = -5.5 (feasible)
        assert g_funcs[1](design) == pytest.approx(-5.5, rel=1e-6)

    def test_missing_evaluator(self, moo_problem):
        registry = {}  # No evaluators
        moo_eval = create_moo_evaluator(moo_problem, registry)

        design = np.array([0.5, 0.0, 0.0, 0.0, 0.0])

        with pytest.raises(ValueError, match="not found"):
            moo_eval.evaluate_objective(moo_problem.objectives[0], design)

    def test_index_out_of_bounds(self):
        variables = [
            Variable(name="x", type="continuous", lower=0.0, upper=1.0)
        ]
        objectives = [
            Objective(name="f1", evaluator_id="eval", index=5),  # Index too large
        ]
        problem = OptimizationProblem(
            problem_id=1,
            name="Bad Index Test",
            variables=variables,
            objectives=objectives,
        )

        registry = {"eval": lambda x: np.array([1.0, 2.0])}  # Only 2 elements
        moo_eval = create_moo_evaluator(problem, registry)

        with pytest.raises(ValueError, match="index 5"):
            moo_eval.evaluate_objective(objectives[0], np.array([0.5]))

    def test_scalar_evaluator(self):
        """Test evaluator that returns scalar (not array)."""
        variables = [
            Variable(name="x", type="continuous", lower=0.0, upper=1.0)
        ]
        objectives = [
            Objective(name="f", evaluator_id="scalar_eval"),
        ]
        problem = OptimizationProblem(
            problem_id=1,
            name="Scalar Test",
            variables=variables,
            objectives=objectives,
        )

        # Evaluator returns float, not array
        registry = {"scalar_eval": lambda x: x[0] ** 2}
        moo_eval = create_moo_evaluator(problem, registry)

        design = np.array([0.5])
        result = moo_eval.evaluate_objective(objectives[0], design)
        assert result == pytest.approx(0.25, rel=1e-6)


class TestMOOEvaluatorIntegration:
    """Integration tests for MOO evaluator with optimization."""

    def test_with_pymoo_objectives(self):
        """Test MOOEvaluator output format matches pymoo expectations."""
        # Create ZDT1-like problem
        variables = [
            Variable(name=f"x{i}", type="continuous", lower=0.0, upper=1.0)
            for i in range(10)
        ]
        objectives = [
            Objective(name="f1", evaluator_id="zdt1", index=0),
            Objective(name="f2", evaluator_id="zdt1", index=1),
        ]
        problem = OptimizationProblem(
            problem_id=1,
            name="ZDT1",
            variables=variables,
            objectives=objectives,
        )

        def zdt1(x):
            n = len(x)
            f1 = x[0]
            g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
            f2 = g * (1.0 - np.sqrt(f1 / g))
            return np.array([f1, f2])

        registry = {"zdt1": zdt1}
        moo_eval = create_moo_evaluator(problem, registry)

        # Get objective functions in format expected by pymoo
        obj_funcs = moo_eval.get_objective_functions()

        # Test on Pareto front (where x_1 = ... = x_n = 0)
        pareto_design = np.zeros(10)
        pareto_design[0] = 0.5

        f1 = obj_funcs[0](pareto_design)
        f2 = obj_funcs[1](pareto_design)

        assert f1 == pytest.approx(0.5, rel=1e-6)
        assert f2 == pytest.approx(1.0 - np.sqrt(0.5), rel=1e-6)

        # Verify trade-off: f2 = 1 - sqrt(f1) on Pareto front
        assert f2 == pytest.approx(1.0 - np.sqrt(f1), rel=1e-6)
