"""
NLPEvaluator - Composite evaluator for NLP problems.

Wraps:
- 1 objective evaluator (FoundryEvaluator)
- 0-N inequality constraint evaluators
- 0-N equality constraint evaluators

Provides scipy.optimize.minimize-compatible interface.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from .evaluator import FoundryEvaluator
from .nlp_schema import NLPProblem, InequalityConstraint, EqualityConstraint


class NLPEvaluator:
    """
    Composite evaluator for NLP (Nonlinear Programming) problems.

    This class wraps multiple FoundryEvaluators (objective + constraints)
    into a single interface compatible with scipy.optimize.minimize.

    Handles:
    - Single objective (minimize or maximize)
    - Multiple inequality constraints (g(x) <= value or g(x) >= value)
    - Multiple equality constraints (h(x) = value)
    - Automatic constraint transformation to scipy format

    Example:
        # Create NLP evaluator from problem specification
        nlp_eval = NLPEvaluator.from_problem(nlp_problem, foundry)

        # Use with scipy
        result = scipy.optimize.minimize(
            fun=nlp_eval.evaluate,
            x0=initial_point,
            method='SLSQP',
            bounds=bounds,
            constraints=nlp_eval.get_scipy_constraints()
        )
    """

    def __init__(
        self,
        problem_spec: NLPProblem,
        objective_evaluator: FoundryEvaluator,
        constraint_evaluators: Dict[str, FoundryEvaluator]
    ):
        """
        Initialize NLPEvaluator.

        Args:
            problem_spec: NLP problem specification
            objective_evaluator: FoundryEvaluator for objective function
            constraint_evaluators: Dict mapping evaluator_id to FoundryEvaluator
        """
        self.problem_spec = problem_spec
        self.objective_eval = objective_evaluator
        self.constraint_evals = constraint_evaluators

        # Extract constraint lists
        self.ineq_constraints = problem_spec.inequality_constraints
        self.eq_constraints = problem_spec.equality_constraints

        # Objective sense
        self.objective_sense = problem_spec.objective_sense

    @classmethod
    def from_problem(
        cls,
        problem: NLPProblem,
        foundry
    ) -> 'NLPEvaluator':
        """
        Create NLPEvaluator from NLPProblem specification.

        Loads all required evaluators from Foundry.

        Args:
            problem: NLP problem specification
            foundry: OptimizationFoundry instance

        Returns:
            NLPEvaluator instance

        Raises:
            ValueError: If any required evaluator not found in Foundry
        """
        # Load objective evaluator
        objective_eval = FoundryEvaluator(
            evaluator_id=problem.objective_evaluator_id,
            foundry=foundry
        )

        # Load constraint evaluators
        constraint_evals = {}
        for evaluator_id in problem.get_all_evaluator_ids():
            if evaluator_id == problem.objective_evaluator_id:
                continue  # Already loaded

            constraint_evals[evaluator_id] = FoundryEvaluator(
                evaluator_id=evaluator_id,
                foundry=foundry
            )

        return cls(
            problem_spec=problem,
            objective_evaluator=objective_eval,
            constraint_evaluators=constraint_evals
        )

    def evaluate(self, x: np.ndarray) -> float:
        """
        Evaluate objective function.

        For minimize: return f(x)
        For maximize: return -f(x)  (scipy minimizes by default)

        Args:
            x: Design vector

        Returns:
            Objective value (transformed based on sense)
        """
        x = np.atleast_1d(x)

        # Evaluate objective
        result = self.objective_eval.evaluate(x)
        obj_value = result.objectives["objective"]

        # Transform for maximization (scipy minimizes)
        if self.objective_sense == "maximize":
            obj_value = -obj_value

        return float(obj_value)

    def get_scipy_constraints(self) -> List[Dict[str, Any]]:
        """
        Generate scipy constraint dictionaries.

        Scipy expects constraints in form:
        - Inequality: g(x) >= 0  (note: >=, not <=)
        - Equality: h(x) = 0

        We transform user specifications to this form.

        Returns:
            List of constraint dicts for scipy.optimize.minimize
        """
        scipy_constraints = []

        # Inequality constraints
        for i, cons in enumerate(self.ineq_constraints):
            def make_constraint_func(cons_spec):
                """Create constraint function with proper closure."""
                def constraint_func(x):
                    x = np.atleast_1d(x)
                    evaluator = self.constraint_evals[cons_spec.evaluator_id]
                    result = evaluator.evaluate(x)
                    g_x = result.objectives["objective"]

                    # Transform to scipy form: g(x) >= 0
                    if cons_spec.constraint_type == "<=":
                        # User: g(x) <= value
                        # Transform: value - g(x) >= 0
                        return float(cons_spec.value - g_x)
                    else:  # ">="
                        # User: g(x) >= value
                        # Transform: g(x) - value >= 0
                        return float(g_x - cons_spec.value)

                return constraint_func

            scipy_constraints.append({
                "type": "ineq",
                "fun": make_constraint_func(cons)
            })

        # Equality constraints
        for i, cons in enumerate(self.eq_constraints):
            def make_constraint_func(cons_spec):
                """Create constraint function with proper closure."""
                def constraint_func(x):
                    x = np.atleast_1d(x)
                    evaluator = self.constraint_evals[cons_spec.evaluator_id]
                    result = evaluator.evaluate(x)
                    h_x = result.objectives["objective"]

                    # Scipy form: h(x) = 0
                    # User: h(x) = value
                    # Transform: h(x) - value = 0
                    return float(h_x - cons_spec.value)

                return constraint_func

            scipy_constraints.append({
                "type": "eq",
                "fun": make_constraint_func(cons)
            })

        return scipy_constraints

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Compute objective gradient (scipy-compatible interface).

        This method provides a simple interface compatible with scipy.optimize.minimize.
        Uses finite difference by default for robustness.

        Args:
            x: Design point

        Returns:
            Gradient vector (same shape as x)
        """
        return self.get_gradient(x, method="finite_difference")

    def get_gradient(self, x: np.ndarray, method: str = "finite_difference") -> np.ndarray:
        """
        Compute objective gradient.

        For gradient-based solvers (SLSQP, IPOPT, etc.).

        Args:
            x: Design point
            method: Gradient computation method
                   - "finite_difference": Numerical approximation
                   - "auto": Use FoundryEvaluator's method

        Returns:
            Gradient vector (same shape as x)
        """
        x = np.atleast_1d(x)

        # Compute gradient
        gradient = self.objective_eval.compute_gradient(x, method=method)

        # Transform for maximization (negate gradient)
        if self.objective_sense == "maximize":
            gradient = -gradient

        return gradient

    def get_constraint_jacobian(self, x: np.ndarray) -> np.ndarray:
        """
        Compute constraint Jacobian matrix.

        For advanced gradient-based solvers.

        Args:
            x: Design point

        Returns:
            Jacobian matrix [num_constraints × num_variables]

        Note: Phase 7+ feature. Currently not used (scipy computes internally).
        """
        # TODO: Implement constraint Jacobian for advanced solvers
        # For now, scipy uses finite differences internally
        raise NotImplementedError(
            "Constraint Jacobian not yet implemented. "
            "Scipy uses finite differences internally."
        )

    @property
    def problem_id(self) -> str:
        """Problem identifier."""
        return self.problem_spec.problem_id

    @property
    def dimension(self) -> int:
        """Problem dimension."""
        return self.problem_spec.dimension

    @property
    def bounds(self) -> List[List[float]]:
        """Variable bounds."""
        return self.problem_spec.bounds

    @property
    def initial_point(self) -> Optional[List[float]]:
        """Initial point."""
        return self.problem_spec.initial_point

    @property
    def num_constraints(self) -> int:
        """Total number of constraints."""
        return self.problem_spec.num_constraints

    @property
    def is_unconstrained(self) -> bool:
        """Whether problem is unconstrained."""
        return self.problem_spec.is_unconstrained

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"NLPEvaluator(\n"
            f"  problem_id={self.problem_id},\n"
            f"  objective={self.objective_sense} {self.problem_spec.objective_evaluator_id},\n"
            f"  dimension={self.dimension},\n"
            f"  inequality_constraints={len(self.ineq_constraints)},\n"
            f"  equality_constraints={len(self.eq_constraints)}\n"
            f")"
        )

    def __repr__(self) -> str:
        """Detailed representation."""
        return str(self)
