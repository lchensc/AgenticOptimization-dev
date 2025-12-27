"""
Multi-Objective Optimization Example: ZDT1 Test Problem

Demonstrates Paola's MOO capabilities with the ZDT1 benchmark problem.

ZDT1 is a bi-objective problem:
    minimize f1(x) = x_1
    minimize f2(x) = g(x) * (1 - sqrt(f1/g))
    where g(x) = 1 + 9 * sum(x_2..x_n) / (n-1)

True Pareto front: f2 = 1 - sqrt(f1), for f1 in [0, 1]

Usage:
    python examples/moo_zdt1.py
"""

import numpy as np
from paola.foundry.problem import (
    OptimizationProblem,
    Variable,
    Objective,
)
from paola.foundry.schema.pareto import ParetoFront, ParetoSolution
from paola.optimizers import PymooBackend


def zdt1_objectives(x: np.ndarray) -> np.ndarray:
    """
    ZDT1 objective functions.

    Args:
        x: Design vector of length n (typically n=30)

    Returns:
        [f1, f2] objective values
    """
    n = len(x)
    f1 = x[0]
    g = 1.0 + 9.0 * np.sum(x[1:]) / (n - 1)
    f2 = g * (1.0 - np.sqrt(f1 / g))
    return np.array([f1, f2])


def create_zdt1_problem(n_vars: int = 30) -> OptimizationProblem:
    """
    Create ZDT1 optimization problem.

    Args:
        n_vars: Number of variables (default: 30)

    Returns:
        OptimizationProblem instance
    """
    variables = [
        Variable(
            name=f"x{i}",
            type="continuous",
            lower=0.0,
            upper=1.0,
        )
        for i in range(n_vars)
    ]

    objectives = [
        Objective(
            name="f1",
            evaluator_id="zdt1",  # Would be registered evaluator
            index=0,  # First element of array return
            sense="minimize",
        ),
        Objective(
            name="f2",
            evaluator_id="zdt1",
            index=1,  # Second element of array return
            sense="minimize",
        ),
    ]

    return OptimizationProblem(
        problem_id=1,
        name="ZDT1 Benchmark",
        variables=variables,
        objectives=objectives,
        description="ZDT1 bi-objective test problem with 30 variables",
        domain_hint="benchmark",
    )


def run_moo_optimization():
    """Run MOO on ZDT1 and display results."""
    print("=" * 60)
    print("Multi-Objective Optimization: ZDT1")
    print("=" * 60)

    # Create problem
    problem = create_zdt1_problem(n_vars=30)
    print(f"\n{problem}")
    print(f"\nProblem class: {problem.problem_class}")

    # Get bounds
    bounds = problem.get_bounds_list()
    n_vars = problem.n_variables

    # Create objective functions
    def f1(x):
        return zdt1_objectives(x)[0]

    def f2(x):
        return zdt1_objectives(x)[1]

    # Run pymoo NSGA-II
    print("\nRunning NSGA-II optimization...")
    backend = PymooBackend()

    if not backend.is_available():
        print("pymoo not installed. Install with: pip install pymoo")
        return

    config = {
        "algorithm": "NSGA-II",
        "n_gen": 100,
        "pop_size": 100,
        "objectives": [f1, f2],
    }

    x0 = np.array([0.5] * n_vars)

    result = backend.optimize(
        objective=f1,  # Primary objective
        bounds=bounds,
        x0=x0,
        config=config,
    )

    print(f"\nOptimization complete!")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Function evaluations: {result.n_function_evals}")

    if result.is_multiobjective:
        print(f"\nPareto front:")
        print(f"  Solutions: {result.n_pareto_solutions}")
        print(f"  Hypervolume: {result.hypervolume:.4f}" if result.hypervolume else "")

        # Create ParetoFront for analysis
        pf = ParetoFront.from_arrays(
            pareto_set=result.pareto_set,
            pareto_front=result.pareto_front,
            objective_names=["f1", "f2"],
            graph_id=1,
            node_id="n1",
            algorithm="NSGA-II",
        )

        # Compute hypervolume
        hv = pf.compute_hypervolume(reference_point=np.array([1.1, 1.1]))
        print(f"  Hypervolume (ref=[1.1, 1.1]): {hv:.4f}")

        # Get knee point
        knee = pf.get_knee_point()
        if knee:
            print(f"\nKnee point (best trade-off):")
            print(f"  f1 = {knee.f[0]:.4f}")
            print(f"  f2 = {knee.f[1]:.4f}")

        # Get extremes
        print("\nExtreme points:")
        extreme_f1 = pf.get_extreme("f1")
        extreme_f2 = pf.get_extreme("f2")
        print(f"  Min f1: f1={extreme_f1.f[0]:.4f}, f2={extreme_f1.f[1]:.4f}")
        print(f"  Min f2: f1={extreme_f2.f[0]:.4f}, f2={extreme_f2.f[1]:.4f}")

        # Get spread
        spread = pf.get_spread()
        print(f"\nSpread:")
        for name, (lo, hi) in spread.items():
            print(f"  {name}: [{lo:.4f}, {hi:.4f}]")

        # Filter example
        filtered = pf.filter_by_objective("f1", max_val=0.3)
        print(f"\nSolutions with f1 <= 0.3: {filtered.n_solutions}")

        # Summary for agent
        print(f"\nSummary (for agent):")
        summary = pf.summary()
        for key, val in summary.items():
            print(f"  {key}: {val}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_moo_optimization()
