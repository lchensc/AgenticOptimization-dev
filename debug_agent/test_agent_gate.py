#!/usr/bin/env python
"""
End-to-end test: Qwen agent solves 10D Rosenbrock using gate pattern.

This uses the recommended gate pattern where scipy runs to completion,
then the agent analyzes the results.

Usage:
    python debug_agent/test_agent_gate.py
"""

import os
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from scipy.optimize import minimize

from aopt.backends.analytical import Rosenbrock
from aopt.optimizers.gate import OptimizationGate
from aopt.tools import (
    register_problem, clear_problem_registry, clear_optimizer_registry,
    register_gate, clear_gate_registry, cache_clear,
    analyze_convergence, get_gradient_quality,
)


def run_optimization_with_gate():
    """Run optimization using gate pattern, then have agent analyze."""

    print("=" * 70)
    print("AGENTIC OPTIMIZATION: 10D Rosenbrock (Gate Pattern)")
    print("=" * 70)

    # Setup
    clear_problem_registry()
    clear_optimizer_registry()
    clear_gate_registry()
    cache_clear()

    problem = Rosenbrock(dimension=10)
    x_opt, f_opt = problem.get_optimum()

    print(f"\nProblem: 10D Rosenbrock")
    print(f"Known optimum: f* = {f_opt}")
    print(f"Initial design: [0, 0, ..., 0]")

    # Create gate
    gate = OptimizationGate(problem_id="rosenbrock_10d", blocking=False)
    register_gate("gate_10d", gate)

    # Run scipy optimization through gate
    print("\n" + "-" * 70)
    print("Phase 1: Running scipy SLSQP optimization...")
    print("-" * 70)

    x0 = np.zeros(10)
    bounds = [(-5, 10)] * 10

    result = minimize(
        fun=gate.wrap_objective(problem.evaluate),
        x0=x0,
        method='SLSQP',
        jac=gate.wrap_gradient(problem.gradient),
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 200}
    )

    print(f"\nScipy result:")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print(f"  Iterations: {result.nit}")
    print(f"  Function evals: {result.nfev}")
    print(f"  Final objective: {result.fun:.6e}")

    # Get gate history
    history = gate.get_history()
    objectives = [h['objective'] for h in history if 'objective' in h]
    gradients = [h['gradient'].tolist() for h in history
                 if 'gradient' in h and h.get('gradient') is not None]

    print(f"\nGate captured {len(history)} calls")

    # Phase 2: Agent analysis
    print("\n" + "-" * 70)
    print("Phase 2: Agent analyzes optimization trajectory...")
    print("-" * 70)

    # Analyze convergence
    if len(objectives) >= 2:
        conv_result = analyze_convergence.invoke({
            "objectives": objectives[-20:],
            "gradients": gradients[-20:] if gradients else None,
            "window_size": 10,
        })
        print(f"\nConvergence analysis:")
        print(f"  Converging: {conv_result['converging']}")
        print(f"  Converged: {conv_result['converged']}")
        print(f"  Stalled: {conv_result['stalled']}")
        print(f"  Recommendation: {conv_result['recommendation']}")

    # Analyze gradient quality
    if len(gradients) >= 2:
        grad_result = get_gradient_quality.invoke({
            "gradients": gradients[-10:],
            "window_size": 5,
        })
        print(f"\nGradient quality:")
        print(f"  Quality: {grad_result['quality']}")
        print(f"  Current norm: {grad_result['current_norm']:.6e}")
        print(f"  Recommendation: {grad_result['recommendation']}")

    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    error_x = np.linalg.norm(result.x - x_opt)
    error_f = abs(result.fun - f_opt)

    print(f"Final objective: {result.fun:.10e}")
    print(f"Error |f - f*|: {error_f:.10e}")
    print(f"Error ||x - x*||: {error_x:.10e}")

    if error_f < 1e-6:
        print("\n✓ SUCCESS: Converged to optimum!")
        return True
    else:
        print(f"\n⚠ Not at optimum (error: {error_f:.2e})")
        return False


def run_with_llm_agent():
    """Run with LLM agent making decisions."""

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY not set!")
        print("Running without LLM...")
        return run_optimization_with_gate()

    print("=" * 70)
    print("AGENTIC OPTIMIZATION with LLM: 10D Rosenbrock")
    print("=" * 70)

    from debug_agent.simple_react_agent import create_simple_react_agent
    from aopt.agent.react_agent import initialize_llm
    from aopt.tools import (
        evaluate_function, compute_gradient,
        gate_get_history, gate_get_statistics,
        analyze_convergence, detect_pattern, get_gradient_quality,
        compute_improvement_statistics,
    )

    # Setup
    clear_problem_registry()
    clear_optimizer_registry()
    clear_gate_registry()
    cache_clear()

    problem = Rosenbrock(dimension=10)
    register_problem("rosenbrock_10d", problem)

    # Create gate
    gate = OptimizationGate(problem_id="rosenbrock_10d", blocking=False)
    register_gate("optimization_gate", gate)

    # Run scipy first (gate captures everything)
    print("\nRunning scipy optimization...")
    x0 = np.zeros(10)
    bounds = [(-5, 10)] * 10

    result = minimize(
        fun=gate.wrap_objective(problem.evaluate),
        x0=x0,
        method='SLSQP',
        jac=gate.wrap_gradient(problem.gradient),
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 200}
    )

    print(f"Scipy completed: {result.message}")
    print(f"Final objective: {result.fun:.6e}")

    # Now have agent analyze
    print("\n" + "-" * 70)
    print("LLM Agent Analyzing Results...")
    print("-" * 70)

    llm = initialize_llm("qwen-flash", temperature=0.0)

    tools = [
        gate_get_history,
        gate_get_statistics,
        analyze_convergence,
        detect_pattern,
        get_gradient_quality,
        compute_improvement_statistics,
    ]

    agent = create_simple_react_agent(llm, tools, max_iterations=10)

    analysis = agent(
        goal=f"""Analyze the optimization results for 10D Rosenbrock.

The optimization has completed with:
- Final objective: {result.fun:.6e}
- Known optimum: f* = 0
- Scipy iterations: {result.nit}

Use gate_get_history with gate_id="optimization_gate" to get the trajectory.
Then use analyze_convergence and get_gradient_quality to assess the results.

Report whether the optimization was successful and provide recommendations.
Say "DONE" when finished with your analysis.""",
        verbose=True
    )

    return analysis.get("success", False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true", help="Use LLM agent for analysis")
    args = parser.parse_args()

    if args.with_llm:
        success = run_with_llm_agent()
    else:
        success = run_optimization_with_gate()

    sys.exit(0 if success else 1)
