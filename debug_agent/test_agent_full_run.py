#!/usr/bin/env python
"""
End-to-end test: Qwen agent solves 10D Rosenbrock in one call.

This test demonstrates the "no-interception" pattern where:
1. Agent receives goal: "Solve a 10D Rosenbrock optimization problem with SLSQP"
2. Agent calls run_scipy_optimization tool once
3. Scipy runs to completion autonomously
4. Agent reports results

No step-by-step interception - the agent just runs the optimizer.

Usage:
    python debug_agent/test_agent_full_run.py
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

from aopt.backends.analytical import Rosenbrock
from aopt.tools import (
    register_problem,
    clear_problem_registry,
    clear_optimizer_registry,
    cache_clear,
    run_scipy_optimization,
    analyze_convergence,
)


def run_without_llm():
    """Run the optimization directly (no LLM) to verify the tool works."""

    print("=" * 70)
    print("TESTING run_scipy_optimization TOOL (No LLM)")
    print("=" * 70)

    # Setup
    clear_problem_registry()
    clear_optimizer_registry()
    cache_clear()

    problem = Rosenbrock(dimension=10)
    register_problem("rosenbrock_10d", problem)

    x_opt, f_opt = problem.get_optimum()
    print(f"\nProblem: 10D Rosenbrock")
    print(f"Known optimum: f* = {f_opt}")
    print(f"Variables: 10")
    print(f"Bounds: [-5, 10] for each")

    # Run the tool directly
    print("\n" + "-" * 70)
    print("Calling run_scipy_optimization tool...")
    print("-" * 70)

    result = run_scipy_optimization.invoke({
        "problem_id": "rosenbrock_10d",
        "algorithm": "SLSQP",
        "bounds": [[-5.0, 10.0]] * 10,
        "initial_design": [0.0] * 10,
        "options": '{"maxiter": 200, "ftol": 1e-9}',
        "use_gradient": True,
    })

    print(f"\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Message: {result['message']}")
    print(f"  Final objective: {result['final_objective']:.6e}")
    print(f"  Iterations: {result['n_iterations']}")
    print(f"  Function evals: {result['n_function_evals']}")
    print(f"  Gradient evals: {result['n_gradient_evals']}")
    print(f"  Elapsed time: {result.get('elapsed_time', 0):.3f}s")

    if result['success']:
        final_x = np.array(result['final_design'])
        error_x = np.linalg.norm(final_x - x_opt)
        error_f = abs(result['final_objective'] - f_opt)

        print(f"\n  Error |f - f*|: {error_f:.10e}")
        print(f"  Error ||x - x*||: {error_x:.10e}")

        if error_f < 1e-6:
            print("\n✓ SUCCESS: Converged to optimum!")
            return True
        else:
            print(f"\n⚠ Not at optimum (error: {error_f:.2e})")
            return False
    else:
        print(f"\n✗ FAILED: {result['message']}")
        return False


def run_with_llm():
    """Run with LLM agent that calls run_scipy_optimization."""

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY not set!")
        print("Running without LLM instead...")
        return run_without_llm()

    print("=" * 70)
    print("AGENTIC OPTIMIZATION: 10D Rosenbrock (Full Run, No Interception)")
    print("=" * 70)

    from debug_agent.simple_react_agent import create_simple_react_agent
    from aopt.agent.react_agent import initialize_llm

    # Setup
    clear_problem_registry()
    clear_optimizer_registry()
    cache_clear()

    problem = Rosenbrock(dimension=10)
    register_problem("rosenbrock_10d", problem)

    x_opt, f_opt = problem.get_optimum()
    print(f"\nProblem: 10D Rosenbrock")
    print(f"Known optimum: f* = {f_opt}")

    # Initialize LLM
    print("\n" + "-" * 70)
    print("Initializing Qwen-flash LLM...")
    print("-" * 70)

    llm = initialize_llm("qwen-flash", temperature=0.0)

    # Only give the agent the tools it needs
    tools = [
        run_scipy_optimization,
        analyze_convergence,
    ]

    print(f"Tools available to agent:")
    for t in tools:
        print(f"  - {t.name}")

    # Create agent
    agent = create_simple_react_agent(llm, tools, max_iterations=5)

    # Run agent
    print("\n" + "-" * 70)
    print("Starting Agent...")
    print("-" * 70)

    result = agent(
        goal="""Solve a 10D Rosenbrock optimization problem with SLSQP.

Problem details:
- Problem ID: "rosenbrock_10d" (already registered)
- Number of variables: 10
- Bounds: [-5, 10] for all variables
- Initial design: [0,0,0,0,0,0,0,0,0,0]
- Known optimum: x*=[1,1,...,1], f*=0

Instructions:
1. Call run_scipy_optimization with:
   - problem_id="rosenbrock_10d"
   - algorithm="SLSQP"
   - bounds=[[-5,10],[-5,10],...] (10 times)
   - initial_design=[0,0,0,0,0,0,0,0,0,0]
   - options='{"maxiter": 200, "ftol": 1e-9}'

2. Report the final objective value and whether optimization was successful.

3. Say "DONE" when finished.""",
        verbose=True
    )

    print("\n" + "=" * 70)
    print("AGENT COMPLETED")
    print("=" * 70)
    print(f"Success: {result.get('success', False)}")
    print(f"Iterations: {result.get('iterations', 0)}")

    return result.get('success', False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--with-llm", action="store_true", help="Use LLM agent")
    args = parser.parse_args()

    if args.with_llm:
        success = run_with_llm()
    else:
        success = run_without_llm()

    sys.exit(0 if success else 1)
