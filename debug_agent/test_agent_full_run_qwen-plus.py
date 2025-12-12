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

from paola.tools import (
    clear_problem_registry,
    clear_optimizer_registry,
    cache_clear,
    create_benchmark_problem,
    run_scipy_optimization,
    analyze_convergence,
)




def run_with_llm():
    """Run with LLM agent that calls run_scipy_optimization."""

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY not set!")
        print("Running without LLM instead...")
        return run_without_llm()

    print("=" * 70)
    print("AGENTIC OPTIMIZATION: 10D Rosenbrock (Full Run, No Interception)")
    print("=" * 70)

    from debug_agent.minimal_react_agent import create_minimal_react_agent
    from paola.agent.react_agent import initialize_llm

    # Setup - clear registries but DON'T pre-register problem
    # The agent must formulate the problem autonomously
    clear_problem_registry()
    clear_optimizer_registry()
    cache_clear()

    print(f"\nProblem: 10D Rosenbrock")
    print(f"Known optimum: f* = 0.0")

    # Initialize LLM
    print("\n" + "-" * 70)
    print("Initializing Qwen-plus LLM...")
    print("-" * 70)

    llm = initialize_llm("qwen-plus", temperature=0.0)

    # Give the agent tools for autonomous problem formulation
    # Agent must call create_benchmark_problem BEFORE run_scipy_optimization
    tools = [
        create_benchmark_problem,
        run_scipy_optimization,
        analyze_convergence,
    ]

    print(f"Tools available to agent:")
    for t in tools:
        print(f"  - {t.name}")

    # Create agent
    agent = create_minimal_react_agent(llm, tools, max_iterations=10)

    # Run agent
    print("\n" + "-" * 70)
    print("Starting Agent...")
    print("-" * 70)

    result = agent(
        goal="""Solve a 10D Rosenbrock optimization problem with SLSQP and BFGS, and compare the results.""",
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

    #parser.add_argument("--with-llm", action="store_true", help="Use LLM agent")
    #args = parser.parse_args()

    #if args.with_llm:
    success = run_with_llm()
    #else:
    #    success = run_without_llm()

    sys.exit(0 if success else 1)
