"""
First LLM Agent Autonomous Optimization Run.

This example demonstrates the agent autonomously optimizing the Rosenbrock function:
1. Agent receives goal in natural language
2. Agent uses tools to create optimizer, evaluate function, analyze convergence
3. Agent decides when to stop based on observations

Requirements:
- Set DASHSCOPE_API_KEY in .env file (for Qwen models)
- Or set ANTHROPIC_API_KEY for Claude models
- Or set OPENAI_API_KEY for GPT models
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from paola import Agent
from paola.backends.analytical import Rosenbrock
from paola.tools import register_problem, clear_problem_registry, clear_optimizer_registry
from paola.tools.cache import cache_clear


def run_agent_optimization():
    """Run agent to optimize Rosenbrock function."""

    # Clear any previous state
    clear_problem_registry()
    clear_optimizer_registry()
    cache_clear()

    # Register the Rosenbrock problem
    problem_id = "rosenbrock_2d"
    rosenbrock = Rosenbrock(dimension=2)
    register_problem(problem_id, rosenbrock)

    print("=" * 60)
    print("AGENTIC OPTIMIZATION: Rosenbrock 2D")
    print("=" * 60)
    print(f"Problem: {problem_id}")
    print(f"Optimum: x* = [1, 1], f* = 0")
    print(f"Initial guess: x0 = [-1, 1]")
    print("=" * 60)

    # Create agent with verbose output
    # Use qwen-flash for fast iteration, qwen-plus for better reasoning
    agent = Agent(
        llm_model="qwen-plus",  # or "claude-sonnet-4" or "gpt-4"
        temperature=0.0,
        verbose=True,
        max_iterations=50  # Safety limit
    )

    # Define the optimization goal
    goal = f"""
Minimize the 2D Rosenbrock function.

Problem ID: {problem_id}

Details:
- Design variables: 2 continuous variables (x1, x2)
- Bounds: [-5, 10] for x1, [-5, 10] for x2
- Initial design: [-1.0, 1.0]
- Known optimum: [1, 1] with objective = 0

Instructions:
1. Create an optimizer using optimizer_create with algorithm="SLSQP"
2. Run optimization loop: propose -> evaluate -> compute_gradient -> update
3. After every 5 iterations, use analyze_convergence to check progress
4. Stop when gradient norm < 1e-5 or no improvement for 5 iterations
5. Report final result with best design and objective value

Available tools: optimizer_create, optimizer_propose, optimizer_update,
evaluate_function, compute_gradient, analyze_convergence, cache_stats
"""

    # Run the agent
    print("\nStarting agent...")
    print("-" * 60)

    try:
        result = agent.run(goal=goal)

        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Converged: {result['converged']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Total time: {result['total_time']:.2f}s")

        # Print final context
        final_context = result.get('final_context', {})
        if final_context:
            print(f"\nFinal Context:")
            for key, value in final_context.items():
                if key not in ['history', 'observations']:  # Skip large items
                    print(f"  {key}: {value}")

        # Print agent reasoning summary
        reasoning = result.get('reasoning_log', [])
        if reasoning:
            print(f"\nAgent Reasoning Summary ({len(reasoning)} entries):")
            for i, r in enumerate(reasoning[-5:]):  # Last 5
                if len(r) > 200:
                    r = r[:200] + "..."
                print(f"  [{i+1}] {r}")

        return result

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_simple_optimization_loop():
    """
    Simpler version: Gate-based optimization using tools directly.

    This demonstrates the gate pattern that the agent would use internally
    for scipy-based optimization, without requiring LLM API keys.
    """
    import numpy as np
    from scipy.optimize import minimize
    from paola.backends.analytical import Rosenbrock
    from paola.optimizers.gate import OptimizationGate
    from paola.tools import analyze_convergence

    print("=" * 60)
    print("GATE-BASED OPTIMIZATION (No LLM)")
    print("=" * 60)

    # Setup problem
    problem = Rosenbrock(dimension=2)
    x_opt, f_opt = problem.get_optimum()
    print(f"Problem: 2D Rosenbrock")
    print(f"Known optimum: x* = [{x_opt[0]:.1f}, {x_opt[1]:.1f}], f* = {f_opt:.1f}")

    # Create gate (non-blocking for analytical problems)
    gate = OptimizationGate(
        problem_id="rosenbrock_2d",
        blocking=False,
    )

    # Wrap objective and gradient
    objective_func = gate.wrap_objective(problem.evaluate)
    gradient_func = gate.wrap_gradient(problem.gradient)

    # Run scipy optimization through gate
    print("\nRunning scipy SLSQP through gate...")
    x0 = np.array([-1.0, 1.0])
    bounds = [(-5, 10), (-5, 10)]

    result = minimize(
        fun=objective_func,
        x0=x0,
        method='SLSQP',
        jac=gradient_func,
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 100}
    )

    print(f"Scipy completed: {result.message}")

    # Get gate history for agent analysis
    history = gate.get_history()
    objectives = [h['objective'] for h in history if 'objective' in h]
    gradients = [[h['gradient'][0], h['gradient'][1]] for h in history
                 if 'gradient' in h and h.get('gradient') is not None]

    # Analyze convergence using our tool
    if len(objectives) >= 2:
        conv_result = analyze_convergence.invoke({
            "objectives": objectives[-10:],
            "gradients": gradients[-10:] if gradients else None,
            "window_size": 5,
        })
        print(f"\nConvergence analysis: {conv_result['recommendation']}")

    # Final results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Initial design: x = [{x0[0]:.4f}, {x0[1]:.4f}]")
    print(f"Initial objective: {objectives[0]:.6f}")
    print(f"Final design: x = [{result.x[0]:.6f}, {result.x[1]:.6f}]")
    print(f"Final objective: {result.fun:.6e}")

    error_x = np.linalg.norm(result.x - x_opt)
    error_f = abs(result.fun - f_opt)
    print(f"\nError from optimum:")
    print(f"  ||x - x*|| = {error_x:.2e}")
    print(f"  |f - f*| = {error_f:.2e}")

    print(f"\nIterations: {result.nit}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Gate intercepted calls: {len(history)}")

    if error_f < 1e-6:
        print("\n✓ Converged to optimum!")
    else:
        print(f"\n⚠ Not at optimum (error: {error_f:.2e})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent Optimization Example")
    parser.add_argument(
        "--mode",
        choices=["agent", "tools"],
        default="tools",
        help="Mode: 'agent' for full LLM agent, 'tools' for direct tool usage"
    )
    args = parser.parse_args()

    if args.mode == "agent":
        # Check for API key
        if not any([
            os.environ.get("DASHSCOPE_API_KEY"),
            os.environ.get("ANTHROPIC_API_KEY"),
            os.environ.get("OPENAI_API_KEY"),
        ]):
            print("WARNING: No API key found!")
            print("Set DASHSCOPE_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY")
            print("Falling back to tools mode...")
            run_simple_optimization_loop()
        else:
            run_agent_optimization()
    else:
        run_simple_optimization_loop()
