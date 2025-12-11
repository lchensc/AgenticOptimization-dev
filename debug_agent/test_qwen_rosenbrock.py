"""
End-to-end test: Qwen-flash agent solves 10D Rosenbrock.

This test demonstrates the full agentic optimization workflow:
1. Initialize Qwen-flash LLM
2. Create ReAct agent with optimization tools
3. Agent autonomously solves: "Solve a 10d Rosenbrock optimization problem with SLSQP"

Requirements:
    export DASHSCOPE_API_KEY=your_key

Usage:
    python debug_agent/test_qwen_rosenbrock.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from scipy.optimize import minimize

from aopt.backends.analytical import Rosenbrock
from aopt.optimizers.gate import OptimizationGate
from aopt.callbacks import CallbackManager, EventType, create_event
from aopt.tools import (
    register_problem,
    clear_problem_registry,
    clear_optimizer_registry,
    register_gate,
    clear_gate_registry,
    cache_clear,
    # Tools for agent (LangChain @tool decorated)
    evaluate_function,
    compute_gradient,
    optimizer_create,
    optimizer_propose,
    optimizer_update,
    optimizer_restart,
    gate_get_history,
    gate_get_statistics,
    analyze_convergence,
    detect_pattern,
    get_gradient_quality,
    compute_improvement_statistics,
)


def setup_problem():
    """Setup the 10D Rosenbrock problem."""
    clear_problem_registry()
    clear_optimizer_registry()
    clear_gate_registry()
    cache_clear()

    problem = Rosenbrock(dimension=10)
    register_problem("rosenbrock_10d", problem)

    return problem


def create_agent_tools():
    """Get the list of tools available to the agent."""
    return [
        evaluate_function,
        compute_gradient,
        optimizer_create,
        optimizer_propose,
        optimizer_update,
        optimizer_restart,
        gate_get_history,
        gate_get_statistics,
        analyze_convergence,
        detect_pattern,
        get_gradient_quality,
        compute_improvement_statistics,
    ]


def run_agent_optimization():
    """Run the full agent optimization."""

    # Check API key
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY not set!")
        print("Please set it: export DASHSCOPE_API_KEY=your_key")
        return None

    print("=" * 70)
    print("AGENTIC OPTIMIZATION: 10D Rosenbrock with Qwen-flash")
    print("=" * 70)

    # Setup problem
    problem = setup_problem()
    x_opt, f_opt = problem.get_optimum()
    print(f"\nProblem: 10D Rosenbrock")
    print(f"Known optimum: x* = [1, 1, ..., 1], f* = {f_opt}")
    print(f"Bounds: [-5, 10] for all variables")

    # Initialize LLM
    print("\n" + "-" * 70)
    print("Initializing Qwen-flash LLM...")

    from aopt.agent.react_agent import initialize_llm, build_aopt_agent

    llm = initialize_llm("qwen-flash", temperature=0.0)
    print(f"LLM initialized: qwen-flash")

    # Get tools
    tools = create_agent_tools()
    print(f"Tools available: {len(tools)}")
    for t in tools:
        print(f"  - {t.name}")

    # Create callback manager for observability
    callback_mgr = CallbackManager()

    # Simple console callback
    def console_callback(event):
        if event.event_type == EventType.AGENT_STEP:
            print(f"\n[Step {event.data.get('step', '?')}]")
        elif event.event_type == EventType.REASONING:
            reasoning = event.data.get('reasoning', '')
            if reasoning:
                # Truncate long reasoning
                if len(reasoning) > 500:
                    reasoning = reasoning[:500] + "..."
                print(f"Agent: {reasoning}")
        elif event.event_type == EventType.TOOL_CALL:
            print(f"  -> Calling: {event.data.get('tool_name', '?')}")
        elif event.event_type == EventType.TOOL_RESULT:
            result = event.data.get('result', {})
            if isinstance(result, dict):
                msg = result.get('message', str(result)[:100])
            else:
                msg = str(result)[:100]
            print(f"  <- Result: {msg}")
        elif event.event_type == EventType.AGENT_DONE:
            print(f"\n[Agent Done] Reason: {event.data.get('reason', 'unknown')}")

    callback_mgr.register(console_callback)

    # Build agent graph
    print("\n" + "-" * 70)
    print("Building ReAct agent...")

    graph = build_aopt_agent(
        tools=tools,
        llm_model="qwen-flash",
        callback_manager=callback_mgr,
        temperature=0.0
    )
    print("Agent graph built successfully")

    # Define the user goal
    user_goal = "Solve a 10d Rosenbrock optimization problem with SLSQP."

    print("\n" + "-" * 70)
    print(f"User Goal: {user_goal}")
    print("-" * 70)

    # Initial state
    initial_state = {
        "messages": [],
        "context": {
            "goal": user_goal,
            "budget_total": 100.0,
            "budget_used": 0.0,
            "problem": {
                "problem_id": "rosenbrock_10d",
                "problem_type": "nonlinear_single",
                "n_variables": 10,
                "bounds": [[-5.0, 10.0]] * 10,
                "known_optimum": {"x": [1.0] * 10, "f": 0.0},
            },
            "optimizer_type": None,
            "iteration": 0,
            "current_objectives": None,
            "best_objectives": None,
            "history": [],
            "observations": {},
            "cache_stats": {"hit_rate": 0.0},
            "budget_status": {
                "total": 100.0,
                "used": 0.0,
                "remaining_pct": 100.0
            }
        },
        "done": False,
        "iteration": 0,
        "callback_manager": callback_mgr
    }

    # Emit start event
    callback_mgr.emit(create_event(
        EventType.AGENT_START,
        iteration=0,
        data={"goal": user_goal, "llm_model": "qwen-flash"}
    ))

    # Run agent
    print("\n" + "=" * 70)
    print("STARTING AGENT EXECUTION")
    print("=" * 70)

    try:
        import time
        start_time = time.time()

        final_state = graph.invoke(initial_state)

        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print("AGENT EXECUTION COMPLETE")
        print("=" * 70)
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Iterations: {final_state.get('iteration', 0)}")
        print(f"Done: {final_state.get('done', False)}")

        # Extract final context
        final_context = final_state.get("context", {})
        print(f"\nFinal Context:")
        print(f"  Best objective: {final_context.get('best_objectives', 'N/A')}")
        print(f"  Budget used: {final_context.get('budget_used', 0):.1f}")

        return final_state

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_manual_optimization():
    """
    Run optimization manually (without LLM) to verify tools work.

    This demonstrates what the agent should do.
    """
    print("=" * 70)
    print("MANUAL OPTIMIZATION (No LLM) - Reference Implementation")
    print("=" * 70)

    # Setup
    problem = setup_problem()
    x_opt, f_opt = problem.get_optimum()

    print(f"\nProblem: 10D Rosenbrock")
    print(f"Known optimum: f* = {f_opt}")

    # Step 1: Create optimizer
    print("\n[Step 1] Creating SLSQP optimizer...")
    result = optimizer_create.invoke({
        "optimizer_id": "slsqp_10d",
        "problem_id": "rosenbrock_10d",
        "algorithm": "SLSQP",
        "bounds": [[-5.0, 10.0]] * 10,
        "initial_design": [0.0] * 10,
        "options": '{"maxiter": 100, "ftol": 1e-9}',
    })
    print(f"  Result: {result['message']}")

    if not result["success"]:
        print(f"  ERROR: {result}")
        return

    # Step 2: Run optimization using gate pattern (more efficient)
    print("\n[Step 2] Running optimization through gate...")

    gate = OptimizationGate(problem_id="rosenbrock_10d", blocking=False)
    register_gate("manual_gate", gate)

    x0 = np.array([0.0] * 10)
    bounds = [(-5, 10)] * 10

    scipy_result = minimize(
        fun=gate.wrap_objective(problem.evaluate),
        x0=x0,
        method='SLSQP',
        jac=gate.wrap_gradient(problem.gradient),
        bounds=bounds,
        options={'ftol': 1e-9, 'maxiter': 100}
    )

    print(f"  Scipy success: {scipy_result.success}")
    print(f"  Scipy message: {scipy_result.message}")
    print(f"  Iterations: {scipy_result.nit}")
    print(f"  Function evals: {scipy_result.nfev}")

    # Step 3: Analyze results
    print("\n[Step 3] Analyzing convergence...")

    history = gate.get_history()
    objectives = [h['objective'] for h in history if 'objective' in h]

    if objectives:
        conv_result = analyze_convergence.invoke({
            "objectives": objectives[-20:],
            "window_size": 5,
        })
        print(f"  Convergence: {conv_result['recommendation']}")

    # Final results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    error_x = np.linalg.norm(scipy_result.x - x_opt)
    error_f = abs(scipy_result.fun - f_opt)

    print(f"Final objective: {scipy_result.fun:.6e}")
    print(f"Error |f - f*|: {error_f:.6e}")
    print(f"Error ||x - x*||: {error_x:.6e}")

    if error_f < 1e-6:
        print("\n✓ Converged to optimum!")
    else:
        print(f"\n⚠ Not at optimum (error: {error_f:.2e})")

    return scipy_result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Qwen agent on 10D Rosenbrock")
    parser.add_argument(
        "--mode",
        choices=["agent", "manual"],
        default="agent",
        help="Mode: 'agent' for LLM agent, 'manual' for reference without LLM"
    )
    args = parser.parse_args()

    if args.mode == "agent":
        result = run_agent_optimization()
    else:
        result = run_manual_optimization()

    if result is None:
        print("\nTest failed!")
        sys.exit(1)
    else:
        print("\nTest completed!")
        sys.exit(0)
