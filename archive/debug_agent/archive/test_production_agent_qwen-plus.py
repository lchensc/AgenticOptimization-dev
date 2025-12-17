"""
Test production agent with same setup as minimal agent test.

Compares production ReAct agent (aopt.agent.Agent) with minimal agent.
Same goal, same tools, same verbose output format.
"""

import logging
from paola.agent.react_agent import build_optimization_agent
from paola.tools.optimizer_tools import run_scipy_optimization
from paola.tools.evaluator_tools import create_benchmark_problem
from paola.tools.observation_tools import analyze_convergence
from paola.callbacks import CallbackManager, RichConsoleCallback
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.WARNING)  # Reduce logging noise
logger = logging.getLogger(__name__)


def test_production_agent_full_run():
    """
    Test production ReAct agent on Rosenbrock problem.

    Same setup as minimal agent test for direct comparison.
    """

    print("\n" + "=" * 70)
    print("PRODUCTION AGENT: 10D Rosenbrock (Full Run)")
    print("=" * 70)
    print()
    print("Problem: 10D Rosenbrock")
    print("Known optimum: f* = 0.0")
    print()

    # Setup tools - agent needs create_benchmark_problem to formulate the problem autonomously
    tools = [create_benchmark_problem, run_scipy_optimization, analyze_convergence]

    print("-" * 70)
    print("Initializing Qwen-plus LLM...")
    print("-" * 70)
    print("Tools available to agent:")
    for tool in tools:
        print(f"  - {tool.name}")
    print()

    # Setup callback manager
    callback_manager = CallbackManager()
    callback_manager.register(RichConsoleCallback(verbose=True))

    # Build agent graph
    print("-" * 70)
    print("Starting Production Agent...")
    print("-" * 70)
    print()

    agent = build_optimization_agent(
        tools=tools,
        llm_model="qwen-plus",
        callback_manager=callback_manager,
        temperature=0.0
    )

    # Same goal as minimal agent test
    goal = "Solve a 10D Rosenbrock optimization problem with SLSQP and BFGS, and compare the results."

    print("=" * 60)
    print(f"Goal: {goal}")
    print(f"Tools: {[t.name for t in tools]}")
    print("=" * 60)
    print()

    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=goal)],
        "context": {
            "goal": goal,
            "iteration": 0
        },
        "done": False,
        "iteration": 0,
        "callback_manager": callback_manager
    }

    # Run agent
    try:
        final_state = agent.invoke(initial_state)

        print()
        print("=" * 70)
        print("AGENT COMPLETED")
        print("=" * 70)
        print(f"Success: {final_state['done']}")
        print(f"Iterations: {final_state['iteration']}")

    except Exception as e:
        print()
        print("=" * 70)
        print("AGENT FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    test_production_agent_full_run()
