"""
Test production agent on 10D Rosenbrock optimization.

This test demonstrates the production Agent class with simple optimization tools,
showing the full agent reasoning and tool usage patterns.
"""

import logging
from paola.agent.agent import Agent
from paola.agent.react_agent import build_optimization_agent
from paola.tools.optimizer_tools import run_scipy_optimization
from paola.tools.observation_tools import analyze_convergence
from paola.callbacks import EventCapture, EventType
from langchain_core.messages import HumanMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_production_agent_rosenbrock():
    """
    Test production agent on 10D Rosenbrock problem.

    Uses the production Agent class with:
    - Custom simple tools (run_scipy_optimization, analyze_convergence)
    - Event capture for monitoring
    - Full agent autonomy
    """

    print("\n" + "=" * 70)
    print("PRODUCTION AGENT TEST: 10D Rosenbrock Optimization")
    print("=" * 70)
    print()
    print("Problem: 10D Rosenbrock")
    print("Known optimum: f* = 0.0 at x* = [1, 1, ..., 1]")
    print()

    # Initialize agent with verbose output
    print("-" * 70)
    print("Initializing Production Agent...")
    print("-" * 70)

    agent = Agent(
        llm_model="qwen-plus",
        temperature=0.0,
        verbose=True,  # Use RichConsoleCallback
        max_iterations=10
    )

    # Add event capture for testing
    capture = EventCapture()
    agent.register_callback(capture)

    # Override tools with our simple optimization tools
    agent.tools = [run_scipy_optimization, analyze_convergence]
    print(f"Tools: {[t.name for t in agent.tools]}")
    print()

    # Build agent graph with custom tools
    print("-" * 70)
    print("Building Agent Graph...")
    print("-" * 70)

    agent.graph = build_optimization_agent(
        tools=agent.tools,
        llm_model=agent.llm_model,
        callback_manager=agent.callback_manager,
        temperature=agent.temperature
    )
    print("Agent graph built successfully")
    print()

    # Define optimization goal
    goal = """Solve a 10D Rosenbrock optimization problem with SLSQP and BFGS, and compare the results."""

    print("-" * 70)
    print("Starting Agent...")
    print("-" * 70)
    print(f"Goal: {goal}")
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
        "callback_manager": agent.callback_manager
    }

    # Run agent
    try:
        final_state = agent.graph.invoke(initial_state)

        print()
        print("=" * 70)
        print("✅ AGENT COMPLETED")
        print("=" * 70)
        print(f"Success: {final_state['done']}")
        print(f"Iterations: {final_state['iteration']}")
        print()

        # Verify events were captured
        print("-" * 70)
        print("Event Summary:")
        print("-" * 70)
        print(f"Total events captured: {len(capture)}")
        print(f"Agent start events: {capture.count(EventType.AGENT_START)}")
        print(f"Agent step events: {capture.count(EventType.AGENT_STEP)}")
        print(f"Agent done events: {capture.count(EventType.AGENT_DONE)}")
        print(f"Tool call events: {capture.count(EventType.TOOL_CALL)}")
        print(f"Tool result events: {capture.count(EventType.TOOL_RESULT)}")
        print(f"Tool error events: {capture.count(EventType.TOOL_ERROR)}")
        print()

        # Verify results
        assert final_state['done'], "Agent should complete successfully"
        assert final_state['iteration'] > 0, "Agent should run at least one iteration"
        assert final_state['iteration'] <= 10, "Agent should not exceed max iterations"

        print("✅ All assertions passed!")

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ AGENT FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    test_production_agent_rosenbrock()
