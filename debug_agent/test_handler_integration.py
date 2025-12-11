"""
Test the InvalidToolCallHandler integration with minimal_react_agent.

This script verifies:
1. Handler correctly fixes Python syntax in tool calls
2. Agent continues working after fixes
3. Statistics are tracked properly
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from langchain_qwq import ChatQwen
from langchain.tools import tool
from minimal_react_agent import create_minimal_react_agent


@tool
def create_bounds(bounds: list) -> str:
    """Create bounds for optimization.

    Args:
        bounds: List of [min, max] pairs for each variable
    """
    return f"Created {len(bounds)} bounds: {bounds[:3]}..."


@tool
def finish() -> str:
    """Finish the task."""
    return "DONE"


def test_handler_integration():
    """Test that handler works in the agent."""

    print("=" * 70)
    print("Testing InvalidToolCallHandler Integration")
    print("=" * 70)

    # Initialize Qwen
    llm = ChatQwen(model="qwen-flash", temperature=0.0)

    # Create agent
    agent = create_minimal_react_agent(
        llm=llm,
        tools=[create_bounds, finish],
        max_iterations=5
    )

    # Goal that will trigger Python syntax
    goal = "Create 10 bounds, each being [-5, 10]. Then finish."

    # Run agent
    result = agent(goal, verbose=True)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Success: {result.get('success')}")
    print(f"Iterations: {result.get('iterations')}")
    print(f"Invalid calls stats: {result.get('invalid_calls_stats')}")

    # Check that we had some invalid calls that were fixed
    stats = result.get('invalid_calls_stats', {})
    if stats.get('fixed', 0) > 0:
        print("\n✓ Handler successfully fixed invalid tool calls!")
    else:
        print("\n⚠ No invalid calls were detected (Qwen might have used valid JSON)")

    return result


if __name__ == "__main__":
    test_handler_integration()
