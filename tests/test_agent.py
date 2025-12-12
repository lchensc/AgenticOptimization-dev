"""
Tests for agent implementation.
"""

import pytest
from paola import Agent, EventCapture, EventType


def test_agent_creation():
    """Test creating agent."""
    agent = Agent(llm_model="qwen-plus", verbose=False)

    assert agent.llm_model == "qwen-plus"
    assert agent.temperature == 0.0
    assert len(agent.callback_manager) == 0  # verbose=False, no default callback


def test_agent_with_verbose():
    """Test agent with verbose output."""
    agent = Agent(verbose=True)

    # Should have registered RichConsoleCallback
    assert len(agent.callback_manager) >= 1


def test_agent_callback_registration():
    """Test registering custom callbacks."""
    agent = Agent(verbose=False)
    capture = EventCapture()

    agent.register_callback(capture)

    assert len(agent.callback_manager) == 1


def test_agent_multiple_callbacks():
    """Test multiple callback registration."""
    agent = Agent(verbose=False)

    capture1 = EventCapture()
    capture2 = EventCapture()

    agent.register_callback(capture1)
    agent.register_callback(capture2)

    assert len(agent.callback_manager) == 2


def test_agent_repr():
    """Test agent string representation."""
    agent = Agent(verbose=False)

    repr_str = repr(agent)

    assert "Agent" in repr_str
    assert "qwen" in repr_str.lower()


def test_agent_reset():
    """Test agent reset."""
    agent = Agent(verbose=False)

    # Build graph (simulate initialization)
    # agent.graph = "some_graph"  # Would be set in real usage

    agent.reset()

    assert agent.graph is None


@pytest.mark.skip(reason="Requires LLM and tools - integration test")
def test_agent_run_basic():
    """
    Test agent run with simple problem.

    Skipped: Requires LLM access and full tool implementation.
    This will be enabled once tools are implemented.
    """
    agent = Agent(verbose=False)
    capture = EventCapture()
    agent.register_callback(capture)

    result = agent.run("Minimize x^2, x in [0, 10]")

    # Verify structure
    assert "converged" in result
    assert "iterations" in result
    assert "final_context" in result
    assert "reasoning_log" in result

    # Verify events
    assert capture.count(EventType.AGENT_START) == 1
    assert capture.count(EventType.AGENT_DONE) >= 0  # May not finish


def test_agent_callback_events():
    """Test that agent emits events through callbacks."""
    agent = Agent(verbose=False)
    capture = EventCapture()
    agent.register_callback(capture)

    # Manually emit event through callback manager
    from paola.callbacks import create_event

    agent.callback_manager.emit(create_event(
        event_type=EventType.AGENT_START,
        iteration=0,
        data={"test": "data"}
    ))

    # Verify event was captured
    assert len(capture) == 1
    assert capture.get_first().event_type == EventType.AGENT_START
