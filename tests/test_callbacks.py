"""
Tests for callback system.
"""

import pytest
import time
import tempfile
from pathlib import Path

from aopt.callbacks import (
    AgentEvent,
    EventType,
    CallbackManager,
    create_event,
    EventCapture,
    FileLogger,
    RichConsoleCallback
)


def test_event_creation():
    """Test creating events."""
    event = AgentEvent(
        event_type=EventType.AGENT_START,
        timestamp=time.time(),
        iteration=0,
        data={"goal": "minimize f"}
    )

    assert event.event_type == EventType.AGENT_START
    assert event.iteration == 0
    assert event.data["goal"] == "minimize f"


def test_create_event_helper():
    """Test event creation helper."""
    event = create_event(
        event_type=EventType.TOOL_CALL,
        iteration=5,
        data={"tool_name": "evaluate"}
    )

    assert event.event_type == EventType.TOOL_CALL
    assert event.iteration == 5
    assert event.timestamp > 0


def test_callback_manager_registration():
    """Test callback registration."""
    manager = CallbackManager()

    def my_callback(event):
        pass

    assert len(manager) == 0

    manager.register(my_callback)
    assert len(manager) == 1

    manager.unregister(my_callback)
    assert len(manager) == 0


def test_callback_manager_emit():
    """Test event emission."""
    manager = CallbackManager()
    received_events = []

    def my_callback(event):
        received_events.append(event)

    manager.register(my_callback)

    event = create_event(EventType.AGENT_START, iteration=0)
    manager.emit(event)

    assert len(received_events) == 1
    assert received_events[0] == event


def test_callback_manager_error_isolation():
    """Test that callback errors don't break emission."""
    manager = CallbackManager()
    received_events = []

    def failing_callback(event):
        raise ValueError("Callback error!")

    def success_callback(event):
        received_events.append(event)

    manager.register(failing_callback)
    manager.register(success_callback)

    event = create_event(EventType.AGENT_START)

    # Should not raise despite failing callback
    manager.emit(event)

    # Success callback should still have received event
    assert len(received_events) == 1


def test_event_capture():
    """Test EventCapture callback."""
    capture = EventCapture()

    # Emit some events
    events = [
        create_event(EventType.AGENT_START, iteration=0),
        create_event(EventType.TOOL_CALL, iteration=1),
        create_event(EventType.TOOL_RESULT, iteration=1),
        create_event(EventType.CACHE_HIT, iteration=2),
        create_event(EventType.AGENT_DONE, iteration=10)
    ]

    for event in events:
        capture(event)

    # Test counting
    assert len(capture) == 5
    assert capture.count(EventType.AGENT_START) == 1
    assert capture.count(EventType.CACHE_HIT) == 1
    assert capture.count(EventType.TOOL_CALL) == 1

    # Test filtering
    tool_events = capture.get_events_by_type(EventType.TOOL_CALL)
    assert len(tool_events) == 1
    assert tool_events[0].event_type == EventType.TOOL_CALL

    # Test first/last
    first = capture.get_first()
    assert first.event_type == EventType.AGENT_START

    last = capture.get_last()
    assert last.event_type == EventType.AGENT_DONE

    # Test summary
    summary = capture.get_event_summary()
    assert summary[EventType.AGENT_START] == 1
    assert summary[EventType.CACHE_HIT] == 1


def test_event_capture_assertions():
    """Test EventCapture assertion methods."""
    capture = EventCapture()

    capture(create_event(EventType.CACHE_HIT))
    capture(create_event(EventType.CACHE_HIT))
    capture(create_event(EventType.AGENT_DONE))

    # Should pass
    capture.assert_count(EventType.CACHE_HIT, 2)
    capture.assert_count(EventType.AGENT_DONE, 1)
    capture.assert_min_count(EventType.CACHE_HIT, 1)
    capture.assert_min_count(EventType.CACHE_HIT, 2)

    # Should fail
    with pytest.raises(AssertionError):
        capture.assert_count(EventType.CACHE_HIT, 5)

    with pytest.raises(AssertionError):
        capture.assert_min_count(EventType.CACHE_HIT, 10)


def test_file_logger():
    """Test FileLogger callback."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"

        logger = FileLogger(str(log_file))

        # Log some events
        events = [
            create_event(EventType.AGENT_START, data={"goal": "minimize"}),
            create_event(EventType.ITERATION_COMPLETE, iteration=1, data={"obj": 0.5}),
            create_event(EventType.AGENT_DONE, iteration=10)
        ]

        for event in events:
            logger(event)

        # Verify file exists and has content
        assert log_file.exists()

        # Verify can load back
        loaded_events = FileLogger.load_from_file(str(log_file))
        assert len(loaded_events) == 3
        assert loaded_events[0].event_type == EventType.AGENT_START
        assert loaded_events[1].iteration == 1
        assert loaded_events[2].event_type == EventType.AGENT_DONE


def test_file_logger_counts():
    """Test FileLogger event counting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"
        logger = FileLogger(str(log_file))

        # Log events
        logger(create_event(EventType.CACHE_HIT))
        logger(create_event(EventType.CACHE_HIT))
        logger(create_event(EventType.TOOL_CALL))

        # Test counts
        assert logger.count(EventType.CACHE_HIT) == 2
        assert logger.count(EventType.TOOL_CALL) == 1
        assert logger.count(EventType.AGENT_DONE) == 0


def test_rich_console_callback():
    """Test RichConsoleCallback doesn't crash."""
    callback = RichConsoleCallback(verbose=False)

    # Should not raise
    callback(create_event(EventType.AGENT_START, data={"goal": "test"}))
    callback(create_event(EventType.TOOL_CALL, data={"tool_name": "test"}))
    callback(create_event(EventType.CACHE_HIT, data={"saved_cost": 0.5}))
    callback(create_event(EventType.AGENT_DONE, data={"reason": "converged"}))


def test_multiple_callbacks():
    """Test using multiple callbacks simultaneously."""
    manager = CallbackManager()

    capture = EventCapture()
    received_by_custom = []

    def custom_callback(event):
        received_by_custom.append(event)

    manager.register(capture)
    manager.register(custom_callback)

    event = create_event(EventType.ITERATION_COMPLETE, iteration=5)
    manager.emit(event)

    # Both should receive
    assert len(capture) == 1
    assert len(received_by_custom) == 1
    assert capture.get_last() == event
    assert received_by_custom[0] == event
