"""
Event capture callback for testing.

Captures all events in memory for assertions and analysis.
"""

from .base import AgentEvent, EventType
from typing import Optional


class EventCapture:
    """
    Callback that captures events for testing assertions.

    Stores all events in memory and provides query methods
    for test assertions.

    Example:
        >>> capture = EventCapture()
        >>> agent.register_callback(capture)
        >>> agent.run("Minimize Rosenbrock")
        >>> assert capture.count(EventType.CACHE_HIT) > 0
        >>> assert capture.count(EventType.AGENT_DONE) == 1
    """

    def __init__(self):
        self.events: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        """
        Capture event.

        Args:
            event: Event to capture
        """
        self.events.append(event)

    def get_events(self) -> list[AgentEvent]:
        """Get all captured events."""
        return self.events.copy()

    def get_events_by_type(self, event_type: EventType) -> list[AgentEvent]:
        """
        Filter events by type.

        Args:
            event_type: Event type to filter

        Returns:
            List of matching events
        """
        return [e for e in self.events if e.event_type == event_type]

    def count(self, event_type: EventType) -> int:
        """
        Count events of specific type.

        Args:
            event_type: Event type to count

        Returns:
            Number of events
        """
        return len(self.get_events_by_type(event_type))

    def get_last(self, event_type: Optional[EventType] = None) -> Optional[AgentEvent]:
        """
        Get last event, optionally filtered by type.

        Args:
            event_type: Optional event type filter

        Returns:
            Last event or None
        """
        if event_type is None:
            return self.events[-1] if self.events else None
        else:
            matching = self.get_events_by_type(event_type)
            return matching[-1] if matching else None

    def get_first(self, event_type: Optional[EventType] = None) -> Optional[AgentEvent]:
        """
        Get first event, optionally filtered by type.

        Args:
            event_type: Optional event type filter

        Returns:
            First event or None
        """
        if event_type is None:
            return self.events[0] if self.events else None
        else:
            matching = self.get_events_by_type(event_type)
            return matching[0] if matching else None

    def clear(self) -> None:
        """Clear all captured events."""
        self.events.clear()

    def __len__(self) -> int:
        """Return number of captured events."""
        return len(self.events)

    def __repr__(self) -> str:
        """String representation."""
        return f"EventCapture({len(self)} events)"

    def get_event_summary(self) -> dict[EventType, int]:
        """
        Get summary of events by type.

        Returns:
            Dictionary mapping event types to counts
        """
        summary = {}
        for event in self.events:
            summary[event.event_type] = summary.get(event.event_type, 0) + 1
        return summary

    def assert_count(self, event_type: EventType, expected: int) -> None:
        """
        Assert that event count matches expected.

        Args:
            event_type: Event type
            expected: Expected count

        Raises:
            AssertionError: If count doesn't match
        """
        actual = self.count(event_type)
        assert actual == expected, \
            f"Expected {expected} {event_type.value} events, got {actual}"

    def assert_min_count(self, event_type: EventType, min_count: int) -> None:
        """
        Assert that event count is at least min_count.

        Args:
            event_type: Event type
            min_count: Minimum expected count

        Raises:
            AssertionError: If count is below minimum
        """
        actual = self.count(event_type)
        assert actual >= min_count, \
            f"Expected at least {min_count} {event_type.value} events, got {actual}"
