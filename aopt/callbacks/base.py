"""
Core callback system for real-time agent event streaming.

Defines event types, event structure, and callback management.
"""

from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Optional, Callable
import time
import logging

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """All event types emitted by agent."""

    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_STEP = "agent_step"           # Start of ReAct cycle
    AGENT_DONE = "agent_done"

    # Formulation
    FORMULATION_START = "formulation_start"
    FORMULATION_COMPLETE = "formulation_complete"
    FORMULATION_QUESTION = "formulation_question"  # Agent needs user input

    # Agent reasoning
    REASONING = "reasoning"             # Agent's thought process

    # Tool execution
    TOOL_CALL = "tool_call"             # About to call tool
    TOOL_RESULT = "tool_result"         # Tool returned result
    TOOL_ERROR = "tool_error"           # Tool failed

    # Optimization progress
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"
    EVALUATION = "evaluation"           # Function evaluation
    CACHE_HIT = "cache_hit"             # Evaluation from cache

    # Convergence & observation
    CONVERGENCE_CHECK = "convergence_check"
    PATTERN_DETECTED = "pattern_detected"  # Agent detected issue

    # Adaptation
    ADAPTATION_START = "adaptation_start"
    ADAPTATION_COMPLETE = "adaptation_complete"
    RESTART = "restart"                 # Optimizer restart

    # Resource tracking
    BUDGET_UPDATE = "budget_update"

    # Storage events
    OPTIMIZATION_COMPLETE = "optimization_complete"
    PROBLEM_CREATED = "problem_created"


class AgentEvent(BaseModel):
    """
    Structured event emitted by agent.

    All callbacks receive AgentEvent instances.

    Example:
        >>> event = AgentEvent(
        ...     event_type=EventType.TOOL_CALL,
        ...     timestamp=time.time(),
        ...     iteration=12,
        ...     data={"tool_name": "evaluate_function", "arguments": {...}}
        ... )
    """

    # Event metadata
    event_type: EventType = Field(..., description="Type of event")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp")
    iteration: int = Field(default=0, description="Current iteration number")

    # Event-specific data
    data: dict[str, Any] = Field(default_factory=dict, description="Event payload")

    # Context snapshot (optional, for rich events)
    context: Optional[dict] = Field(None, description="Agent context snapshot")

    class Config:
        # Allow serialization to JSON for file logging
        json_encoders = {
            EventType: lambda v: v.value
        }


# Type alias for callback functions
CallbackFunction = Callable[[AgentEvent], None]


class CallbackManager:
    """
    Manages multiple callbacks, handles errors.

    Allows registering multiple callbacks that run in order.
    If one callback fails, others still execute (error isolation).

    Example:
        >>> manager = CallbackManager()
        >>> manager.register(my_callback)
        >>> manager.register(another_callback)
        >>> manager.emit(AgentEvent(...))  # Both callbacks executed
    """

    def __init__(self):
        self.callbacks: list[CallbackFunction] = []

    def register(self, callback: CallbackFunction) -> None:
        """
        Add callback to list.

        Args:
            callback: Function that receives AgentEvent
        """
        if not callable(callback):
            raise TypeError(f"Callback must be callable, got {type(callback)}")
        self.callbacks.append(callback)
        logger.debug(f"Registered callback: {callback}")

    def unregister(self, callback: CallbackFunction) -> None:
        """Remove callback from list."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.debug(f"Unregistered callback: {callback}")

    def emit(self, event: AgentEvent) -> None:
        """
        Send event to all registered callbacks.

        Catches exceptions to prevent callback failures from
        breaking the optimization.

        Args:
            event: Event to emit
        """
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                # Log error but continue - don't let callback failures break optimization
                logger.error(
                    f"Callback {callback.__name__ if hasattr(callback, '__name__') else callback} "
                    f"failed with error: {e}",
                    exc_info=True
                )

    def clear(self) -> None:
        """Remove all callbacks."""
        self.callbacks.clear()
        logger.debug("Cleared all callbacks")

    def __len__(self) -> int:
        """Return number of registered callbacks."""
        return len(self.callbacks)


def create_event(
    event_type: EventType,
    iteration: int = 0,
    data: Optional[dict] = None,
    context: Optional[dict] = None
) -> AgentEvent:
    """
    Convenience function to create events.

    Args:
        event_type: Type of event
        iteration: Current iteration
        data: Event payload
        context: Optional context snapshot

    Returns:
        AgentEvent instance
    """
    return AgentEvent(
        event_type=event_type,
        timestamp=time.time(),
        iteration=iteration,
        data=data or {},
        context=context
    )
