"""
Callback system for real-time agent event streaming.
"""

from .base import (
    AgentEvent,
    EventType,
    CallbackFunction,
    CallbackManager,
    create_event
)
from .rich_console import RichConsoleCallback
from .file_logger import FileLogger
from .capture import EventCapture

__all__ = [
    # Core
    "AgentEvent",
    "EventType",
    "CallbackFunction",
    "CallbackManager",
    "create_event",
    # Implementations
    "RichConsoleCallback",
    "FileLogger",
    "EventCapture",
]
