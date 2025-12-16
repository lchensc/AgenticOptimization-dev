"""
Paola - Package for Agentic Optimization with Learning and Analysis
"""

__version__ = "0.1.0"

from .agent import Agent
from .callbacks import (
    EventType,
    AgentEvent,
    EventCapture,
    RichConsoleCallback,
    FileLogger
)

__all__ = [
    "Agent",
    "EventType",
    "AgentEvent",
    "EventCapture",
    "RichConsoleCallback",
    "FileLogger",
]
