"""
Paola - Package for Agentic Optimization with Learning and Analysis
"""

__version__ = "0.3.1"

from .agent import build_optimization_agent, build_conversational_agent
from .callbacks import (
    EventType,
    AgentEvent,
    EventCapture,
    RichConsoleCallback,
    FileLogger
)

__all__ = [
    "build_optimization_agent",
    "build_conversational_agent",
    "EventType",
    "AgentEvent",
    "EventCapture",
    "RichConsoleCallback",
    "FileLogger",
]
