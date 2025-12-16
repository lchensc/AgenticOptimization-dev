"""
Agent module - Agent implementations for Paola.

Provides two agent types:
- ReAct agent: Autonomous task completion (original)
- Conversational agent: Interactive request-response (like Claude Code)
"""

from .agent import Agent
from .react_agent import build_optimization_agent, AgentState
from .conversational_agent import build_conversational_agent

__all__ = [
    "Agent",
    "build_optimization_agent",  # ReAct agent (autonomous)
    "build_conversational_agent",  # Conversational agent (interactive)
    "AgentState"
]
