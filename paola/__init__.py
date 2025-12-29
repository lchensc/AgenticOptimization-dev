"""
Paola - Package for Agentic Optimization with Learning and Analysis

v0.2.0 Recording API:
    import paola

    # Start new graph
    f = paola.objective(problem_id=7, goal="Minimize drag")

    # Run optimization
    from scipy.optimize import minimize
    result = minimize(f, x0, method='SLSQP')

    # Checkpoint (returns summary for agent inspection)
    summary = paola.checkpoint(f, script=SCRIPT, reasoning="Initial attempt")

    # Continue graph from checkpoint
    f = paola.continue_graph(42, parent_node="n1", edge_type="warm_start")

    # Finalize
    paola.finalize_graph(42)
"""

__version__ = "0.5.0"  # v0.2.0 redesign

# Recording API (v0.2.0)
from .api import (
    objective,
    checkpoint,
    continue_graph,
    complete,
    finalize_graph,
    get_foundry,
    set_foundry_dir,
)

# Legacy agent API
from .agent import build_optimization_agent, build_conversational_agent
from .callbacks import (
    EventType,
    AgentEvent,
    EventCapture,
    RichConsoleCallback,
    FileLogger
)

__all__ = [
    # Recording API (v0.2.0)
    "objective",
    "checkpoint",
    "continue_graph",
    "complete",
    "finalize_graph",
    "get_foundry",
    "set_foundry_dir",
    # Legacy agent API
    "build_optimization_agent",
    "build_conversational_agent",
    "EventType",
    "AgentEvent",
    "EventCapture",
    "RichConsoleCallback",
    "FileLogger",
]
