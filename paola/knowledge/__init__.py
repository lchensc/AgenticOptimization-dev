"""
Knowledge module for optimization insights.

SKELETON IMPLEMENTATION - Interfaces defined, minimal implementation.

This module enables organizational learning across optimization runs:
- Agent stores insights from successful optimizations
- Agent retrieves insights when starting similar problems
- Knowledge accumulates over time

Current Status:
    SKELETON - Ready for iteration with real data

Usage:
    from paola.knowledge import KnowledgeBase

    # Create knowledge base
    kb = KnowledgeBase()

    # Store insight
    insight_id = kb.store_insight(
        problem_signature={"dimensions": 10, "problem_type": "nonlinear"},
        strategy={"algorithm": "SLSQP", "settings": {"ftol": 1e-6}},
        outcome={"success": True, "iterations": 45}
    )

    # Retrieve insights (currently returns empty - needs embeddings)
    insights = kb.retrieve_insights(
        problem_signature={"dimensions": 10, "problem_type": "nonlinear"}
    )
"""

from .knowledge_base import KnowledgeBase
from .storage import KnowledgeStorage, MemoryKnowledgeStorage, FileKnowledgeStorage

__all__ = [
    "KnowledgeBase",
    "KnowledgeStorage",
    "MemoryKnowledgeStorage",
    "FileKnowledgeStorage",
]
