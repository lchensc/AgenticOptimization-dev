"""
Paola Recording API - Public interface for optimization recording.

This module provides the user-facing API for the v0.2.0 recording infrastructure:
- objective(): Start new graph, return RecordingObjective
- checkpoint(): End node, persist, return summary
- continue_graph(): Resume graph from parent node
- complete(): Shorthand for checkpoint() + finalize_graph()
- finalize_graph(): Mark graph as complete

Design:
- LLM writes Python optimization code directly
- One LLM turn = one node
- Multi-turn pattern enables dynamic decision-making

Example:
    import paola
    from scipy.optimize import minimize

    # Turn 1: Start optimization
    f = paola.objective(problem_id=7, goal="Minimize drag")
    result = minimize(f, x0, method='SLSQP')
    summary = paola.checkpoint(f, script=SCRIPT, reasoning="Initial attempt")
    print(json.dumps(summary))  # Returns to agent

    # Turn 2: Continue from checkpoint
    f = paola.continue_graph(42, parent_node="n1", edge_type="warm_start")
    result = minimize(f, f.get_warm_start(), method='L-BFGS-B')
    summary = paola.complete(f, script=SCRIPT, reasoning="Refinement")
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from paola.recording import RecordingObjective, EvaluationCache
from paola.foundry import OptimizationFoundry
from paola.foundry.storage import FileStorage


# Module-level foundry instance (lazy initialization)
_foundry: Optional[OptimizationFoundry] = None
_foundry_base_dir: str = ".paola_foundry"


def _get_foundry() -> OptimizationFoundry:
    """Get or create the module-level Foundry instance."""
    global _foundry
    if _foundry is None:
        storage = FileStorage(base_dir=_foundry_base_dir)
        _foundry = OptimizationFoundry(storage=storage)
    return _foundry


# Module-level journal instance (lazy initialization)
_journal = None


def _get_journal():
    """Get or create the module-level Journal instance."""
    global _journal
    if _journal is None:
        from paola.foundry.journal import GraphJournal
        journal_path = Path(_foundry_base_dir) / "journal.jsonl"
        _journal = GraphJournal(journal_path)
    return _journal


def set_foundry_dir(base_dir: str) -> None:
    """
    Set the base directory for Foundry storage.

    Must be called before any API functions if using non-default directory.

    Args:
        base_dir: Path to storage directory (default: ".paola_foundry")
    """
    global _foundry, _foundry_base_dir, _journal
    _foundry_base_dir = base_dir
    _foundry = None  # Reset to force re-initialization
    _journal = None  # Reset journal as well


def _get_scripts_dir(graph_id: int) -> Path:
    """Get scripts directory for a graph."""
    return Path(_foundry_base_dir) / "scripts" / f"graph_{graph_id:04d}"


def _get_cache_dir(graph_id: int) -> Path:
    """Get cache directory for a graph."""
    return Path(_foundry_base_dir) / "cache" / f"graph_{graph_id:04d}"


# Track active RecordingObjectives by (graph_id, node_id)
_active_objectives: Dict[tuple, RecordingObjective] = {}

# Store checkpoint summaries for warm-start support
# Key: (graph_id, node_id), Value: summary dict with best_x
_checkpoint_summaries: Dict[tuple, Dict[str, Any]] = {}


def objective(
    problem_id: int,
    goal: Optional[str] = None,
) -> RecordingObjective:
    """
    Start new optimization graph and return RecordingObjective.

    This creates a new graph in Foundry and returns a callable objective
    that records all evaluations automatically.

    Args:
        problem_id: Problem ID (from create_nlp_problem or registered problems)
        goal: Natural language optimization goal

    Returns:
        RecordingObjective: Callable that wraps the problem's evaluator

    Example:
        f = paola.objective(problem_id=7, goal="Minimize drag coefficient")
        result = minimize(f, x0, method='SLSQP')

    Note:
        The graph is created immediately (for crash recovery).
        Call checkpoint() or complete() to persist node data.
    """
    foundry = _get_foundry()

    # Get problem evaluator
    evaluator = foundry.get_problem_evaluator(problem_id)
    if evaluator is None:
        raise ValueError(f"Problem {problem_id} not found in Foundry")

    # Create graph
    graph = foundry.create_graph(problem_id=problem_id, goal=goal)
    graph_id = graph.graph_id
    node_id = "n1"  # First node

    # Setup directories
    cache_dir = _get_cache_dir(graph_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = _get_scripts_dir(graph_id)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Create RecordingObjective
    # The evaluator's __call__ method is the actual function
    eval_func = evaluator.evaluate if hasattr(evaluator, 'evaluate') else evaluator

    recording_obj = RecordingObjective(
        evaluator=eval_func,
        graph_id=graph_id,
        node_id=node_id,
        cache_dir=cache_dir,
        problem_id=problem_id,
        goal=goal,
        parent_best_x=None,  # First node has no parent
    )

    # Track for later use
    _active_objectives[(graph_id, node_id)] = recording_obj

    # Log graph creation to journal for cross-process visibility
    journal = _get_journal()
    journal.append({
        "type": "graph_created",
        "graph_id": graph_id,
        "problem_id": problem_id,
        "goal": goal,
    })

    return recording_obj


def checkpoint(
    f: RecordingObjective,
    script: str,
    reasoning: Optional[str] = None,
    optimizer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End current node, persist data, return summary for LLM inspection.

    This saves:
    - Node evaluation data to graph
    - Script to scripts/graph_XXXX/nX.py
    - Summary for agent to inspect and decide next action

    Args:
        f: The RecordingObjective from objective() or continue_graph()
        script: Python source code that was executed
        reasoning: LLM's reasoning for this optimization strategy
        optimizer: LLM-reported optimizer (e.g., "scipy:SLSQP")

    Returns:
        Summary dict with:
        - graph_id: Graph identifier
        - node_id: Node identifier within graph
        - best_x: Best design found
        - best_f: Best objective value
        - n_evaluations: Number of evaluations
        - optimizer: Optimizer used (reported by LLM)
        - optimizer_source: "reported" (always in MVP)
        - has_crashes: Whether any evaluations crashed
        - status: "ok" or "has_crashes"

    Example:
        summary = paola.checkpoint(f,
            script=SCRIPT,
            reasoning="SLSQP chosen for smooth convex problem"
        )
        # Agent inspects summary and decides next action
    """
    graph_id = f.graph_id
    node_id = f.node_id

    # Save script
    scripts_dir = _get_scripts_dir(graph_id)
    script_path = scripts_dir / f"{node_id}.py"
    script_path.write_text(script)

    # Get summary from RecordingObjective
    obj_summary = f.summary()

    # Build checkpoint summary for agent
    summary = {
        "graph_id": graph_id,
        "node_id": node_id,
        "problem_id": obj_summary.get("problem_id"),
        "goal": obj_summary.get("goal"),
        "best_x": obj_summary.get("best_x"),
        "best_f": obj_summary.get("best_f"),
        "n_evaluations": obj_summary.get("n_evaluations", 0),
        "n_actual_evals": obj_summary.get("n_actual_evals", 0),
        "n_cache_hits": obj_summary.get("n_cache_hits", 0),
        "cache_hit_rate": obj_summary.get("cache_hit_rate", 0.0),
        "total_eval_time": obj_summary.get("total_eval_time", 0.0),
        "wall_time": obj_summary.get("wall_time", 0.0),
        "optimizer": optimizer or "unknown",
        "optimizer_source": "reported",  # MVP: always LLM-reported
        "reasoning": reasoning,
        "script_ref": f"scripts/graph_{graph_id:04d}/{node_id}.py",
        "has_crashes": obj_summary.get("has_crashes", False),
        "status": "has_crashes" if obj_summary.get("has_crashes") else "ok",
    }

    # Add crash info if present
    if summary["has_crashes"]:
        crash_info = f.get_crash_info()
        if crash_info:
            last_crash = crash_info[-1]
            summary["last_crash"] = {
                "x": last_crash.get("x"),
                "error": last_crash.get("error"),
                "error_type": last_crash.get("error_type"),
            }

    # Store checkpoint summary for warm-start support
    _checkpoint_summaries[(graph_id, node_id)] = summary

    # Clean up active objective tracking
    if (graph_id, node_id) in _active_objectives:
        del _active_objectives[(graph_id, node_id)]

    # Log to journal for cross-process visibility
    journal = _get_journal()
    journal.append({
        "type": "checkpoint",
        "graph_id": graph_id,
        "node_id": node_id,
        "best_f": summary.get("best_f"),
        "best_x": summary.get("best_x"),
        "n_evals": summary.get("n_evaluations"),
        "optimizer": optimizer,
    })

    return summary


def continue_graph(
    graph_id: int,
    parent_node: str,
    edge_type: str = "warm_start",
) -> RecordingObjective:
    """
    Resume graph from parent node for next turn.

    This continues an existing graph by creating a new node that
    inherits context from the parent node.

    Args:
        graph_id: Graph ID to continue
        parent_node: Node ID to continue from (e.g., "n1")
        edge_type: Relationship to parent:
            - "warm_start": Use parent's best_x as starting point
            - "restart": Fresh start with knowledge of parent result
            - "branch": Explore alternative from same starting point
            - "refine": Tighten tolerances to polish solution

    Returns:
        RecordingObjective: Callable with parent's best_x available via get_warm_start()

    Example:
        f = paola.continue_graph(42, parent_node="n1", edge_type="warm_start")
        x0 = f.get_warm_start()  # Parent's best solution
        result = minimize(f, x0, method='L-BFGS-B')
    """
    foundry = _get_foundry()
    journal = _get_journal()

    # First, try to get parent's best_x from checkpoint summaries (same-session)
    parent_best_x = None
    checkpoint_key = (graph_id, parent_node)
    if checkpoint_key in _checkpoint_summaries:
        parent_summary = _checkpoint_summaries[checkpoint_key]
        if parent_summary.get("best_x"):
            parent_best_x = np.array(parent_summary["best_x"])

    # Second, try journal (cross-process fallback)
    if parent_best_x is None:
        journal_best_x = journal.get_node_best_x(graph_id, parent_node)
        if journal_best_x is not None:
            parent_best_x = np.array(journal_best_x)

    # Load graph record to get parent node info
    record = foundry.load_graph_record(graph_id)
    if record is None:
        # Check if graph is still active
        active = foundry.get_graph(graph_id)
        if active is None:
            raise ValueError(f"Graph {graph_id} not found")
        # For active graphs, use checkpoint summaries (already checked above)
        problem_id = active.problem_id
        goal = active.goal
    else:
        problem_id = record.problem_id
        goal = record.goal

        # If we didn't find best_x from checkpoint, try graph record
        if parent_best_x is None:
            if record.nodes and parent_node in record.nodes:
                parent_info = record.nodes[parent_node]
                if hasattr(parent_info, 'best_x') and parent_info.best_x is not None:
                    parent_best_x = np.array(parent_info.best_x)
                elif isinstance(parent_info, dict) and parent_info.get('best_x'):
                    parent_best_x = np.array(parent_info['best_x'])

    # Get problem evaluator
    evaluator = foundry.get_problem_evaluator(problem_id)
    if evaluator is None:
        raise ValueError(f"Problem {problem_id} not found in Foundry")

    # Determine next node ID
    # Count existing nodes from checkpoint summaries + graph record + journal
    checkpoint_nodes = {
        k[1] for k in _checkpoint_summaries.keys()
        if k[0] == graph_id
    }
    record_nodes = set(record.nodes.keys()) if record and record.nodes else set()
    journal_node_count = journal.count_nodes(graph_id)
    # Take max of all sources
    existing_count = max(
        len(checkpoint_nodes | record_nodes),
        journal_node_count,
        1  # At least 1 for first node
    )
    node_id = f"n{existing_count + 1}"

    # Get/create cache directory (shared across nodes in same graph)
    cache_dir = _get_cache_dir(graph_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create RecordingObjective
    eval_func = evaluator.evaluate if hasattr(evaluator, 'evaluate') else evaluator

    recording_obj = RecordingObjective(
        evaluator=eval_func,
        graph_id=graph_id,
        node_id=node_id,
        cache_dir=cache_dir,
        problem_id=problem_id,
        goal=goal,
        parent_best_x=parent_best_x,
    )

    # Track for later use
    _active_objectives[(graph_id, node_id)] = recording_obj

    # Log continue to journal for cross-process visibility
    journal.append({
        "type": "continue",
        "graph_id": graph_id,
        "parent_node": parent_node,
        "edge_type": edge_type,
        "new_node_id": node_id,
    })

    return recording_obj


def complete(
    f: RecordingObjective,
    script: str,
    reasoning: Optional[str] = None,
    optimizer: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Shorthand for checkpoint() + finalize_graph().

    Use this when finishing a simple single-node optimization.

    Args:
        f: The RecordingObjective
        script: Python source code that was executed
        reasoning: LLM's reasoning for this optimization strategy
        optimizer: LLM-reported optimizer

    Returns:
        Summary dict (same as checkpoint(), plus finalized=True)

    Example:
        summary = paola.complete(f,
            script=SCRIPT,
            reasoning="Direct gradient solve for smooth problem"
        )
    """
    # First checkpoint
    summary = checkpoint(f, script, reasoning, optimizer)

    # Then finalize graph
    finalize_graph(summary["graph_id"])

    summary["finalized"] = True
    return summary


def finalize_graph(graph_id: int) -> Optional[Dict[str, Any]]:
    """
    Mark graph as complete.

    This persists the graph to storage and removes it from active tracking.

    Args:
        graph_id: Graph ID to finalize

    Returns:
        Final graph summary or None if graph not found

    Note:
        After finalization, the graph can be queried via query_past_graphs()
        but cannot be continued with continue_graph().
    """
    foundry = _get_foundry()

    # Finalize via foundry (handles persistence)
    result = foundry.finalize_graph(graph_id)

    if result is None:
        return None

    # Log finalization to journal
    journal = _get_journal()
    journal.append({
        "type": "finalized",
        "graph_id": graph_id,
    })

    return {
        "graph_id": graph_id,
        "status": "finalized",
        "pattern": getattr(result, 'pattern', 'unknown'),
        "final_objective": getattr(result, 'final_objective', None),
    }


# Convenience function to get Foundry for advanced use
def get_foundry() -> OptimizationFoundry:
    """
    Get the Foundry instance for advanced operations.

    Returns:
        OptimizationFoundry instance

    Example:
        foundry = paola.get_foundry()
        records = foundry.query_graphs(problem_id=7, limit=10)
    """
    return _get_foundry()
