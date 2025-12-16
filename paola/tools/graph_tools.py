"""
Tools for agent to manage optimization graphs.

v0.3.0: Graph-based architecture
- Graph = complete optimization task (may involve multiple nodes)
- Node = single optimizer execution

Uses OptimizationFoundry with dependency injection for data foundation.

Design Principle: "Graph externalizes state, agent makes decisions."
- The graph helps the agent track state (node IDs, not x0 values)
- The agent explicitly decides which node to continue from
- The system does NOT automatically select "best" - that's the agent's decision
"""

from typing import Dict, Any, Optional
from langchain_core.tools import tool

from ..foundry import OptimizationFoundry

# Global foundry reference (set by build_tools)
_FOUNDRY: Optional[OptimizationFoundry] = None


def set_foundry(foundry: OptimizationFoundry) -> None:
    """
    Set global foundry reference for tools.

    This is called by build_tools() during initialization.

    Args:
        foundry: OptimizationFoundry instance
    """
    global _FOUNDRY
    _FOUNDRY = foundry


def get_foundry() -> Optional[OptimizationFoundry]:
    """
    Get current foundry reference.

    Returns:
        Current foundry instance or None if not set
    """
    return _FOUNDRY


@tool
def start_graph(
    problem_id: str,
    goal: str = "",
) -> Dict[str, Any]:
    """
    Start a new optimization graph.

    Must be called before any optimization. Creates a graph to track
    the optimization process. The returned graph_id must be passed
    to run_optimization.

    Args:
        problem_id: Problem identifier (from registered problems)
        goal: Natural language description of optimization goal

    Returns:
        Dict with:
            - success: bool
            - graph_id: int - Use this in run_optimization
            - problem_id: str
            - message: str

    Example:
        result = start_graph(
            problem_id="rosenbrock_10d",
            goal="Minimize Rosenbrock function with multi-start strategy"
        )
        # Returns: {"success": True, "graph_id": 1, ...}

        # Then run optimization
        result = run_optimization(
            graph_id=1,
            optimizer="scipy:L-BFGS-B",
            init_strategy="random"
        )
    """
    try:
        if _FOUNDRY is None:
            return {
                "success": False,
                "message": "Foundry not initialized. Call set_foundry() first.",
            }

        # Create graph
        graph = _FOUNDRY.create_graph(
            problem_id=problem_id,
            goal=goal if goal else None,
        )

        return {
            "success": True,
            "graph_id": graph.graph_id,
            "problem_id": problem_id,
            "message": f"Created graph #{graph.graph_id} for problem '{problem_id}'.",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error creating graph: {str(e)}",
        }


@tool
def get_graph_state(graph_id: int) -> Dict[str, Any]:
    """
    Get current state of an optimization graph for decision making.

    Use this to understand what has been tried and decide next steps.
    The agent should use this information to explicitly decide which
    node to continue from.

    Args:
        graph_id: Graph ID from start_graph

    Returns:
        Dict with:
            - success: bool
            - graph_id: int
            - problem_id: str
            - status: str ("active" or "completed")
            - n_nodes: int
            - n_edges: int
            - pattern: str - "empty", "single", "multistart", "chain", "tree", "dag"
            - nodes: List of node summaries
                - node_id: str (e.g., "n1", "n2")
                - optimizer: str
                - status: str
                - best_objective: float
                - n_evaluations: int
            - best_node: Best node summary (or None)
                - node_id: str
                - optimizer: str
                - best_objective: float
                - best_x: List[float]
            - leaf_nodes: Potential continuation points
            - message: str

    Example:
        state = get_graph_state(graph_id=1)
        # state["nodes"] shows all optimization runs
        # state["best_node"] shows the best result found
        # state["leaf_nodes"] shows nodes you can continue from

        # Agent reasons: "n2 has best objective (0.08), I'll warm-start from n2"
        run_optimization(graph_id=1, optimizer="scipy:SLSQP",
                         parent_node="n2", edge_type="warm_start")
    """
    try:
        if _FOUNDRY is None:
            return {
                "success": False,
                "message": "Foundry not initialized.",
            }

        # Check active graphs first
        graph = _FOUNDRY.get_graph(graph_id)
        if graph is not None:
            state = graph.get_graph_state()
            return {
                "success": True,
                "status": "active",
                **state,
                "message": f"Graph #{graph_id} is active with {state['n_nodes']} node(s).",
            }

        # Check completed graphs
        record = _FOUNDRY.load_graph(graph_id)
        if record is not None:
            # Build state from completed graph
            node_summaries = []
            for node in record.nodes.values():
                node_summaries.append({
                    "node_id": node.node_id,
                    "optimizer": node.optimizer,
                    "status": node.status,
                    "best_objective": node.best_objective,
                    "n_evaluations": node.n_evaluations,
                })

            best = record.get_best_node()
            best_summary = None
            if best:
                best_summary = {
                    "node_id": best.node_id,
                    "optimizer": best.optimizer,
                    "best_objective": best.best_objective,
                    "best_x": best.best_x,
                }

            leaf_ids = record.get_leaf_nodes()
            leaf_summaries = [
                {
                    "node_id": nid,
                    "optimizer": record.nodes[nid].optimizer,
                    "best_objective": record.nodes[nid].best_objective,
                }
                for nid in leaf_ids
            ]

            return {
                "success": True,
                "status": "completed",
                "graph_id": graph_id,
                "problem_id": record.problem_id,
                "n_nodes": len(record.nodes),
                "n_edges": len(record.edges),
                "pattern": record.detect_pattern(),
                "nodes": node_summaries,
                "best_node": best_summary,
                "leaf_nodes": leaf_summaries,
                "final_objective": record.final_objective,
                "total_evaluations": record.total_evaluations,
                "message": f"Graph #{graph_id} completed with {len(record.nodes)} node(s).",
            }

        return {
            "success": False,
            "message": f"Graph {graph_id} not found.",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting graph state: {str(e)}",
        }


@tool
def finalize_graph(
    graph_id: int,
    success: bool,
    notes: str = "",
) -> Dict[str, Any]:
    """
    Finalize optimization graph and persist to storage.

    Call this when the optimization task is complete (whether successful
    or not). The graph record will be saved for future analysis.

    Args:
        graph_id: Graph ID from start_graph
        success: Whether the optimization achieved its goal
        notes: Optional final notes or analysis

    Returns:
        Dict with:
            - success: bool
            - graph_id: int
            - final_objective: float (best found)
            - total_evaluations: int
            - n_nodes: int (number of optimization runs)
            - pattern: str (graph pattern)
            - message: str

    Example:
        result = finalize_graph(
            graph_id=1,
            success=True,
            notes="Multi-start + refinement achieved objective 3.98"
        )
    """
    try:
        if _FOUNDRY is None:
            return {
                "success": False,
                "message": "Foundry not initialized.",
            }

        graph = _FOUNDRY.get_graph(graph_id)

        if graph is None:
            return {
                "success": False,
                "message": f"Graph {graph_id} not found or already finalized.",
            }

        # Record notes as a decision if provided
        if notes:
            graph.record_decision(
                decision_type="finalize",
                reasoning=notes,
            )

        # Finalize graph
        record = _FOUNDRY.finalize_graph(graph_id, success)

        if record is None:
            return {
                "success": False,
                "message": f"Failed to finalize graph {graph_id}.",
            }

        return {
            "success": True,
            "graph_id": graph_id,
            "final_objective": record.final_objective,
            "total_evaluations": record.total_evaluations,
            "n_nodes": len(record.nodes),
            "pattern": record.detect_pattern(),
            "total_wall_time": record.total_wall_time,
            "message": (
                f"Graph #{graph_id} finalized. "
                f"Best objective: {record.final_objective:.6e}, "
                f"{len(record.nodes)} node(s), "
                f"{record.total_evaluations} evaluations."
            ),
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error finalizing graph: {str(e)}",
        }
