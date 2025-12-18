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
    problem_id: int,
    goal: str = "",
) -> Dict[str, Any]:
    """
    Start a new optimization graph.

    Must be called before any optimization. Creates a graph to track
    the optimization process. The returned graph_id must be passed
    to run_optimization.

    Args:
        problem_id: Numeric problem ID (from create_nlp_problem)
        goal: Natural language description of optimization goal

    Returns:
        Dict with:
            - success: bool
            - graph_id: int - Use this in run_optimization
            - problem_id: int
            - message: str

    Example:
        result = start_graph(
            problem_id=1,
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
            "message": f"Created graph #{graph.graph_id} for problem #{problem_id}.",
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

        # Format final objective (handle None case)
        if record.final_objective is not None:
            obj_str = f"{record.final_objective:.6e}"
        else:
            obj_str = "N/A"

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
                f"Best objective: {obj_str}, "
                f"{len(record.nodes)} node(s), "
                f"{record.total_evaluations} evaluations."
            ),
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error finalizing graph: {str(e)}",
        }


@tool
def query_past_graphs(
    problem_id: Optional[int] = None,  # v0.4.7: Changed from problem_pattern: str
    n_dimensions: Optional[int] = None,
    success: Optional[bool] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Query past optimization graphs for cross-graph learning.

    Use this to learn from BOTH successful and failed past optimizations:
    - Successes: What strategies worked, which to replicate
    - Failures: What to avoid, what didn't work for this problem type

    Returns compact summaries optimized for reasoning about:
    - What optimizer configurations were used
    - What patterns (chain, multistart) were effective
    - How efficient the strategy was (evaluations, wall time)
    - Why certain approaches failed (valuable negative knowledge)

    Args:
        problem_id: Filter by exact problem ID (int). Omit to query all problems.
        n_dimensions: Filter by problem dimensions (e.g., 50)
        success: Filter by success status. Omit to get BOTH successes and failures.
        limit: Maximum results to return (default: 5)

    Returns:
        Dict with:
            - success: bool
            - n_results: int
            - graphs: List of graph summaries with:
                - graph_id: int
                - problem_id: int
                - problem_signature: {n_dimensions, n_constraints, bounds_range}
                - pattern: str (chain, multistart, etc.)
                - strategy: List of optimizer configs used
                - outcome: {success, final_objective, total_evaluations}
            - message: str

    Example:
        # Get ALL past graphs for a problem (both successes and failures)
        result = query_past_graphs(problem_id=7)
        # Learn: what worked AND what didn't work

        # Find what worked for similar high-dimensional problems
        result = query_past_graphs(n_dimensions=50, success=True)
        # Results: "Graph #42 used TPE→L-BFGS-B chain, achieved 0.001"

        # Find what FAILED to avoid repeating mistakes
        result = query_past_graphs(problem_id=7, success=False)
        # Learn: "SLSQP alone failed, maybe need global search first"
    """
    try:
        if _FOUNDRY is None:
            return {
                "success": False,
                "message": "Foundry not initialized.",
            }

        # Query using foundry (returns GraphRecords)
        records = _FOUNDRY.query_graphs(
            problem_id=problem_id,
            success=success,
            n_dimensions=n_dimensions,
            limit=limit,
        )

        # Build compact summaries for LLM
        graph_summaries = []
        for record in records:
            # Build problem signature summary
            sig_summary = None
            if record.problem_signature:
                sig_summary = {
                    "n_dimensions": record.problem_signature.n_dimensions,
                    "n_constraints": record.problem_signature.n_constraints,
                    "bounds_range": list(record.problem_signature.bounds_range),
                    "domain_hint": record.problem_signature.domain_hint,
                }

            # Build strategy summary (ordered list of optimizers with configs)
            strategy = []
            for node_id in sorted(record.nodes.keys()):
                node = record.nodes[node_id]
                strategy.append({
                    "node_id": node.node_id,
                    "optimizer": node.optimizer,
                    "config": node.config,
                    "init_strategy": node.init_strategy,
                    "edge_type": node.edge_type,
                    "n_evaluations": node.n_evaluations,
                    "start_objective": node.start_objective,
                    "best_objective": node.best_objective,
                })

            # Build outcome summary
            outcome = {
                "success": record.success,
                "final_objective": record.final_objective,
                "total_evaluations": record.total_evaluations,
                "total_wall_time": record.total_wall_time,
            }

            graph_summaries.append({
                "graph_id": record.graph_id,
                "problem_id": record.problem_id,
                "goal": record.goal,
                "problem_signature": sig_summary,
                "pattern": record.pattern,
                "strategy": strategy,
                "outcome": outcome,
            })

        return {
            "success": True,
            "n_results": len(graph_summaries),
            "graphs": graph_summaries,
            "message": f"Found {len(graph_summaries)} matching graph(s).",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error querying graphs: {str(e)}",
        }


@tool
def get_past_graph(graph_id: int) -> Dict[str, Any]:
    """
    Get detailed strategy information from a completed optimization graph.

    Use this to learn from past optimizations and compose better strategies.
    Returns full configuration details including optimizer configs, init strategies,
    and edge types - everything needed to understand and replicate the strategy.

    Args:
        graph_id: Graph ID to retrieve

    Returns:
        Dict with:
            - success: bool
            - graph_id: int
            - problem_id: str
            - goal: str (original optimization goal)
            - problem_signature: Problem characteristics
                - n_dimensions: int
                - n_constraints: int
                - bounds_range: [min, max]
                - domain_hint: str (if any)
            - pattern: str (single, multistart, chain, tree, dag)
            - strategy: List of nodes with full details
                - node_id: str
                - optimizer: str
                - config: Dict (FULL optimizer configuration)
                - init_strategy: str (center, random, warm_start)
                - parent_node: str (if any)
                - edge_type: str (warm_start, restart, refine, branch, explore)
                - n_evaluations: int
                - start_objective: float
                - best_objective: float
            - edges: List of edge connections
            - outcome: Final results
                - success: bool
                - final_objective: float
                - total_evaluations: int
                - total_wall_time: float
            - message: str

    Example:
        # Learn from a successful past optimization
        past = get_past_graph(graph_id=11)

        # Agent sees: Graph #11 used TPE→L-BFGS-B chain on ackley_30d
        # with TPE config: {max_iterations: 300}
        # achieved 5.7e-07 in 28591 evals

        # Agent composes improved strategy for same problem:
        start_graph(problem_id="ackley_30d",
                    goal="Improve on graph #11 strategy")
        run_optimization(..., optimizer="optuna:TPE",
                         config={"max_iterations": 500})  # More exploration
    """
    try:
        if _FOUNDRY is None:
            return {
                "success": False,
                "message": "Foundry not initialized.",
            }

        # Load GraphRecord (Tier 1)
        record = _FOUNDRY.load_graph_record(graph_id)

        if record is None:
            return {
                "success": False,
                "message": f"Graph #{graph_id} not found.",
            }

        # Build problem signature summary
        sig_summary = None
        if record.problem_signature:
            sig_summary = {
                "n_dimensions": record.problem_signature.n_dimensions,
                "n_constraints": record.problem_signature.n_constraints,
                "bounds_range": list(record.problem_signature.bounds_range),
                "domain_hint": record.problem_signature.domain_hint,
            }

        # Build detailed strategy (ordered by node_id)
        strategy = []
        for node_id in sorted(record.nodes.keys()):
            node = record.nodes[node_id]
            strategy.append({
                "node_id": node.node_id,
                "optimizer": node.optimizer,
                "optimizer_family": node.optimizer_family,
                "config": node.config,  # FULL configuration
                "init_strategy": node.init_strategy,
                "parent_node": node.parent_node,
                "edge_type": node.edge_type,
                "status": node.status,
                "n_evaluations": node.n_evaluations,
                "wall_time": node.wall_time,
                "start_objective": node.start_objective,
                "best_objective": node.best_objective,
            })

        # Build edges list
        edges = [
            {"source": e.source, "target": e.target, "edge_type": e.edge_type}
            for e in record.edges
        ]

        # Build outcome summary
        outcome = {
            "success": record.success,
            "final_objective": record.final_objective,
            "total_evaluations": record.total_evaluations,
            "total_wall_time": record.total_wall_time,
        }

        return {
            "success": True,
            "graph_id": record.graph_id,
            "problem_id": record.problem_id,
            "goal": record.goal,
            "problem_signature": sig_summary,
            "pattern": record.pattern,
            "strategy": strategy,
            "edges": edges,
            "outcome": outcome,
            "message": f"Graph #{graph_id}: {record.problem_id}, {len(strategy)} nodes, "
                       f"pattern={record.pattern}, final_obj="
                       f"{f'{record.final_objective:.2e}' if record.final_objective is not None else 'N/A'}",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error loading graph: {str(e)}",
        }
