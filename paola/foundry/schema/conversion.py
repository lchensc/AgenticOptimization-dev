"""
Conversion utilities between OptimizationGraph and two-tier storage.

Converts:
- OptimizationGraph → (GraphRecord, GraphDetail)
- (GraphRecord, GraphDetail) → OptimizationGraph (for backward compatibility)
"""

from typing import Tuple, Optional, Dict, Any, List

from .graph import OptimizationGraph, OptimizationNode
from .graph_record import (
    ProblemSignature,
    NodeSummary,
    EdgeSummary,
    GraphRecord,
)
from .graph_detail import (
    ConvergencePoint,
    XPoint,
    NodeDetail,
    GraphDetail,
)


def extract_init_strategy(node: OptimizationNode) -> str:
    """Extract initialization strategy from node."""
    if node.initialization is None:
        return "unknown"

    spec = getattr(node.initialization, 'specification', None)
    if spec is None:
        return "unknown"

    if isinstance(spec, dict):
        return spec.get("type", "unknown")

    return "unknown"


def extract_convergence_and_trajectory(
    node: OptimizationNode
) -> Tuple[List[ConvergencePoint], List[XPoint], Optional[float]]:
    """
    Extract convergence history and x trajectory from node.

    Returns:
        (convergence_history, x_history, start_objective)
    """
    convergence = []
    x_history = []
    start_objective = None

    if node.progress is None:
        return convergence, x_history, start_objective

    # Handle different progress types (gradient, bayesian, population, cmaes)
    iterations = getattr(node.progress, 'iterations', None)
    if iterations is None:
        # Try trials for bayesian
        iterations = getattr(node.progress, 'trials', None)
    if iterations is None:
        # Try generations for population/cmaes
        iterations = getattr(node.progress, 'generations', None)

    if iterations is None:
        return convergence, x_history, start_objective

    for i, iter_data in enumerate(iterations):
        # Get iteration number
        iter_num = getattr(iter_data, 'iteration', None)
        if iter_num is None:
            iter_num = getattr(iter_data, 'trial_number', None)
        if iter_num is None:
            iter_num = getattr(iter_data, 'generation', None)
        if iter_num is None:
            iter_num = i + 1

        # Get objective
        obj = getattr(iter_data, 'objective', None)
        if obj is None:
            obj = getattr(iter_data, 'best_objective', None)

        if obj is not None:
            convergence.append(ConvergencePoint(
                iteration=iter_num,
                objective=obj,
            ))
            if start_objective is None:
                start_objective = obj

        # Get design/x
        design = getattr(iter_data, 'design', None)
        if design is None:
            design = getattr(iter_data, 'best_design', None)

        if design is not None:
            x_history.append(XPoint(
                iteration=iter_num,
                x=list(design),
            ))

    return convergence, x_history, start_objective


def split_graph(
    graph: OptimizationGraph,
    problem_signature: Optional[ProblemSignature] = None,
) -> Tuple[GraphRecord, GraphDetail]:
    """
    Convert OptimizationGraph to two-tier representation.

    Args:
        graph: Full optimization graph
        problem_signature: Optional problem signature (if available)

    Returns:
        (GraphRecord, GraphDetail) tuple
    """
    # Build node summaries and details
    node_summaries: Dict[str, NodeSummary] = {}
    node_details: Dict[str, NodeDetail] = {}

    for node_id, node in graph.nodes.items():
        # Extract convergence and trajectory for Tier 2
        convergence, x_history, start_objective = extract_convergence_and_trajectory(node)

        # Get parent node and edge type from edges
        parent_node = None
        edge_type = None
        for edge in graph.edges:
            if edge.target == node_id:
                parent_node = edge.source
                edge_type = edge.edge_type
                break

        # Get x0 and best_x
        x0 = None
        if node.initialization is not None:
            x0 = getattr(node.initialization, 'x0', None)
            if x0 is not None:
                x0 = list(x0)

        best_x = list(node.best_x) if node.best_x else None

        # Node summary (Tier 1) - minimal, no trajectory
        node_summaries[node_id] = NodeSummary(
            node_id=node_id,
            optimizer=node.optimizer,
            optimizer_family=node.optimizer_family,
            config=node.config,  # FULL config - key for learning!
            init_strategy=extract_init_strategy(node),
            parent_node=parent_node,
            edge_type=edge_type,
            status=node.status,
            n_evaluations=node.n_evaluations,
            wall_time=node.wall_time,
            start_objective=start_objective,
            best_objective=node.best_objective,
        )

        # Node detail (Tier 2) - full trajectory
        node_details[node_id] = NodeDetail(
            node_id=node_id,
            x0=x0,
            best_x=best_x,
            convergence_history=convergence,
            x_history=x_history,
        )

    # Build edge summaries
    edge_summaries = [
        EdgeSummary(
            source=e.source,
            target=e.target,
            edge_type=e.edge_type,
        )
        for e in graph.edges
    ]

    # Build decisions list (just the dict representation)
    decisions = [d.to_dict() for d in graph.decisions]

    # Build GraphRecord (Tier 1)
    record = GraphRecord(
        graph_id=graph.graph_id,
        problem_id=graph.problem_id,
        created_at=graph.created_at,
        goal=graph.goal,
        problem_signature=problem_signature,
        pattern=graph.detect_pattern(),
        edges=edge_summaries,
        nodes=node_summaries,
        status=graph.status,
        success=graph.success,
        final_objective=graph.final_objective,
        final_x=list(graph.final_x) if graph.final_x else None,
        total_evaluations=graph.total_evaluations,
        total_wall_time=graph.total_wall_time,
        decisions=decisions,
    )

    # Build GraphDetail (Tier 2)
    detail = GraphDetail(
        graph_id=graph.graph_id,
        nodes=node_details,
    )

    return record, detail


def create_problem_signature(
    n_dimensions: int,
    bounds: List[Tuple[float, float]],
    n_constraints: int = 0,
    constraint_types: Optional[List[str]] = None,
    domain_hint: Optional[str] = None,
) -> ProblemSignature:
    """
    Create a ProblemSignature from problem information.

    Args:
        n_dimensions: Number of design variables
        bounds: List of (lower, upper) bounds
        n_constraints: Number of constraints
        constraint_types: Types of constraints (e.g., ["equality", "inequality"])
        domain_hint: Optional domain hint (e.g., "rosenbrock", "ackley")

    Returns:
        ProblemSignature instance
    """
    # Calculate bounds range
    if bounds:
        all_bounds = [b for pair in bounds for b in pair]
        bounds_range = (min(all_bounds), max(all_bounds))
    else:
        bounds_range = (0.0, 0.0)

    return ProblemSignature(
        n_dimensions=n_dimensions,
        n_constraints=n_constraints,
        bounds_range=bounds_range,
        constraint_types=constraint_types or [],
        domain_hint=domain_hint,
    )
