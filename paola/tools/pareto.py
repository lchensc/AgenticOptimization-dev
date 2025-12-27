"""
Pareto front query and analysis tools.

Provides tools for querying and analyzing Pareto fronts from MOO optimization.
"""

from typing import Optional, Dict, Any, List
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)

# Module-level storage reference (set by set_pareto_storage)
_PARETO_STORAGE = None


def set_pareto_storage(storage):
    """Set the Pareto storage instance for tools to use."""
    global _PARETO_STORAGE
    _PARETO_STORAGE = storage


def get_pareto_storage():
    """Get the current Pareto storage instance."""
    return _PARETO_STORAGE


@tool
def query_pareto(
    graph_id: int,
    node_id: Optional[str] = None,
    select: str = "all",
    objective_filter: Optional[str] = None,
    target: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Query Pareto front from MOO optimization.

    Args:
        graph_id: Graph containing MOO results
        node_id: Specific node (default: node with best hypervolume)
        select: Query type - "all", "knee", "extreme", "closest", "diverse"
        objective_filter: Filter by objective, e.g., "drag<0.05" or "weight>0.3,weight<0.8"
        target: For "closest" - target values, e.g., "drag=0.04,weight=0.5"
        limit: Max solutions to return

    Returns:
        solutions, n_total, hypervolume, objective_names, summary

    Examples:
        query_pareto(graph_id=1, select="knee")
        query_pareto(graph_id=1, objective_filter="drag<0.05")
        query_pareto(graph_id=1, select="closest", target="drag=0.04,weight=0.5")
        query_pareto(graph_id=1, select="extreme", objective_filter="drag")
    """
    if _PARETO_STORAGE is None:
        return {
            "success": False,
            "message": "Pareto storage not initialized",
        }

    try:
        # Find node if not specified
        if node_id is None:
            best = _PARETO_STORAGE.get_best_hypervolume(graph_id)
            if best is None:
                return {
                    "success": False,
                    "message": f"No Pareto fronts found for graph {graph_id}",
                }
            node_id = best["node_id"]

        # Load Pareto front
        pf = _PARETO_STORAGE.load(graph_id, node_id)
        if pf is None:
            return {
                "success": False,
                "message": f"Pareto front not found: graph {graph_id}, node {node_id}",
            }

        # Apply objective filter
        if objective_filter:
            pf = _apply_objective_filter(pf, objective_filter)

        # Select solutions based on query type
        if select == "knee":
            sol = pf.get_knee_point()
            if sol:
                return {
                    "success": True,
                    "select": "knee",
                    "solution": {
                        "x": sol.x.tolist(),
                        "f": sol.f.tolist(),
                        "objectives": dict(zip(pf.objective_names, sol.f.tolist())),
                    },
                    "n_total": pf.n_solutions,
                    "hypervolume": pf.hypervolume,
                    "objective_names": pf.objective_names,
                }

        elif select == "extreme":
            # Get extreme for each objective (or specified one)
            if objective_filter and "," not in objective_filter and "<" not in objective_filter and ">" not in objective_filter:
                # Just objective name provided
                sol = pf.get_extreme(objective_filter)
                if sol:
                    return {
                        "success": True,
                        "select": f"extreme_{objective_filter}",
                        "solution": {
                            "x": sol.x.tolist(),
                            "f": sol.f.tolist(),
                            "objectives": dict(zip(pf.objective_names, sol.f.tolist())),
                        },
                        "n_total": pf.n_solutions,
                        "hypervolume": pf.hypervolume,
                    }
            else:
                # Get all extremes
                extremes = {}
                for name in pf.objective_names:
                    sol = pf.get_extreme(name)
                    if sol:
                        extremes[name] = {
                            "x": sol.x.tolist(),
                            "f": sol.f.tolist(),
                            "objectives": dict(zip(pf.objective_names, sol.f.tolist())),
                        }
                return {
                    "success": True,
                    "select": "extreme",
                    "extremes": extremes,
                    "n_total": pf.n_solutions,
                    "hypervolume": pf.hypervolume,
                }

        elif select == "closest":
            if not target:
                return {
                    "success": False,
                    "message": "target required for select='closest'",
                }
            target_dict = _parse_target(target)
            sol = pf.get_closest_to(target_dict)
            if sol:
                return {
                    "success": True,
                    "select": "closest",
                    "target": target_dict,
                    "solution": {
                        "x": sol.x.tolist(),
                        "f": sol.f.tolist(),
                        "objectives": dict(zip(pf.objective_names, sol.f.tolist())),
                    },
                    "n_total": pf.n_solutions,
                    "hypervolume": pf.hypervolume,
                }

        elif select == "diverse":
            solutions = pf.get_most_diverse(n=limit)
            return {
                "success": True,
                "select": "diverse",
                "solutions": [
                    {
                        "x": s.x.tolist(),
                        "f": s.f.tolist(),
                        "objectives": dict(zip(pf.objective_names, s.f.tolist())),
                        "crowding_distance": s.crowding_distance,
                    }
                    for s in solutions
                ],
                "n_returned": len(solutions),
                "n_total": pf.n_solutions,
                "hypervolume": pf.hypervolume,
            }

        else:  # "all"
            solutions = pf.solutions[:limit]
            return {
                "success": True,
                "select": "all",
                "solutions": [
                    {
                        "x": s.x.tolist(),
                        "f": s.f.tolist(),
                        "objectives": dict(zip(pf.objective_names, s.f.tolist())),
                    }
                    for s in solutions
                ],
                "n_returned": len(solutions),
                "n_total": pf.n_solutions,
                "hypervolume": pf.hypervolume,
                "objective_names": pf.objective_names,
                "summary": pf.summary(),
            }

    except Exception as e:
        logger.error(f"query_pareto error: {e}")
        return {
            "success": False,
            "message": str(e),
        }


@tool
def compare_pareto_fronts(
    graph_id_1: int,
    node_id_1: str,
    graph_id_2: int,
    node_id_2: str,
) -> Dict[str, Any]:
    """
    Compare two Pareto fronts.

    Args:
        graph_id_1: First graph ID
        node_id_1: First node ID
        graph_id_2: Second graph ID
        node_id_2: Second node ID

    Returns:
        Comparison with hypervolume difference, solution counts, recommendation
    """
    if _PARETO_STORAGE is None:
        return {
            "success": False,
            "message": "Pareto storage not initialized",
        }

    try:
        comparison = _PARETO_STORAGE.compare(
            graph_id_1, node_id_1,
            graph_id_2, node_id_2,
        )
        comparison["success"] = True
        return comparison

    except Exception as e:
        logger.error(f"compare_pareto_fronts error: {e}")
        return {
            "success": False,
            "message": str(e),
        }


@tool
def list_pareto_fronts(
    graph_id: Optional[int] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    List available Pareto fronts.

    Args:
        graph_id: Filter by graph ID (optional)
        limit: Maximum results

    Returns:
        List of Pareto front summaries
    """
    if _PARETO_STORAGE is None:
        return {
            "success": False,
            "message": "Pareto storage not initialized",
        }

    try:
        all_fronts = _PARETO_STORAGE.list_all()

        # Filter by graph_id if specified
        if graph_id is not None:
            all_fronts = [f for f in all_fronts if f["graph_id"] == graph_id]

        # Limit results
        all_fronts = all_fronts[:limit]

        return {
            "success": True,
            "pareto_fronts": all_fronts,
            "n_total": len(all_fronts),
            "stats": _PARETO_STORAGE.get_stats(),
        }

    except Exception as e:
        logger.error(f"list_pareto_fronts error: {e}")
        return {
            "success": False,
            "message": str(e),
        }


def _apply_objective_filter(pf, filter_str: str):
    """
    Apply objective filter to Pareto front.

    Filter format: "objective<value" or "objective>value" or "obj1<val1,obj2>val2"
    """
    filters = filter_str.split(",")

    for f in filters:
        f = f.strip()
        if "<" in f:
            parts = f.split("<")
            obj_name = parts[0].strip()
            max_val = float(parts[1].strip())
            pf = pf.filter_by_objective(obj_name, max_val=max_val)
        elif ">" in f:
            parts = f.split(">")
            obj_name = parts[0].strip()
            min_val = float(parts[1].strip())
            pf = pf.filter_by_objective(obj_name, min_val=min_val)

    return pf


def _parse_target(target_str: str) -> Dict[str, float]:
    """
    Parse target string to dict.

    Format: "objective=value,objective2=value2"
    """
    result = {}
    for part in target_str.split(","):
        part = part.strip()
        if "=" in part:
            name, val = part.split("=")
            result[name.strip()] = float(val.strip())
    return result


# Export tools list
PARETO_TOOLS = [
    query_pareto,
    compare_pareto_fronts,
    list_pareto_fronts,
]
