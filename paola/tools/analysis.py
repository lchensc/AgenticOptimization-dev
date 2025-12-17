"""
Analysis tools for agent.

Provides convergence and efficiency analysis for optimization graphs.
Works with the graph-based architecture (v0.4.0).
"""

from typing import Dict, Any, Optional, List
from langchain_core.tools import tool

from .graph_tools import get_foundry


def _compute_convergence_metrics(objectives: List[float]) -> Dict[str, Any]:
    """
    Compute convergence metrics from objective history.

    Args:
        objectives: List of objective values (iteration order)

    Returns:
        Convergence metrics dictionary
    """
    if not objectives:
        return {
            "iterations_total": 0,
            "rate": 0.0,
            "is_stalled": True,
            "improvement_last_10": 0.0,
            "best_objective": None,
            "final_objective": None,
        }

    n = len(objectives)
    best = min(objectives)
    final = objectives[-1]

    # Compute improvement rate
    if n >= 2:
        total_improvement = objectives[0] - best
        rate = total_improvement / n if total_improvement > 0 else 0.0
    else:
        rate = 0.0

    # Check last 10 iterations for stalling
    last_10 = objectives[-10:] if n >= 10 else objectives
    if len(last_10) >= 2:
        improvement_last_10 = last_10[0] - min(last_10)
        is_stalled = improvement_last_10 < 1e-8 * abs(last_10[0]) if last_10[0] != 0 else improvement_last_10 < 1e-10
    else:
        improvement_last_10 = 0.0
        is_stalled = True

    return {
        "iterations_total": n,
        "rate": rate,
        "is_stalled": is_stalled,
        "improvement_last_10": improvement_last_10,
        "best_objective": best,
        "final_objective": final,
    }


def _compute_efficiency_metrics(objectives: List[float]) -> Dict[str, Any]:
    """
    Compute efficiency metrics from objective history.

    Args:
        objectives: List of objective values

    Returns:
        Efficiency metrics dictionary
    """
    if not objectives:
        return {
            "evaluations": 0,
            "improvement_per_eval": 0.0,
        }

    n = len(objectives)
    total_improvement = objectives[0] - min(objectives) if n >= 1 else 0.0
    improvement_per_eval = total_improvement / n if n > 0 else 0.0

    return {
        "evaluations": n,
        "improvement_per_eval": improvement_per_eval,
    }


@tool
def analyze_convergence(graph_id: int, node_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Deterministic convergence analysis for an optimization graph or node.

    Analyzes convergence behavior from the optimization history.
    Returns metrics about convergence rate, stalling, and improvement.

    Use this for quick checks during or after optimization.

    Args:
        graph_id: Graph identifier
        node_id: Optional node ID (e.g., "n1"). If not provided, analyzes best node.

    Returns:
        {
            "success": bool,
            "graph_id": int,
            "node_id": str,
            "convergence": {
                "iterations_total": int,
                "rate": float,              # Average improvement rate
                "is_stalled": bool,         # True if stalled
                "improvement_last_10": float,
                "best_objective": float,
                "final_objective": float,
            }
        }

    Example:
        # Analyze convergence of graph 1
        result = analyze_convergence(graph_id=1)

        if result["convergence"]["is_stalled"]:
            # Optimization stalled, need strategy change
            pass
    """
    try:
        foundry = get_foundry()
        if foundry is None:
            return {"success": False, "error": "Foundry not initialized"}

        # Load graph detail (has convergence history)
        detail = foundry.load_graph_detail(graph_id)
        if detail is None:
            return {"success": False, "error": f"Graph {graph_id} not found or has no detail data"}

        # If no node specified, find the best node
        if node_id is None:
            graph = foundry.load_graph(graph_id)
            if graph is None:
                return {"success": False, "error": f"Graph {graph_id} not found"}

            best_node = graph.get_best_node()
            if best_node is None:
                return {"success": False, "error": f"Graph {graph_id} has no completed nodes"}
            node_id = best_node.node_id

        # Get node detail
        node_detail = detail.get_node_detail(node_id)
        if node_detail is None:
            return {"success": False, "error": f"Node {node_id} not found in graph {graph_id}"}

        # Compute metrics from convergence history
        objectives = node_detail.get_objectives()
        convergence = _compute_convergence_metrics(objectives)

        return {
            "success": True,
            "graph_id": graph_id,
            "node_id": node_id,
            "convergence": convergence,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def analyze_efficiency(graph_id: int, node_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Deterministic efficiency analysis for an optimization graph or node.

    Analyzes how efficiently the optimizer is using evaluations.

    Args:
        graph_id: Graph identifier
        node_id: Optional node ID (e.g., "n1"). If not provided, analyzes best node.

    Returns:
        {
            "success": bool,
            "graph_id": int,
            "node_id": str,
            "efficiency": {
                "evaluations": int,
                "improvement_per_eval": float,
            }
        }
    """
    try:
        foundry = get_foundry()
        if foundry is None:
            return {"success": False, "error": "Foundry not initialized"}

        # Load graph detail
        detail = foundry.load_graph_detail(graph_id)
        if detail is None:
            return {"success": False, "error": f"Graph {graph_id} not found or has no detail data"}

        # If no node specified, find the best node
        if node_id is None:
            graph = foundry.load_graph(graph_id)
            if graph is None:
                return {"success": False, "error": f"Graph {graph_id} not found"}

            best_node = graph.get_best_node()
            if best_node is None:
                return {"success": False, "error": f"Graph {graph_id} has no completed nodes"}
            node_id = best_node.node_id

        # Get node detail
        node_detail = detail.get_node_detail(node_id)
        if node_detail is None:
            return {"success": False, "error": f"Node {node_id} not found in graph {graph_id}"}

        # Compute metrics
        objectives = node_detail.get_objectives()
        efficiency = _compute_efficiency_metrics(objectives)

        return {
            "success": True,
            "graph_id": graph_id,
            "node_id": node_id,
            "efficiency": efficiency,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def get_all_metrics(graph_id: int, node_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get all deterministic metrics for a graph node.

    Returns complete set of metrics: convergence and efficiency.
    Use this to get a comprehensive overview.

    Args:
        graph_id: Graph identifier
        node_id: Optional node ID. If not provided, analyzes best node.

    Returns:
        Complete metrics dictionary with convergence and efficiency categories.

    Example:
        # Check graph health
        metrics = get_all_metrics(graph_id=1)

        print(f"Iterations: {metrics['convergence']['iterations_total']}")
        print(f"Stalled: {metrics['convergence']['is_stalled']}")
        print(f"Evaluations: {metrics['efficiency']['evaluations']}")
    """
    try:
        foundry = get_foundry()
        if foundry is None:
            return {"success": False, "error": "Foundry not initialized"}

        # Load graph and detail
        graph = foundry.load_graph(graph_id)
        if graph is None:
            return {"success": False, "error": f"Graph {graph_id} not found"}

        detail = foundry.load_graph_detail(graph_id)
        if detail is None:
            return {"success": False, "error": f"Graph {graph_id} has no detail data"}

        # If no node specified, find the best node
        if node_id is None:
            best_node = graph.get_best_node()
            if best_node is None:
                return {"success": False, "error": f"Graph {graph_id} has no completed nodes"}
            node_id = best_node.node_id

        # Get node info
        node = graph.get_node(node_id)
        if node is None:
            return {"success": False, "error": f"Node {node_id} not found in graph {graph_id}"}

        # Get node detail
        node_detail = detail.get_node_detail(node_id)
        if node_detail is None:
            return {"success": False, "error": f"Node {node_id} has no detail data"}

        # Compute metrics
        objectives = node_detail.get_objectives()
        convergence = _compute_convergence_metrics(objectives)
        efficiency = _compute_efficiency_metrics(objectives)

        return {
            "success": True,
            "graph_id": graph_id,
            "node_id": node_id,
            "optimizer": node.optimizer,
            "convergence": convergence,
            "efficiency": efficiency,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def analyze_run_with_ai(
    graph_id: int,
    node_id: Optional[str] = None,
    focus: str = "overall"
) -> Dict[str, Any]:
    """
    AI-powered strategic analysis (costs ~$0.02-0.05).

    Uses LLM to reason over optimization data and provide
    diagnosis + actionable recommendations.

    NOTE: This is a placeholder. Full AI analysis requires LLM integration.

    Use this when:
    - Deterministic metrics show issues (stalling, poor convergence)
    - You need strategic advice (should I restart? switch algorithms?)
    - You want insights about what's happening

    Args:
        graph_id: Graph to analyze
        node_id: Optional specific node
        focus: What to analyze:
            - "convergence": Why converging slowly/fast?
            - "efficiency": Why so many evaluations?
            - "algorithm": Should we switch algorithms?
            - "overall": Holistic diagnosis (default)

    Returns:
        {
            "success": bool,
            "diagnosis": str,
            "recommendations": list,
        }
    """
    try:
        # First get deterministic metrics
        metrics = get_all_metrics.invoke({"graph_id": graph_id, "node_id": node_id})

        if not metrics.get("success"):
            return metrics

        # Simple rule-based diagnosis (placeholder for full AI analysis)
        convergence = metrics.get("convergence", {})
        efficiency = metrics.get("efficiency", {})

        diagnosis_parts = []
        recommendations = []

        # Check stalling
        if convergence.get("is_stalled"):
            diagnosis_parts.append("Optimization appears stalled - no significant improvement in recent iterations.")
            recommendations.append({
                "action": "Consider restarting with different initialization or algorithm",
                "priority": "high"
            })

        # Check convergence rate
        rate = convergence.get("rate", 0)
        if rate < 1e-6 and not convergence.get("is_stalled"):
            diagnosis_parts.append("Convergence rate is very slow.")
            recommendations.append({
                "action": "Consider using a more aggressive step size or different algorithm",
                "priority": "medium"
            })

        # Check efficiency
        evals = efficiency.get("evaluations", 0)
        improvement_per_eval = efficiency.get("improvement_per_eval", 0)
        if evals > 100 and improvement_per_eval < 1e-6:
            diagnosis_parts.append(f"Used {evals} evaluations with minimal improvement per evaluation.")
            recommendations.append({
                "action": "Consider reducing evaluation budget or using surrogate model",
                "priority": "medium"
            })

        if not diagnosis_parts:
            diagnosis_parts.append("Optimization appears to be progressing normally.")

        return {
            "success": True,
            "graph_id": graph_id,
            "node_id": metrics.get("node_id"),
            "focus": focus,
            "diagnosis": " ".join(diagnosis_parts),
            "recommendations": recommendations,
            "metrics_summary": {
                "iterations": convergence.get("iterations_total"),
                "best_objective": convergence.get("best_objective"),
                "is_stalled": convergence.get("is_stalled"),
                "evaluations": efficiency.get("evaluations"),
            },
            "note": "Full AI analysis not yet implemented. This is rule-based diagnosis."
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
