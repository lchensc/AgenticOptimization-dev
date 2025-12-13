"""
Analysis tools for agent.

Thin wrappers around paola.analysis module.
Exposes both deterministic and AI-powered analysis to agent.
"""

from typing import Dict, Any
from langchain_core.tools import tool

from ..foundry import OptimizationFoundry
from ..analysis import compute_metrics, ai_analyze

# Global foundry reference (set by build_tools)
from .run_tools import get_foundry


@tool
def analyze_convergence(run_id: int) -> Dict[str, Any]:
    """
    Deterministic convergence analysis (fast, free).

    Analyzes convergence behavior of an optimization run.
    Returns metrics about convergence rate, stalling, improvement.

    Use this for quick checks during optimization.

    Args:
        run_id: Run identifier

    Returns:
        {
            "iterations_total": int,
            "rate": float,              # Average improvement rate
            "is_stalled": bool,         # True if stalled
            "improvement_last_10": float,
        }

    Example:
        # Agent monitors optimization
        metrics = analyze_convergence(run_id=1)

        if metrics["is_stalled"]:
            # Optimization stalled, need strategy change
            pass
    """
    foundry = get_foundry()
    if foundry is None:
        return {"error": "Foundry not initialized"}

    run = foundry.load_run(run_id)
    if run is None:
        return {"error": f"Run {run_id} not found"}

    metrics = compute_metrics(run)
    return metrics["convergence"]


@tool
def analyze_efficiency(run_id: int) -> Dict[str, Any]:
    """
    Deterministic efficiency analysis (fast, free).

    Analyzes how efficiently the optimizer is using evaluations.

    Args:
        run_id: Run identifier

    Returns:
        {
            "evaluations": int,
            "improvement_per_eval": float,
        }
    """
    foundry = get_foundry()
    if foundry is None:
        return {"error": "Foundry not initialized"}

    run = foundry.load_run(run_id)
    if run is None:
        return {"error": f"Run {run_id} not found"}

    metrics = compute_metrics(run)
    return metrics["efficiency"]


@tool
def get_all_metrics(run_id: int) -> Dict[str, Any]:
    """
    Get all deterministic metrics for a run (fast, free).

    Returns complete set of metrics: convergence, gradient,
    constraints, efficiency, objective.

    Use this to get comprehensive overview of run health.

    Args:
        run_id: Run identifier

    Returns:
        Complete metrics dictionary with all categories

    Example:
        # Agent checks run health
        metrics = get_all_metrics(run_id=1)

        print(f"Convergence rate: {metrics['convergence']['rate']}")
        print(f"Gradient quality: {metrics['gradient']['quality']}")
        print(f"Evaluations: {metrics['efficiency']['evaluations']}")
    """
    foundry = get_foundry()
    if foundry is None:
        return {"error": "Foundry not initialized"}

    run = foundry.load_run(run_id)
    if run is None:
        return {"error": f"Run {run_id} not found"}

    return compute_metrics(run)


@tool
def analyze_run_with_ai(
    run_id: int,
    focus: str = "overall"
) -> Dict[str, Any]:
    """
    AI-powered strategic analysis (costs ~$0.02-0.05).

    Uses LLM to reason over optimization data and provide
    diagnosis + actionable recommendations.

    Use this when:
    - Deterministic metrics show issues (stalling, violations)
    - You need strategic advice (should I restart? switch algorithms?)
    - You want to extract insights for knowledge base

    Args:
        run_id: Run to analyze
        focus: What to analyze:
            - "convergence": Why converging slowly/fast?
            - "feasibility": Why violating constraints?
            - "efficiency": Why so many evaluations?
            - "algorithm": Should we switch algorithms?
            - "overall": Holistic diagnosis (default)

    Returns:
        {
            "diagnosis": str,           # What's happening
            "root_cause": str,          # Why it's happening
            "confidence": "low|medium|high",
            "evidence": [str],
            "recommendations": [
                {
                    "action": str,      # Tool to call
                    "args": dict,       # Arguments
                    "rationale": str,   # Why this helps
                    "priority": int,
                    "expected_impact": str,
                }
            ],
            "metadata": {
                "model": str,
                "timestamp": str,
                "cost_estimate": float,
            }
        }

    Example:
        # Agent first checks deterministic metrics
        metrics = analyze_convergence(run_id=1)

        # If stalled, ask for AI diagnosis
        if metrics["is_stalled"]:
            insights = analyze_run_with_ai(run_id=1, focus="convergence")

            # Execute recommendations
            for rec in insights["recommendations"]:
                if rec["action"] == "optimizer_restart":
                    optimizer_restart(**rec["args"])
    """
    foundry = get_foundry()
    if foundry is None:
        return {"error": "Foundry not initialized"}

    run = foundry.load_run(run_id)
    if run is None:
        return {"error": f"Run {run_id} not found"}

    # Compute deterministic metrics first (fast, feeds into AI)
    metrics = compute_metrics(run)

    # AI analysis (slow, costs money)
    insights = ai_analyze(run, metrics, focus=focus)

    return insights
