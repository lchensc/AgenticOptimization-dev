"""
Deterministic metrics computation.

All metrics are computed from run/node data - fast, free, reproducible.
These metrics serve as:
1. Display in CLI (/show command)
2. Real-time monitoring during optimization
3. Input to AI analysis (ai_analyze)
"""

from typing import Dict, Any, List, Union
import numpy as np


def compute_metrics(run: Union[Dict, Any]) -> Dict[str, Any]:
    """
    Compute all deterministic metrics for a run.

    This is the foundation for both display and AI analysis.
    Fast (milliseconds), free, reproducible.

    Args:
        run: Optimization run record

    Returns:
        {
            "convergence": {
                "rate": float,              # (f_n - f_n-1) / f_n-1
                "is_stalled": bool,         # Rate < threshold for N iters
                "improvement_last_10": float,
                "iterations_total": int,
            },
            "gradient": {
                "norm": float,
                "variance": float,          # Variance over last N gradients
                "quality": "good" | "noisy" | "flat",
            },
            "constraints": {
                "violations": [
                    {"name": str, "value": float, "bound": float, "margin": float}
                ],
                "active_count": int,
            },
            "efficiency": {
                "evaluations": int,
                "improvement_per_eval": float,
            },
            "objective": {
                "current": float,
                "best": float,
                "worst": float,
                "improvement_from_start": float,
                "improvement_rate": float,
            }
        }

    Example:
        # CLI: Fast display
        metrics = compute_metrics(run)
        display_table(metrics)

        # Agent: Monitor during optimization
        metrics = compute_metrics(run)
        if metrics["convergence"]["is_stalled"]:
            # Trigger adaptation
    """
    # Extract iterations from result_data (supports dict or object interface)
    if isinstance(run, dict):
        result_data = run.get("result_data", {})
        n_evaluations = run.get("n_evaluations", 0)
        objective_value = run.get("objective_value", 0.0)
    else:
        result_data = getattr(run, "result_data", {}) or {}
        n_evaluations = getattr(run, "n_evaluations", 0)
        objective_value = getattr(run, "objective_value", 0.0)

    iterations = result_data.get("iterations", [])

    if not iterations:
        # No iteration data, return minimal metrics
        return {
            "convergence": {"iterations_total": 0, "rate": 0.0, "is_stalled": False, "improvement_last_10": 0.0},
            "gradient": {"norm": 0.0, "variance": 0.0, "quality": "unknown"},
            "constraints": {"violations": [], "active_count": 0},
            "efficiency": {"evaluations": n_evaluations, "improvement_per_eval": 0.0},
            "objective": {
                "current": objective_value,
                "best": objective_value,
                "worst": objective_value,
                "improvement_from_start": 0.0,
                "improvement_rate": 0.0,
            }
        }

    return {
        "convergence": _compute_convergence_metrics(iterations),
        "gradient": _compute_gradient_metrics(iterations),
        "constraints": _compute_constraint_metrics(iterations),
        "efficiency": _compute_efficiency_metrics(n_evaluations, iterations),
        "objective": _compute_objective_metrics(iterations),
    }


def _compute_convergence_metrics(iterations: List[Dict]) -> Dict[str, Any]:
    """
    Compute convergence metrics.

    Args:
        iterations: List of iteration records

    Returns:
        Convergence metrics dict
    """
    n_iters = len(iterations)

    if n_iters < 2:
        return {
            "iterations_total": n_iters,
            "rate": 0.0,
            "is_stalled": False,
            "improvement_last_10": 0.0,
        }

    # Extract objectives
    objectives = [it["objective"] for it in iterations]

    # Convergence rate (average improvement rate)
    improvements = []
    for i in range(1, len(objectives)):
        if objectives[i-1] != 0:
            rate = (objectives[i-1] - objectives[i]) / abs(objectives[i-1])
            improvements.append(rate)

    avg_rate = np.mean(improvements) if improvements else 0.0

    # Check if stalled (last 5 iterations show < 1e-6 improvement)
    is_stalled = False
    if n_iters >= 5:
        last_5_objs = objectives[-5:]
        improvement_last_5 = abs(max(last_5_objs) - min(last_5_objs))
        is_stalled = improvement_last_5 < 1e-6

    # Improvement in last 10 iterations
    improvement_last_10 = 0.0
    if n_iters >= 10:
        improvement_last_10 = objectives[-10] - objectives[-1]
    elif n_iters >= 2:
        improvement_last_10 = objectives[0] - objectives[-1]

    return {
        "iterations_total": n_iters,
        "rate": float(avg_rate),
        "is_stalled": bool(is_stalled),
        "improvement_last_10": float(improvement_last_10),
    }


def _compute_gradient_metrics(iterations: List[Dict]) -> Dict[str, Any]:
    """
    Compute gradient metrics.

    Args:
        iterations: List of iteration records

    Returns:
        Gradient metrics dict
    """
    # Extract gradients (may be None)
    gradients = []
    for it in iterations:
        grad = it.get("gradient")
        if grad is not None:
            gradients.append(np.array(grad))

    if not gradients:
        return {
            "norm": 0.0,
            "variance": 0.0,
            "quality": "unknown",
        }

    # Gradient norm (last iteration)
    grad_norm = float(np.linalg.norm(gradients[-1]))

    # Gradient variance (over last 10 gradients)
    variance = 0.0
    quality = "good"

    if len(gradients) >= 10:
        last_10_grads = gradients[-10:]
        # Compute variance of gradient norms
        grad_norms = [np.linalg.norm(g) for g in last_10_grads]
        variance = float(np.var(grad_norms))

        # Quality assessment
        if variance > 1e-2:
            quality = "noisy"
        elif grad_norm < 1e-8:
            quality = "flat"
        else:
            quality = "good"

    return {
        "norm": grad_norm,
        "variance": variance,
        "quality": quality,
    }


def _compute_constraint_metrics(iterations: List[Dict]) -> Dict[str, Any]:
    """
    Compute constraint metrics.

    Args:
        iterations: List of iteration records

    Returns:
        Constraint metrics dict
    """
    # Check last iteration for constraint values
    last_iter = iterations[-1]
    constraints = last_iter.get("constraints", {})

    violations = []
    if constraints:
        for name, value in constraints.items():
            # Simple violation check (assumes constraints should be >= 0)
            # In practice, would need constraint bounds from problem
            if value < 0:
                violations.append({
                    "name": name,
                    "value": float(value),
                    "bound": 0.0,
                    "margin": float(value),
                })

    return {
        "violations": violations,
        "active_count": len(violations),
    }


def _compute_efficiency_metrics(n_evaluations: int, iterations: List[Dict]) -> Dict[str, Any]:
    """
    Compute efficiency metrics.

    Args:
        n_evaluations: Number of function evaluations
        iterations: List of iteration records

    Returns:
        Efficiency metrics dict
    """
    n_evals = n_evaluations

    # Improvement per evaluation
    improvement_per_eval = 0.0
    if n_evals > 0 and len(iterations) >= 2:
        total_improvement = iterations[0]["objective"] - iterations[-1]["objective"]
        improvement_per_eval = total_improvement / n_evals

    return {
        "evaluations": n_evals,
        "improvement_per_eval": float(improvement_per_eval),
    }


def _compute_objective_metrics(iterations: List[Dict]) -> Dict[str, Any]:
    """
    Compute objective metrics.

    Args:
        iterations: List of iteration records

    Returns:
        Objective metrics dict
    """
    objectives = [it["objective"] for it in iterations]

    current = objectives[-1] if objectives else float('inf')
    best = min(objectives) if objectives else float('inf')
    worst = max(objectives) if objectives else float('inf')

    # Improvement from start
    improvement_from_start = 0.0
    if len(objectives) >= 2:
        improvement_from_start = objectives[0] - objectives[-1]

    # Improvement rate (linear fit)
    improvement_rate = 0.0
    if len(objectives) >= 2:
        # Simple linear regression
        x = np.arange(len(objectives))
        y = np.array(objectives)
        improvement_rate = float((y[0] - y[-1]) / len(objectives))

    return {
        "current": float(current),
        "best": float(best),
        "worst": float(worst),
        "improvement_from_start": float(improvement_from_start),
        "improvement_rate": improvement_rate,
    }
