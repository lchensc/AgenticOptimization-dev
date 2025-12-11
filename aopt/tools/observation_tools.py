"""
Observation and analysis tools for the agentic optimization platform.

Provides LangChain @tool decorated functions for observing optimization health:
- analyze_convergence: Analyze convergence metrics
- detect_pattern: Detect optimization patterns and issues
- check_feasibility: Check constraint feasibility
- get_gradient_quality: Analyze gradient reliability

These tools enable the agent to continuously monitor optimization progress
and make informed decisions about adaptation strategies.
"""

from typing import Optional, Dict, Any, List
import numpy as np
from langchain_core.tools import tool


@tool
def analyze_convergence(
    objectives: List[float],
    gradients: Optional[List[List[float]]] = None,
    window_size: int = 5,
) -> Dict[str, Any]:
    """
    Analyze optimization convergence health.

    Use this tool periodically to assess if optimization is converging well.
    Helps detect:
    - Convergence (decreasing objective, small gradient)
    - Stalling (no improvement)
    - Divergence (increasing objective)
    - Oscillation (alternating up/down)

    Args:
        objectives: List of recent objective values (most recent last)
        gradients: Optional list of gradient vectors (for gradient norm analysis)
        window_size: Number of recent iterations to analyze (default: 5)

    Returns:
        Dict with:
            - converging: bool - True if objective is decreasing
            - converged: bool - True if gradient is small and stable
            - stalled: bool - True if no significant improvement
            - diverging: bool - True if objective is increasing
            - oscillating: bool - True if objective alternates
            - improvement_rate: float - average improvement per iteration
            - gradient_norm: float - current gradient norm (if provided)
            - gradient_trend: str - "decreasing", "stable", "increasing"
            - recommendation: str - suggested action
            - details: dict - detailed metrics

    Example:
        result = analyze_convergence(
            objectives=[1.5, 1.2, 0.9, 0.85, 0.84],
            gradients=[[0.5, 0.3], [0.3, 0.2], [0.1, 0.1], [0.05, 0.05], [0.01, 0.01]]
        )
        if result["converged"]:
            print("Optimization converged!")
        elif result["stalled"]:
            print("Consider adapting strategy")
    """
    if len(objectives) < 2:
        return {
            "success": True,
            "converging": False,
            "converged": False,
            "stalled": False,
            "diverging": False,
            "oscillating": False,
            "recommendation": "Not enough data. Continue optimization.",
            "details": {"n_objectives": len(objectives)},
            "message": "Need at least 2 objective values for analysis",
        }

    # Use last window_size values
    recent_obj = objectives[-window_size:] if len(objectives) >= window_size else objectives

    # Calculate metrics
    improvements = [recent_obj[i] - recent_obj[i+1] for i in range(len(recent_obj)-1)]
    avg_improvement = np.mean(improvements) if improvements else 0.0
    total_improvement = recent_obj[0] - recent_obj[-1]

    # Analyze patterns
    n_improvements = sum(1 for imp in improvements if imp > 1e-10)
    n_worsening = sum(1 for imp in improvements if imp < -1e-10)

    converging = avg_improvement > 1e-8
    diverging = avg_improvement < -1e-8 and n_worsening > len(improvements) // 2
    oscillating = n_improvements > 0 and n_worsening > 0 and abs(n_improvements - n_worsening) <= 1
    stalled = abs(avg_improvement) < 1e-10 and abs(total_improvement) < 1e-8

    # Gradient analysis
    gradient_norm = None
    gradient_trend = "unknown"
    gradient_norms = []

    if gradients and len(gradients) > 0:
        gradient_norms = [np.linalg.norm(g) for g in gradients[-window_size:]]
        gradient_norm = gradient_norms[-1]

        if len(gradient_norms) >= 2:
            grad_improvements = [gradient_norms[i] - gradient_norms[i+1] for i in range(len(gradient_norms)-1)]
            avg_grad_improvement = np.mean(grad_improvements)

            if avg_grad_improvement > 1e-6:
                gradient_trend = "decreasing"
            elif avg_grad_improvement < -1e-6:
                gradient_trend = "increasing"
            else:
                gradient_trend = "stable"

    # Check for convergence
    converged = False
    if gradient_norm is not None:
        converged = gradient_norm < 1e-6 and stalled

    # Generate recommendation
    if converged:
        recommendation = "CONVERGED: Optimization has converged. Consider stopping or verifying with high-fidelity."
    elif diverging:
        recommendation = "DIVERGING: Objective is increasing. Consider: 1) Restart from best design, 2) Check problem formulation, 3) Reduce step size."
    elif oscillating:
        recommendation = "OSCILLATING: Objective alternates. Consider: 1) Reduce step size, 2) Switch to more stable algorithm."
    elif stalled:
        recommendation = "STALLED: No improvement. Consider: 1) Restart with different initial point, 2) Check for local minimum, 3) Try different optimizer."
    elif converging:
        recommendation = "CONVERGING: Good progress. Continue optimization."
    else:
        recommendation = "UNCERTAIN: Continue monitoring."

    return {
        "success": True,
        "converging": converging,
        "converged": converged,
        "stalled": stalled,
        "diverging": diverging,
        "oscillating": oscillating,
        "improvement_rate": float(avg_improvement),
        "total_improvement": float(total_improvement),
        "gradient_norm": float(gradient_norm) if gradient_norm is not None else None,
        "gradient_trend": gradient_trend,
        "recommendation": recommendation,
        "details": {
            "n_iterations_analyzed": len(recent_obj),
            "n_improvements": n_improvements,
            "n_worsening": n_worsening,
            "gradient_norms": [float(g) for g in gradient_norms] if gradient_norms else None,
        },
        "message": recommendation,
    }


@tool
def detect_pattern(
    objectives: List[float],
    constraints: Optional[List[Dict[str, float]]] = None,
    gradients: Optional[List[List[float]]] = None,
) -> Dict[str, Any]:
    """
    Detect common optimization patterns and issues.

    Use this tool to identify specific problems that require intervention:
    - Constraint boundary stuck (optimizer can't escape infeasible region)
    - Gradient noise (unreliable gradient information)
    - Trust region collapse (optimizer step size too small)
    - Cycling (returning to previous designs)

    Args:
        objectives: List of objective values
        constraints: Optional list of constraint dicts (e.g., [{"CL": 0.49}, {"CL": 0.48}])
        gradients: Optional list of gradient vectors

    Returns:
        Dict with:
            - patterns_detected: List[str] - detected pattern names
            - constraint_stuck: bool - repeated constraint violations
            - gradient_noise: bool - high gradient variance
            - trust_region_collapse: bool - very small steps
            - cycling: bool - returning to previous designs
            - severity: str - "none", "low", "medium", "high"
            - adaptations_suggested: List[str] - recommended actions
            - details: dict - pattern-specific details

    Example:
        result = detect_pattern(
            objectives=[1.0, 0.95, 0.94, 0.94, 0.94],
            constraints=[{"CL": 0.49}, {"CL": 0.48}, {"CL": 0.49}, {"CL": 0.49}, {"CL": 0.49}]
        )
        if result["constraint_stuck"]:
            print("Constraint violation detected!")
            print(f"Suggestion: {result['adaptations_suggested']}")
    """
    patterns_detected = []
    adaptations_suggested = []
    details = {}

    # Pattern 1: Constraint boundary stuck
    constraint_stuck = False
    if constraints and len(constraints) >= 3:
        # Check for repeated violations of same constraint
        constraint_names = set()
        for c in constraints:
            constraint_names.update(c.keys())

        for name in constraint_names:
            violations = []
            for c in constraints[-5:]:  # Last 5 iterations
                if name in c:
                    # Assuming constraint < 0 means satisfied, > 0 means violated
                    # Or for bound constraints like CL >= 0.5, check if consistently below
                    violations.append(c[name])

            if len(violations) >= 3:
                # Check if consistently violated (all similar values)
                violation_std = np.std(violations)
                violation_mean = np.mean(violations)

                # If all values are close and consistently violating
                if violation_std < 0.02 * abs(violation_mean) if violation_mean != 0 else violation_std < 0.01:
                    constraint_stuck = True
                    patterns_detected.append(f"constraint_stuck:{name}")
                    details[f"constraint_{name}_violations"] = violations
                    adaptations_suggested.append(
                        f"Tighten constraint '{name}' by 2-5% to force optimizer into feasible region"
                    )

    # Pattern 2: Gradient noise
    gradient_noise = False
    if gradients and len(gradients) >= 3:
        gradient_norms = [np.linalg.norm(g) for g in gradients[-5:]]

        if len(gradient_norms) >= 3:
            # Check variance of gradient norms
            grad_std = np.std(gradient_norms)
            grad_mean = np.mean(gradient_norms)

            # High variance relative to mean indicates noise
            if grad_mean > 1e-10:
                variance_ratio = grad_std / grad_mean
                if variance_ratio > 0.3:  # >30% variance
                    gradient_noise = True
                    patterns_detected.append("gradient_noise")
                    details["gradient_variance_ratio"] = float(variance_ratio)
                    adaptations_suggested.append(
                        "Switch to finite-difference gradients for more stability"
                    )

    # Pattern 3: Trust region collapse (very small steps)
    trust_region_collapse = False
    if len(objectives) >= 5:
        recent_obj = objectives[-5:]
        improvements = [abs(recent_obj[i] - recent_obj[i+1]) for i in range(len(recent_obj)-1)]

        if all(imp < 1e-10 for imp in improvements):
            # Check if gradient is still large (not converged)
            if gradients and len(gradients) > 0:
                current_grad_norm = np.linalg.norm(gradients[-1])
                if current_grad_norm > 1e-4:  # Gradient still significant
                    trust_region_collapse = True
                    patterns_detected.append("trust_region_collapse")
                    details["gradient_norm_at_collapse"] = float(current_grad_norm)
                    adaptations_suggested.append(
                        "Restart optimizer - trust region has collapsed but gradient is still large"
                    )

    # Pattern 4: Cycling (returning to previous designs)
    cycling = False
    # This would require design history, which we don't have in this simple version
    # Could be added if designs are passed in

    # Determine severity
    n_patterns = len(patterns_detected)
    if n_patterns == 0:
        severity = "none"
    elif n_patterns == 1:
        severity = "low"
    elif n_patterns == 2:
        severity = "medium"
    else:
        severity = "high"

    return {
        "success": True,
        "patterns_detected": patterns_detected,
        "constraint_stuck": constraint_stuck,
        "gradient_noise": gradient_noise,
        "trust_region_collapse": trust_region_collapse,
        "cycling": cycling,
        "severity": severity,
        "adaptations_suggested": adaptations_suggested,
        "details": details,
        "message": (
            f"Detected {n_patterns} pattern(s): {patterns_detected}" if patterns_detected
            else "No problematic patterns detected"
        ),
    }


@tool
def check_feasibility(
    design: List[float],
    constraint_values: str,
    tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Check if a design satisfies all constraints.

    Use this tool to verify constraint satisfaction before accepting a design
    as the final solution or when diagnosing feasibility issues.

    Args:
        design: Design vector to check
        constraint_values: JSON string with constraint definitions, e.g.:
            '{"CL": {"constraint_type": ">=", "bound": 0.5, "value": 0.52},
              "thickness": {"constraint_type": ">=", "bound": 0.12, "value": 0.15}}'
            Where constraint_type is ">=", "<=", or "=="
        tolerance: Tolerance for constraint satisfaction (default: 1e-6)

    Returns:
        Dict with:
            - feasible: bool - True if all constraints satisfied
            - n_violated: int - number of violated constraints
            - violations: List[Dict] - details of violated constraints
            - satisfied: List[str] - names of satisfied constraints
            - margin: Dict[str, float] - margin for each constraint (positive = satisfied)
            - message: str

    Example:
        result = check_feasibility(
            design=[1.0, 2.0],
            constraint_values='{"CL": {"constraint_type": ">=", "bound": 0.5, "value": 0.49}}'
        )
        if not result["feasible"]:
            for v in result["violations"]:
                print(f"Constraint {v['name']} violated")
    """
    import json

    try:
        constraints = json.loads(constraint_values)
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "message": f"Invalid constraint_values JSON: {e}",
        }
    violations = []
    satisfied = []
    margins = {}

    for name, constraint in constraints.items():
        c_type = constraint.get("constraint_type", constraint.get("type", "<="))
        bound = constraint.get("bound", 0.0)
        value = constraint.get("value", 0.0)

        # Calculate margin (positive = satisfied)
        if c_type == ">=":
            margin = value - bound
            is_satisfied = margin >= -tolerance
        elif c_type == "<=":
            margin = bound - value
            is_satisfied = margin >= -tolerance
        elif c_type == "==":
            margin = -abs(value - bound)
            is_satisfied = abs(value - bound) <= tolerance
        else:
            margin = 0.0
            is_satisfied = True  # Unknown constraint type

        margins[name] = float(margin)

        if is_satisfied:
            satisfied.append(name)
        else:
            violations.append({
                "name": name,
                "type": c_type,
                "bound": bound,
                "value": value,
                "margin": float(margin),
                "violation_pct": abs(margin / bound) * 100 if bound != 0 else abs(margin) * 100,
            })

    feasible = len(violations) == 0

    return {
        "success": True,
        "feasible": feasible,
        "n_violated": len(violations),
        "n_satisfied": len(satisfied),
        "violations": violations,
        "satisfied": satisfied,
        "margins": margins,
        "message": (
            "All constraints satisfied" if feasible
            else f"{len(violations)} constraint(s) violated: {[v['name'] for v in violations]}"
        ),
    }


@tool
def get_gradient_quality(
    gradients: List[List[float]],
    objectives: Optional[List[float]] = None,
    window_size: int = 5,
) -> Dict[str, Any]:
    """
    Analyze gradient quality and reliability.

    Use this tool to detect gradient issues that may cause optimization problems:
    - High variance (numerical noise from CFD shocks, mesh issues)
    - Near-zero gradients (possible local minimum or numerical issues)
    - Inconsistent direction (gradient switching direction frequently)

    Args:
        gradients: List of gradient vectors (most recent last)
        objectives: Optional corresponding objective values
        window_size: Number of gradients to analyze (default: 5)

    Returns:
        Dict with:
            - quality: str - "good", "marginal", "poor"
            - variance: float - variance of gradient norms
            - variance_ratio: float - variance / mean (normalized)
            - mean_norm: float - average gradient norm
            - current_norm: float - most recent gradient norm
            - direction_consistency: float - 0 to 1 (1 = consistent direction)
            - near_zero: bool - True if gradient is very small
            - recommendation: str - suggested action
            - details: dict

    Example:
        result = get_gradient_quality(
            gradients=[
                [0.5, 0.3], [0.48, 0.31], [0.8, -0.2],  # Last one inconsistent
                [0.45, 0.28], [0.43, 0.27]
            ]
        )
        if result["quality"] == "poor":
            print("Consider switching to finite-difference gradients")
    """
    if len(gradients) < 2:
        return {
            "success": True,
            "quality": "unknown",
            "recommendation": "Need at least 2 gradients for quality analysis",
            "message": "Insufficient gradient data",
        }

    # Use last window_size gradients
    recent_grads = gradients[-window_size:] if len(gradients) >= window_size else gradients
    gradient_arrays = [np.array(g) for g in recent_grads]

    # Calculate norms
    norms = [np.linalg.norm(g) for g in gradient_arrays]
    mean_norm = np.mean(norms)
    current_norm = norms[-1]
    variance = np.var(norms)

    # Variance ratio (normalized variance)
    variance_ratio = np.std(norms) / mean_norm if mean_norm > 1e-10 else 0.0

    # Direction consistency (cosine similarity between consecutive gradients)
    direction_consistencies = []
    for i in range(len(gradient_arrays) - 1):
        g1, g2 = gradient_arrays[i], gradient_arrays[i+1]
        norm1, norm2 = np.linalg.norm(g1), np.linalg.norm(g2)
        if norm1 > 1e-10 and norm2 > 1e-10:
            cosine_sim = np.dot(g1, g2) / (norm1 * norm2)
            direction_consistencies.append(cosine_sim)

    direction_consistency = np.mean(direction_consistencies) if direction_consistencies else 1.0

    # Near-zero check
    near_zero = current_norm < 1e-6

    # Determine quality
    if variance_ratio > 0.5 or direction_consistency < 0.5:
        quality = "poor"
        recommendation = "Gradient is unreliable. Switch to finite-difference method."
    elif variance_ratio > 0.3 or direction_consistency < 0.7:
        quality = "marginal"
        recommendation = "Gradient shows some noise. Monitor closely, consider FD if worsens."
    elif near_zero:
        quality = "converged"
        recommendation = "Gradient near zero - likely converged or at saddle point."
    else:
        quality = "good"
        recommendation = "Gradient quality is good. Continue with current method."

    return {
        "success": True,
        "quality": quality,
        "variance": float(variance),
        "variance_ratio": float(variance_ratio),
        "mean_norm": float(mean_norm),
        "current_norm": float(current_norm),
        "direction_consistency": float(direction_consistency),
        "near_zero": near_zero,
        "recommendation": recommendation,
        "details": {
            "n_gradients_analyzed": len(recent_grads),
            "norms": [float(n) for n in norms],
            "direction_similarities": [float(d) for d in direction_consistencies],
        },
        "message": f"Gradient quality: {quality}. {recommendation}",
    }


@tool
def compute_improvement_statistics(
    objectives: List[float],
    budget_used: float,
    budget_total: float,
) -> Dict[str, Any]:
    """
    Compute optimization improvement statistics.

    Use this tool to understand optimization efficiency and decide whether
    to continue, stop, or adapt strategy.

    Args:
        objectives: List of objective values (most recent last)
        budget_used: CPU hours or evaluations used so far
        budget_total: Total budget allowed

    Returns:
        Dict with:
            - total_improvement: float - improvement from start
            - total_improvement_pct: float - percentage improvement
            - recent_improvement: float - improvement in last 5 iterations
            - improvement_per_budget: float - improvement per unit budget
            - budget_remaining_pct: float - percentage of budget remaining
            - efficiency: str - "high", "medium", "low"
            - continue_recommendation: bool - whether to continue
            - message: str

    Example:
        result = compute_improvement_statistics(
            objectives=[10.0, 8.0, 6.0, 5.5, 5.2, 5.1],
            budget_used=30.0,
            budget_total=100.0
        )
        if not result["continue_recommendation"]:
            print("Consider stopping - low efficiency")
    """
    if len(objectives) < 2:
        return {
            "success": True,
            "message": "Need at least 2 objectives for statistics",
            "continue_recommendation": True,
        }

    initial_obj = objectives[0]
    current_obj = objectives[-1]

    # Total improvement
    total_improvement = initial_obj - current_obj
    total_improvement_pct = (total_improvement / abs(initial_obj)) * 100 if initial_obj != 0 else 0.0

    # Recent improvement (last 5 iterations)
    recent_obj = objectives[-5:] if len(objectives) >= 5 else objectives
    recent_improvement = recent_obj[0] - recent_obj[-1]

    # Efficiency metrics
    budget_remaining_pct = ((budget_total - budget_used) / budget_total) * 100 if budget_total > 0 else 0.0
    improvement_per_budget = total_improvement / budget_used if budget_used > 0 else 0.0

    # Determine efficiency
    if improvement_per_budget > 0.1:  # Arbitrary threshold
        efficiency = "high"
    elif improvement_per_budget > 0.01:
        efficiency = "medium"
    else:
        efficiency = "low"

    # Recommendation
    continue_recommendation = True
    if budget_remaining_pct < 10 and recent_improvement < 0.001 * abs(current_obj):
        continue_recommendation = False
    elif efficiency == "low" and budget_remaining_pct < 30:
        continue_recommendation = False

    return {
        "success": True,
        "initial_objective": float(initial_obj),
        "current_objective": float(current_obj),
        "total_improvement": float(total_improvement),
        "total_improvement_pct": float(total_improvement_pct),
        "recent_improvement": float(recent_improvement),
        "improvement_per_budget": float(improvement_per_budget),
        "budget_used": float(budget_used),
        "budget_total": float(budget_total),
        "budget_remaining_pct": float(budget_remaining_pct),
        "efficiency": efficiency,
        "continue_recommendation": continue_recommendation,
        "message": (
            f"Total improvement: {total_improvement_pct:.1f}%. "
            f"Efficiency: {efficiency}. "
            f"Budget remaining: {budget_remaining_pct:.0f}%. "
            f"{'Continue' if continue_recommendation else 'Consider stopping'}."
        ),
    }
