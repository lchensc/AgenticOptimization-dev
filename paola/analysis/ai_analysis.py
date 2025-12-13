"""
AI-powered strategic analysis.

Uses LLM to reason over deterministic metrics and provide:
- Strategic diagnosis (what's happening and why)
- Actionable recommendations (what to do next)
- Structured output (agent can execute)

Cost: ~$0.02-0.05 per analysis
Latency: 5-10 seconds
"""

from typing import Dict, Any, Literal, Optional, List
from datetime import datetime
import json
import re

from ..foundry import RunRecord
from ..agent.react_agent import initialize_llm
from langchain_core.messages import HumanMessage

# Analysis focus types
AnalysisFocus = Literal[
    "convergence",      # Why converging slowly/fast?
    "feasibility",      # Why violating constraints?
    "efficiency",       # Why so many evaluations?
    "algorithm",        # Should we switch algorithms?
    "overall",          # Holistic diagnosis
]


def ai_analyze(
    run: RunRecord,
    deterministic_metrics: Dict[str, Any],
    focus: AnalysisFocus = "overall",
    llm_model: str = "qwen-plus",
    force_reanalysis: bool = False
) -> Dict[str, Any]:
    """
    AI-powered strategic analysis of optimization run.

    Uses LLM to reason over deterministic metrics and provide
    strategic diagnosis + actionable recommendations.

    Args:
        run: Optimization run record
        deterministic_metrics: Pre-computed metrics from compute_metrics()
        focus: What aspect to analyze
        llm_model: LLM to use for reasoning
        force_reanalysis: Ignore cached insights

    Returns:
        {
            "diagnosis": str,           # What's happening (2-3 sentences)
            "root_cause": str,          # Why it's happening (1-2 sentences)
            "confidence": "low" | "medium" | "high",
            "evidence": [str],          # Supporting evidence from metrics
            "recommendations": [
                {
                    "action": str,      # Tool name (e.g., "constraint_adjust")
                    "args": dict,       # Tool arguments
                    "rationale": str,   # Why this helps
                    "priority": int,    # Execution order (1=first)
                    "expected_impact": str,  # What should change
                }
            ],
            "metadata": {
                "model": str,
                "timestamp": str,
                "focus": str,
                "cost_estimate": float,
            }
        }

    Example:
        from paola.analysis import compute_metrics, ai_analyze

        # Deterministic first (instant, free)
        metrics = compute_metrics(run)

        # AI if needed (costs money, strategic)
        if metrics["convergence"]["is_stalled"]:
            insights = ai_analyze(run, metrics, focus="convergence")

            # Execute recommendations
            for rec in insights["recommendations"]:
                if rec["action"] == "constraint_adjust":
                    constraint_adjust_bounds(**rec["args"])
    """
    # Check cache (unless force_reanalysis)
    if not force_reanalysis and run.ai_insights:
        cached = run.ai_insights
        if cached.get("metadata", {}).get("focus") == focus:
            # Check if recent (within 1 hour)
            cached_time = datetime.fromisoformat(cached.get("metadata", {}).get("timestamp", "2000-01-01"))
            age_seconds = (datetime.now() - cached_time).total_seconds()
            if age_seconds < 3600:  # 1 hour
                return cached

    # Build structured prompt
    prompt = _build_analysis_prompt(run, deterministic_metrics, focus)

    # Call LLM
    try:
        llm = initialize_llm(llm_model, temperature=0.0)
        response = llm.invoke([HumanMessage(content=prompt)])

        # Parse structured response
        insights = _parse_analysis_response(response.content)

        # Add metadata
        insights["metadata"] = {
            "model": llm_model,
            "timestamp": datetime.now().isoformat(),
            "focus": focus,
            "cost_estimate": _estimate_cost(llm_model, prompt),
        }

        return insights

    except Exception as e:
        # Fallback on error
        return {
            "diagnosis": f"AI analysis failed: {str(e)}",
            "root_cause": "Error during LLM invocation",
            "confidence": "low",
            "evidence": [],
            "recommendations": [],
            "metadata": {
                "model": llm_model,
                "timestamp": datetime.now().isoformat(),
                "focus": focus,
                "error": str(e),
            }
        }


def _build_analysis_prompt(
    run: RunRecord,
    metrics: Dict[str, Any],
    focus: AnalysisFocus
) -> str:
    """
    Build structured prompt for AI analysis.

    Prompt structure:
    1. Problem context (algorithm, problem name)
    2. Current status (iterations, objective values)
    3. Deterministic metrics (formatted for readability)
    4. Analysis focus
    5. Output format (JSON schema)

    Returns:
        Structured prompt string
    """
    metrics_formatted = _format_metrics_for_llm(metrics)
    recent_iters = run.result_data.get("iterations", [])[-10:]

    prompt = f"""You are an expert optimization analyst. Analyze this run and provide strategic recommendations.

PROBLEM:
- Algorithm: {run.algorithm}
- Problem: {run.problem_name}

CURRENT STATUS:
- Total iterations: {metrics['convergence']['iterations_total']}
- Current objective: {metrics['objective']['current']:.6f}
- Best objective: {metrics['objective']['best']:.6f}
- Improvement from start: {metrics['objective']['improvement_from_start']:.6f}
- Success: {run.success}

DETERMINISTIC METRICS:
{metrics_formatted}

RECENT ITERATIONS (last 10):
{_format_recent_iterations(recent_iters)}

ANALYSIS FOCUS: {focus}

YOUR TASK:
1. Diagnose what's happening in this optimization
2. Identify root cause of current behavior
3. Provide concrete, actionable recommendations

Output MUST be valid JSON with this structure:
{{
    "diagnosis": "Brief explanation of current state (2-3 sentences)",
    "root_cause": "Why this is happening (1-2 sentences)",
    "confidence": "low|medium|high",
    "evidence": [
        "Metric or pattern that supports diagnosis",
        "Another piece of evidence"
    ],
    "recommendations": [
        {{
            "action": "constraint_adjust|optimizer_restart|algorithm_switch",
            "args": {{"param_name": "value"}},
            "rationale": "Why this action will help",
            "priority": 1,
            "expected_impact": "What should change"
        }}
    ]
}}

IMPORTANT:
- Base diagnosis on provided metrics (evidence-driven)
- Recommendations must be specific (e.g., "restart from iteration 45" not "restart")
- action field must be a tool name that agent can call
- Provide 1-3 recommendations ordered by priority
- If optimization succeeded, focus on what worked well
"""
    return prompt


def _format_metrics_for_llm(metrics: Dict[str, Any]) -> str:
    """Format metrics in readable way for LLM."""
    lines = []

    # Convergence
    conv = metrics["convergence"]
    lines.append(f"Convergence:")
    lines.append(f"  - Rate: {conv['rate']:.4f}")
    lines.append(f"  - Stalled: {'YES' if conv['is_stalled'] else 'NO'}")
    lines.append(f"  - Improvement (last 10): {conv['improvement_last_10']:.6f}")

    # Gradient
    grad = metrics["gradient"]
    lines.append(f"\nGradient:")
    lines.append(f"  - Norm: {grad['norm']:.6e}")
    lines.append(f"  - Variance: {grad['variance']:.6e}")
    lines.append(f"  - Quality: {grad['quality']}")

    # Constraints
    const = metrics["constraints"]
    lines.append(f"\nConstraints:")
    if const["violations"]:
        lines.append(f"  - Violations: {len(const['violations'])}")
        for v in const["violations"][:3]:  # Show first 3
            lines.append(f"    • {v['name']}: value={v['value']:.4f}, margin={v['margin']:.4f}")
    else:
        lines.append(f"  - All satisfied ✓")

    # Efficiency
    eff = metrics["efficiency"]
    lines.append(f"\nEfficiency:")
    lines.append(f"  - Total evaluations: {eff['evaluations']}")
    lines.append(f"  - Improvement per eval: {eff['improvement_per_eval']:.6f}")

    return "\n".join(lines)


def _format_recent_iterations(iterations: List[Dict]) -> str:
    """Format recent iterations for LLM."""
    if not iterations:
        return "No iteration data available"

    lines = []
    for it in iterations:
        iter_num = it.get("iteration", "?")
        obj = it.get("objective", float('inf'))
        lines.append(f"  Iter {iter_num}: obj={obj:.6f}")

    return "\n".join(lines)


def _parse_analysis_response(response: str) -> Dict[str, Any]:
    """
    Parse LLM response into structured format.

    Handles JSON in markdown code blocks and direct JSON.
    """
    # Try to extract JSON (handle markdown wrapping)
    json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response

    try:
        insights = json.loads(json_str)

        # Validate required fields
        required = ["diagnosis", "root_cause", "recommendations"]
        for field in required:
            if field not in insights:
                # Provide default
                if field == "recommendations":
                    insights[field] = []
                else:
                    insights[field] = f"Missing {field}"

        # Ensure confidence field exists
        if "confidence" not in insights:
            insights["confidence"] = "medium"

        # Ensure evidence field exists
        if "evidence" not in insights:
            insights["evidence"] = []

        return insights

    except (json.JSONDecodeError, ValueError) as e:
        # Fallback: return unstructured
        return {
            "diagnosis": response[:500] if len(response) > 500 else response,
            "root_cause": "Unable to parse structured response",
            "confidence": "low",
            "evidence": [],
            "recommendations": [],
            "parse_error": str(e)
        }


def _estimate_cost(llm_model: str, prompt: str) -> float:
    """
    Estimate cost of analysis.

    Very rough estimates based on typical pricing.
    """
    # Rough token estimate (1 token ≈ 4 chars)
    input_tokens = len(prompt) / 4
    output_tokens = 500  # Typical output size

    # Rough pricing (per 1M tokens)
    pricing = {
        "qwen-flash": {"input": 0.10, "output": 0.10},
        "qwen-plus": {"input": 0.50, "output": 0.50},
        "qwen-max": {"input": 2.00, "output": 2.00},
    }

    model_pricing = pricing.get(llm_model, {"input": 0.50, "output": 0.50})

    cost = (input_tokens / 1_000_000 * model_pricing["input"]) + \
           (output_tokens / 1_000_000 * model_pricing["output"])

    return round(cost, 4)
