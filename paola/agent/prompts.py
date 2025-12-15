"""
Prompts for optimization agent.

Separated from agent logic for clarity and maintainability.
"""

from typing import Dict, Any, List, Optional


def build_optimization_prompt(context: dict, tools: list = None) -> str:
    """
    Build prompt with current optimization state.

    Args:
        context: Current optimization context
        tools: List of actually available tools (if None, lists generic tools)

    Returns:
        Formatted prompt string
    """
    # Get budget status
    budget_status = context.get('budget_status', {})
    budget_text = f"{budget_status.get('used', 0):.1f} / {budget_status.get('total', 'Unknown')} CPU hours"

    # Get cache stats
    cache_stats = context.get('cache_stats', {})
    cache_hit_rate = cache_stats.get('hit_rate', 0.0)

    return f"""
You are an autonomous optimization agent.

**Goal:** {context.get('goal', 'Not set')}

**Status:**
- Problem: {format_problem(context.get('problem', {}))}
- Optimizer: {context.get('optimizer_type', 'Not created')}
- Iteration: {context.get('iteration', 0)}
- Current objective: {context.get('current_objectives', 'N/A')}
- Best objective: {context.get('best_objectives', 'N/A')}

**Resources:**
- Budget: {budget_text}
- Cache hit rate: {cache_hit_rate:.1%}
- Evaluations: {context.get('total_evaluations', 0)}

**History:**
{format_history(context.get('history', [])[-5:])}

**Observations:**
{format_observations(context.get('observations', {}))}

**Tools:**
{format_tools(tools)}

**Instructions:**
1. Tool arguments must be valid JSON. Do NOT use Python syntax like `[...] * 10` or `True/False`. Write out arrays fully.
2. Evaluators must be registered in Foundry before use (check foundry_list_evaluators)

Decide next action. Use tools or respond "DONE".
"""


def format_problem(problem: dict) -> str:
    """
    Format problem for prompt.

    Args:
        problem: Problem dictionary with objectives, variables, constraints

    Returns:
        Formatted problem description
    """
    if not problem:
        return "Not formulated yet"

    lines = []

    # Objectives
    objectives = problem.get('objectives', [])
    if objectives:
        obj_strs = [f"{obj.get('name')} ({obj.get('sense')})" for obj in objectives]
        lines.append(f"Objectives: {', '.join(obj_strs)}")

    # Variables
    variables = problem.get('variables', [])
    if variables:
        lines.append(f"Variables: {len(variables)} design variables")

    # Constraints
    constraints = problem.get('constraints', [])
    if constraints:
        lines.append(f"Constraints: {len(constraints)} constraints")

    return '\n'.join(lines) if lines else "Empty problem"


def format_history(history: list) -> str:
    """
    Format recent history for prompt.

    Args:
        history: List of iteration records

    Returns:
        Formatted history string
    """
    if not history:
        return "No history yet"

    lines = []
    for entry in history:
        iter_num = entry.get('iteration', '?')
        obj = entry.get('objective', 'N/A')
        lines.append(f"  Iter {iter_num}: obj={obj}")

    return '\n'.join(lines) if lines else "No history"


def format_observations(observations: dict) -> str:
    """
    Format observations for prompt.

    Args:
        observations: Dictionary of observation metrics

    Returns:
        Formatted observations string
    """
    if not observations:
        return "No observations yet"

    lines = []
    for key, value in observations.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.6e}")
        else:
            lines.append(f"  {key}: {value}")

    return '\n'.join(lines) if lines else "No observations"


def format_tools(tools: list = None) -> str:
    """
    Format available tools for prompt.

    Args:
        tools: List of actual tools bound to agent (if None, uses default list)

    Returns:
        Formatted string listing available tools
    """
    if tools is None:
        # Fallback to default tool list
        return _get_default_tool_list()

    # Dynamic tool listing based on actual bound tools
    lines = []
    for tool in tools:
        tool_name = tool.name if hasattr(tool, 'name') else str(tool)
        tool_desc = tool.description if hasattr(tool, 'description') else "No description"

        # Truncate long descriptions
        if len(tool_desc) > 80:
            tool_desc = tool_desc[:77] + "..."

        lines.append(f"- {tool_name}: {tool_desc}")

    return '\n'.join(lines) if lines else "No tools available"


def _get_default_tool_list() -> str:
    """
    Default tool list (for when tools not provided).

    This is a fallback - prefer using actual tools when available.

    Returns:
        Formatted tool list string
    """
    return """
**Problem Formulation:**
- create_nlp_problem: Create NLP from registered evaluators

**Run Management:**
- start_optimization_run: Start new optimization run
- finalize_optimization_run: Finalize completed run
- get_active_runs: Get active optimization runs

**Optimization:**
- run_scipy_optimization: Run SciPy optimizer (SLSQP, BFGS, etc.)

**Analysis (Deterministic - Fast & Free):**
- analyze_convergence: Check convergence rate, stalling, improvement
- analyze_efficiency: Check evaluations and improvement per eval
- get_all_metrics: Get complete metric suite (convergence, gradient, constraints, efficiency)

**Analysis (AI-Powered - Strategic, ~$0.02-0.05):**
- analyze_run_with_ai: AI diagnosis with actionable recommendations

**Knowledge (Skeleton - Not Yet Implemented):**
- store_optimization_insight: Store insight from completed optimization
- retrieve_optimization_knowledge: Retrieve insights from similar problems
- list_all_knowledge: List all stored insights
"""
