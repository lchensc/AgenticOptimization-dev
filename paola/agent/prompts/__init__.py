"""Agent prompts for various tasks."""

from .optimization import (
    build_optimization_prompt,
    format_problem,
    format_history,
    format_observations,
    format_tools,
)

from .evaluator_registration import (
    EVALUATOR_REGISTRATION_PROMPT,
    REGISTRATION_EXAMPLE
)

__all__ = [
    # Optimization prompts
    "build_optimization_prompt",
    "format_problem",
    "format_history",
    "format_observations",
    "format_tools",
    # Registration prompts
    "EVALUATOR_REGISTRATION_PROMPT",
    "REGISTRATION_EXAMPLE",
]
