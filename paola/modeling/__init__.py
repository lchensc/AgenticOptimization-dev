"""
Problem Modeling Module - Mathematical modeling of optimization problems.

This module provides LLM-powered assistance for translating real-world problems
into mathematical optimization formulations.

Key capabilities:
- Natural language parsing: "minimize x^2 + 3x subject to x > 1"
- Code parsing: Python functions, classes
- Structured parsing: JSON, YAML, dict
- Mathematical modeling: Real-world description → formulation
- Problem reformulation: Scaling, penalty methods, transformations

Realistic positioning:
- LLMs ARE capable at mathematical modeling (don't shy away)
- PAOLA CAN help users formulate problems
- NOT claiming to be better than domain experts at complex problems
- Most first users already have formulations ready
"""

from .parsers import ProblemParser
from .validation import validate_problem

__all__ = [
    "ProblemParser",
    "validate_problem",
]
