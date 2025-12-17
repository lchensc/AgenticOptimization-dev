"""
Paola Skills Infrastructure.

Skills are external knowledge packages that enable Paola to become
an expert in optimization. Features:

- **Progressive disclosure**: Load knowledge on-demand (token efficient)
- **LLM-agnostic**: Works with any LLM via tool interface
- **Graph-aware**: Configurations know about Paola's graph architecture
- **Learning-ready**: Supports auto-generated learned skills

Skill Categories:
- optimizers: Optimizer expertise (IPOPT, SciPy, Optuna)
- domains: Domain expertise (aerodynamics, structures, MDO)
- patterns: Optimization patterns (warm-starting, multi-fidelity)
- learned: Auto-generated from successful optimizations

Usage:
    from paola.skills import get_skill_tools
    tools = get_skill_tools()  # Register with agent
"""

from .loader import SkillLoader
from .index import SkillIndex
from .tools import (
    list_skills,
    load_skill,
    query_skills,
    get_skill_tools,
)

__all__ = [
    # Classes
    "SkillLoader",
    "SkillIndex",
    # Tools
    "list_skills",
    "load_skill",
    "query_skills",
    "get_skill_tools",
]
