"""
Skill Tools for Paola agent.

LangChain @tool decorated functions for skill discovery and loading.
These tools provide LLM-agnostic access to Paola's skill system.
"""

from typing import Optional
from langchain_core.tools import tool

from .loader import SkillLoader
from .index import SkillIndex

# Global instances (initialized on first use)
_loader: Optional[SkillLoader] = None
_index: Optional[SkillIndex] = None


def _get_loader() -> SkillLoader:
    """Get or create the global SkillLoader."""
    global _loader
    if _loader is None:
        _loader = SkillLoader()
    return _loader


def _get_index() -> SkillIndex:
    """Get or create the global SkillIndex."""
    global _index
    if _index is None:
        _index = SkillIndex()
    return _index


@tool
def list_skills(category: Optional[str] = None) -> str:
    """
    List available Paola skills for optimizer configuration expertise.

    Use when:
    - You need to configure a specific optimizer (IPOPT, SciPy, Optuna, NLopt)
    - The user asks for non-default settings (tolerances, warm-start, scaling)
    - Optimization is failing and you need diagnostic guidance
    - You're unsure which optimizer options exist

    Don't use when:
    - Running simple optimizations with default settings
    - The problem is straightforward (small dimension, no constraints)

    Args:
        category: Optional filter ("optimizers", "domains", "patterns", "learned")

    Returns:
        Formatted list of skills with name, description, and when_to_use
    """
    try:
        index = _get_index()
        skills = index.list_skills(category)

        if not skills:
            if category:
                return f"No skills found in category: {category}"
            return "No skills found."

        lines = []
        if category:
            lines.append(f"## Skills in '{category}' category:\n")
        else:
            lines.append("## Available Paola Skills:\n")

        for skill in skills:
            lines.append(f"### {skill['name']}")
            lines.append(f"**Category**: {skill.get('category', 'unknown')}")
            lines.append(f"**Description**: {skill.get('description', 'N/A')}")

            when_to_use = skill.get('when_to_use', [])
            if when_to_use:
                lines.append("**When to use**:")
                if isinstance(when_to_use, list):
                    for item in when_to_use[:3]:  # Limit to 3 items
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"  {when_to_use}")

            lines.append("")

        lines.append("\nUse `load_skill(skill_name)` to get detailed information.")
        return "\n".join(lines)

    except Exception as e:
        return f"Error listing skills: {str(e)}"


@tool
def load_skill(skill_name: str, section: str = "overview") -> str:
    """
    Load detailed optimizer configuration knowledge.

    Use when:
    - Configuring IPOPT, SciPy, Optuna, or NLopt beyond defaults
    - You need specific option values (tolerances, max iterations, scaling)
    - Setting up warm-start from a previous optimization
    - Diagnosing why optimization failed or converged slowly

    Progressive disclosure (load only what you need):
    - "overview": Start here - capabilities and common configurations
    - "options": Full option reference when you need specific settings
    - "options.<category>": Just one category (e.g., "options.warm_start")

    Args:
        skill_name: "ipopt", "scipy", "optuna", or "nlopt"
        section: "overview" (default), "options", "options.<category>", "paola"

    Returns:
        Skill content with configuration guidance
    """
    try:
        loader = _get_loader()

        if section == "overview":
            # Load overview (Level 2)
            metadata = loader.load_metadata(skill_name)
            overview = loader.load_overview(skill_name)

            lines = [
                f"# {metadata['name'].upper()} Skill\n",
                f"**Category**: {metadata.get('category', 'unknown')}",
                f"**Description**: {metadata.get('description', 'N/A')}\n",
            ]

            # When to use
            when_to_use = metadata.get('when_to_use', [])
            if when_to_use:
                lines.append("**When to use**:")
                if isinstance(when_to_use, list):
                    for item in when_to_use:
                        lines.append(f"- {item}")
                else:
                    lines.append(f"  {when_to_use}")
                lines.append("")

            # When NOT to use
            when_not = metadata.get('when_not_to_use', [])
            if when_not:
                lines.append("**When NOT to use**:")
                if isinstance(when_not, list):
                    for item in when_not:
                        lines.append(f"- {item}")
                else:
                    lines.append(f"  {when_not}")
                lines.append("")

            lines.append("---\n")
            lines.append(overview)

            return "\n".join(lines)

        else:
            # Load specific section (Level 3)
            content = loader.load_section(skill_name, section)
            return f"# {skill_name.upper()} - {section}\n\n{content}"

    except FileNotFoundError as e:
        return f"Skill not found: {skill_name}. Use list_skills() to see available skills."
    except Exception as e:
        return f"Error loading skill: {str(e)}"


@tool
def query_skills(query: str, limit: int = 3) -> str:
    """
    Search skills when you're unsure which optimizer skill to use.

    Use when:
    - You don't know which optimizer has the feature you need
    - Looking for cross-optimizer knowledge (e.g., "warm-start" across all)
    - The user mentions a concept but not a specific optimizer

    Prefer load_skill() when you already know the optimizer name.

    Args:
        query: What you're looking for (e.g., "warm-start", "constraint handling")
        limit: Max results (default: 3)

    Returns:
        Matching skills ranked by relevance
    """
    try:
        index = _get_index()
        results = index.search(query, limit=limit)

        if not results:
            return f"No skills found matching: '{query}'"

        lines = [f"## Skills matching: '{query}'\n"]

        for result in results:
            lines.append(f"### {result['name']} (relevance: {result['relevance_score']})")
            lines.append(f"**Category**: {result.get('category', 'unknown')}")
            lines.append(f"**Description**: {result.get('description', 'N/A')}")

            when_to_use = result.get('when_to_use', [])
            if when_to_use:
                lines.append("**When to use**:")
                if isinstance(when_to_use, list):
                    for item in when_to_use[:2]:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"  {str(when_to_use)[:100]}")

            lines.append("")

        lines.append("\nUse `load_skill(skill_name)` for detailed information.")
        return "\n".join(lines)

    except Exception as e:
        return f"Error searching skills: {str(e)}"


# Export all tools
def get_skill_tools():
    """Get all skill tools for agent registration."""
    return [
        list_skills,
        load_skill,
        query_skills,
    ]
