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
    List available Paola skills with their descriptions.

    This is the entry point for discovering what optimization expertise
    Paola has. Call this first to see what skills are available, then
    use load_skill() to get detailed knowledge.

    Args:
        category: Optional filter for skill category
            - "optimizers": Optimizer expertise (IPOPT, SciPy, Optuna)
            - "domains": Domain expertise (aerodynamics, structures, MDO)
            - "patterns": Optimization patterns (warm-starting, multi-fidelity)
            - "learned": Knowledge from past optimizations
            - None: List all skills

    Returns:
        Formatted list of skills with name, description, and when_to_use

    Example:
        list_skills()               # All skills
        list_skills("optimizers")   # Just optimizer skills
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
    Load detailed knowledge from a Paola skill.

    Skills use progressive disclosure - start with "overview" and
    drill down to specific sections as needed.

    Args:
        skill_name: Skill to load (e.g., "ipopt", "scipy", "aerodynamics")
        section: Which section to load
            - "overview": Main guidance and capabilities (default)
            - "options": Full option/parameter reference
            - "options.<category>": Specific option category
              (e.g., "options.warm_start", "options.hessian")
            - "paola": Paola integration details (graph edges, learning)

    Returns:
        Skill content formatted for reading

    Example:
        load_skill("ipopt")                       # Overview
        load_skill("ipopt", "options")            # All options
        load_skill("ipopt", "options.warm_start") # Just warm-start options
        load_skill("ipopt", "paola")              # Paola integration
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
    Search across all Paola skills for relevant knowledge.

    Use this when you're not sure which skill has the information,
    or to find related knowledge across multiple skills.

    Args:
        query: Natural language query describing what you're looking for
        limit: Maximum number of results (default: 3)

    Returns:
        Relevant skills matching the query, ranked by relevance

    Example:
        query_skills("warm-start configuration")
        query_skills("how to handle infeasible constraints")
        query_skills("large-scale optimization")
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
