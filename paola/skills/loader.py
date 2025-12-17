"""
Skill Loader for Paola Skills infrastructure.

Loads and parses skill files (skill.yaml, options.yaml) with caching for performance.
"""

from pathlib import Path
from typing import Any, Optional
import yaml
import logging

logger = logging.getLogger(__name__)


class SkillLoader:
    """
    Loads and parses skill files.

    Implements progressive disclosure:
    - Level 1: Metadata (name, description, when_to_use)
    - Level 2: Overview (full skill.yaml overview section)
    - Level 3: Sections (options.yaml categories, configurations.yaml)
    """

    def __init__(self, skills_root: Optional[Path] = None):
        """
        Initialize SkillLoader.

        Args:
            skills_root: Root directory for skills. Defaults to paola/skills/
        """
        if skills_root is None:
            # Default to the skills directory in the paola package
            skills_root = Path(__file__).parent

        self.skills_root = Path(skills_root)
        self._cache: dict[str, dict] = {}
        self._registry: Optional[dict] = None

    def _load_yaml(self, path: Path) -> dict:
        """Load a YAML file with caching."""
        cache_key = str(path)
        if cache_key not in self._cache:
            if not path.exists():
                raise FileNotFoundError(f"Skill file not found: {path}")
            with open(path, 'r') as f:
                self._cache[cache_key] = yaml.safe_load(f)
        return self._cache[cache_key]

    def _get_registry(self) -> dict:
        """Load the skill registry."""
        if self._registry is None:
            registry_path = self.skills_root / "registry.yaml"
            self._registry = self._load_yaml(registry_path)
        return self._registry

    def _get_skill_path(self, skill_name: str) -> Path:
        """Get the directory path for a skill."""
        registry = self._get_registry()
        skills = registry.get("skills", {})

        if skill_name not in skills:
            raise ValueError(f"Unknown skill: {skill_name}")

        skill_info = skills[skill_name]
        return self.skills_root / skill_info["path"]

    def load_metadata(self, skill_name: str) -> dict:
        """
        Load Level 1 metadata for a skill.

        Returns minimal info for skill discovery: name, description, when_to_use.
        This is what list_skills() returns.

        Args:
            skill_name: Name of the skill (e.g., "ipopt")

        Returns:
            Dict with name, description, when_to_use, keywords
        """
        skill_path = self._get_skill_path(skill_name)
        skill_yaml = self._load_yaml(skill_path / "skill.yaml")

        return {
            "name": skill_yaml.get("name", skill_name),
            "category": skill_yaml.get("category", "unknown"),
            "description": skill_yaml.get("description", ""),
            "when_to_use": skill_yaml.get("when_to_use", []),
            "when_not_to_use": skill_yaml.get("when_not_to_use", []),
            "keywords": skill_yaml.get("keywords", []),
        }

    def load_overview(self, skill_name: str) -> str:
        """
        Load Level 2 overview for a skill.

        Returns the full overview section from skill.yaml.

        Args:
            skill_name: Name of the skill

        Returns:
            Overview text (markdown formatted)
        """
        skill_path = self._get_skill_path(skill_name)
        skill_yaml = self._load_yaml(skill_path / "skill.yaml")

        return skill_yaml.get("overview", f"No overview available for {skill_name}")

    def load_section(self, skill_name: str, section: str) -> str:
        """
        Load Level 3 specific section from a skill.

        Supports:
        - "options": All options from options.yaml
        - "options.<category>": Specific option category (e.g., "options.warm_start")
        - "configurations": All configurations from configurations.yaml
        - "paola": Paola integration info from skill.yaml

        Args:
            skill_name: Name of the skill
            section: Section to load

        Returns:
            Section content as formatted string
        """
        skill_path = self._get_skill_path(skill_name)

        if section == "options":
            # Load all options
            options_yaml = self._load_yaml(skill_path / "options.yaml")
            return self._format_options(options_yaml)

        elif section.startswith("options."):
            # Load specific option category
            category = section.split(".", 1)[1]
            options_yaml = self._load_yaml(skill_path / "options.yaml")
            if category not in options_yaml:
                available = list(options_yaml.keys())
                return f"Unknown option category: {category}. Available: {available}"
            return self._format_options({category: options_yaml[category]})

        elif section == "paola":
            # Load Paola integration info
            skill_yaml = self._load_yaml(skill_path / "skill.yaml")
            paola_info = skill_yaml.get("paola", {})
            return self._format_paola_integration(paola_info)

        else:
            return f"Unknown section: {section}. Available: options, options.<category>, paola"

    def _format_options(self, options: dict) -> str:
        """Format options dict as readable text."""
        lines = []
        for category, opts in options.items():
            lines.append(f"## {category.replace('_', ' ').title()}\n")
            for opt_name, opt_info in opts.items():
                lines.append(f"### {opt_name}")
                lines.append(f"- **Type**: {opt_info.get('type', 'unknown')}")
                lines.append(f"- **Default**: {opt_info.get('default', 'N/A')}")
                if 'options' in opt_info:
                    lines.append(f"- **Options**: {opt_info['options']}")
                if 'range' in opt_info:
                    lines.append(f"- **Range**: {opt_info['range']}")
                lines.append(f"- **Description**: {opt_info.get('description', 'N/A')}")
                if 'when_to_adjust' in opt_info:
                    lines.append(f"- **When to adjust**: {opt_info['when_to_adjust']}")
                if 'when_to_use' in opt_info:
                    lines.append(f"- **When to use**: {opt_info['when_to_use']}")
                if 'recommendation' in opt_info:
                    lines.append(f"- **Recommendation**: {opt_info['recommendation']}")
                if 'paola_integration' in opt_info:
                    lines.append(f"- **Paola integration**: {opt_info['paola_integration']}")
                lines.append("")
        return "\n".join(lines)

    def _format_paola_integration(self, paola_info: dict) -> str:
        """Format Paola integration info as readable text."""
        lines = ["## Paola Integration\n"]

        lines.append(f"- **Optimizer name**: {paola_info.get('optimizer_name', 'N/A')}")
        lines.append(f"- **Backend**: {paola_info.get('backend', 'N/A')}")
        lines.append(f"- **Requires gradient**: {paola_info.get('requires_gradient', 'N/A')}")
        lines.append(f"- **Supports constraints**: {paola_info.get('supports_constraints', 'N/A')}")
        lines.append(f"- **Supports warm start**: {paola_info.get('supports_warm_start', 'N/A')}")

        if 'graph_integration' in paola_info:
            lines.append("\n### Graph Integration")
            gi = paola_info['graph_integration']
            if 'edge_configurations' in gi:
                lines.append("\n**Edge Configurations**:")
                for edge_type, cfg in gi['edge_configurations'].items():
                    lines.append(f"- **{edge_type}**: {cfg.get('description', '')} → use config `{cfg.get('use_config', 'N/A')}`")
            if 'typical_patterns' in gi:
                lines.append("\n**Typical Patterns**:")
                for pattern in gi['typical_patterns']:
                    lines.append(f"- **{pattern['pattern']}**: {pattern['description']}")
                    lines.append(f"  Example: {pattern.get('example', 'N/A')}")

        if 'evaluation_guidance' in paola_info:
            lines.append("\n### Evaluation Guidance")
            eg = paola_info['evaluation_guidance']
            lines.append(f"- **Typical cost**: {eg.get('typical_cost', 'N/A')}")
            lines.append(f"- **Gradient cost**: {eg.get('gradient_cost', 'N/A')}")
            lines.append(f"- **Hessian cost**: {eg.get('hessian_cost', 'N/A')}")
            if 'recommendation' in eg:
                lines.append(f"- **Recommendation**: {eg['recommendation']}")

        return "\n".join(lines)

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
        self._registry = None
