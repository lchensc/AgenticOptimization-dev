"""
Skill Index for Paola Skills infrastructure.

Indexes skills for discovery and search operations.
"""

from pathlib import Path
from typing import Optional
import yaml
import logging

logger = logging.getLogger(__name__)


class SkillIndex:
    """
    Indexes skills for discovery and search.

    Provides:
    - Skill listing by category
    - Keyword-based search
    - Skill path resolution
    """

    def __init__(self, skills_root: Optional[Path] = None):
        """
        Initialize SkillIndex.

        Args:
            skills_root: Root directory for skills. Defaults to paola/skills/
        """
        if skills_root is None:
            skills_root = Path(__file__).parent

        self.skills_root = Path(skills_root)
        self._registry: Optional[dict] = None
        self._index: Optional[dict] = None

    def _load_registry(self) -> dict:
        """Load the skill registry."""
        if self._registry is None:
            registry_path = self.skills_root / "registry.yaml"
            if not registry_path.exists():
                raise FileNotFoundError(f"Registry not found: {registry_path}")
            with open(registry_path, 'r') as f:
                self._registry = yaml.safe_load(f)
        return self._registry

    def build_index(self) -> None:
        """
        Build the skill index from registry and skill files.

        This loads metadata from all registered skills for fast lookup.
        """
        from .loader import SkillLoader

        registry = self._load_registry()
        loader = SkillLoader(self.skills_root)

        self._index = {
            "skills": {},
            "by_category": {},
            "by_keyword": {},
        }

        for skill_name, skill_info in registry.get("skills", {}).items():
            try:
                metadata = loader.load_metadata(skill_name)

                # Store full metadata
                self._index["skills"][skill_name] = {
                    **metadata,
                    "path": skill_info.get("path", ""),
                }

                # Index by category
                category = metadata.get("category", "unknown")
                if category not in self._index["by_category"]:
                    self._index["by_category"][category] = []
                self._index["by_category"][category].append(skill_name)

                # Index by keyword
                for keyword in metadata.get("keywords", []):
                    keyword_lower = keyword.lower()
                    if keyword_lower not in self._index["by_keyword"]:
                        self._index["by_keyword"][keyword_lower] = []
                    self._index["by_keyword"][keyword_lower].append(skill_name)

            except Exception as e:
                logger.warning(f"Failed to index skill {skill_name}: {e}")

    def _ensure_index(self) -> None:
        """Ensure the index is built."""
        if self._index is None:
            self.build_index()

    def list_skills(self, category: Optional[str] = None) -> list[dict]:
        """
        List skills, optionally filtered by category.

        Args:
            category: Optional category filter (e.g., "optimizers", "domains")

        Returns:
            List of skill metadata dicts
        """
        self._ensure_index()

        if category:
            skill_names = self._index["by_category"].get(category, [])
        else:
            skill_names = list(self._index["skills"].keys())

        return [self._index["skills"][name] for name in skill_names]

    def list_categories(self) -> list[dict]:
        """
        List all skill categories.

        Returns:
            List of category info dicts
        """
        registry = self._load_registry()
        categories = registry.get("categories", {})

        result = []
        for name, info in categories.items():
            result.append({
                "name": name,
                "description": info.get("description", ""),
                "skill_count": len(info.get("skills", [])),
            })
        return result

    def search(self, query: str, limit: int = 3) -> list[dict]:
        """
        Search skills by keyword.

        Currently uses simple keyword matching.
        Future: embedding-based semantic search.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching skill metadata with relevance scores
        """
        self._ensure_index()

        # Tokenize query
        query_terms = query.lower().split()

        # Score each skill
        scores = {}
        for skill_name, metadata in self._index["skills"].items():
            score = 0

            # Check name
            if any(term in skill_name.lower() for term in query_terms):
                score += 10

            # Check description
            description = metadata.get("description", "").lower()
            for term in query_terms:
                if term in description:
                    score += 5

            # Check keywords
            keywords = [k.lower() for k in metadata.get("keywords", [])]
            for term in query_terms:
                if term in keywords:
                    score += 8
                elif any(term in kw for kw in keywords):
                    score += 3

            # Check when_to_use
            when_to_use = metadata.get("when_to_use", [])
            if isinstance(when_to_use, list):
                when_to_use_text = " ".join(when_to_use).lower()
            else:
                when_to_use_text = str(when_to_use).lower()
            for term in query_terms:
                if term in when_to_use_text:
                    score += 4

            if score > 0:
                scores[skill_name] = score

        # Sort by score and return top results
        sorted_skills = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for skill_name, score in sorted_skills[:limit]:
            result = self._index["skills"][skill_name].copy()
            result["relevance_score"] = score
            results.append(result)

        return results

    def get_skill_path(self, skill_name: str) -> Path:
        """
        Get the directory path for a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            Path to skill directory
        """
        registry = self._load_registry()
        skills = registry.get("skills", {})

        if skill_name not in skills:
            raise ValueError(f"Unknown skill: {skill_name}")

        return self.skills_root / skills[skill_name]["path"]

    def skill_exists(self, skill_name: str) -> bool:
        """
        Check if a skill exists.

        Args:
            skill_name: Name of the skill

        Returns:
            True if skill exists
        """
        registry = self._load_registry()
        return skill_name in registry.get("skills", {})

    def clear_cache(self) -> None:
        """Clear the internal cache and index."""
        self._registry = None
        self._index = None
