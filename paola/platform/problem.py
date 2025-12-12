"""Problem definition for optimization."""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import json


@dataclass
class Problem:
    """
    Optimization problem metadata.

    Stores problem definition and characteristics for
    retrieval and knowledge matching.
    """
    problem_id: str
    name: str
    dimensions: int
    problem_type: str  # "linear", "quadratic", "nonlinear", etc.
    created_at: str  # ISO format
    metadata: Dict[str, Any]  # Additional characteristics

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Problem':
        """Deserialize from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'Problem':
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))

    def signature(self) -> Dict[str, Any]:
        """
        Extract problem signature for knowledge matching.

        Returns:
            Dict with problem characteristics for embedding
        """
        return {
            "problem_type": self.problem_type,
            "dimensions": self.dimensions,
            **self.metadata.get("characteristics", {})
        }
