"""Data models for optimization storage."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import json


@dataclass
class OptimizationRun:
    """Serializable optimization run record."""
    run_id: int
    problem_id: str
    problem_name: str
    algorithm: str
    objective_value: float
    success: bool
    n_evaluations: int
    timestamp: str  # ISO format
    duration: float
    result_data: Dict[str, Any]  # Full scipy result as dict

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationRun':
        """Deserialize from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'OptimizationRun':
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Problem:
    """Benchmark problem metadata."""
    problem_id: str
    name: str
    dimensions: int
    problem_type: str
    created_at: str
    metadata: Dict[str, Any]

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
