"""
Abstract base classes for polymorphic optimization components.

Each optimizer family (gradient, bayesian, population, cmaes) implements
these ABCs with family-specific data structures.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class InitializationComponent(ABC):
    """
    Base class for run initialization data.

    Stores what was requested (specification) for reproducibility.
    Family-specific subclasses add actual initialization values.
    """

    specification: Dict[str, Any]

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InitializationComponent':
        """Deserialize from dictionary."""
        pass


@dataclass
class ProgressComponent(ABC):
    """
    Base class for optimization progress data.

    Family-specific subclasses store iterations/trials/generations
    with appropriate metrics.
    """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressComponent':
        """Deserialize from dictionary."""
        pass


@dataclass
class ResultComponent(ABC):
    """
    Base class for detailed result data.

    Family-specific subclasses store termination info and
    final optimizer state.
    """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResultComponent':
        """Deserialize from dictionary."""
        pass
