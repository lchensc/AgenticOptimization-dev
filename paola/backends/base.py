"""
Base classes for evaluation backends.

Defines abstract interface for all evaluation backends:
- Analytical test functions
- User-provided functions
- CFD/FEA simulations
- ML training
- External APIs
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class EvaluationResult:
    """
    Result from evaluation backend.

    Contains:
    - objectives: Dict mapping objective names to values
    - constraints: Dict mapping constraint names to values
    - cost: Computational cost (CPU hours or arbitrary units)
    - metadata: Additional information
    """
    objectives: Dict[str, float]
    constraints: Dict[str, float]
    cost: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class EvaluationBackend(ABC):
    """
    Abstract interface for evaluation backends.

    Backends can be:
    1. User-provided functions (most common)
    2. Analytical test functions
    3. Physics engines (CFD, FEA)
    4. ML model training
    5. External APIs
    """

    @abstractmethod
    def evaluate(self, design: np.ndarray) -> EvaluationResult:
        """
        Evaluate objective and constraints at design point.

        Args:
            design: Design vector (numpy array)

        Returns:
            EvaluationResult with objectives, constraints, cost
        """
        pass

    @abstractmethod
    def compute_gradient(
        self,
        design: np.ndarray,
        method: str = "auto"
    ) -> np.ndarray:
        """
        Compute gradient of objective.

        Args:
            design: Design point
            method: Gradient method
                   - "auto": Use best available
                   - "adjoint": Adjoint solver (CFD/FEA)
                   - "finite_difference": Numerical approximation
                   - "symbolic": Symbolic differentiation

        Returns:
            Gradient array (same shape as design)
        """
        pass

    @property
    @abstractmethod
    def supports_gradients(self) -> bool:
        """Whether this backend provides gradients."""
        pass

    @property
    @abstractmethod
    def cost_per_evaluation(self) -> float:
        """Computational cost per evaluation (CPU hours or arbitrary units)."""
        pass

    @property
    def domain(self) -> str:
        """Problem domain (for knowledge classification)."""
        return "unknown"

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(domain={self.domain})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"{self.__class__.__name__}("
            f"domain={self.domain}, "
            f"gradients={self.supports_gradients}, "
            f"cost={self.cost_per_evaluation})"
        )
