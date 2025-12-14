"""
PAOLA capabilities for evaluators.

Built-in features that enhance all registered evaluators:
- Observation gates: Monitor every evaluation
- Evaluation caching: Avoid redundant expensive evaluations
- Interjection support: User/agent can interrupt
- Performance tracking: Learn execution patterns
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime

from ..backends.base import EvaluationResult


class EvaluationObserver:
    """
    Observation gates for monitoring evaluations.

    Logs every evaluation before and after execution.
    Provides hooks for:
    - Pre-evaluation checks (bounds, similarity to failed designs)
    - Post-evaluation analysis (anomalies, performance issues)
    - Interjection triggers (user/agent wants to intervene)
    """

    def __init__(self, evaluator_id: str, logger: Optional[logging.Logger] = None):
        """
        Initialize observer.

        Args:
            evaluator_id: ID of evaluator being observed
            logger: Optional logger (creates one if not provided)
        """
        self.evaluator_id = evaluator_id

        if logger is None:
            logger = logging.getLogger(f"paola.evaluator.{evaluator_id}")
            logger.setLevel(logging.INFO)

        self.logger = logger
        self.evaluation_count = 0
        self.interjection_callback = None  # Set by agent if needed

    def before_evaluation(self, design: np.ndarray, evaluator_id: str):
        """
        Called before every evaluation.

        Logs design point and performs pre-checks.

        Args:
            design: Design vector to be evaluated
            evaluator_id: ID of evaluator
        """
        self.evaluation_count += 1

        self.logger.info(
            f"[{self.evaluation_count}] Pre-evaluation: "
            f"design={design[:3]}{'...' if len(design) > 3 else ''}"
        )

        # Future: Add pre-checks
        # - Check if design is in bounds
        # - Check if similar to previously failed designs
        # - Estimate cost and check budget

    def after_evaluation(
        self,
        design: np.ndarray,
        result: EvaluationResult,
        execution_time: float,
        evaluator_id: str
    ) -> bool:
        """
        Called after every evaluation.

        Logs result and checks for anomalies.
        Returns whether to continue or trigger interjection.

        Args:
            design: Design vector that was evaluated
            result: Evaluation result
            execution_time: Time taken (seconds)
            evaluator_id: ID of evaluator

        Returns:
            True: Continue normally
            False: Trigger interjection (agent/user wants to intervene)
        """
        # Log result
        obj_values = list(result.objectives.values())
        self.logger.info(
            f"[{self.evaluation_count}] Post-evaluation: "
            f"objectives={obj_values[:3]}{'...' if len(obj_values) > 3 else ''}, "
            f"time={execution_time:.3f}s"
        )

        # Check for anomalies
        if execution_time > 3600:  # > 1 hour
            self.logger.warning(f"Evaluation took {execution_time:.1f}s (> 1 hour)")

        if any(np.isnan(v) or np.isinf(v) for v in obj_values):
            self.logger.error(f"NaN or Inf in objectives: {result.objectives}")

        # Check if interjection requested
        if self.interjection_callback:
            should_interject = self.interjection_callback(design, result)
            if should_interject:
                self.logger.info("Interjection requested by callback")
                return False

        return True  # Continue normally

    def set_interjection_callback(self, callback):
        """
        Set callback for interjection decisions.

        Callback signature: (design, result) -> bool
        Returns True to trigger interjection, False to continue.

        Args:
            callback: Function that decides whether to interject
        """
        self.interjection_callback = callback


class EvaluationCache:
    """
    Cache for expensive evaluations.

    Stores evaluation results indexed by design vector.
    Supports tolerance-based lookup (designs within epsilon are considered same).
    """

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize cache.

        Args:
            tolerance: Tolerance for design vector comparison
        """
        self.tolerance = tolerance
        self._cache: Dict[str, EvaluationResult] = {}
        self.hits = 0
        self.misses = 0

    def _design_key(self, design: np.ndarray) -> str:
        """
        Generate cache key from design vector.

        Rounds to tolerance to enable tolerance-based lookups.

        Args:
            design: Design vector

        Returns:
            Cache key string
        """
        # Round to tolerance precision
        rounded = np.round(design / self.tolerance) * self.tolerance
        return str(rounded.tolist())

    def get(self, design: np.ndarray) -> Optional[EvaluationResult]:
        """
        Retrieve cached result for design.

        Args:
            design: Design vector

        Returns:
            Cached EvaluationResult if found, None otherwise
        """
        key = self._design_key(design)

        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        else:
            self.misses += 1
            return None

    def store(self, design: np.ndarray, result: EvaluationResult):
        """
        Store evaluation result in cache.

        Args:
            design: Design vector
            result: Evaluation result
        """
        key = self._design_key(design)
        self._cache[key] = result

    def clear(self):
        """Clear cache."""
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        """Number of cached evaluations."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __str__(self) -> str:
        """String representation."""
        return (
            f"EvaluationCache(size={self.size}, "
            f"hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.1%})"
        )


class PerformanceTracker:
    """
    Track performance metrics for evaluators.

    Learns execution patterns over time:
    - Median execution time
    - Standard deviation
    - Success rate
    - Cost accumulation
    """

    def __init__(self, evaluator_id: str):
        """
        Initialize tracker.

        Args:
            evaluator_id: ID of evaluator being tracked
        """
        self.evaluator_id = evaluator_id
        self.execution_times = []
        self.successes = 0
        self.failures = 0
        self.total_cost = 0.0

    def record_evaluation(
        self,
        execution_time: float,
        success: bool,
        cost: float = 1.0
    ):
        """
        Record evaluation performance.

        Args:
            execution_time: Time taken (seconds)
            success: Whether evaluation succeeded
            cost: Computational cost
        """
        self.execution_times.append(execution_time)

        if success:
            self.successes += 1
        else:
            self.failures += 1

        self.total_cost += cost

    @property
    def total_calls(self) -> int:
        """Total number of evaluations."""
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        """Success rate."""
        total = self.total_calls
        return self.successes / total if total > 0 else 0.0

    @property
    def median_time(self) -> Optional[float]:
        """Median execution time."""
        if not self.execution_times:
            return None
        return float(np.median(self.execution_times))

    @property
    def std_time(self) -> Optional[float]:
        """Standard deviation of execution time."""
        if len(self.execution_times) < 2:
            return None
        return float(np.std(self.execution_times))

    def to_dict(self) -> Dict[str, Any]:
        """
        Export metrics as dictionary.

        Returns:
            Dictionary with performance metrics
        """
        return {
            'total_calls': self.total_calls,
            'successes': self.successes,
            'failures': self.failures,
            'success_rate': self.success_rate,
            'median_time': self.median_time,
            'std_time': self.std_time,
            'total_cost': self.total_cost
        }

    def __str__(self) -> str:
        """String representation."""
        return (
            f"PerformanceTracker(\n"
            f"  evaluator_id={self.evaluator_id},\n"
            f"  calls={self.total_calls},\n"
            f"  success_rate={self.success_rate:.1%},\n"
            f"  median_time={self.median_time:.3f}s,\n"
            f"  total_cost={self.total_cost:.1f}\n"
            f")"
        )
