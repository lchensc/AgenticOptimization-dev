"""
RecordingObjective - Core component for automatic evaluation capture.

The RecordingObjective wraps any evaluator function and automatically:
- Captures all (x, f(x)) pairs
- Logs to JSONL immediately (crash-safe)
- Uses cache to avoid redundant evaluations
- Tracks best solution found

This is to optimization what tensors are to autodiff.
"""

import json
import fcntl
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np

from paola.recording.cache import EvaluationCache, ArrayHasher


@dataclass
class EvaluationRecord:
    """Record of a single evaluation."""
    x: np.ndarray
    f: Optional[float]
    status: str  # "ok", "pending", "crash", "cached"
    timestamp: float
    eval_time: Optional[float] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = {
            "x": self.x.tolist(),
            "status": self.status,
            "timestamp": self.timestamp,
        }
        if self.f is not None:
            d["f"] = self.f
        if self.eval_time is not None:
            d["eval_time"] = self.eval_time
        if self.error is not None:
            d["error"] = self.error
        if self.error_type is not None:
            d["error_type"] = self.error_type
        if self.cached:
            d["cached"] = True
        return d


class RecordingObjective:
    """
    Wrapper that records all evaluations of an objective function.

    Key features:
    - Automatic logging to JSONL (crash-safe, incremental)
    - Cache integration for expensive evaluations
    - Best solution tracking
    - Warm-start support from parent nodes

    Usage:
        # Create recording objective
        f = RecordingObjective(
            evaluator=my_function,
            graph_id=42,
            node_id="n1",
            cache_dir=Path(".paola_foundry/cache/graph_42"),
        )

        # Use like normal function - all calls are recorded
        result = minimize(f, x0, method='SLSQP')

        # Get best solution
        best_x, best_f = f.get_best()

    The log file (evaluations.jsonl) captures every attempted evaluation,
    including crashes, enabling post-mortem analysis and problem reformulation.
    """

    def __init__(
        self,
        evaluator: Callable[[np.ndarray], float],
        graph_id: int,
        node_id: str,
        cache_dir: Path,
        problem_id: Optional[int] = None,
        goal: Optional[str] = None,
        parent_best_x: Optional[np.ndarray] = None,
        parent_node: Optional[str] = None,
        edge_type: Optional[str] = None,
        log_file: Optional[Path] = None,
        use_cache: bool = True,
        hasher: Optional[ArrayHasher] = None,
    ):
        """
        Initialize recording objective.

        Args:
            evaluator: The underlying objective function f(x) -> float
            graph_id: ID of the optimization graph
            node_id: ID of this node within the graph (e.g., "n1")
            cache_dir: Directory for cache storage
            problem_id: Optional problem ID for metadata
            goal: Optional goal description
            parent_best_x: Best x from parent node (for warm-start)
            parent_node: Parent node ID for edge tracking (v0.2.1)
            edge_type: Edge type to parent (warm_start, restart, etc.) (v0.2.1)
            log_file: Path to evaluation log (default: cache_dir/evaluations.jsonl)
            use_cache: Whether to use evaluation cache
            hasher: ArrayHasher for cache (default: tolerance=1e-10)
        """
        self.evaluator = evaluator
        self.graph_id = graph_id
        self.node_id = node_id
        self.problem_id = problem_id
        self.goal = goal
        self._parent_best_x = parent_best_x.copy() if parent_best_x is not None else None
        # Parent relationship for journal-based finalization (v0.2.1)
        self._parent_node = parent_node
        self._edge_type = edge_type

        # Cache setup
        self._hasher = hasher or ArrayHasher()
        self._use_cache = use_cache
        if use_cache:
            self._cache = EvaluationCache(
                cache_dir=cache_dir,
                hasher=self._hasher,
                load_existing=True,
            )
        else:
            self._cache = None

        # Log file for this node's evaluations
        self._log_file = log_file or (Path(cache_dir) / "evaluations.jsonl")
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        # In-memory records (subset - just this node's evaluations)
        self._evaluations: List[EvaluationRecord] = []

        # Best tracking
        self._best_x: Optional[np.ndarray] = None
        self._best_f: Optional[float] = None

        # Stats
        self._n_calls = 0
        self._n_cache_hits = 0
        self._total_eval_time = 0.0
        self._start_time = time.time()

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate objective at x with automatic recording.

        This method:
        1. Logs the attempt BEFORE calling evaluator (so crashes are recorded)
        2. Checks cache for existing result
        3. Calls evaluator if not cached
        4. Updates log with result or error
        5. Updates best solution if improved

        Args:
            x: Design point (numpy array)

        Returns:
            Objective value f(x)

        Raises:
            Any exception from the evaluator (after logging the crash)
        """
        self._n_calls += 1
        x = np.asarray(x)
        timestamp = time.time()

        # Check cache first
        if self._use_cache and self._cache is not None:
            cached_result = self._cache.get(x)
            if cached_result is not None:
                f, _ = cached_result
                self._n_cache_hits += 1

                # Record cache hit
                record = EvaluationRecord(
                    x=x.copy(),
                    f=f,
                    status="cached",
                    timestamp=timestamp,
                    cached=True,
                )
                self._evaluations.append(record)
                self._append_to_log(record)

                # Update best if needed
                self._update_best(x, f)

                return f

        # Log BEFORE calling evaluator (so we know what was attempted on crash)
        pending_record = EvaluationRecord(
            x=x.copy(),
            f=None,
            status="pending",
            timestamp=timestamp,
        )
        self._append_to_log(pending_record)

        # Call evaluator
        try:
            start = time.time()
            f = float(self.evaluator(x))
            eval_time = time.time() - start
            self._total_eval_time += eval_time

            # Update record to success
            record = EvaluationRecord(
                x=x.copy(),
                f=f,
                status="ok",
                timestamp=timestamp,
                eval_time=eval_time,
            )
            self._evaluations.append(record)
            self._update_log_status(pending_record, record)

            # Store in cache
            if self._use_cache and self._cache is not None:
                self._cache.put(x, f, eval_time=eval_time)

            # Update best
            self._update_best(x, f)

            return f

        except Exception as e:
            # Log the crash
            crash_record = EvaluationRecord(
                x=x.copy(),
                f=None,
                status="crash",
                timestamp=timestamp,
                error=str(e),
                error_type=type(e).__name__,
            )
            self._evaluations.append(crash_record)
            self._update_log_status(pending_record, crash_record)

            # Re-raise so optimizer knows eval failed
            raise

    def _append_to_log(self, record: EvaluationRecord) -> None:
        """Append record to JSONL log with file locking."""
        with open(self._log_file, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                entry = {
                    "graph_id": self.graph_id,
                    "node_id": self.node_id,
                    **record.to_dict(),
                }
                f.write(json.dumps(entry) + '\n')
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _update_log_status(
        self,
        pending: EvaluationRecord,
        final: EvaluationRecord,
    ) -> None:
        """
        Update the last pending entry in log to final status.

        For simplicity in MVP, we just append the final record.
        The log may have duplicate x entries (pending then ok/crash).
        Analysis tools should use the latest status for each x.
        """
        self._append_to_log(final)

    def _update_best(self, x: np.ndarray, f: float) -> None:
        """Update best solution if improved."""
        if self._best_f is None or f < self._best_f:
            self._best_x = x.copy()
            self._best_f = f

    def get_best(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        Get best solution found.

        Returns:
            (best_x, best_f) or (None, None) if no evaluations yet
        """
        return self._best_x, self._best_f

    def get_warm_start(self) -> Optional[np.ndarray]:
        """
        Get warm-start point (parent node's best_x).

        Returns:
            Parent's best_x if available, None otherwise
        """
        return self._parent_best_x.copy() if self._parent_best_x is not None else None

    @property
    def n_evaluations(self) -> int:
        """Number of evaluations (including cache hits)."""
        return len(self._evaluations)

    @property
    def n_calls(self) -> int:
        """Number of __call__ invocations."""
        return self._n_calls

    @property
    def n_cache_hits(self) -> int:
        """Number of cache hits."""
        return self._n_cache_hits

    @property
    def n_actual_evals(self) -> int:
        """Number of actual evaluator calls (excluding cache hits)."""
        return self._n_calls - self._n_cache_hits

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        return self._n_cache_hits / self._n_calls if self._n_calls > 0 else 0.0

    @property
    def total_eval_time(self) -> float:
        """Total time spent in evaluator (seconds)."""
        return self._total_eval_time

    @property
    def wall_time(self) -> float:
        """Wall time since creation (seconds)."""
        return time.time() - self._start_time

    def get_evaluations(self) -> List[Dict[str, Any]]:
        """Get all evaluation records as dicts."""
        return [r.to_dict() for r in self._evaluations]

    def get_successful_evaluations(self) -> List[Tuple[np.ndarray, float]]:
        """Get (x, f) pairs for successful evaluations."""
        return [
            (r.x, r.f)
            for r in self._evaluations
            if r.status in ("ok", "cached") and r.f is not None
        ]

    def get_crash_info(self) -> List[Dict[str, Any]]:
        """Get info about crashed evaluations."""
        return [
            {
                "x": r.x.tolist(),
                "error": r.error,
                "error_type": r.error_type,
                "timestamp": r.timestamp,
            }
            for r in self._evaluations
            if r.status == "crash"
        ]

    def summary(self) -> Dict[str, Any]:
        """
        Get summary of this recording objective.

        This is what checkpoint() returns to the agent.
        """
        best_x, best_f = self.get_best()

        # Count by status
        status_counts = {}
        for r in self._evaluations:
            status_counts[r.status] = status_counts.get(r.status, 0) + 1

        return {
            "graph_id": self.graph_id,
            "node_id": self.node_id,
            "problem_id": self.problem_id,
            "goal": self.goal,
            "best_x": best_x.tolist() if best_x is not None else None,
            "best_f": best_f,
            "n_evaluations": self.n_evaluations,
            "n_cache_hits": self.n_cache_hits,
            "n_actual_evals": self.n_actual_evals,
            "cache_hit_rate": self.cache_hit_rate,
            "total_eval_time": self.total_eval_time,
            "wall_time": self.wall_time,
            "status_counts": status_counts,
            "has_crashes": any(r.status == "crash" for r in self._evaluations),
        }
