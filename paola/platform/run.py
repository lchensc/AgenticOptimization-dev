"""
Optimization run classes.

Two representations:
- Run: Active handle for tracking in-progress optimization
- RunRecord: Immutable storage record for completed runs
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List
import json
import numpy as np


@dataclass
class IterationRecord:
    """Single optimization iteration."""
    iteration: int
    design: List[float]
    objective: float
    gradient: Optional[List[float]] = None
    constraints: Optional[Dict[str, float]] = None
    timestamp: str = ""


@dataclass
class RunRecord:
    """
    Immutable optimization run record for storage.

    This is the serialized representation of a completed run.
    Used for persistence and querying.

    Replaces: storage.models.OptimizationRun
    """
    run_id: int
    problem_id: str
    problem_name: str
    algorithm: str
    objective_value: float
    success: bool
    n_evaluations: int
    timestamp: str  # ISO format
    duration: float  # seconds
    result_data: Dict[str, Any]  # Full scipy result + iterations
    ai_insights: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunRecord':
        """Deserialize from dictionary."""
        return cls(**data)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'RunRecord':
        """Deserialize from JSON."""
        return cls.from_dict(json.loads(json_str))


class Run:
    """
    Active optimization run handle.

    Provides methods for tracking progress during optimization.
    Auto-persists to storage on every update.

    This is the in-memory active object that tools interact with.

    Replaces: runs.active_run.OptimizationRun
    """

    def __init__(
        self,
        run_id: int,
        problem_id: str,
        problem_name: str,
        algorithm: str,
        storage: 'StorageBackend',
        description: str = ""
    ):
        """
        Initialize active run.

        Args:
            run_id: Unique run identifier
            problem_id: Problem being optimized
            problem_name: Human-readable problem name
            algorithm: Optimization algorithm
            storage: Storage backend for persistence
            description: Optional description
        """
        self.run_id = run_id
        self.problem_id = problem_id
        self.problem_name = problem_name
        self.algorithm = algorithm
        self.storage = storage
        self.description = description

        # Tracking state
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None
        self.iterations: List[IterationRecord] = []
        self.result: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}

        # AI insights cache
        self.ai_insights: Optional[Dict[str, Any]] = None

        # Finalization state
        self.finalized = False

    def record_iteration(
        self,
        design: np.ndarray,
        objective: float,
        gradient: Optional[np.ndarray] = None,
        constraints: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Record an optimization iteration.

        Auto-persists to storage after recording.

        Args:
            design: Design vector
            objective: Objective function value
            gradient: Optional gradient vector
            constraints: Optional constraint values
        """
        iteration_data = IterationRecord(
            iteration=len(self.iterations) + 1,
            design=design.tolist() if isinstance(design, np.ndarray) else list(design),
            objective=float(objective),
            gradient=gradient.tolist() if gradient is not None and isinstance(gradient, np.ndarray) else None,
            constraints=constraints,
            timestamp=datetime.now().isoformat()
        )

        self.iterations.append(iteration_data)

        # Auto-persist on every iteration
        self._persist()

    def finalize(
        self,
        result: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Finalize run with optimization result.

        Args:
            result: scipy.optimize.OptimizeResult
            metadata: Optional additional metadata
        """
        if self.finalized:
            raise RuntimeError(f"Run {self.run_id} already finalized")

        self.result = result
        self.end_time = datetime.now()
        self.finalized = True

        if metadata:
            self.metadata.update(metadata)

        # Final persist
        self._persist()

    def add_ai_insights(self, insights: Dict[str, Any]) -> None:
        """
        Cache AI analysis insights with run.

        Args:
            insights: AI analysis results from paola.analysis.ai_analyze()
        """
        self.ai_insights = insights
        self._persist()

    def get_current_best(self) -> Optional[Dict[str, Any]]:
        """
        Get current best objective and design.

        Returns:
            Dict with objective, design, iteration or None if no iterations
        """
        if not self.iterations:
            return None

        best_iter = min(self.iterations, key=lambda it: it.objective)
        return {
            "objective": best_iter.objective,
            "design": best_iter.design,
            "iteration": best_iter.iteration
        }

    def to_record(self) -> RunRecord:
        """
        Convert to immutable storage record.

        Returns:
            RunRecord for persistence
        """
        # Compute duration
        if self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        else:
            duration = (datetime.now() - self.start_time).total_seconds()

        # Extract result data
        if self.result:
            objective_value = float(self.result.fun)
            success = bool(self.result.success)
            n_evaluations = int(self.result.nfev)
            result_data = self._serialize_result(self.result)
        else:
            # In-progress run
            objective_value = self.iterations[-1].objective if self.iterations else float('inf')
            success = False
            n_evaluations = len(self.iterations)
            result_data = {
                "in_progress": True,
                "iterations": [asdict(it) for it in self.iterations[-5:]] if len(self.iterations) > 5 else [asdict(it) for it in self.iterations]
            }

        return RunRecord(
            run_id=self.run_id,
            problem_id=self.problem_id,
            problem_name=self.problem_name,
            algorithm=self.algorithm,
            objective_value=objective_value,
            success=success,
            n_evaluations=n_evaluations,
            timestamp=self.start_time.isoformat(),
            duration=duration,
            result_data=result_data,
            ai_insights=self.ai_insights
        )

    def _persist(self) -> None:
        """Persist current state to storage."""
        record = self.to_record()
        self.storage.save_run(record)

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        """
        Serialize scipy OptimizeResult to dict.

        Args:
            result: scipy.optimize.OptimizeResult

        Returns:
            Serialized result dictionary
        """
        return {
            "fun": float(result.fun),
            "x": result.x.tolist() if isinstance(result.x, np.ndarray) else list(result.x),
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "nit": int(getattr(result, 'nit', 0)),
            "njev": int(getattr(result, 'njev', 0)),
            "iterations": [asdict(it) for it in self.iterations],  # Store all iterations
        }

    def __repr__(self) -> str:
        """String representation."""
        status = "finalized" if self.finalized else "in-progress"
        return f"Run(id={self.run_id}, {self.algorithm} on {self.problem_name}, status={status})"
