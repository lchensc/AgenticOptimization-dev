"""Storage callback that writes optimization events to persistent storage."""

from datetime import datetime
from typing import Any, Dict
import numpy as np

from ..callbacks import AgentEvent, EventType
from ..storage import StorageBackend, OptimizationRun, Problem


class StorageCallback:
    """
    Callback that writes optimization events to storage.
    This is the ONLY component that writes to storage.
    """

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    def __call__(self, event: AgentEvent):
        """Handle events and persist to storage."""
        if event.event_type == EventType.OPTIMIZATION_COMPLETE:
            self._save_run(event.data)
        elif event.event_type == EventType.PROBLEM_CREATED:
            self._save_problem(event.data)

    def _save_run(self, data: Dict[str, Any]):
        """Save optimization run to storage."""
        run_id = self.storage.get_next_run_id()

        result = data["result"]

        # Get problem name from storage if available
        problem = self.storage.load_problem(data["problem_id"])
        problem_name = problem.name if problem else data.get("problem_name", "Unknown")

        run = OptimizationRun(
            run_id=run_id,
            problem_id=data["problem_id"],
            problem_name=problem_name,
            algorithm=data["algorithm"],
            objective_value=float(result.fun),
            success=bool(result.success),
            n_evaluations=int(result.nfev),
            timestamp=data["timestamp"].isoformat(),
            duration=data["duration"],
            result_data=self._serialize_result(result)
        )

        self.storage.save_run(run)

    def _save_problem(self, data: Dict[str, Any]):
        """Save problem metadata to storage."""
        problem = Problem(
            problem_id=data["problem_id"],
            name=data["name"],
            dimensions=data["dimensions"],
            problem_type=data.get("problem_type", "unconstrained"),
            created_at=datetime.now().isoformat(),
            metadata=data.get("metadata", {})
        )

        self.storage.save_problem(problem)

    def _serialize_result(self, result) -> Dict[str, Any]:
        """Convert scipy OptimizeResult to dict."""
        return {
            "fun": float(result.fun),
            "x": result.x.tolist() if isinstance(result.x, np.ndarray) else list(result.x),
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "nit": int(getattr(result, 'nit', 0)),
            "njev": int(getattr(result, 'njev', 0)),
        }
