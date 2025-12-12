"""
Active optimization run that tracks and persists optimization progress.

This is an active object managed by the agent, not a passive data model.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import numpy as np

from ..storage import StorageBackend, OptimizationRun as RunModel


class OptimizationRun:
    """
    Active optimization run managed by agent.

    Auto-persists to storage as optimization progresses.
    Agent creates this explicitly via start_optimization_run tool.
    """

    def __init__(
        self,
        run_id: int,
        problem_id: str,
        problem_name: str,
        algorithm: str,
        storage: StorageBackend,
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
        self.iterations: List[Dict[str, Any]] = []
        self.result: Optional[Any] = None  # scipy OptimizeResult
        self.metadata: Dict[str, Any] = {}

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

        Args:
            design: Design vector
            objective: Objective function value
            gradient: Optional gradient vector
            constraints: Optional constraint values
        """
        iteration_data = {
            "iteration": len(self.iterations) + 1,
            "design": design.tolist() if isinstance(design, np.ndarray) else list(design),
            "objective": float(objective),
            "gradient": gradient.tolist() if gradient is not None and isinstance(gradient, np.ndarray) else None,
            "constraints": constraints,
            "timestamp": datetime.now().isoformat()
        }

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

    def _persist(self) -> None:
        """Persist current state to storage."""
        model = self._to_model()
        self.storage.save_run(model)

    def _to_model(self) -> RunModel:
        """Convert to storage model."""
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
            objective_value = self.iterations[-1]["objective"] if self.iterations else float('inf')
            success = False
            n_evaluations = len(self.iterations)
            result_data = {
                "in_progress": True,
                "iterations": self.iterations[-5:] if len(self.iterations) > 5 else self.iterations
            }

        return RunModel(
            run_id=self.run_id,
            problem_id=self.problem_id,
            problem_name=self.problem_name,
            algorithm=self.algorithm,
            objective_value=objective_value,
            success=success,
            n_evaluations=n_evaluations,
            timestamp=self.start_time.isoformat(),
            duration=duration,
            result_data=result_data
        )

    def _serialize_result(self, result: Any) -> Dict[str, Any]:
        """Serialize scipy OptimizeResult to dict."""
        return {
            "fun": float(result.fun),
            "x": result.x.tolist() if isinstance(result.x, np.ndarray) else list(result.x),
            "success": bool(result.success),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "nit": int(getattr(result, 'nit', 0)),
            "njev": int(getattr(result, 'njev', 0)),
            "iterations": self.iterations,  # Store all iterations for complete convergence history
        }

    def get_current_best(self) -> Optional[Dict[str, Any]]:
        """Get current best objective and design."""
        if not self.iterations:
            return None

        best_iter = min(self.iterations, key=lambda it: it["objective"])
        return {
            "objective": best_iter["objective"],
            "design": best_iter["design"],
            "iteration": best_iter["iteration"]
        }

    def __repr__(self) -> str:
        status = "finalized" if self.finalized else "in-progress"
        return f"OptimizationRun(id={self.run_id}, {self.algorithm} on {self.problem_name}, status={status})"
