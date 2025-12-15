"""
Active session and run management for in-progress optimizations.

ActiveSession: Handle for in-progress optimization session
ActiveRun: Handle for in-progress single optimizer execution
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

from .schema import (
    SessionRecord,
    OptimizationRun,
    PaolaDecision,
    InitializationComponent,
    ProgressComponent,
    ResultComponent,
    COMPONENT_REGISTRY,
)


@dataclass
class ActiveRun:
    """
    Handle for in-progress single optimizer execution.

    Collects iteration data during optimization and finalizes to OptimizationRun.
    """

    run_id: int
    optimizer: str
    optimizer_family: str
    initialization: InitializationComponent
    warm_start_from: Optional[int] = None
    start_time: datetime = field(default_factory=datetime.now)
    raw_iterations: List[Dict[str, Any]] = field(default_factory=list)
    _best_objective: float = field(default=float('inf'))
    _best_design: Optional[List[float]] = None

    def record_iteration(self, data: Dict[str, Any]):
        """
        Record an iteration/trial/generation.

        Args:
            data: Iteration data dict. Expected keys depend on family:
                - gradient: iteration, objective, design, gradient_norm, step_size
                - bayesian: trial_number, design, objective, state
                - population: generation, best_objective, best_design, mean_objective
                - cmaes: generation, best_objective, best_design, mean, sigma
        """
        self.raw_iterations.append(data)

        # Track best
        obj = data.get("objective") or data.get("best_objective")
        if obj is not None and obj < self._best_objective:
            self._best_objective = obj
            self._best_design = data.get("design") or data.get("best_design")

    def get_best(self) -> Dict[str, Any]:
        """Get current best objective and design."""
        return {
            "objective": self._best_objective,
            "design": self._best_design,
        }

    def finalize(
        self,
        progress: ProgressComponent,
        result: ResultComponent,
        best_objective: float,
        best_design: List[float],
        success: bool = True,
    ) -> OptimizationRun:
        """
        Finalize run and return immutable OptimizationRun.

        Args:
            progress: Family-specific progress component with iteration data
            result: Family-specific result component
            best_objective: Final best objective value
            best_design: Final best design vector
            success: Whether run succeeded

        Returns:
            Immutable OptimizationRun record
        """
        wall_time = (datetime.now() - self.start_time).total_seconds()

        return OptimizationRun(
            run_id=self.run_id,
            optimizer=self.optimizer,
            optimizer_family=self.optimizer_family,
            warm_start_from=self.warm_start_from,
            n_evaluations=len(self.raw_iterations),
            wall_time=wall_time,
            run_success=success,
            best_objective=best_objective,
            best_design=best_design,
            initialization=self.initialization,
            progress=progress,
            result=result,
        )


class ActiveSession:
    """
    Handle for in-progress optimization session.

    Manages runs as they are created and completed.
    Tracks Paola's strategic decisions.
    """

    def __init__(
        self,
        session_id: int,
        problem_id: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.session_id = session_id
        self.problem_id = problem_id
        self.config = config or {}
        self.runs: List[OptimizationRun] = []
        self.decisions: List[PaolaDecision] = []
        self.current_run: Optional[ActiveRun] = None
        self.start_time = datetime.now()

    def start_run(
        self,
        optimizer: str,
        initialization: InitializationComponent,
        warm_start_from: Optional[int] = None,
    ) -> ActiveRun:
        """
        Start a new optimization run.

        Args:
            optimizer: Optimizer spec (e.g., "scipy:SLSQP", "optuna:TPE")
            initialization: Family-specific initialization component
            warm_start_from: Run ID to warm-start from, or None

        Returns:
            ActiveRun handle for recording iterations
        """
        if self.current_run is not None:
            raise RuntimeError(
                f"Run {self.current_run.run_id} is still active. "
                "Complete it before starting a new run."
            )

        run_id = len(self.runs) + 1
        family = COMPONENT_REGISTRY.get_family(optimizer)

        self.current_run = ActiveRun(
            run_id=run_id,
            optimizer=optimizer,
            optimizer_family=family,
            initialization=initialization,
            warm_start_from=warm_start_from,
        )
        return self.current_run

    def complete_run(
        self,
        progress: ProgressComponent,
        result: ResultComponent,
        best_objective: float,
        best_design: List[float],
        success: bool = True,
    ) -> OptimizationRun:
        """
        Complete current run and add to session.

        Args:
            progress: Family-specific progress component
            result: Family-specific result component
            best_objective: Final best objective
            best_design: Final best design
            success: Whether run succeeded

        Returns:
            Completed OptimizationRun record
        """
        if self.current_run is None:
            raise RuntimeError("No active run to complete")

        run = self.current_run.finalize(
            progress=progress,
            result=result,
            best_objective=best_objective,
            best_design=best_design,
            success=success,
        )
        self.runs.append(run)
        self.current_run = None
        return run

    def record_decision(
        self,
        decision_type: str,
        reasoning: str,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Record Paola's strategic decision.

        Args:
            decision_type: Type of decision (start_run, switch_optimizer, terminate)
            reasoning: Paola's reasoning for the decision
            metrics: Metrics that informed the decision
        """
        decision = PaolaDecision(
            timestamp=datetime.now().isoformat(),
            decision_type=decision_type,
            reasoning=reasoning,
            from_run=len(self.runs) if self.runs else None,
            to_run=len(self.runs) + 1 if decision_type != "terminate" else None,
            metrics_at_decision=metrics or {},
        )
        self.decisions.append(decision)

    def get_best_run(self) -> Optional[OptimizationRun]:
        """Get run with best objective so far."""
        if not self.runs:
            return None
        return min(self.runs, key=lambda r: r.best_objective)

    def finalize(self, success: bool) -> SessionRecord:
        """
        Finalize session and return immutable SessionRecord.

        Args:
            success: Whether overall optimization was successful

        Returns:
            Immutable SessionRecord
        """
        if self.current_run is not None:
            raise RuntimeError(
                f"Run {self.current_run.run_id} is still active. "
                "Complete it before finalizing session."
            )

        # Compute overall metrics
        total_evals = sum(r.n_evaluations for r in self.runs)
        total_time = (datetime.now() - self.start_time).total_seconds()

        # Get best result
        best_run = self.get_best_run()
        if best_run:
            final_objective = best_run.best_objective
            final_design = best_run.best_design
        else:
            final_objective = float('inf')
            final_design = []

        return SessionRecord(
            session_id=self.session_id,
            problem_id=self.problem_id,
            created_at=self.start_time.isoformat(),
            config=self.config,
            runs=self.runs,
            success=success,
            final_objective=final_objective,
            final_design=final_design,
            total_evaluations=total_evals,
            total_wall_time=total_time,
            decisions=self.decisions,
        )
