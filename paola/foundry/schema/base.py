"""
Base schema classes for optimization sessions and runs.

Session = complete optimization task (may involve multiple runs)
Run = single optimizer execution within a session
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .components import InitializationComponent, ProgressComponent, ResultComponent


@dataclass
class PaolaDecision:
    """
    Record of Paola's strategic decision during optimization.

    Captures why Paola switched runs, changed strategy, etc.
    Important for learning and explainability.
    """

    timestamp: str
    decision_type: str  # "start_run", "switch_optimizer", "terminate"
    reasoning: str
    from_run: Optional[int]
    to_run: Optional[int]
    metrics_at_decision: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "decision_type": self.decision_type,
            "reasoning": self.reasoning,
            "from_run": self.from_run,
            "to_run": self.to_run,
            "metrics_at_decision": self.metrics_at_decision,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaolaDecision':
        return cls(
            timestamp=data["timestamp"],
            decision_type=data["decision_type"],
            reasoning=data["reasoning"],
            from_run=data.get("from_run"),
            to_run=data.get("to_run"),
            metrics_at_decision=data.get("metrics_at_decision", {}),
        )


@dataclass
class OptimizationRun:
    """
    Single optimizer execution within a session.

    Each run uses one optimizer from one family.
    Runs can warm-start from previous runs.
    """

    run_id: int
    optimizer: str  # Full spec: "scipy:SLSQP", "optuna:TPE"
    optimizer_family: str  # Family: "gradient", "bayesian", etc.
    warm_start_from: Optional[int]
    n_evaluations: int
    wall_time: float
    run_success: bool
    best_objective: float
    best_design: List[float]
    initialization: InitializationComponent
    progress: ProgressComponent
    result: ResultComponent

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "run_id": self.run_id,
            "optimizer": self.optimizer,
            "optimizer_family": self.optimizer_family,
            "warm_start_from": self.warm_start_from,
            "n_evaluations": self.n_evaluations,
            "wall_time": self.wall_time,
            "run_success": self.run_success,
            "best_objective": self.best_objective,
            "best_design": self.best_design,
            "initialization": self.initialization.to_dict(),
            "progress": self.progress.to_dict(),
            "result": self.result.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationRun':
        """Deserialize from dictionary using component registry."""
        from .registry import COMPONENT_REGISTRY

        family = data["optimizer_family"]

        init, progress, result = COMPONENT_REGISTRY.deserialize_components(
            family=family,
            init_data=data["initialization"],
            progress_data=data["progress"],
            result_data=data["result"],
        )

        return cls(
            run_id=data["run_id"],
            optimizer=data["optimizer"],
            optimizer_family=family,
            warm_start_from=data.get("warm_start_from"),
            n_evaluations=data["n_evaluations"],
            wall_time=data["wall_time"],
            run_success=data["run_success"],
            best_objective=data["best_objective"],
            best_design=data["best_design"],
            initialization=init,
            progress=progress,
            result=result,
        )


@dataclass
class SessionRecord:
    """
    Complete optimization session record.

    A Session represents Paola's complete effort to solve an optimization problem.
    It may involve multiple runs using different optimizers.
    """

    session_id: int
    problem_id: str
    created_at: str
    config: Dict[str, Any]
    runs: List[OptimizationRun]
    success: bool
    final_objective: float
    final_design: List[float]
    total_evaluations: int
    total_wall_time: float
    decisions: List[PaolaDecision] = field(default_factory=list)

    def get_run(self, run_id: int) -> Optional[OptimizationRun]:
        """Get run by ID."""
        for run in self.runs:
            if run.run_id == run_id:
                return run
        return None

    def get_best_run(self) -> Optional[OptimizationRun]:
        """Get run that found the best objective."""
        if not self.runs:
            return None
        return min(self.runs, key=lambda r: r.best_objective)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "session_id": self.session_id,
            "problem_id": self.problem_id,
            "created_at": self.created_at,
            "config": self.config,
            "runs": [r.to_dict() for r in self.runs],
            "success": self.success,
            "final_objective": self.final_objective,
            "final_design": self.final_design,
            "total_evaluations": self.total_evaluations,
            "total_wall_time": self.total_wall_time,
            "decisions": [d.to_dict() for d in self.decisions],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionRecord':
        """Deserialize from dictionary."""
        runs = [OptimizationRun.from_dict(r) for r in data["runs"]]
        decisions = [
            PaolaDecision.from_dict(d) for d in data.get("decisions", [])
        ]
        return cls(
            session_id=data["session_id"],
            problem_id=data["problem_id"],
            created_at=data["created_at"],
            config=data.get("config", {}),
            runs=runs,
            success=data["success"],
            final_objective=data["final_objective"],
            final_design=data["final_design"],
            total_evaluations=data["total_evaluations"],
            total_wall_time=data["total_wall_time"],
            decisions=decisions,
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'SessionRecord':
        """Deserialize from JSON string."""
        import json
        return cls.from_dict(json.loads(json_str))
