"""
Bayesian optimizer family components.

Covers: Optuna (TPE, CMA-ES sampler), Bayesian Optimization with GP.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from .components import InitializationComponent, ProgressComponent, ResultComponent


@dataclass
class BayesianInitialization(InitializationComponent):
    """Initialization for Bayesian optimizers."""

    specification: Dict[str, Any]
    warm_start_trials: Optional[List[Dict[str, Any]]] = None
    n_initial_random: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "bayesian",
            "specification": self.specification,
            "warm_start_trials": self.warm_start_trials,
            "n_initial_random": self.n_initial_random,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BayesianInitialization':
        return cls(
            specification=data["specification"],
            warm_start_trials=data.get("warm_start_trials"),
            n_initial_random=data.get("n_initial_random", 10),
        )


@dataclass
class Trial:
    """Single trial record for Bayesian optimizer."""

    trial_number: int
    design: List[float]
    objective: float
    state: str  # "complete", "pruned", "failed"
    duration_seconds: Optional[float] = None


@dataclass
class BayesianProgress(ProgressComponent):
    """Progress data for Bayesian optimizers."""

    trials: List[Trial] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "bayesian",
            "trials": [
                {
                    "trial_number": t.trial_number,
                    "design": t.design,
                    "objective": t.objective,
                    "state": t.state,
                    "duration_seconds": t.duration_seconds,
                }
                for t in self.trials
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BayesianProgress':
        trials = [
            Trial(
                trial_number=t["trial_number"],
                design=t["design"],
                objective=t["objective"],
                state=t["state"],
                duration_seconds=t.get("duration_seconds"),
            )
            for t in data["trials"]
        ]
        return cls(trials=trials)

    def add_trial(
        self,
        trial_number: int,
        design: List[float],
        objective: float,
        state: str = "complete",
        duration_seconds: Optional[float] = None,
    ):
        """Add a trial record."""
        self.trials.append(
            Trial(
                trial_number=trial_number,
                design=design,
                objective=objective,
                state=state,
                duration_seconds=duration_seconds,
            )
        )


@dataclass
class BayesianResult(ResultComponent):
    """Detailed result for Bayesian optimizers."""

    termination_reason: str
    best_trial_number: int
    n_complete_trials: int
    n_pruned_trials: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": "bayesian",
            "termination_reason": self.termination_reason,
            "best_trial_number": self.best_trial_number,
            "n_complete_trials": self.n_complete_trials,
            "n_pruned_trials": self.n_pruned_trials,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BayesianResult':
        return cls(
            termination_reason=data["termination_reason"],
            best_trial_number=data["best_trial_number"],
            n_complete_trials=data["n_complete_trials"],
            n_pruned_trials=data["n_pruned_trials"],
        )
