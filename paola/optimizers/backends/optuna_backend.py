"""
Optuna optimization backend.

Wraps Optuna for black-box optimization with support for:
- Multiple samplers: TPE, CMA-ES, GP, NSGA-II, NSGA-III, QMC, Random, Grid
- Multiple pruners: Median, Hyperband, SuccessiveHalving, etc.
- Warm-starting via enqueue_trial
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
import logging

from ..base import OptimizerBackend
from ..result import OptimizationResult

logger = logging.getLogger(__name__)


class OptunaBackend(OptimizerBackend):
    """
    Optuna optimization backend.

    Supports black-box optimization with various samplers:
    - TPE: Tree-structured Parzen Estimator (default, good general-purpose)
    - CMA-ES: Covariance Matrix Adaptation (good for continuous optimization)
    - GP: Gaussian Process (good for expensive evaluations)
    - NSGA-II, NSGA-III: Multi-objective (via Optuna's built-in support)
    - QMC: Quasi-Monte Carlo (good for exploration)
    - Random: Random sampling (baseline)
    - Grid: Grid search (requires search_space)
    """

    SAMPLERS = ["TPE", "CMA-ES", "GP", "NSGA-II", "NSGA-III", "QMC", "Random", "Grid"]
    PRUNERS = ["Median", "Hyperband", "SuccessiveHalving", "Threshold", "Percentile", "Patient", "Nop"]

    # Known invalid options that users commonly try to pass incorrectly
    INVALID_SAMPLER_OPTIONS = {
        "bounds": "Bounds are defined at problem creation, not via sampler_options",
        "maximize": "Direction is determined by Optuna study, not sampler",
        "initial_step_size": "Use 'sigma0' for CMA-ES initial standard deviation",
    }

    @property
    def name(self) -> str:
        return "optuna"

    @property
    def family(self) -> str:
        return "bayesian"

    @property
    def supports_constraints(self) -> bool:
        return False  # Optuna handles constraints differently

    def is_available(self) -> bool:
        try:
            import optuna
            return True
        except ImportError:
            return False

    def get_methods(self) -> List[str]:
        return self.SAMPLERS

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Optuna",
            "samplers": self.SAMPLERS,
            "pruners": self.PRUNERS,
            "skill": "optuna",
        }

    def _create_sampler(
        self,
        sampler_name: str,
        seed: Optional[int],
        sampler_options: Dict[str, Any],
        source_trials: Optional[List] = None,
    ):
        """Create Optuna sampler with options."""
        import optuna

        opts = sampler_options.copy() if sampler_options else {}

        # Validate options and provide helpful error messages
        for invalid_opt, message in self.INVALID_SAMPLER_OPTIONS.items():
            if invalid_opt in opts:
                raise ValueError(f"Invalid sampler option '{invalid_opt}': {message}")

        if seed is not None:
            opts["seed"] = seed

        if sampler_name == "TPE":
            return optuna.samplers.TPESampler(**opts)
        elif sampler_name == "CMA-ES":
            # CMA-ES supports warm-starting via source_trials
            if source_trials:
                opts["source_trials"] = source_trials
            return optuna.samplers.CmaEsSampler(**opts)
        elif sampler_name == "GP":
            return optuna.samplers.GPSampler(**opts)
        elif sampler_name == "NSGA-II":
            return optuna.samplers.NSGAIISampler(**opts)
        elif sampler_name == "NSGA-III":
            return optuna.samplers.NSGAIIISampler(**opts)
        elif sampler_name == "QMC":
            return optuna.samplers.QMCSampler(**opts)
        elif sampler_name == "Random":
            return optuna.samplers.RandomSampler(**opts)
        elif sampler_name == "Grid":
            # Grid sampler requires search_space parameter
            if "search_space" not in opts:
                logger.warning("Grid sampler requires search_space, falling back to TPE")
                return optuna.samplers.TPESampler(seed=seed)
            return optuna.samplers.GridSampler(**opts)
        else:
            logger.warning(f"Unknown sampler '{sampler_name}', using TPE")
            return optuna.samplers.TPESampler(seed=seed)

    def _create_pruner(self, pruner_name: Optional[str], pruner_options: Dict[str, Any]):
        """Create Optuna pruner with options."""
        import optuna

        if not pruner_name or pruner_name == "Nop":
            return optuna.pruners.NopPruner()

        opts = pruner_options.copy() if pruner_options else {}

        if pruner_name == "Median":
            return optuna.pruners.MedianPruner(**opts)
        elif pruner_name == "Hyperband":
            return optuna.pruners.HyperbandPruner(**opts)
        elif pruner_name == "SuccessiveHalving":
            return optuna.pruners.SuccessiveHalvingPruner(**opts)
        elif pruner_name == "Threshold":
            return optuna.pruners.ThresholdPruner(**opts)
        elif pruner_name == "Percentile":
            return optuna.pruners.PercentilePruner(**opts)
        elif pruner_name == "Patient":
            wrapped = opts.pop("wrapped_pruner", None)
            if wrapped is None:
                wrapped = optuna.pruners.MedianPruner()
            return optuna.pruners.PatientPruner(wrapped_pruner=wrapped, **opts)
        else:
            logger.warning(f"Unknown pruner '{pruner_name}', using NopPruner")
            return optuna.pruners.NopPruner()

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: List[List[float]],
        x0: np.ndarray,
        config: Dict[str, Any],
        constraints: Optional[List[Dict[str, Any]]] = None,
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> OptimizationResult:
        """
        Run Optuna optimization.

        Args:
            objective: Function f(x) -> float to minimize
            bounds: Variable bounds [[lb, ub], ...]
            x0: Initial design point (can be enqueued as first trial)
            config: Optuna configuration (sampler, n_trials, etc.)
            constraints: Not supported by Optuna
            gradient: Not used by Optuna

        Returns:
            OptimizationResult with solution and statistics
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            return OptimizationResult.from_failure(
                message="Optuna not available. Install with: pip install optuna",
                x0=x0,
            )

        # Extract config options
        sampler_name = config.get("sampler", "TPE")
        n_trials = config.get("n_trials", config.get("max_iterations", 100))
        seed = config.get("seed")
        sampler_options = config.get("sampler_options", {})
        pruner_name = config.get("pruner")
        pruner_options = config.get("pruner_options", {})

        # Warm-start support
        source_trials = config.get("source_trials")  # For CMA-ES
        enqueue_points = config.get("enqueue_points")  # List of designs to try first

        # Create sampler and pruner
        try:
            sampler = self._create_sampler(sampler_name, seed, sampler_options, source_trials)
            pruner = self._create_pruner(pruner_name, pruner_options)
        except Exception as e:
            return OptimizationResult.from_failure(
                message=f"Failed to create sampler/pruner: {str(e)}",
                x0=x0,
            )

        # Tracking state
        n_evals = 0
        history = []
        best_x = x0.copy()
        best_f = float("inf")

        def optuna_objective(trial):
            nonlocal n_evals, best_x, best_f

            x = []
            for i, (lb, ub) in enumerate(bounds):
                x.append(trial.suggest_float(f"x{i}", lb, ub))
            x = np.array(x)

            n_evals += 1
            val = float(objective(x))

            # Record trial
            history.append({
                "iteration": n_evals,
                "trial": trial.number,
                "objective": val,
                "design": x.tolist(),
                "pruned": False,
            })

            if val < best_f:
                best_f = val
                best_x = x.copy()

            return val

        try:
            # Create study
            study = optuna.create_study(
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
            )

            # Enqueue x0 as first trial
            if x0 is not None and len(x0) == len(bounds):
                params = {f"x{i}": float(v) for i, v in enumerate(x0)}
                study.enqueue_trial(params)

            # Enqueue additional points for warm-starting
            if enqueue_points:
                for point in enqueue_points:
                    if isinstance(point, (list, np.ndarray)):
                        params = {f"x{i}": float(v) for i, v in enumerate(point)}
                    else:
                        params = point
                    study.enqueue_trial(params)

            # Run optimization
            study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)

            # Count pruned trials
            n_pruned = len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.PRUNED
            ])

            # Get best result
            if study.best_trial is not None:
                final_x = np.array([study.best_params[f"x{i}"] for i in range(len(bounds))])
                final_f = study.best_value
            else:
                final_x = best_x
                final_f = best_f

            # Build message with stats
            completed = len([
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ])
            msg = f"Completed {completed} trials"
            if n_pruned > 0:
                msg += f", {n_pruned} pruned"

            return OptimizationResult(
                success=True,
                message=msg,
                best_x=final_x,
                best_f=float(final_f),
                n_iterations=n_trials,
                n_function_evals=n_evals,
                n_gradient_evals=0,
                history=history,
                raw_result=study,
            )

        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}")
            return OptimizationResult.from_failure(
                message=f"Optuna optimization failed: {str(e)}",
                x0=x0,
                n_evals=n_evals,
                history=history,
            )
