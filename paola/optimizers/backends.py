"""
Optimizer backends for run-to-completion optimization.

These backends wrap optimization libraries (SciPy, IPOPT, Optuna) and provide
a simple interface for running optimizations. The LLM agent decides which
backend and configuration to use based on its reasoning.

Architecture:
- Each backend implements `optimize()` which runs to completion
- The `run_optimization` tool routes to the appropriate backend
- Configuration is passed from the LLM (not hardcoded)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Standardized result from any optimizer backend."""

    success: bool
    message: str
    final_design: np.ndarray
    final_objective: float
    n_iterations: int
    n_function_evals: int
    n_gradient_evals: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    raw_result: Any = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for tool response."""
        return {
            "success": self.success,
            "message": self.message,
            "final_design": self.final_design.tolist() if isinstance(self.final_design, np.ndarray) else list(self.final_design),
            "final_objective": float(self.final_objective),
            "n_iterations": self.n_iterations,
            "n_function_evals": self.n_function_evals,
            "n_gradient_evals": self.n_gradient_evals,
            "history": self.history[-20:] if len(self.history) > 20 else self.history,
        }


class OptimizerBackend(ABC):
    """Abstract base class for optimizer backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name (e.g., 'scipy', 'ipopt', 'optuna')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is installed."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get backend capabilities."""
        pass

    @abstractmethod
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: List[List[float]],
        x0: np.ndarray,
        config: Dict[str, Any],
        constraints: Optional[List[Dict[str, Any]]] = None,
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> OptimizationResult:
        """Execute optimization to completion."""
        pass


class SciPyBackend(OptimizerBackend):
    """SciPy optimization backend."""

    METHODS = {
        "SLSQP": {"gradient": True, "constraints": True, "bounds": True},
        "L-BFGS-B": {"gradient": True, "constraints": False, "bounds": True},
        "trust-constr": {"gradient": True, "constraints": True, "bounds": True},
        "COBYLA": {"gradient": False, "constraints": True, "bounds": False},
        "Nelder-Mead": {"gradient": False, "constraints": False, "bounds": False},
        "Powell": {"gradient": False, "constraints": False, "bounds": True},
        "BFGS": {"gradient": True, "constraints": False, "bounds": False},
        "CG": {"gradient": True, "constraints": False, "bounds": False},
    }

    @property
    def name(self) -> str:
        return "scipy"

    def is_available(self) -> bool:
        try:
            import scipy.optimize
            return True
        except ImportError:
            return False

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "SciPy",
            "type": "general",
            "methods": list(self.METHODS.keys()),
            "supports_constraints": True,
            "supports_bounds": True,
            "requires_gradients": False,
            "key_options": ["maxiter", "ftol", "gtol", "disp"],
        }

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: List[List[float]],
        x0: np.ndarray,
        config: Dict[str, Any],
        constraints: Optional[List[Dict[str, Any]]] = None,
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> OptimizationResult:
        from scipy.optimize import minimize

        method = config.get("method", "SLSQP")
        # Filter out non-option parameters
        excluded_keys = ["method", "max_iterations"]
        options = {k: v for k, v in config.items() if k not in excluded_keys}

        # Ensure maxiter is set (convert max_iterations to maxiter)
        if "maxiter" not in options:
            options["maxiter"] = config.get("max_iterations", 100)

        # Track evaluations
        n_evals = [0]
        n_grad_evals = [0]
        history = []

        def wrapped_objective(x):
            n_evals[0] += 1
            val = objective(x)
            history.append({
                "iteration": n_evals[0],
                "objective": float(val),
                "design": x.tolist() if hasattr(x, 'tolist') else list(x),
            })
            return val

        def wrapped_gradient(x):
            n_grad_evals[0] += 1
            return gradient(x)

        # Prepare bounds
        scipy_bounds = [(b[0], b[1]) for b in bounds] if bounds else None

        # Prepare gradient
        jac = wrapped_gradient if gradient and self.METHODS.get(method, {}).get("gradient", False) else None

        # Prepare constraints
        scipy_constraints = None
        if constraints and self.METHODS.get(method, {}).get("constraints", False):
            scipy_constraints = constraints

        try:
            result = minimize(
                fun=wrapped_objective,
                x0=x0,
                method=method,
                jac=jac,
                bounds=scipy_bounds if self.METHODS.get(method, {}).get("bounds", False) else None,
                constraints=scipy_constraints,
                options=options,
            )

            return OptimizationResult(
                success=result.success,
                message=str(result.message) if hasattr(result, 'message') else "completed",
                final_design=result.x,
                final_objective=float(result.fun),
                n_iterations=int(result.nit) if hasattr(result, 'nit') else n_evals[0],
                n_function_evals=n_evals[0],
                n_gradient_evals=n_grad_evals[0],
                history=history,
                raw_result=result,
            )
        except Exception as e:
            return OptimizationResult(
                success=False,
                message=f"Optimization failed: {str(e)}",
                final_design=x0,
                final_objective=float('inf'),
                n_iterations=0,
                n_function_evals=n_evals[0],
                n_gradient_evals=n_grad_evals[0],
                history=history,
            )


class IPOPTBackend(OptimizerBackend):
    """IPOPT interior-point optimizer backend."""

    @property
    def name(self) -> str:
        return "ipopt"

    def is_available(self) -> bool:
        try:
            import cyipopt
            return True
        except ImportError:
            return False

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "IPOPT",
            "type": "interior-point",
            "supports_constraints": True,
            "supports_bounds": True,
            "requires_gradients": True,
            "key_options": [
                "tol", "max_iter", "mu_strategy", "mu_init",
                "linear_solver", "nlp_scaling_method", "print_level"
            ],
        }

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: List[List[float]],
        x0: np.ndarray,
        config: Dict[str, Any],
        constraints: Optional[List[Dict[str, Any]]] = None,
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> OptimizationResult:
        try:
            from cyipopt import minimize_ipopt
        except ImportError:
            return OptimizationResult(
                success=False,
                message="IPOPT not available. Install with: pip install cyipopt",
                final_design=x0,
                final_objective=float('inf'),
                n_iterations=0,
                n_function_evals=0,
            )

        # Track evaluations
        n_evals = [0]
        n_grad_evals = [0]
        history = []

        def wrapped_objective(x):
            n_evals[0] += 1
            val = objective(x)
            history.append({
                "iteration": n_evals[0],
                "objective": float(val),
                "design": x.tolist() if hasattr(x, 'tolist') else list(x),
            })
            return val

        def wrapped_gradient(x):
            if gradient:
                n_grad_evals[0] += 1
                return gradient(x)
            return None

        # Prepare bounds
        ipopt_bounds = [(b[0], b[1]) for b in bounds]

        # Prepare IPOPT options
        options = {}
        option_mapping = {
            "tol": "tol",
            "max_iter": "max_iter",
            "maxiter": "max_iter",
            "mu_strategy": "mu_strategy",
            "mu_init": "mu_init",
            "linear_solver": "linear_solver",
            "nlp_scaling_method": "nlp_scaling_method",
            "print_level": "print_level",
        }

        for key, ipopt_key in option_mapping.items():
            if key in config:
                options[ipopt_key] = config[key]

        # Set defaults
        if "max_iter" not in options:
            options["max_iter"] = config.get("max_iterations", 100)
        if "print_level" not in options:
            options["print_level"] = 0

        try:
            result = minimize_ipopt(
                fun=wrapped_objective,
                x0=x0,
                jac=wrapped_gradient if gradient else None,
                bounds=ipopt_bounds,
                constraints=constraints,
                options=options,
            )

            return OptimizationResult(
                success=result.success,
                message=str(result.message) if hasattr(result, 'message') else "completed",
                final_design=result.x,
                final_objective=float(result.fun),
                n_iterations=int(result.nit) if hasattr(result, 'nit') else n_evals[0],
                n_function_evals=n_evals[0],
                n_gradient_evals=n_grad_evals[0],
                history=history,
                raw_result=result,
            )
        except Exception as e:
            return OptimizationResult(
                success=False,
                message=f"IPOPT optimization failed: {str(e)}",
                final_design=x0,
                final_objective=float('inf'),
                n_iterations=0,
                n_function_evals=n_evals[0],
                n_gradient_evals=n_grad_evals[0],
                history=history,
            )


class OptunaBackend(OptimizerBackend):
    """Optuna Bayesian optimization backend."""

    SAMPLERS = ["TPE", "CMA-ES", "Random", "Grid"]

    @property
    def name(self) -> str:
        return "optuna"

    def is_available(self) -> bool:
        try:
            import optuna
            return True
        except ImportError:
            return False

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "Optuna",
            "type": "bayesian",
            "samplers": self.SAMPLERS,
            "supports_constraints": False,
            "supports_bounds": True,
            "requires_gradients": False,
            "key_options": ["n_trials", "sampler", "seed"],
        }

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        bounds: List[List[float]],
        x0: np.ndarray,
        config: Dict[str, Any],
        constraints: Optional[List[Dict[str, Any]]] = None,
        gradient: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> OptimizationResult:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            return OptimizationResult(
                success=False,
                message="Optuna not available. Install with: pip install optuna",
                final_design=x0,
                final_objective=float('inf'),
                n_iterations=0,
                n_function_evals=0,
            )

        sampler_name = config.get("sampler", "TPE")
        n_trials = config.get("n_trials", config.get("max_iterations", 100))
        seed = config.get("seed")

        # Create sampler
        if sampler_name == "TPE":
            sampler = optuna.samplers.TPESampler(seed=seed)
        elif sampler_name == "CMA-ES":
            sampler = optuna.samplers.CmaEsSampler(seed=seed)
        elif sampler_name == "Random":
            sampler = optuna.samplers.RandomSampler(seed=seed)
        else:
            sampler = optuna.samplers.TPESampler(seed=seed)

        # Track evaluations
        n_evals = [0]
        history = []
        best_x = [x0.copy()]
        best_f = [float('inf')]

        def optuna_objective(trial):
            x = []
            for i, (lb, ub) in enumerate(bounds):
                x.append(trial.suggest_float(f"x{i}", lb, ub))
            x = np.array(x)

            n_evals[0] += 1
            val = objective(x)

            # Record trial with design vector for proper run tracking
            # Note: Optuna trials are independent samples, not trajectory steps
            history.append({
                "iteration": n_evals[0],
                "trial": n_evals[0],
                "objective": float(val),
                "design": x.tolist(),  # Include design for run recording
            })

            if val < best_f[0]:
                best_f[0] = val
                best_x[0] = x.copy()

            return val

        try:
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)

            # Get best result
            final_x = np.array([study.best_params[f"x{i}"] for i in range(len(bounds))])
            final_f = study.best_value

            return OptimizationResult(
                success=True,
                message=f"Completed {n_trials} trials",
                final_design=final_x,
                final_objective=float(final_f),
                n_iterations=n_trials,
                n_function_evals=n_evals[0],
                n_gradient_evals=0,
                history=history,
                raw_result=study,
            )
        except Exception as e:
            return OptimizationResult(
                success=False,
                message=f"Optuna optimization failed: {str(e)}",
                final_design=best_x[0],
                final_objective=best_f[0],
                n_iterations=n_evals[0],
                n_function_evals=n_evals[0],
                n_gradient_evals=0,
                history=history,
            )


# Backend registry
_BACKENDS: Dict[str, OptimizerBackend] = {}


def _initialize_backends():
    """Initialize and register all backends."""
    global _BACKENDS
    _BACKENDS = {
        "scipy": SciPyBackend(),
        "ipopt": IPOPTBackend(),
        "optuna": OptunaBackend(),
    }


def get_backend(name: str) -> Optional[OptimizerBackend]:
    """
    Get backend by name.

    Args:
        name: Backend name, optionally with method (e.g., "scipy:SLSQP", "optuna:TPE")

    Returns:
        Backend instance or None if not found
    """
    if not _BACKENDS:
        _initialize_backends()

    # Handle "backend:method" format
    backend_name = name.split(":")[0] if ":" in name else name
    return _BACKENDS.get(backend_name)


def list_backends() -> Dict[str, Dict[str, Any]]:
    """List all backends with their availability and capabilities."""
    if not _BACKENDS:
        _initialize_backends()

    result = {}
    for name, backend in _BACKENDS.items():
        info = backend.get_info()
        info["available"] = backend.is_available()
        result[name] = info
    return result


def get_available_backends() -> List[str]:
    """Get list of available (installed) backend names."""
    if not _BACKENDS:
        _initialize_backends()

    return [name for name, backend in _BACKENDS.items() if backend.is_available()]
