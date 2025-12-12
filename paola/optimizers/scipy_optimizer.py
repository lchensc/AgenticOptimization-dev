"""
Scipy optimizer wrapper for the agentic optimization platform.

Wraps scipy.optimize algorithms (SLSQP, L-BFGS-B, COBYLA) to provide
iteration-level control for the agent.
"""

from typing import Optional, Dict, Any, Tuple, Callable
import numpy as np
from scipy.optimize import minimize, OptimizeResult
import copy

from paola.optimizers.base import BaseOptimizer, OptimizerState


class ScipyOptimizer(BaseOptimizer):
    """
    Wrapper for scipy.optimize algorithms with iteration-level control.

    Supported algorithms:
    - SLSQP: Sequential Least Squares Programming (gradient-based, constrained)
    - L-BFGS-B: Limited-memory BFGS with bounds (gradient-based, box constraints)
    - COBYLA: Constrained Optimization BY Linear Approximation (derivative-free)
    """

    # Default options for each algorithm
    DEFAULT_OPTIONS = {
        "SLSQP": {
            "ftol": 1e-6,
            "maxiter": 100,
            "disp": False,
        },
        "L-BFGS-B": {
            "ftol": 1e-6,
            "gtol": 1e-5,
            "maxiter": 100,
            "disp": False,
        },
        "COBYLA": {
            "rhobeg": 1.0,
            "tol": 1e-6,
            "maxiter": 1000,
            "disp": False,
        },
    }

    def __init__(
        self,
        problem_id: str,
        algorithm: str,
        bounds: Tuple[np.ndarray, np.ndarray],
        initial_design: Optional[np.ndarray] = None,
        options: Optional[Dict[str, Any]] = None,
        constraints: Optional[list] = None,
    ):
        """
        Initialize scipy optimizer.

        Args:
            problem_id: Unique problem identifier
            algorithm: One of "SLSQP", "L-BFGS-B", "COBYLA"
            bounds: (lower_bounds, upper_bounds) tuple
            initial_design: Starting design point
            options: Algorithm-specific options (merged with defaults)
            constraints: List of constraint dicts (scipy format)
        """
        # Validate algorithm
        if algorithm not in self.DEFAULT_OPTIONS:
            raise ValueError(
                f"Unknown algorithm: {algorithm}. "
                f"Supported: {list(self.DEFAULT_OPTIONS.keys())}"
            )

        # Merge options with defaults
        merged_options = self.DEFAULT_OPTIONS[algorithm].copy()
        if options:
            merged_options.update(options)

        super().__init__(problem_id, algorithm, bounds, initial_design, merged_options)

        self.constraints = constraints or []

        # For iteration-level control
        self._iteration_callback_data = []
        self._proposed_design = None
        self._waiting_for_evaluation = False

        # Initialize at starting point
        if initial_design is not None:
            self._proposed_design = initial_design.copy()
            self._waiting_for_evaluation = True

    def propose_design(self) -> np.ndarray:
        """
        Propose next design to evaluate.

        Returns:
            Design vector to evaluate

        Raises:
            StopIteration: If optimizer has converged
            RuntimeError: If called while waiting for update
        """
        if self.state.converged:
            raise StopIteration(
                f"Optimizer converged: {self.state.convergence_reason}"
            )

        if self._waiting_for_evaluation:
            # Return the already-proposed design
            return self._proposed_design.copy()

        # First iteration - return initial design
        if self._proposed_design is None:
            if self.initial_design is None:
                # Generate random initial design within bounds
                lower, upper = self.bounds
                self._proposed_design = lower + np.random.rand(len(lower)) * (upper - lower)
            else:
                self._proposed_design = self.initial_design.copy()
            self._waiting_for_evaluation = True
            return self._proposed_design.copy()

        # Subsequent iterations - compute gradient descent step
        # (Simple implementation for testing; real scipy integration would be more complex)
        if self.state.current_gradient is not None:
            # Gradient descent with adaptive step size
            gradient_norm = np.linalg.norm(self.state.current_gradient)

            # Normalize gradient and use adaptive step size based on function value
            if gradient_norm > 1e-10:
                direction = -self.state.current_gradient / gradient_norm

                # Adaptive step size: smaller when objective is small, larger when large
                # Also scale by 1/gradient_norm to avoid taking huge steps
                base_step = min(0.1 * abs(self.state.current_objective) / (gradient_norm + 1e-10), 1.0)
                base_step = max(base_step, 1e-4)  # Minimum step size

                step_size = base_step
                new_design = self.state.current_design + step_size * direction

                # Enforce bounds
                lower, upper = self.bounds
                new_design = np.clip(new_design, lower, upper)

                self._proposed_design = new_design
            else:
                # Gradient too small, consider converged
                raise StopIteration("Gradient norm too small")
        else:
            # If no gradient available, can't propose new design
            raise RuntimeError("No gradient available to propose new design")

        self._waiting_for_evaluation = True
        return self._proposed_design.copy()

    def update(
        self,
        design: np.ndarray,
        objective: float,
        gradient: Optional[np.ndarray] = None,
        constraints: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Update optimizer with evaluation results.

        For scipy optimizers, this doesn't actually "update" an internal state
        in the traditional sense. Instead, we store the evaluation and will
        run scipy's minimize on next propose_design() call with maxiter=1.

        Args:
            design: Evaluated design
            objective: Objective value
            gradient: Gradient vector (if available)
            constraints: Constraint values (if any)

        Returns:
            Update info dict
        """
        if not self._waiting_for_evaluation:
            raise RuntimeError("No design proposal pending - call propose_design() first")

        # Update state
        self.state.iteration += 1
        self.state.current_design = design.copy()
        self.state.current_objective = objective
        self.state.current_gradient = gradient.copy() if gradient is not None else None
        self.state.current_constraints = constraints.copy() if constraints else None

        # Update best
        improvement = self.state.best_objective - objective
        self._update_best(design, objective)

        # Record evaluation
        gradient_norm = np.linalg.norm(gradient) if gradient is not None else None
        self._record_evaluation(
            design, objective, gradient, constraints,
            metadata={"gradient_norm": gradient_norm}
        )

        # Check convergence
        converged = False
        reason = None

        if gradient is not None:
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm < self.options.get("gtol", 1e-5):
                converged = True
                reason = f"Gradient norm {gradient_norm:.2e} below tolerance"

        if self.state.iteration >= self.options.get("maxiter", 100):
            converged = True
            reason = f"Maximum iterations {self.options['maxiter']} reached"

        if abs(improvement) < self.options.get("ftol", 1e-6) and self.state.iteration > 1:
            converged = True
            reason = f"Objective improvement {abs(improvement):.2e} below tolerance"

        self.state.converged = converged
        self.state.convergence_reason = reason

        self._waiting_for_evaluation = False

        return {
            "converged": converged,
            "reason": reason,
            "improvement": improvement,
            "gradient_norm": gradient_norm,
            "step_size": None,  # Would need previous design to compute
            "iteration": self.state.iteration,
        }

    def run_to_completion(
        self,
        objective_func: Callable[[np.ndarray], float],
        gradient_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> OptimizeResult:
        """
        Run optimizer to completion (traditional scipy mode).

        This is provided for comparison/testing - the agent typically uses
        propose_design/update for iteration-level control.

        Args:
            objective_func: Function that evaluates objective
            gradient_func: Function that computes gradient (if available)

        Returns:
            scipy OptimizeResult
        """
        # Prepare bounds in scipy format
        bounds_scipy = list(zip(self.bounds[0], self.bounds[1]))

        # Prepare jac argument
        jac = gradient_func if gradient_func is not None else None
        if self.algorithm == "COBYLA":
            jac = None  # COBYLA doesn't use gradients

        # Run optimization
        result = minimize(
            fun=objective_func,
            x0=self.initial_design if self.initial_design is not None else
               (self.bounds[0] + self.bounds[1]) / 2,
            method=self.algorithm,
            jac=jac,
            bounds=bounds_scipy if self.algorithm != "COBYLA" else None,
            constraints=self.constraints if self.algorithm in ["SLSQP", "COBYLA"] else None,
            options=self.options,
        )

        # Update state with final result
        self.state.current_design = result.x.copy()
        self.state.current_objective = result.fun
        self.state.iteration = result.nit
        self.state.converged = result.success
        self.state.convergence_reason = result.message
        self._update_best(result.x, result.fun)

        return result

    def checkpoint(self) -> Dict[str, Any]:
        """Create checkpoint of current state."""
        return {
            "state": self.state.to_dict(),
            "history": [
                {
                    "iteration": h["iteration"],
                    "design": h["design"].tolist(),
                    "objective": h["objective"],
                    "gradient": h["gradient"].tolist() if h["gradient"] is not None else None,
                    "constraints": h["constraints"],
                    "is_best": h["is_best"],
                }
                for h in self.history
            ],
            "algorithm": self.algorithm,
            "options": self.options,
            "constraints": self.constraints,
            "bounds": [self.bounds[0].tolist(), self.bounds[1].tolist()],
            "proposed_design": self._proposed_design.tolist() if self._proposed_design is not None else None,
            "waiting_for_evaluation": self._waiting_for_evaluation,
        }

    def restore(self, checkpoint: Dict[str, Any]):
        """Restore optimizer from checkpoint."""
        self.state = OptimizerState.from_dict(checkpoint["state"])

        # Restore history
        self.history = [
            {
                "iteration": h["iteration"],
                "design": np.array(h["design"]),
                "objective": h["objective"],
                "gradient": np.array(h["gradient"]) if h["gradient"] is not None else None,
                "constraints": h["constraints"],
                "is_best": h["is_best"],
            }
            for h in checkpoint["history"]
        ]

        self.algorithm = checkpoint["algorithm"]
        self.options = checkpoint["options"]
        self.constraints = checkpoint["constraints"]
        self.bounds = (np.array(checkpoint["bounds"][0]), np.array(checkpoint["bounds"][1]))
        self._proposed_design = np.array(checkpoint["proposed_design"]) if checkpoint["proposed_design"] else None
        self._waiting_for_evaluation = checkpoint["waiting_for_evaluation"]

    def restart_from_design(
        self,
        design: np.ndarray,
        objective: float,
        gradient: Optional[np.ndarray] = None,
        new_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Restart optimizer from specified design.

        Resets optimizer state but preserves evaluation history.
        Updates options if provided.

        Args:
            design: Design to restart from
            objective: Objective at restart design
            gradient: Gradient at restart design
            new_options: New algorithm options (merged with existing)
        """
        # Update options if provided
        if new_options:
            self.options.update(new_options)

        # Reset state but keep history
        old_iteration = self.state.iteration
        self.state = OptimizerState(
            current_design=design.copy(),
            current_objective=objective,
            current_gradient=gradient.copy() if gradient is not None else None,
            best_design=design.copy(),
            best_objective=objective,
            iteration=old_iteration,  # Keep iteration count
            converged=False,
            convergence_reason=None,
        )

        # Set as proposed design
        self._proposed_design = design.copy()
        self._waiting_for_evaluation = False

        # Record restart in history
        self._record_evaluation(
            design, objective, gradient,
            metadata={
                "event": "restart",
                "gradient_norm": np.linalg.norm(gradient) if gradient is not None else None,
            }
        )


# Factory function
def create_scipy_optimizer(
    problem_id: str,
    algorithm: str,
    bounds: Tuple[np.ndarray, np.ndarray],
    initial_design: Optional[np.ndarray] = None,
    options: Optional[Dict[str, Any]] = None,
    constraints: Optional[list] = None,
) -> ScipyOptimizer:
    """
    Create scipy optimizer instance.

    Args:
        problem_id: Unique problem identifier
        algorithm: One of "SLSQP", "L-BFGS-B", "COBYLA"
        bounds: (lower_bounds, upper_bounds) tuple
        initial_design: Starting design point
        options: Algorithm-specific options
        constraints: List of constraint dicts

    Returns:
        ScipyOptimizer instance
    """
    return ScipyOptimizer(
        problem_id=problem_id,
        algorithm=algorithm,
        bounds=bounds,
        initial_design=initial_design,
        options=options,
        constraints=constraints,
    )
