"""
Optimization gate for agent observation and control.

Provides iteration-level interception of optimizer function calls,
allowing the agent to observe progress and control execution.

Two modes:
- Blocking (engineering): Pauses at each evaluation, waits for agent decision
- Non-blocking (analytical): Logs iterations, runs to completion
"""

from typing import Optional, Dict, Any, Callable
import numpy as np
import threading
from enum import Enum
import time
from datetime import datetime


class GateAction(Enum):
    """Actions the agent can take at the gate."""
    CONTINUE = "continue"  # Let optimizer proceed normally
    STOP = "stop"          # Stop optimization (converged or give up)
    RESTART = "restart"    # Restart with new settings


class GateSignal(Exception):
    """Base exception for gate control signals."""
    pass


class StopOptimizationSignal(GateSignal):
    """Signal to stop optimization."""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class RestartOptimizationSignal(GateSignal):
    """Signal to restart optimization with new settings."""
    def __init__(self, restart_from: np.ndarray, new_options: Dict[str, Any], reason: str):
        self.restart_from = restart_from
        self.new_options = new_options
        self.reason = reason
        super().__init__(reason)


class OptimizationGate:
    """
    Gate that intercepts optimizer function calls for agent control.

    Wraps objective/gradient functions to pause at each iteration,
    emit events, and wait for agent decisions.

    Two modes:
    - blocking=True (engineering): Pauses and waits for agent after each call
    - blocking=False (analytical): Logs calls but returns immediately

    Usage:
        gate = OptimizationGate(
            problem_id="rosenbrock",
            blocking=False,  # Analytical mode
            callback_manager=callback_mgr
        )

        result = scipy.optimize.minimize(
            fun=gate.wrap_objective(problem),
            jac=gate.wrap_gradient(problem),
            x0=x0,
            method='SLSQP'
        )

        # Agent reviews trajectory
        history = gate.get_history()
    """

    def __init__(
        self,
        problem_id: str,
        blocking: bool = False,
        callback_manager: Optional[Any] = None,
        timeout: float = 300.0,  # 5 minutes default timeout for blocking
    ):
        """
        Initialize optimization gate.

        Args:
            problem_id: Problem identifier
            blocking: If True, pause and wait for agent at each iteration
            callback_manager: Callback manager for emitting events
            timeout: Timeout in seconds for blocking mode (prevents deadlock)
        """
        self.problem_id = problem_id
        self.blocking = blocking
        self.callback_manager = callback_manager
        self.timeout = timeout

        # State
        self.iteration = 0
        self.history: list[Dict[str, Any]] = []
        self.current_action = GateAction.CONTINUE
        self.restart_settings: Optional[Dict[str, Any]] = None
        self.stop_reason: Optional[str] = None

        # Threading for blocking mode
        self._continue_event = threading.Event()
        self._continue_event.set()  # Initially allow continuation

        # Statistics
        self.total_wait_time = 0.0
        self.n_pauses = 0

    def wrap_objective(self, objective_func: Callable) -> Callable:
        """
        Wrap objective function with gate interception.

        Returns a function that scipy can call, which internally:
        1. Evaluates objective (with caching)
        2. Emits event for agent observation
        3. Waits for agent decision (if blocking)
        4. Returns objective or raises signal

        Args:
            objective_func: Original objective function

        Returns:
            Wrapped objective function
        """
        def observable_objective(x: np.ndarray) -> float:
            """
            Wrapped objective that allows agent observation.

            This is what scipy calls - it thinks it's just evaluating f(x).
            Actually, the agent can observe and control at this point.
            """
            # Evaluate objective
            obj_value = objective_func(x)

            # Record iteration
            iter_data = {
                'iteration': self.iteration,
                'design': x.copy(),
                'objective': obj_value,
                'timestamp': datetime.now().isoformat(),
                'call_type': 'objective',
            }
            self.history.append(iter_data)

            # Emit event for agent observation
            if self.callback_manager:
                from ..callbacks import EventType, create_event
                event = create_event(
                    EventType.ITERATION_COMPLETE,
                    iteration=self.iteration,
                    data={
                        'design': x.tolist(),
                        'objective': float(obj_value),
                        'problem_id': self.problem_id,
                        'gate_mode': 'blocking' if self.blocking else 'non-blocking',
                    }
                )
                self.callback_manager.emit(event)

            self.iteration += 1

            # Blocking mode: wait for agent decision
            if self.blocking:
                wait_start = time.time()
                self.n_pauses += 1

                # Wait for agent to call continue/stop/restart
                if not self._continue_event.wait(timeout=self.timeout):
                    raise TimeoutError(
                        f"Gate timeout after {self.timeout}s waiting for agent decision"
                    )

                wait_time = time.time() - wait_start
                self.total_wait_time += wait_time

                # Check action set by agent
                if self.current_action == GateAction.STOP:
                    raise StopOptimizationSignal(self.stop_reason)

                elif self.current_action == GateAction.RESTART:
                    raise RestartOptimizationSignal(
                        restart_from=self.restart_settings['restart_from'],
                        new_options=self.restart_settings['new_options'],
                        reason=self.restart_settings['reason']
                    )

                # Reset for next iteration
                self._continue_event.clear()

            # Return objective to scipy
            return obj_value

        return observable_objective

    def wrap_gradient(self, gradient_func: Callable) -> Callable:
        """
        Wrap gradient function with gate interception.

        Similar to wrap_objective but for gradient calls.
        """
        def observable_gradient(x: np.ndarray) -> np.ndarray:
            """Wrapped gradient that allows agent observation."""
            # Evaluate gradient
            grad_value = gradient_func(x)

            # Record gradient call
            grad_norm = np.linalg.norm(grad_value)
            grad_data = {
                'iteration': self.iteration,
                'design': x.copy(),
                'gradient': grad_value.copy(),
                'gradient_norm': grad_norm,
                'timestamp': datetime.now().isoformat(),
                'call_type': 'gradient',
            }

            # Update last history entry if it's for same design
            if (self.history and
                np.allclose(self.history[-1]['design'], x, rtol=1e-9)):
                self.history[-1].update({
                    'gradient': grad_value.copy(),
                    'gradient_norm': grad_norm,
                })
            else:
                self.history.append(grad_data)

            # Emit gradient event
            if self.callback_manager:
                from ..callbacks import EventType, create_event
                event = create_event(
                    EventType.TOOL_RESULT,
                    iteration=self.iteration,
                    data={
                        'tool': 'compute_gradient',
                        'gradient_norm': float(grad_norm),
                        'problem_id': self.problem_id,
                    }
                )
                self.callback_manager.emit(event)

            # Note: gradient calls typically don't block separately
            # The objective call already handled blocking

            return grad_value

        return observable_gradient

    def agent_continue(self):
        """
        Agent decision: continue optimization.

        Called by agent tool to signal optimizer should proceed.
        Only relevant in blocking mode.
        """
        self.current_action = GateAction.CONTINUE
        self._continue_event.set()

    def agent_stop(self, reason: str):
        """
        Agent decision: stop optimization.

        Args:
            reason: Why agent decided to stop
        """
        self.current_action = GateAction.STOP
        self.stop_reason = reason
        self._continue_event.set()

    def agent_restart(
        self,
        restart_from: np.ndarray,
        new_options: Dict[str, Any],
        reason: str
    ):
        """
        Agent decision: restart optimization with new settings.

        Args:
            restart_from: Design to restart from
            new_options: New optimizer options
            reason: Why agent decided to restart
        """
        self.current_action = GateAction.RESTART
        self.restart_settings = {
            'restart_from': restart_from,
            'new_options': new_options,
            'reason': reason,
        }
        self._continue_event.set()

    def get_history(self) -> list[Dict[str, Any]]:
        """Get full iteration history."""
        return self.history.copy()

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get latest iteration data."""
        return self.history[-1] if self.history else None

    def get_statistics(self) -> Dict[str, Any]:
        """Get gate statistics."""
        return {
            'total_iterations': self.iteration,
            'blocking_mode': self.blocking,
            'n_pauses': self.n_pauses,
            'total_wait_time': self.total_wait_time,
            'avg_wait_time': self.total_wait_time / self.n_pauses if self.n_pauses > 0 else 0.0,
        }

    def reset(self):
        """Reset gate for new optimization run."""
        self.iteration = 0
        self.history.clear()
        self.current_action = GateAction.CONTINUE
        self.restart_settings = None
        self.stop_reason = None
        self._continue_event.set()
        self.total_wait_time = 0.0
        self.n_pauses = 0
