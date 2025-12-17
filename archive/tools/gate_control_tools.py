"""
Gate control tools for the agentic optimization platform.

Provides LangChain @tool decorated functions for controlling optimization gates:
- gate_continue: Resume blocked optimization
- gate_stop: Stop optimization early
- gate_restart_from: Restart optimization with new settings

These tools enable the agent to control optimization execution in blocking mode,
where expensive engineering simulations require agent approval at each iteration.
"""

from typing import Optional, Dict, Any, List
import numpy as np
from langchain_core.tools import tool

from paola.optimizers.gate import OptimizationGate, GateAction


# Global gate registry
_GATE_REGISTRY: Dict[str, OptimizationGate] = {}


def register_gate(gate_id: str, gate: OptimizationGate):
    """Register a gate for agent control."""
    _GATE_REGISTRY[gate_id] = gate


def _get_gate(gate_id: str) -> OptimizationGate:
    """Get gate from registry."""
    if gate_id not in _GATE_REGISTRY:
        raise ValueError(
            f"Gate '{gate_id}' not found. "
            f"Available: {list(_GATE_REGISTRY.keys())}"
        )
    return _GATE_REGISTRY[gate_id]


@tool
def gate_continue(gate_id: str) -> Dict[str, Any]:
    """
    Continue optimization (let optimizer proceed to next iteration).

    Use this tool when you've reviewed the current iteration and want the
    optimizer to proceed. Only relevant in blocking mode (expensive simulations).

    In non-blocking mode, optimization runs to completion automatically and
    you don't need this tool.

    Args:
        gate_id: Gate identifier

    Returns:
        Dict with:
            - success: bool
            - iteration: int - current iteration when continued
            - message: str

    Example:
        # Agent reviews iteration, decides to continue
        result = gate_continue("gate_1")
    """
    try:
        gate = _get_gate(gate_id)

        if not gate.blocking:
            return {
                "success": False,
                "message": f"Gate '{gate_id}' is in non-blocking mode. No need to call gate_continue.",
            }

        current_iter = gate.iteration
        gate.agent_continue()

        return {
            "success": True,
            "iteration": current_iter,
            "message": f"Continued optimization from iteration {current_iter}",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error continuing gate: {str(e)}",
        }


@tool
def gate_stop(gate_id: str, reason: str) -> Dict[str, Any]:
    """
    Stop optimization early.

    Use this tool when you decide the optimization should stop:
    - Converged (gradient small, no improvement)
    - Budget exhausted
    - Detected hopeless situation (diverging, stuck)
    - Satisfied with current result

    Args:
        gate_id: Gate identifier
        reason: Why you're stopping (for logging and analysis)

    Returns:
        Dict with:
            - success: bool
            - final_iteration: int
            - best_objective: float (if available)
            - message: str

    Example:
        # Agent decides optimization has converged
        result = gate_stop("gate_1", reason="Gradient norm < 1e-6, converged")
    """
    try:
        gate = _get_gate(gate_id)

        final_iter = gate.iteration
        gate.agent_stop(reason)

        # Get best result from history
        best_obj = None
        if gate.history:
            objectives = [h.get('objective') for h in gate.history if h.get('objective') is not None]
            if objectives:
                best_obj = min(objectives)  # Assuming minimization

        return {
            "success": True,
            "final_iteration": final_iter,
            "best_objective": best_obj,
            "reason": reason,
            "message": f"Stopped optimization at iteration {final_iter}. Reason: {reason}",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error stopping gate: {str(e)}",
        }


@tool
def gate_restart_from(
    gate_id: str,
    restart_design: List[float],
    reason: str,
    new_options: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Restart optimization from a specific design with new settings.

    Use this tool for strategic restarts when:
    - Constraint violations detected (tighten constraints)
    - Gradient noise detected (switch gradient method)
    - Optimizer stuck (change algorithm parameters)

    IMPORTANT: This is a strategic restart, not arbitrary experimentation.
    Restart from a known good position with informed changes.

    Args:
        gate_id: Gate identifier
        restart_design: Design vector to restart from
        reason: Why you're restarting (for logging)
        new_options: Optional JSON string with new optimizer options

    Returns:
        Dict with:
            - success: bool
            - restart_from: List[float]
            - iteration_at_restart: int
            - message: str

    Example:
        # Agent detects constraint violations, restarts with tighter bounds
        result = gate_restart_from(
            gate_id="gate_1",
            restart_design=[1.0, 2.0],
            reason="CL constraint violated 5 times, tightening to 0.51",
            new_options='{"constraint_bounds": {"CL": 0.51}}'
        )
    """
    import json

    try:
        gate = _get_gate(gate_id)

        restart_array = np.array(restart_design)
        iter_at_restart = gate.iteration

        # Parse new options
        new_options_dict = {}
        if new_options:
            try:
                new_options_dict = json.loads(new_options)
            except json.JSONDecodeError as e:
                return {
                    "success": False,
                    "message": f"Invalid new_options JSON: {e}",
                }

        gate.agent_restart(
            restart_from=restart_array,
            new_options=new_options_dict,
            reason=reason
        )

        return {
            "success": True,
            "restart_from": restart_design,
            "iteration_at_restart": iter_at_restart,
            "new_options": new_options_dict,
            "reason": reason,
            "message": f"Initiated restart from iteration {iter_at_restart}. Reason: {reason}",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error restarting gate: {str(e)}",
        }


@tool
def gate_get_history(gate_id: str, last_n: Optional[int] = None) -> Dict[str, Any]:
    """
    Get optimization history from gate.

    Use this to review past iterations and analyze optimization progress.
    Useful for detecting patterns, convergence issues, or constraint violations.

    Args:
        gate_id: Gate identifier
        last_n: If specified, only return last N iterations (default: all)

    Returns:
        Dict with:
            - success: bool
            - history: List[Dict] - iteration data
            - total_iterations: int
            - best_objective: float
            - best_design: List[float]
            - message: str

    Example:
        # Agent reviews last 5 iterations to check for patterns
        result = gate_get_history("gate_1", last_n=5)
        for entry in result["history"]:
            print(f"Iter {entry['iteration']}: obj={entry['objective']:.6f}")
    """
    try:
        gate = _get_gate(gate_id)

        history = gate.get_history()
        if last_n is not None:
            history = history[-last_n:]

        # Find best
        best_obj = None
        best_design = None
        for h in gate.history:
            obj = h.get('objective')
            if obj is not None:
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best_design = h.get('design')

        # Convert numpy arrays to lists for JSON serialization
        serializable_history = []
        for h in history:
            entry = {}
            for k, v in h.items():
                if isinstance(v, np.ndarray):
                    entry[k] = v.tolist()
                else:
                    entry[k] = v
            serializable_history.append(entry)

        return {
            "success": True,
            "history": serializable_history,
            "total_iterations": gate.iteration,
            "best_objective": best_obj,
            "best_design": best_design.tolist() if isinstance(best_design, np.ndarray) else best_design,
            "message": f"Retrieved {len(serializable_history)} iterations (total: {gate.iteration})",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting gate history: {str(e)}",
        }


@tool
def gate_get_statistics(gate_id: str) -> Dict[str, Any]:
    """
    Get gate statistics for performance analysis.

    Use this to understand optimization performance:
    - Total iterations
    - Time spent waiting (in blocking mode)
    - Number of pauses

    Args:
        gate_id: Gate identifier

    Returns:
        Dict with gate statistics

    Example:
        stats = gate_get_statistics("gate_1")
        print(f"Total iterations: {stats['total_iterations']}")
        print(f"Avg wait time: {stats['avg_wait_time']:.2f}s")
    """
    try:
        gate = _get_gate(gate_id)
        stats = gate.get_statistics()

        return {
            "success": True,
            "gate_id": gate_id,
            "problem_id": gate.problem_id,
            **stats,
            "message": f"Gate '{gate_id}' statistics retrieved",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting gate statistics: {str(e)}",
        }


# Utility functions
def clear_gate_registry():
    """Clear all gates from registry."""
    _GATE_REGISTRY.clear()


def get_gate_by_id(gate_id: str) -> Optional[OptimizationGate]:
    """Get gate instance by ID."""
    return _GATE_REGISTRY.get(gate_id)
