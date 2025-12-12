"""
Tools for agent to manage optimization runs.

Provides explicit run creation and finalization.

REFACTORED: Now uses OptimizationPlatform with dependency injection
instead of RunManager singleton.
"""

from typing import Dict, Any, Optional
from langchain_core.tools import tool

from ..platform import OptimizationPlatform
from .evaluator_tools import _get_problem

# Global platform reference (set by build_tools)
_PLATFORM: Optional[OptimizationPlatform] = None


def set_platform(platform: OptimizationPlatform) -> None:
    """
    Set global platform reference for tools.

    This is called by build_tools() during initialization.

    Args:
        platform: OptimizationPlatform instance
    """
    global _PLATFORM
    _PLATFORM = platform


def get_platform() -> Optional[OptimizationPlatform]:
    """
    Get current platform reference.

    Returns:
        Current platform instance or None if not set
    """
    return _PLATFORM


@tool
def start_optimization_run(
    problem_id: str,
    algorithm: str,
    description: str = ""
) -> Dict[str, Any]:
    """
    Start a new tracked optimization run.

    Use this tool BEFORE running optimization to create a tracked run.
    All optimization results will be recorded to this run for later analysis.

    This creates a persistent record that survives across sessions and can be
    inspected using CLI commands (/runs, /show, /plot, etc.) or notebooks.

    Args:
        problem_id: Problem identifier (from create_benchmark_problem)
        algorithm: Optimization algorithm name (e.g., "SLSQP", "BFGS")
        description: Optional description of this optimization attempt

    Returns:
        Dict with:
            - success: bool
            - run_id: int - Use this ID in run_scipy_optimization
            - problem_name: str
            - algorithm: str
            - message: str

    Example:
        # Agent receives: "Optimize 10D Rosenbrock with SLSQP"
        # Agent calls:
        problem_result = create_benchmark_problem(
            problem_id="rosenbrock_10d",
            function_name="rosenbrock",
            dimension=10
        )

        run_result = start_optimization_run(
            problem_id="rosenbrock_10d",
            algorithm="SLSQP",
            description="First attempt with default settings"
        )
        # Returns: {"success": True, "run_id": 1, ...}

        opt_result = run_scipy_optimization(
            problem_id="rosenbrock_10d",
            algorithm="SLSQP",
            bounds=[[-5, 10]] * 10,
            run_id=1  # Link to this run
        )
    """
    try:
        # Get platform
        if _PLATFORM is None:
            return {
                "success": False,
                "message": "Platform not initialized. Call set_platform() first."
            }

        # Get problem to extract name
        try:
            problem = _get_problem(problem_id)
            problem_name = getattr(problem, 'name', problem_id)
        except ValueError:
            # Problem not found, use ID as name
            problem_name = problem_id

        # Create run
        run = _PLATFORM.create_run(
            problem_id=problem_id,
            problem_name=problem_name,
            algorithm=algorithm,
            description=description
        )

        return {
            "success": True,
            "run_id": run.run_id,
            "problem_id": problem_id,
            "problem_name": problem_name,
            "algorithm": algorithm,
            "message": f"Created run #{run.run_id}: {algorithm} on {problem_name}. Use run_id={run.run_id} in optimization tools."
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error creating run: {str(e)}"
        }


@tool
def finalize_optimization_run(
    run_id: int,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Finalize optimization run and add final notes.

    Use this after optimization completes to add final analysis or notes.
    The run is automatically finalized when optimization completes, but this
    tool allows adding additional metadata.

    Args:
        run_id: Run identifier from start_optimization_run
        notes: Optional final notes or analysis

    Returns:
        Dict with:
            - success: bool
            - run_id: int
            - message: str

    Example:
        finalize_result = finalize_optimization_run(
            run_id=1,
            notes="Converged successfully. Final objective: 0.0234"
        )
    """
    try:
        if _PLATFORM is None:
            return {
                "success": False,
                "message": "Platform not initialized. Call set_platform() first."
            }

        run = _PLATFORM.get_run(run_id)

        if run is None:
            return {
                "success": False,
                "message": f"Run {run_id} not found or already finalized"
            }

        # Add notes to metadata
        if notes:
            run.metadata["final_notes"] = notes

        # Remove from active registry
        _PLATFORM.finalize_run(run_id)

        return {
            "success": True,
            "run_id": run_id,
            "message": f"Run #{run_id} finalized"
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error finalizing run: {str(e)}"
        }


@tool
def get_active_runs() -> Dict[str, Any]:
    """
    Get list of currently active (in-progress) optimization runs.

    Returns:
        Dict with:
            - success: bool
            - active_runs: List[Dict] - List of active run info
            - count: int
            - message: str

    Example:
        result = get_active_runs()
        # Returns: {"success": True, "active_runs": [...], "count": 2}
    """
    try:
        if _PLATFORM is None:
            return {
                "success": False,
                "message": "Platform not initialized. Call set_platform() first."
            }

        active_runs = _PLATFORM.get_active_runs()

        runs_info = []
        for run_id, run in active_runs.items():
            best = run.get_current_best()
            runs_info.append({
                "run_id": run_id,
                "problem_id": run.problem_id,
                "problem_name": run.problem_name,
                "algorithm": run.algorithm,
                "iterations": len(run.iterations),
                "current_best": best["objective"] if best else None,
                "description": run.description
            })

        return {
            "success": True,
            "active_runs": runs_info,
            "count": len(runs_info),
            "message": f"Found {len(runs_info)} active run(s)"
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting active runs: {str(e)}"
        }
