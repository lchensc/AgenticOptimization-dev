"""
Tools for agent to manage optimization sessions.

v0.2.0: Session-based architecture
- Session = complete optimization task (may involve multiple runs)
- Run = single optimizer execution

Uses OptimizationFoundry with dependency injection for data foundation.
"""

from typing import Dict, Any, Optional
from langchain_core.tools import tool

from ..foundry import OptimizationFoundry

# Global foundry reference (set by build_tools)
_FOUNDRY: Optional[OptimizationFoundry] = None


def set_foundry(foundry: OptimizationFoundry) -> None:
    """
    Set global foundry reference for tools.

    This is called by build_tools() during initialization.

    Args:
        foundry: OptimizationFoundry instance
    """
    global _FOUNDRY
    _FOUNDRY = foundry


def get_foundry() -> Optional[OptimizationFoundry]:
    """
    Get current foundry reference.

    Returns:
        Current foundry instance or None if not set
    """
    return _FOUNDRY


@tool
def start_session(
    problem_id: str,
    goal: str = "",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Start a new optimization session.

    A session represents a complete optimization task that may involve
    multiple optimizer runs (e.g., global exploration followed by
    local refinement with warm-starting).

    Use this BEFORE running any optimization. The returned session_id
    must be passed to run_optimization.

    Args:
        problem_id: Problem identifier (from create_nlp_problem)
        goal: Optional description of optimization goal
        config: Optional session configuration

    Returns:
        Dict with:
            - success: bool
            - session_id: int - Use this in run_optimization
            - problem_id: str
            - message: str

    Example:
        session = start_session(
            problem_id="wing_v2",
            goal="Minimize drag while maintaining lift >= 0.5"
        )
        # Returns: {"success": True, "session_id": 1, ...}

        # Then run optimization
        result = run_optimization(
            session_id=1,
            optimizer="scipy:SLSQP",
            ...
        )
    """
    try:
        if _FOUNDRY is None:
            return {
                "success": False,
                "message": "Foundry not initialized. Call set_foundry() first.",
            }

        # Build config
        session_config = config or {}
        if goal:
            session_config["goal"] = goal

        # Create session
        session = _FOUNDRY.create_session(
            problem_id=problem_id,
            config=session_config,
        )

        return {
            "success": True,
            "session_id": session.session_id,
            "problem_id": problem_id,
            "message": f"Created session #{session.session_id} for problem '{problem_id}'.",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error creating session: {str(e)}",
        }


@tool
def finalize_session(
    session_id: int,
    success: bool,
    notes: str = "",
) -> Dict[str, Any]:
    """
    Finalize optimization session and persist to storage.

    Call this when the optimization task is complete (whether successful
    or not). The session record will be saved for future analysis.

    Args:
        session_id: Session identifier from start_session
        success: Whether the optimization was successful overall
        notes: Optional final notes or analysis

    Returns:
        Dict with:
            - success: bool
            - session_id: int
            - final_objective: float (best found)
            - total_evaluations: int
            - n_runs: int (number of optimizer runs)
            - message: str

    Example:
        result = finalize_session(
            session_id=1,
            success=True,
            notes="Converged after 3 runs: Bayesian -> SLSQP -> CMA-ES"
        )
    """
    try:
        if _FOUNDRY is None:
            return {
                "success": False,
                "message": "Foundry not initialized. Call set_foundry() first.",
            }

        session = _FOUNDRY.get_session(session_id)

        if session is None:
            return {
                "success": False,
                "message": f"Session {session_id} not found or already finalized.",
            }

        # Add notes to config
        if notes:
            session.config["final_notes"] = notes

        # Finalize session
        record = _FOUNDRY.finalize_session(session_id, success)

        if record is None:
            return {
                "success": False,
                "message": f"Failed to finalize session {session_id}.",
            }

        return {
            "success": True,
            "session_id": session_id,
            "final_objective": record.final_objective,
            "total_evaluations": record.total_evaluations,
            "n_runs": len(record.runs),
            "total_wall_time": record.total_wall_time,
            "message": f"Session #{session_id} finalized. Best objective: {record.final_objective:.6e}, {len(record.runs)} run(s), {record.total_evaluations} evaluations.",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error finalizing session: {str(e)}",
        }


@tool
def get_active_sessions() -> Dict[str, Any]:
    """
    Get list of currently active (in-progress) optimization sessions.

    Returns:
        Dict with:
            - success: bool
            - sessions: List[Dict] - List of active session info
            - count: int
            - message: str

    Example:
        result = get_active_sessions()
        # Returns: {"success": True, "sessions": [...], "count": 2}
    """
    try:
        if _FOUNDRY is None:
            return {
                "success": False,
                "message": "Foundry not initialized. Call set_foundry() first.",
            }

        active_sessions = _FOUNDRY.get_active_sessions()

        sessions_info = []
        for session_id, session in active_sessions.items():
            best_run = session.get_best_run()
            sessions_info.append({
                "session_id": session_id,
                "problem_id": session.problem_id,
                "n_runs": len(session.runs),
                "current_run_active": session.current_run is not None,
                "best_objective": best_run.best_objective if best_run else None,
                "config": session.config,
            })

        return {
            "success": True,
            "sessions": sessions_info,
            "count": len(sessions_info),
            "message": f"Found {len(sessions_info)} active session(s).",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting active sessions: {str(e)}",
        }


@tool
def get_session_info(session_id: int) -> Dict[str, Any]:
    """
    Get detailed information about a session (active or completed).

    Args:
        session_id: Session identifier

    Returns:
        Dict with session details including runs, decisions, and results.

    Example:
        info = get_session_info(session_id=1)
    """
    try:
        if _FOUNDRY is None:
            return {
                "success": False,
                "message": "Foundry not initialized. Call set_foundry() first.",
            }

        # Check active sessions first
        session = _FOUNDRY.get_session(session_id)
        if session is not None:
            best_run = session.get_best_run()
            return {
                "success": True,
                "session_id": session_id,
                "status": "active",
                "problem_id": session.problem_id,
                "config": session.config,
                "n_runs": len(session.runs),
                "n_decisions": len(session.decisions),
                "best_objective": best_run.best_objective if best_run else None,
                "runs": [
                    {
                        "run_id": r.run_id,
                        "optimizer": r.optimizer,
                        "optimizer_family": r.optimizer_family,
                        "n_evaluations": r.n_evaluations,
                        "best_objective": r.best_objective,
                        "run_success": r.run_success,
                    }
                    for r in session.runs
                ],
                "message": f"Session #{session_id} is active with {len(session.runs)} run(s).",
            }

        # Check completed sessions
        record = _FOUNDRY.load_session(session_id)
        if record is not None:
            return {
                "success": True,
                "session_id": session_id,
                "status": "completed",
                "problem_id": record.problem_id,
                "config": record.config,
                "success_status": record.success,
                "final_objective": record.final_objective,
                "total_evaluations": record.total_evaluations,
                "total_wall_time": record.total_wall_time,
                "n_runs": len(record.runs),
                "n_decisions": len(record.decisions),
                "runs": [
                    {
                        "run_id": r.run_id,
                        "optimizer": r.optimizer,
                        "optimizer_family": r.optimizer_family,
                        "n_evaluations": r.n_evaluations,
                        "best_objective": r.best_objective,
                        "run_success": r.run_success,
                        "warm_start_from": r.warm_start_from,
                    }
                    for r in record.runs
                ],
                "decisions": [
                    {
                        "timestamp": d.timestamp,
                        "decision_type": d.decision_type,
                        "reasoning": d.reasoning,
                    }
                    for d in record.decisions
                ],
                "message": f"Session #{session_id} completed with {len(record.runs)} run(s).",
            }

        return {
            "success": False,
            "message": f"Session {session_id} not found.",
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting session info: {str(e)}",
        }
