"""
Agent tools for evaluator registration.

Minimal tool set - LLM does the reasoning.
"""

import subprocess
import sys
import tempfile
from typing import Dict, Any
from pathlib import Path
from langchain_core.tools import tool


@tool
def read_file(file_path: str) -> Dict[str, Any]:
    """
    Read file contents.

    Args:
        file_path: Path to file

    Returns:
        {"success": True, "contents": "..."}
        {"success": False, "error": "..."}
    """
    try:
        with open(file_path, 'r') as f:
            contents = f.read()

        return {
            "success": True,
            "contents": contents,
            "file_type": Path(file_path).suffix
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
def execute_python(code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute Python code in subprocess.

    Args:
        code: Python code to execute
        timeout: Timeout in seconds

    Returns:
        {"success": True, "stdout": "...", "stderr": "..."}
        {"success": False, "error": "..."}
    """
    try:
        # Write code to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute with timeout
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            success = result.returncode == 0

            return {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }

        finally:
            # Clean up
            Path(temp_file).unlink()

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Execution timed out after {timeout} seconds"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
def foundry_store_evaluator(
    config: Dict[str, Any],
    test_result: Dict[str, Any] = {"success": True}
) -> Dict[str, Any]:
    """
    Store evaluator configuration in Foundry.

    Args:
        config: Evaluator configuration dict
        test_result: Result from testing (for validation)

    Returns:
        {"success": True, "evaluator_id": "..."}
        {"success": False, "error": "..."}
    """
    try:
        from paola.foundry import (
            OptimizationFoundry,
            FileStorage,
            EvaluatorConfig
        )

        # Create config from dict
        evaluator_config = EvaluatorConfig.from_dict(config)

        # Get or create foundry instance
        # In production, this would be passed in or retrieved from context
        # For now, use default storage
        storage = FileStorage(base_dir=".paola_data")
        foundry = OptimizationFoundry(storage=storage)

        # Store evaluator
        evaluator_id = foundry.register_evaluator(evaluator_config)

        return {
            "success": True,
            "evaluator_id": evaluator_id,
            "message": f"Registered evaluator '{evaluator_config.name}' (ID: {evaluator_id})"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
def foundry_list_evaluators(
    evaluator_type: str = None,
    status: str = None
) -> Dict[str, Any]:
    """
    List registered evaluators.

    Args:
        evaluator_type: Filter by type (python_function, cli_executable)
        status: Filter by status

    Returns:
        {"success": True, "evaluators": [...]}
    """
    try:
        from paola.foundry import OptimizationFoundry, FileStorage

        storage = FileStorage(base_dir=".paola_data")
        foundry = OptimizationFoundry(storage=storage)

        evaluators = foundry.list_evaluators(
            evaluator_type=evaluator_type,
            status=status
        )

        # Convert to serializable dicts
        evaluators_list = [
            {
                "evaluator_id": e.evaluator_id,
                "name": e.name,
                "type": e.source.type,
                "status": e.status
            }
            for e in evaluators
        ]

        return {
            "success": True,
            "evaluators": evaluators_list,
            "count": len(evaluators_list)
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
def foundry_get_evaluator(evaluator_id: str) -> Dict[str, Any]:
    """
    Get evaluator configuration details.

    Args:
        evaluator_id: Evaluator ID

    Returns:
        {"success": True, "config": {...}}
    """
    try:
        from paola.foundry import OptimizationFoundry, FileStorage

        storage = FileStorage(base_dir=".paola_data")
        foundry = OptimizationFoundry(storage=storage)

        config_dict = foundry.get_evaluator_config(evaluator_id)

        return {
            "success": True,
            "config": config_dict
        }

    except KeyError:
        return {
            "success": False,
            "error": f"Evaluator not found: {evaluator_id}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Export all tools
ALL_REGISTRATION_TOOLS = [
    read_file,
    execute_python,
    foundry_store_evaluator,
    foundry_list_evaluators,
    foundry_get_evaluator
]
