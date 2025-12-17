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
def write_file(file_path: str, content: str) -> Dict[str, Any]:
    """
    Write content to file.

    Args:
        file_path: Path to file
        content: Content to write

    Returns:
        {"success": True, "file_path": "...", "bytes_written": ...}
        {"success": False, "error": "..."}
    """
    try:
        # Create parent directories if needed
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Write file
        with open(file_path, 'w') as f:
            bytes_written = f.write(content)

        return {
            "success": True,
            "file_path": file_path,
            "bytes_written": bytes_written
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
    evaluator_id: str,
    name: str,
    file_path: str,
    callable_name: str = "evaluate",
    description: str = None
) -> Dict[str, Any]:
    """
    Register Python function evaluator in Foundry.

    This stores metadata about your evaluator so it can be used in optimization problems.
    The evaluator file must exist and contain the specified callable.

    Args:
        evaluator_id: Unique ID (e.g., "rosenbrock_eval")
        name: Human-readable name (e.g., "Rosenbrock 2D Function")
        file_path: Absolute path to .py file (e.g., ".paola_data/evaluators/rosenbrock.py")
        callable_name: Function name to call (default: "evaluate")
        description: Optional description of what this evaluator does

    Returns:
        {"success": True, "evaluator_id": "...", "message": "..."}
        {"success": False, "error": "..."}

    Example:
        foundry_store_evaluator(
            evaluator_id="rosenbrock_eval",
            name="Rosenbrock 2D",
            file_path=".paola_data/evaluators/rosenbrock.py",
            callable_name="evaluate",
            description="2D Rosenbrock function with minimum at (1, 1)"
        )
    """
    try:
        from paola.foundry import (
            OptimizationFoundry,
            FileStorage,
            create_python_function_config
        )
        from datetime import datetime

        # Validate file exists
        from pathlib import Path
        if not Path(file_path).exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        # Create configuration using convenience function
        config = create_python_function_config(
            evaluator_id=evaluator_id,
            name=name,
            file_path=file_path,
            callable_name=callable_name,
            metadata={
                "description": description or name,
                "tags": [],
                "domain": "optimization"
            },
            lineage={
                "registered_at": datetime.now(),
                "registered_by": "agent"
            }
        )

        # Get or create foundry instance (uses unified .paola_foundry)
        storage = FileStorage()
        foundry = OptimizationFoundry(storage=storage)

        # Store evaluator
        stored_id = foundry.register_evaluator(config)

        return {
            "success": True,
            "evaluator_id": stored_id,
            "message": f"Registered evaluator '{name}' (ID: {stored_id})"
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"{str(e)}\n{traceback.format_exc()}"
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

        storage = FileStorage()  # uses unified .paola_foundry
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

        storage = FileStorage()  # uses unified .paola_foundry
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
