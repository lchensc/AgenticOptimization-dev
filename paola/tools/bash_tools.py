"""
Bash tool for general scripting.

Like Claude Code's Bash tool, this empowers the agent with general
shell capabilities for running scripts, inspecting files, git operations,
package management, etc.

v0.2.1: Replaces execute_python for code execution.
Agent writes Python scripts to files, then runs via `bash("python script.py")`.
"""

import os
import subprocess
from typing import Dict, Any
from langchain_core.tools import tool


@tool
def bash(
    command: str,
    timeout: int = 120,
    description: str = "",
) -> Dict[str, Any]:
    """
    Execute bash command.

    Use this for:
    - Running optimization scripts: `python scripts/my_optimization.py`
    - Testing code: `python -m pytest tests/`
    - Inspecting files: `ls -la`, `cat file.py`
    - Git operations: `git status`, `git diff`
    - Package management: `pip list | grep scipy`

    Args:
        command: Bash command to execute
        timeout: Timeout in seconds (default: 120s for optimization runs)
        description: Optional description of what the command does

    Returns:
        success: Whether command succeeded (returncode == 0)
        stdout: Command standard output
        stderr: Command standard error
        returncode: Exit code
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=os.getcwd(),
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "returncode": -1,
            "error": f"Timeout after {timeout}s",
        }

    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
            "error": str(e),
        }
