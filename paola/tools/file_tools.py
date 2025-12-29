"""
File operation tools.

Tools for file operations:
- read_file: Read file contents
- write_file: Write content to file

v0.2.1: Removed execute_python. Use bash tool instead.
"""

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
        success, contents, file_type
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
        success, file_path, bytes_written
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
