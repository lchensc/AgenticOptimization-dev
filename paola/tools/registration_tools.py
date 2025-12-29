"""
Agent tools for evaluator registration.

Re-exports tools from reorganized modules for backward compatibility.

v0.2.1: Replaced execute_python with bash tool.
"""

# Re-export from new modules
from paola.tools.file_tools import (
    read_file,
    write_file,
)
from paola.tools.bash_tools import bash
from paola.tools.evaluator import (
    foundry_store_evaluator,
    foundry_list_evaluators,
    foundry_get_evaluator,
)

# Export all tools
ALL_REGISTRATION_TOOLS = [
    read_file,
    bash,
    foundry_store_evaluator,
    foundry_list_evaluators,
    foundry_get_evaluator,
]
