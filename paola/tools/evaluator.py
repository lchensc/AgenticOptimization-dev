"""
Evaluator registration tools.

Tools for managing evaluator functions in Foundry:
- foundry_store_evaluator: Register evaluator function
- foundry_list_evaluators: List registered evaluators
- foundry_get_evaluator: Get evaluator configuration
"""

from typing import Dict, Any, Optional, List
from langchain_core.tools import tool


@tool
def foundry_store_evaluator(
    evaluator_id: str,
    name: str,
    file_path: str,
    callable_name: str = "evaluate",
    description: str = None,
    n_outputs: int = 1,
    output_names: List[str] = None
) -> Dict[str, Any]:
    """
    Register evaluator function in Foundry.

    Args:
        evaluator_id: Unique ID
        name: Human-readable name
        file_path: Path to .py file
        callable_name: Function name (default: "evaluate")
        description: Description
        n_outputs: Number of output values (default: 1). For MOO, set to number of objectives.
        output_names: Names for each output (e.g., ["f1", "f2"] or ["drag", "lift"])

    Example:
        # Register single-objective evaluator
        foundry_store_evaluator("sharpe_eval", "Sharpe Ratio", "portfolio.py", "evaluate")

        # Register multi-objective evaluator (MOO)
        foundry_store_evaluator(
            "moo_eval", "MOO Objectives", "moo.py", "evaluate",
            n_outputs=2, output_names=["f1", "f2"]
        )

        # Register constraint function
        foundry_store_evaluator("bond_constraint", "Min Bonds", "portfolio.py", "constraint_min_bonds")
    """
    try:
        from paola.foundry import (
            OptimizationFoundry,
            FileStorage,
            create_python_function_config
        )
        from paola.foundry.evaluator_schema import (
            EvaluatorInterface,
            OutputInterface,
        )
        from datetime import datetime
        from pathlib import Path

        # Validate file exists
        if not Path(file_path).exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        # Determine output format based on n_outputs
        output_format = "array" if n_outputs > 1 else "auto"

        # Create interface with explicit n_outputs
        interface = EvaluatorInterface(
            output=OutputInterface(
                format=output_format,
                n_outputs=n_outputs,
                output_names=output_names,
            )
        )

        # Create configuration using convenience function
        config = create_python_function_config(
            evaluator_id=evaluator_id,
            name=name,
            file_path=file_path,
            callable_name=callable_name,
            interface=interface,
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
        success, evaluators, count
    """
    try:
        from paola.foundry import OptimizationFoundry, FileStorage

        storage = FileStorage()
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
        success, config
    """
    try:
        from paola.foundry import OptimizationFoundry, FileStorage

        storage = FileStorage()
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
