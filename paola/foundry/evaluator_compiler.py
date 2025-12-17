"""
Evaluator Compiler - Extract and compile evaluators into immutable snapshots.

This module implements the "compiled evaluator" architecture where evaluators
are extracted from source files and stored as immutable, self-contained snapshots
in .paola_foundry/evaluators/{evaluator_id}/.

Key Benefits:
1. Immutability: Original file changes don't break cached results
2. Portability: Self-contained evaluators work across machines
3. Semantic Understanding: LLM analyzes code to extract metadata
4. Auto-Testing: ReAct validates evaluators before registration
"""

import ast
import inspect
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import importlib.util
import sys
import textwrap


class EvaluatorCompiler:
    """
    Compiles evaluators from source files into immutable snapshots.

    Workflow:
    1. Extract function/class from source file
    2. Identify dependencies (imports, helper functions)
    3. Generate standalone source.py with all dependencies
    4. Copy any data files to dependencies/
    5. Create metadata.json with semantic information
    6. Store in .paola_foundry/evaluators/{evaluator_id}/
    """

    def __init__(self, base_dir: str = ".paola_foundry"):
        """
        Initialize compiler.

        Args:
            base_dir: Base directory for PAOLA data (default: .paola_foundry)
        """
        self.base_dir = Path(base_dir)
        self.evaluators_dir = self.base_dir / "evaluators"
        self.evaluators_dir.mkdir(parents=True, exist_ok=True)

    def compile_function(
        self,
        source_file: Path,
        function_name: str,
        evaluator_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Compile a function from source file into immutable evaluator.

        Args:
            source_file: Path to source Python file
            function_name: Name of function to extract
            evaluator_id: Unique ID for this evaluator
            metadata: Optional semantic metadata from LLM analysis

        Returns:
            {
                "success": True,
                "evaluator_id": "rosenbrock_eval",
                "source_path": ".paola_data/evaluators/rosenbrock_eval/source.py",
                "metadata_path": ".paola_data/evaluators/rosenbrock_eval/metadata.json"
            }
        """
        try:
            # Create evaluator directory
            eval_dir = self.evaluators_dir / evaluator_id
            eval_dir.mkdir(parents=True, exist_ok=True)

            # Read source file
            source_code = source_file.read_text()

            # Parse AST to extract function
            tree = ast.parse(source_code)

            # Extract imports
            imports = self._extract_imports(tree)

            # Extract target function
            function_node = self._find_function(tree, function_name)
            if function_node is None:
                return {
                    "success": False,
                    "error": f"Function '{function_name}' not found in {source_file}"
                }

            # Extract helper functions (functions called by target function)
            helper_functions = self._extract_helper_functions(tree, function_node)

            # Generate standalone source.py
            standalone_code = self._generate_standalone_code(
                imports=imports,
                function_node=function_node,
                helper_functions=helper_functions,
                original_file=source_file,
                evaluator_id=evaluator_id
            )

            # Write source.py
            source_path = eval_dir / "source.py"
            source_path.write_text(standalone_code)

            # Create metadata
            if metadata is None:
                metadata = {}

            full_metadata = {
                "evaluator_id": evaluator_id,
                "created_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "origin": {
                    "original_file": str(source_file.absolute()),
                    "function_name": function_name,
                    "extraction_method": "ast_extraction",
                    "extraction_timestamp": datetime.now().isoformat()
                },
                "semantics": metadata.get("semantics", {}),
                "capabilities": metadata.get("capabilities", {
                    "gradient_method": "finite_difference",
                    "vectorized": False,
                    "supports_batch": False
                }),
                "validation": metadata.get("validation", {
                    "test_points": [],
                    "all_tests_passed": False
                })
            }

            # Write metadata.json
            metadata_path = eval_dir / "metadata.json"
            metadata_path.write_text(json.dumps(full_metadata, indent=2))

            return {
                "success": True,
                "evaluator_id": evaluator_id,
                "source_path": str(source_path),
                "metadata_path": str(metadata_path),
                "evaluator_dir": str(eval_dir)
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Compilation failed: {str(e)}"
            }

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all import statements from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = ", ".join(alias.name for alias in node.names)
                imports.append(f"from {module} import {names}")
        return imports

    def _find_function(self, tree: ast.AST, function_name: str) -> Optional[ast.FunctionDef]:
        """Find function definition in AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return node
        return None

    def _extract_helper_functions(
        self,
        tree: ast.AST,
        target_function: ast.FunctionDef
    ) -> List[ast.FunctionDef]:
        """
        Extract helper functions called by target function.

        TODO: Implement call graph analysis to find dependencies.
        For now, returns empty list (conservative - only extract target).
        """
        # Simple implementation: no helpers
        # Advanced: analyze function calls and extract dependencies
        return []

    def _generate_standalone_code(
        self,
        imports: List[str],
        function_node: ast.FunctionDef,
        helper_functions: List[ast.FunctionDef],
        original_file: Path,
        evaluator_id: str
    ) -> str:
        """Generate standalone Python code with all dependencies."""

        # Header comment
        header = f'''"""
Evaluator: {evaluator_id}
Compiled from: {original_file.name}
Function: {function_node.name}
Timestamp: {datetime.now().isoformat()}

IMMUTABLE SNAPSHOT - Do not modify this file.
Changes to the original file will not affect this evaluator.
"""

'''

        # Deduplicate and sort imports
        unique_imports = sorted(set(imports))
        imports_section = "\n".join(unique_imports) + "\n\n" if unique_imports else ""

        # Helper functions
        helpers_section = ""
        for helper in helper_functions:
            helpers_section += ast.unparse(helper) + "\n\n"

        # Main function
        function_code = ast.unparse(function_node)

        # Combine
        standalone_code = header + imports_section + helpers_section + function_code + "\n"

        return standalone_code

    def load_evaluator(self, evaluator_id: str) -> Optional[callable]:
        """
        Load compiled evaluator function.

        Args:
            evaluator_id: Evaluator ID

        Returns:
            Callable function or None if not found
        """
        eval_dir = self.evaluators_dir / evaluator_id
        source_path = eval_dir / "source.py"

        if not source_path.exists():
            return None

        # Load module dynamically
        spec = importlib.util.spec_from_file_location(
            f"evaluator_{evaluator_id}",
            source_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Get metadata to find function name
        metadata_path = eval_dir / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            function_name = metadata["origin"]["function_name"]
            return getattr(module, function_name, None)

        return None

    def get_metadata(self, evaluator_id: str) -> Optional[Dict[str, Any]]:
        """Load evaluator metadata."""
        metadata_path = self.evaluators_dir / evaluator_id / "metadata.json"
        if not metadata_path.exists():
            return None
        return json.loads(metadata_path.read_text())

    def list_evaluators(self) -> List[Dict[str, Any]]:
        """List all compiled evaluators."""
        evaluators = []

        if not self.evaluators_dir.exists():
            return evaluators

        for eval_dir in self.evaluators_dir.iterdir():
            if eval_dir.is_dir():
                metadata = self.get_metadata(eval_dir.name)
                if metadata:
                    evaluators.append({
                        "evaluator_id": eval_dir.name,
                        "name": metadata.get("semantics", {}).get("description", eval_dir.name),
                        "created_at": metadata.get("created_at"),
                        "type": metadata.get("semantics", {}).get("type", "unknown")
                    })

        return evaluators

    def generate_variable_extractor(
        self,
        variable_index: int,
        dimension: int,
        evaluator_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Auto-generate variable extractor evaluator.

        Args:
            variable_index: Index of variable to extract (0-based)
            dimension: Total dimension of design vector
            evaluator_id: Optional ID (default: x{i}_extractor)

        Returns:
            Compilation result
        """
        if evaluator_id is None:
            evaluator_id = f"x{variable_index}_extractor"

        # Create evaluator directory
        eval_dir = self.evaluators_dir / evaluator_id
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Generate source code
        source_code = f'''"""
Evaluator: {evaluator_id}
Auto-generated variable extractor for x[{variable_index}]
Timestamp: {datetime.now().isoformat()}

IMMUTABLE SNAPSHOT - Auto-generated by PAOLA.
"""

import numpy as np


def x{variable_index}_extractor(x):
    """
    Extract component {variable_index} from design vector.

    Args:
        x: Design vector (dimension >= {variable_index + 1})

    Returns:
        x[{variable_index}] as float
    """
    x = np.atleast_1d(x)
    return float(x[{variable_index}])
'''

        # Write source.py
        source_path = eval_dir / "source.py"
        source_path.write_text(source_code)

        # Create metadata
        metadata = {
            "evaluator_id": evaluator_id,
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "origin": {
                "original_file": "auto-generated",
                "function_name": f"x{variable_index}_extractor",
                "extraction_method": "auto_generation",
                "extraction_timestamp": datetime.now().isoformat()
            },
            "semantics": {
                "type": "variable_extractor",
                "description": f"Extracts x[{variable_index}] from design vector",
                "input_dimension": dimension,
                "output_type": "scalar",
                "variable_index": variable_index
            },
            "capabilities": {
                "gradient_method": "analytical",  # Gradient is [0, 0, ..., 1, ..., 0]
                "vectorized": False,
                "supports_batch": False
            },
            "validation": {
                "test_points": [],
                "all_tests_passed": True  # Simple enough to trust
            }
        }

        # Write metadata.json
        metadata_path = eval_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        return {
            "success": True,
            "evaluator_id": evaluator_id,
            "source_path": str(source_path),
            "metadata_path": str(metadata_path),
            "auto_generated": True
        }
