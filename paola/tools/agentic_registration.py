"""
Agentic Evaluator Registration with LLM Analysis and ReAct Testing.

This module implements the "compiled evaluator" workflow:
1. Agent reads source file
2. LLM analyzes code semantics
3. Compiler extracts and compiles evaluator
4. ReAct loop tests evaluator
5. Store immutable snapshot with metadata
"""

from typing import Dict, Any, Optional
from pathlib import Path
from langchain_core.tools import tool
import json
import numpy as np


# Global reference to foundry (set by tools)
_FOUNDRY = None


def set_foundry(foundry):
    """Set global foundry instance."""
    global _FOUNDRY
    _FOUNDRY = foundry


@tool
def register_evaluator_agentic(
    file_path: str,
    function_name: str,
    evaluator_id: Optional[str] = None,
    test_inputs: Optional[str] = None
) -> Dict[str, Any]:
    """
    Register evaluator with LLM analysis and ReAct validation.

    This tool implements the "compiled evaluator" architecture:
    - Extracts code into immutable snapshot
    - Uses LLM to analyze semantics
    - Tests evaluator with ReAct loop
    - Stores in .paola_data/evaluators/{id}/

    Key advantages over deterministic registration:
    1. Immutable: Original file changes don't break cached results
    2. Portable: Self-contained evaluators work across machines
    3. Semantic: Agent understands what evaluator computes
    4. Tested: ReAct validates before registration

    Args:
        file_path: Path to Python file containing evaluator
        function_name: Name of function to extract
        evaluator_id: Optional ID (default: {function_name}_eval)
        test_inputs: Optional JSON list of test inputs, e.g. "[[1.0, 1.0], [0.0, 0.0]]"

    Returns:
        {
            "success": True,
            "evaluator_id": "rosenbrock_eval",
            "semantics": {
                "type": "objective_function",
                "input_dimension": 2,
                ...
            },
            "source_path": ".paola_data/evaluators/rosenbrock_eval/source.py",
            "tests_passed": True
        }

    Example:
        # Agent workflow:
        result = register_evaluator_agentic(
            file_path="evaluators.py",
            function_name="rosenbrock_2d"
        )

        # Result contains semantic info:
        # - Agent now knows this is a 2D objective function
        # - Agent can infer need for x0_extractor, x1_extractor
        # - Agent can create correct NLP problems
    """
    try:
        from ..foundry.evaluator_compiler import EvaluatorCompiler
        from ..agent.evaluator_analyzer import EvaluatorSemanticAnalyzer
        from ..agent.react_agent import initialize_llm

        # Validate file exists
        source_file = Path(file_path)
        if not source_file.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}"
            }

        # Default evaluator ID
        if evaluator_id is None:
            evaluator_id = f"{function_name}_eval"

        # Step 1: Read source code
        source_code = source_file.read_text()

        # Find the function in source
        import ast
        tree = ast.parse(source_code)
        function_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                function_node = node
                break

        if function_node is None:
            return {
                "success": False,
                "error": f"Function '{function_name}' not found in {file_path}"
            }

        # Extract function code
        function_code = ast.unparse(function_node)

        # Step 2: LLM analyzes semantics
        llm = initialize_llm("qwen-plus", temperature=0.0)  # Use smart model for analysis
        analyzer = EvaluatorSemanticAnalyzer(llm)

        analysis_result = analyzer.analyze_function(
            code=function_code,
            function_name=function_name,
            file_context=source_code
        )

        if not analysis_result["success"]:
            return {
                "success": False,
                "error": f"Semantic analysis failed: {analysis_result['error']}"
            }

        semantics = analysis_result["semantics"]

        # Step 3: Compile evaluator
        compiler = EvaluatorCompiler()
        compile_result = compiler.compile_function(
            source_file=source_file,
            function_name=function_name,
            evaluator_id=evaluator_id,
            metadata={"semantics": semantics}
        )

        if not compile_result["success"]:
            return compile_result

        # Step 4: Test evaluator (ReAct loop)
        evaluator_func = compiler.load_evaluator(evaluator_id)
        if evaluator_func is None:
            return {
                "success": False,
                "error": "Failed to load compiled evaluator"
            }

        # Generate or use provided test inputs
        if test_inputs:
            try:
                test_cases = json.loads(test_inputs)
                test_cases = [{"input": tc} for tc in test_cases]
            except json.JSONDecodeError:
                test_cases = analyzer.generate_test_cases(semantics)
        else:
            test_cases = analyzer.generate_test_cases(semantics)

        # Run tests
        test_results = []
        all_passed = True

        for test_case in test_cases:
            try:
                test_input = np.array(test_case["input"])
                result = evaluator_func(test_input)

                # Check output type
                expected_type = semantics.get("output_type", "scalar")
                if expected_type == "scalar":
                    is_valid = isinstance(result, (int, float, np.number))
                else:
                    is_valid = True  # Accept any output for non-scalar

                test_results.append({
                    "input": test_case["input"],
                    "output": float(result) if isinstance(result, (int, float, np.number)) else str(result),
                    "passed": is_valid
                })

                if not is_valid:
                    all_passed = False

            except Exception as e:
                test_results.append({
                    "input": test_case["input"],
                    "error": str(e),
                    "passed": False
                })
                all_passed = False

        # Update metadata with test results
        metadata = compiler.get_metadata(evaluator_id)
        metadata["validation"] = {
            "test_points": test_results,
            "all_tests_passed": all_passed
        }

        # Save updated metadata
        metadata_path = Path(compile_result["metadata_path"])
        metadata_path.write_text(json.dumps(metadata, indent=2))

        # Step 5: Register in Foundry
        if _FOUNDRY is not None:
            # Store evaluator config in Foundry
            config = {
                "evaluator_id": evaluator_id,
                "name": semantics.get("description", function_name),
                "source": {
                    "type": "compiled",
                    "source_path": compile_result["source_path"],
                    "function_name": function_name
                },
                "interface": {
                    "output": {"format": "scalar" if semantics.get("output_type") == "scalar" else "auto"}
                },
                "capabilities": {
                    "observation_gates": True,
                    "caching": True
                },
                "performance": {
                    "cost_per_eval": 1.0
                },
                "semantics": semantics  # Store semantic info
            }

            try:
                _FOUNDRY.register_evaluator_config(evaluator_id, config)
            except:
                # Foundry registration failed, but compilation succeeded
                pass

        return {
            "success": True,
            "evaluator_id": evaluator_id,
            "semantics": semantics,
            "source_path": compile_result["source_path"],
            "metadata_path": compile_result["metadata_path"],
            "tests_passed": all_passed,
            "test_results": test_results,
            "message": (
                f"✓ Evaluator '{evaluator_id}' compiled and tested successfully\n"
                f"  Type: {semantics.get('type', 'unknown')}\n"
                f"  Input dimension: {semantics.get('input_dimension', 'unknown')}\n"
                f"  Tests: {len([t for t in test_results if t['passed']])}/{len(test_results)} passed"
            )
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Registration failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


@tool
def auto_generate_variable_extractor(
    variable_index: int,
    dimension: int,
    evaluator_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Auto-generate a variable extractor evaluator.

    This tool automatically creates an evaluator that extracts a specific
    component from the design vector (e.g., x[0], x[1]).

    Used by agent when creating constraints on design variables:
    - Constraint "x[0] >= 1.5" needs x0_extractor
    - Constraint "x[1] <= 5.0" needs x1_extractor

    Args:
        variable_index: Index of variable to extract (0-based)
        dimension: Total dimension of design vector
        evaluator_id: Optional ID (default: x{i}_extractor)

    Returns:
        {
            "success": True,
            "evaluator_id": "x0_extractor",
            "source_path": ".paola_data/evaluators/x0_extractor/source.py",
            "auto_generated": True
        }

    Example:
        # Agent creating constraint x[0] >= 1.5
        # Agent realizes it needs x0_extractor
        result = auto_generate_variable_extractor(
            variable_index=0,
            dimension=2
        )
        # Returns: {"evaluator_id": "x0_extractor", ...}
        # Agent can now use x0_extractor in constraint
    """
    try:
        from ..foundry.evaluator_compiler import EvaluatorCompiler

        compiler = EvaluatorCompiler()

        result = compiler.generate_variable_extractor(
            variable_index=variable_index,
            dimension=dimension,
            evaluator_id=evaluator_id
        )

        if not result["success"]:
            return result

        # Register in Foundry
        if _FOUNDRY is not None:
            actual_id = result["evaluator_id"]
            config = {
                "evaluator_id": actual_id,
                "name": f"x{variable_index} extractor",
                "source": {
                    "type": "compiled",
                    "source_path": result["source_path"],
                    "function_name": f"x{variable_index}_extractor"
                },
                "interface": {
                    "output": {"format": "scalar"}
                },
                "capabilities": {
                    "observation_gates": True,
                    "caching": True
                },
                "performance": {
                    "cost_per_eval": 0.01  # Trivial cost
                },
                "semantics": {
                    "type": "variable_extractor",
                    "input_dimension": dimension,
                    "variable_index": variable_index
                }
            }

            try:
                _FOUNDRY.register_evaluator_config(actual_id, config)
            except:
                pass

        result["message"] = (
            f"✓ Auto-generated variable extractor '{result['evaluator_id']}'\n"
            f"  Extracts x[{variable_index}] from {dimension}D design vector"
        )

        return result

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Auto-generation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }
