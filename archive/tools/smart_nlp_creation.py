"""
Smart NLP Problem Creation with Semantic Understanding.

This tool enhances create_nlp_problem by:
1. Using semantic metadata from compiled evaluators
2. Auto-generating variable extractors for constraints
3. Inferring problem structure from evaluator semantics
"""

from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
import re


@tool
def create_nlp_problem_smart(
    problem_id: str,
    objective_evaluator_id: str,
    constraints: List[str],
    bounds: Optional[List[List[float]]] = None,
    objective_sense: str = "minimize"
) -> Dict[str, Any]:
    """
    Create NLP problem with semantic understanding and auto-generation.

    This tool is smarter than create_nlp_problem:
    - Reads semantic metadata from compiled evaluators
    - Auto-generates variable extractors for constraints like "x[0] >= 1.5"
    - Infers bounds from evaluator semantics if not provided
    - Validates constraint specifications against objective structure

    Args:
        problem_id: Unique problem identifier
        objective_evaluator_id: Evaluator ID for objective function
        constraints: List of constraint specifications:
            - Natural language: "x[0] >= 1.5", "x[1] <= 5.0"
            - Evaluator-based: "lift_eval >= 1000", "stress_eval <= 200"
        bounds: Optional variable bounds (inferred from semantics if not provided)
        objective_sense: "minimize" or "maximize"

    Returns:
        {
            "success": True,
            "problem_id": "rosenbrock_constrained",
            "auto_generated_extractors": ["x0_extractor"],
            "dimension": 2,
            "num_constraints": 1
        }

    Example:
        # Agent workflow:
        result = create_nlp_problem_smart(
            problem_id="rosenbrock_constrained",
            objective_evaluator_id="rosenbrock_eval",
            constraints=["x[0] >= 1.5"]  # ← Agent detects this needs x0_extractor
        )

        # Tool automatically:
        # 1. Reads rosenbrock_eval semantics: input_dimension=2
        # 2. Parses constraint: needs x[0]
        # 3. Auto-generates x0_extractor
        # 4. Creates NLP with correct constraint
    """
    try:
        from ..foundry.evaluator_compiler import EvaluatorCompiler
        from ..tools.evaluator_tools import create_nlp_problem
        from ..tools.agentic_registration import auto_generate_variable_extractor

        compiler = EvaluatorCompiler()

        # Step 1: Get objective evaluator semantics
        objective_metadata = compiler.get_metadata(objective_evaluator_id)
        if objective_metadata is None:
            return {
                "success": False,
                "error": f"Evaluator '{objective_evaluator_id}' not found. Use register_evaluator_agentic first."
            }

        objective_semantics = objective_metadata.get("semantics", {})

        # Validate it's an objective function
        eval_type = objective_semantics.get("type", "unknown")
        if eval_type == "variable_extractor":
            return {
                "success": False,
                "error": f"'{objective_evaluator_id}' is a variable extractor, not an objective function"
            }

        # Infer dimension from semantics
        dimension = objective_semantics.get("input_dimension")
        if dimension == "variable":
            # Can't infer, need user input
            if bounds is None:
                return {
                    "success": False,
                    "error": "Cannot infer dimension. Please provide 'bounds' parameter."
                }
            dimension = len(bounds)

        # Infer bounds if not provided
        if bounds is None:
            # Use standard bounds based on function type
            function_name = objective_metadata["origin"]["function_name"]
            if "rosenbrock" in function_name.lower():
                bounds = [[-5, 10]] * dimension
            else:
                # Conservative default
                bounds = [[-10, 10]] * dimension

        # Step 2: Parse constraints and auto-generate extractors
        inequality_constraints = []
        auto_generated_extractors = []

        for constraint_spec in constraints:
            constraint_spec = constraint_spec.strip()

            # Parse constraint specification
            # Pattern 1: x[i] >= value or x[i] <= value
            var_pattern = r'x\[(\d+)\]\s*([<>=]+)\s*([-+]?\d+\.?\d*)'
            var_match = re.match(var_pattern, constraint_spec)

            if var_match:
                # Variable constraint like "x[0] >= 1.5"
                var_index = int(var_match.group(1))
                operator = var_match.group(2)
                value = float(var_match.group(3))

                # Check dimension
                if var_index >= dimension:
                    return {
                        "success": False,
                        "error": f"Constraint references x[{var_index}] but dimension is {dimension}"
                    }

                # Check if extractor exists
                extractor_id = f"x{var_index}_extractor"
                extractor_metadata = compiler.get_metadata(extractor_id)

                if extractor_metadata is None:
                    # Auto-generate extractor
                    gen_result = auto_generate_variable_extractor.invoke({
                        "variable_index": var_index,
                        "dimension": dimension,
                        "evaluator_id": extractor_id
                    })

                    if not gen_result["success"]:
                        return {
                            "success": False,
                            "error": f"Failed to auto-generate {extractor_id}: {gen_result['error']}"
                        }

                    auto_generated_extractors.append(extractor_id)

                # Convert operator
                if operator in [">=", "≥"]:
                    constraint_type = ">="
                elif operator in ["<=", "≤"]:
                    constraint_type = "<="
                else:
                    return {
                        "success": False,
                        "error": f"Unsupported operator '{operator}'. Use >= or <="
                    }

                inequality_constraints.append({
                    "name": f"x{var_index}_{constraint_type.replace('=', '').replace('<', 'lt').replace('>', 'gt')}_{value}",
                    "evaluator_id": extractor_id,
                    "type": constraint_type,
                    "value": value
                })

            else:
                # Pattern 2: evaluator_id >= value or evaluator_id <= value
                eval_pattern = r'(\w+)\s*([<>=]+)\s*([-+]?\d+\.?\d*)'
                eval_match = re.match(eval_pattern, constraint_spec)

                if eval_match:
                    eval_id = eval_match.group(1)
                    operator = eval_match.group(2)
                    value = float(eval_match.group(3))

                    # Validate evaluator exists
                    eval_metadata = compiler.get_metadata(eval_id)
                    if eval_metadata is None:
                        return {
                            "success": False,
                            "error": f"Constraint evaluator '{eval_id}' not found"
                        }

                    # Convert operator
                    if operator in [">=", "≥"]:
                        constraint_type = ">="
                    elif operator in ["<=", "≤"]:
                        constraint_type = "<="
                    else:
                        return {
                            "success": False,
                            "error": f"Unsupported operator '{operator}'. Use >= or <="
                        }

                    inequality_constraints.append({
                        "name": f"{eval_id}_{constraint_type.replace('=', '').replace('<', 'lt').replace('>', 'gt')}_{value}",
                        "evaluator_id": eval_id,
                        "type": constraint_type,
                        "value": value
                    })

                else:
                    return {
                        "success": False,
                        "error": f"Cannot parse constraint '{constraint_spec}'. Use format: 'x[i] >= value' or 'evaluator_id >= value'"
                    }

        # Step 3: Create NLP problem using standard tool
        result = create_nlp_problem.invoke({
            "problem_id": problem_id,
            "objective_evaluator_id": objective_evaluator_id,
            "bounds": bounds,
            "objective_sense": objective_sense,
            "inequality_constraints": inequality_constraints,
            "equality_constraints": None
        })

        if not result["success"]:
            return result

        # Add auto-generation info
        result["auto_generated_extractors"] = auto_generated_extractors
        result["objective_semantics"] = {
            "type": eval_type,
            "dimension": dimension,
            "description": objective_semantics.get("description", "")
        }

        # Enhanced message
        if auto_generated_extractors:
            result["message"] += f"\n\n✓ Auto-generated extractors: {', '.join(auto_generated_extractors)}"

        return result

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Smart NLP creation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }
