"""
LLM-Based Evaluator Semantic Analyzer.

Uses LLM to analyze evaluator code and extract semantic information:
- Input dimensions
- Output type
- Function type (objective, constraint, variable extractor)
- Mathematical properties
- Dependencies
"""

from typing import Dict, Any, Optional
import json


class EvaluatorSemanticAnalyzer:
    """
    Analyzes evaluator code using LLM to extract semantic information.

    This enables the agent to:
    - Understand what the evaluator computes
    - Identify input/output structure
    - Determine appropriate use (objective vs constraint)
    - Auto-generate variable extractors when needed
    """

    def __init__(self, llm):
        """
        Initialize analyzer.

        Args:
            llm: LangChain LLM instance for analysis
        """
        self.llm = llm

    def analyze_function(
        self,
        code: str,
        function_name: str,
        file_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze function code to extract semantic information.

        Args:
            code: Function source code
            function_name: Name of function to analyze
            file_context: Optional full file content for context

        Returns:
            {
                "type": "objective_function" | "constraint_function" | "variable_extractor",
                "description": "Human-readable description",
                "input_dimension": int or "variable",
                "input_variables": ["x[0]", "x[1]", ...],
                "output_type": "scalar" | "vector" | "dict",
                "formula": "Mathematical formula if simple enough",
                "properties": {
                    "convex": bool or None,
                    "smooth": bool or None,
                    "differentiable": bool or None
                },
                "dependencies": ["numpy", "scipy", ...]
            }
        """

        analysis_prompt = f"""Analyze this Python function and extract semantic information for an optimization evaluator.

Function Code:
```python
{code}
```

Function Name: {function_name}

{f"File Context:\n```python\n{file_context}\n```\n" if file_context else ""}

Please analyze and provide:

1. **Type**: What is this function used for?
   - "objective_function": Computes an objective to minimize/maximize
   - "constraint_function": Computes a constraint value
   - "variable_extractor": Simply extracts a design variable component

2. **Description**: One-sentence description of what it computes

3. **Input Structure**:
   - What is the input parameter (usually 'x' or 'design')?
   - What dimension is it? (e.g., 2 for 2D, or "variable" if it works for any dimension)
   - Which components are accessed? (e.g., ["x[0]", "x[1]"])

4. **Output Type**:
   - "scalar": Returns a single number
   - "vector": Returns an array
   - "dict": Returns a dictionary

5. **Formula**: If simple enough, write the mathematical formula (e.g., "(1-x0)^2 + 100*(x1-x0^2)^2")

6. **Properties** (if determinable):
   - Is it convex?
   - Is it smooth (continuously differentiable)?
   - Is it differentiable?

7. **Dependencies**: List of imported modules used (e.g., ["numpy", "scipy"])

IMPORTANT: Respond ONLY with a valid JSON object, no additional text:

{{
  "type": "objective_function",
  "description": "...",
  "input_dimension": 2,
  "input_variables": ["x[0]", "x[1]"],
  "output_type": "scalar",
  "formula": "...",
  "properties": {{
    "convex": false,
    "smooth": true,
    "differentiable": true
  }},
  "dependencies": ["numpy"]
}}
"""

        try:
            # Query LLM
            response = self.llm.invoke(analysis_prompt)

            # Extract content
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)

            # Parse JSON response
            # Handle markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove ```
            response_text = response_text.strip()

            semantics = json.loads(response_text)

            return {
                "success": True,
                "semantics": semantics
            }

        except json.JSONDecodeError as e:
            return {
                "success": False,
                "error": f"Failed to parse LLM response as JSON: {e}",
                "llm_response": response_text if 'response_text' in locals() else "No response"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }

    def infer_variable_extractors_needed(
        self,
        objective_semantics: Dict[str, Any],
        constraint_specs: list
    ) -> list:
        """
        Infer which variable extractors need to be created for constraints.

        Args:
            objective_semantics: Semantic analysis of objective function
            constraint_specs: List of constraint specifications in natural language

        Returns:
            List of variable indices that need extractors

        Example:
            objective_semantics = {"input_dimension": 2, "input_variables": ["x[0]", "x[1]"]}
            constraint_specs = ["x[0] >= 1.5", "x[1] <= 5.0"]
            Returns: [0, 1]  # Need x0_extractor and x1_extractor
        """

        input_dim = objective_semantics.get("input_dimension")
        if input_dim == "variable" or input_dim is None:
            # Can't infer, need user input
            return []

        extractors_needed = []

        for constraint_spec in constraint_specs:
            # Parse constraint like "x[0] >= 1.5"
            constraint_spec = constraint_spec.strip()

            # Simple parsing: look for x[i] pattern
            import re
            match = re.search(r'x\[(\d+)\]', constraint_spec)
            if match:
                index = int(match.group(1))
                if index < input_dim and index not in extractors_needed:
                    extractors_needed.append(index)

        return sorted(extractors_needed)

    def generate_test_cases(
        self,
        semantics: Dict[str, Any],
        num_tests: int = 3
    ) -> list:
        """
        Generate test cases based on semantic analysis.

        Args:
            semantics: Semantic information from analyze_function
            num_tests: Number of test cases to generate

        Returns:
            [
                {"input": [1.0, 1.0], "expected_type": "float"},
                {"input": [0.0, 0.0], "expected_type": "float"},
                ...
            ]
        """

        input_dim = semantics.get("input_dimension")
        output_type = semantics.get("output_type", "scalar")

        if input_dim == "variable":
            # Use 2D for testing
            input_dim = 2

        test_cases = []

        # Test case 1: Zeros
        test_cases.append({
            "input": [0.0] * input_dim,
            "expected_type": output_type
        })

        # Test case 2: Ones
        test_cases.append({
            "input": [1.0] * input_dim,
            "expected_type": output_type
        })

        # Test case 3: Random values
        import random
        test_cases.append({
            "input": [random.uniform(-1, 1) for _ in range(input_dim)],
            "expected_type": output_type
        })

        return test_cases[:num_tests]
