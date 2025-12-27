"""
FoundryEvaluator - Universal evaluator with PAOLA capabilities built-in.

This is the core infrastructure that handles ALL registered evaluators:
- Python functions (Level 0)
- CLI executables (Level 1)
- Complex workflows (Level 2-3)

Built-in PAOLA capabilities:
- Observation gates (before/after every evaluation)
- Interjection points (user/agent can interrupt)
- Evaluation caching (avoid redundant expensive evaluations)
- Cost tracking (monitor computational budget)
- Performance metrics (learn execution patterns)
- Lineage tracking (which runs use this evaluator)
"""

import time
import importlib.util
import subprocess
import json
from typing import Callable, Dict, Any, Optional
from pathlib import Path
import numpy as np

from ..backends.base import EvaluationBackend, EvaluationResult


class InterjectionRequested(Exception):
    """Exception raised when observation gate triggers interjection."""
    pass


class EvaluationError(Exception):
    """Exception raised when evaluation fails with context."""

    def __init__(self, message: str, evaluator_id: str, design: np.ndarray, original_error: Exception):
        self.evaluator_id = evaluator_id
        self.design = design
        self.original_error = original_error
        super().__init__(message)


class FoundryEvaluator(EvaluationBackend):
    """
    Universal evaluator with PAOLA capabilities built-in.

    This class handles ANY registered evaluator through configuration.
    No per-user wrapper code - just configuration-driven execution.

    Example:
        # Create from configuration
        evaluator = FoundryEvaluator(evaluator_id="eval_abc123", foundry=foundry)

        # Evaluate with all PAOLA capabilities
        result = evaluator.evaluate(design)
        # - Observation gates logged
        # - Cache checked/updated
        # - Cost tracked
        # - Interjection point available
    """

    def __init__(
        self,
        evaluator_id: str,
        foundry,  # OptimizationFoundry instance
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize FoundryEvaluator from configuration.

        Args:
            evaluator_id: Unique identifier for this evaluator
            foundry: OptimizationFoundry instance
            config: Optional pre-loaded configuration (otherwise loads from foundry)
        """
        self.evaluator_id = evaluator_id
        self.foundry = foundry

        # Load configuration
        if config is None:
            self.config = foundry.get_evaluator_config(evaluator_id)
        else:
            self.config = config

        # Load user's function based on configuration
        self._user_callable = self._load_user_function()

        # Setup PAOLA capabilities
        from .capabilities import EvaluationObserver, EvaluationCache

        self._observer = None
        if self.config.get('capabilities', {}).get('observation_gates', True):
            self._observer = EvaluationObserver(evaluator_id=evaluator_id)

        self._cache = None
        if self.config.get('capabilities', {}).get('caching', True):
            self._cache = EvaluationCache()

    @classmethod
    def from_config(cls, config: Dict[str, Any], foundry=None):
        """
        Create FoundryEvaluator directly from configuration.

        Useful for testing without foundry database.

        Args:
            config: Evaluator configuration dict
            foundry: Optional foundry instance

        Returns:
            FoundryEvaluator instance
        """
        evaluator_id = config.get('evaluator_id', 'test_evaluator')
        return cls(evaluator_id=evaluator_id, foundry=foundry, config=config)

    def evaluate(self, design: np.ndarray) -> EvaluationResult:
        """
        Evaluate with full PAOLA capabilities.

        Flow:
        1. Pre-evaluation observation gate
        2. Check cache
        3. Call user's function directly
        4. Post-evaluation observation gate (interjection point)
        5. Update cache and metrics
        6. Return result

        Args:
            design: Design vector (numpy array)

        Returns:
            EvaluationResult with objectives, constraints, cost

        Raises:
            EvaluationError: If user's function fails
            InterjectionRequested: If observation gate triggers interjection
        """

        # 1. Pre-evaluation observation gate
        if self._observer:
            self._observer.before_evaluation(design, self.evaluator_id)

        # 2. Check cache
        if self._cache:
            cached = self._cache.get(design)
            if cached is not None:
                return cached

        # 3. Call user's function
        start_time = time.time()

        try:
            raw_result = self._user_callable(design)
        except Exception as e:
            # Wrap with context
            raise EvaluationError(
                f"Evaluation failed for {self.config.get('name', 'unknown')}",
                evaluator_id=self.evaluator_id,
                design=design,
                original_error=e
            ) from e

        execution_time = time.time() - start_time

        # 4. Parse result based on discovered interface
        result = self._parse_result(raw_result, execution_time)

        # 5. Post-evaluation observation gate + interjection point
        if self._observer:
            should_continue = self._observer.after_evaluation(
                design=design,
                result=result,
                execution_time=execution_time,
                evaluator_id=self.evaluator_id
            )

            if not should_continue:
                raise InterjectionRequested(
                    "Observation gate triggered interjection after evaluation"
                )

        # 6. Update cache and metrics
        if self._cache:
            self._cache.store(design, result)

        # Update performance metrics in foundry (if available)
        if self.foundry:
            self.foundry.update_evaluator_performance(
                evaluator_id=self.evaluator_id,
                execution_time=execution_time,
                success=True
            )

        return result

    def _load_user_function(self) -> Callable:
        """
        Import user's function based on configuration.

        Handles different source types:
        - python_function: Direct import from file
        - cli_executable: Create subprocess callable
        - api_endpoint: Create HTTP callable (future)

        Returns:
            Callable that evaluates the design
        """
        source = self.config.get('source', {})
        source_type = source.get('type')

        if source_type == 'python_function':
            return self._load_python_function(source)

        elif source_type == 'cli_executable':
            return self._create_cli_callable(source)

        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    def _load_python_function(self, source: Dict[str, Any]) -> Callable:
        """
        Load Python function from file.

        Args:
            source: Source configuration with file_path and callable_name

        Returns:
            Callable function
        """
        file_path = source.get('file_path')
        callable_name = source.get('callable_name')

        if not file_path or not callable_name:
            raise ValueError("Python function source requires file_path and callable_name")

        # Import module from file
        spec = importlib.util.spec_from_file_location("user_module", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get callable
        if not hasattr(module, callable_name):
            raise AttributeError(
                f"Module {file_path} does not have function '{callable_name}'"
            )

        return getattr(module, callable_name)

    def _create_cli_callable(self, source: Dict[str, Any]) -> Callable:
        """
        Create callable that executes CLI command.

        Args:
            source: Source configuration with command, input/output details

        Returns:
            Callable that runs command and parses output
        """
        command = source.get('command')
        input_file = source.get('input_file', 'input.txt')
        output_file = source.get('output_file', 'output.txt')
        input_format = source.get('input_format', 'text')
        output_format = source.get('output_format', 'text')
        working_dir = source.get('working_directory')

        def cli_evaluator(design: np.ndarray):
            """Execute CLI command with design input."""

            # Convert to Path if working_dir specified
            work_path = Path(working_dir) if working_dir else Path.cwd()
            input_path = work_path / input_file
            output_path = work_path / output_file

            # Write input based on format
            if input_format == 'text':
                np.savetxt(input_path, design)
            elif input_format == 'json':
                with open(input_path, 'w') as f:
                    json.dump(design.tolist(), f)
            else:
                raise ValueError(f"Unsupported input format: {input_format}")

            # Run command
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_path,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"CLI command failed with code {result.returncode}\n"
                    f"stderr: {result.stderr}"
                )

            # Read output based on format
            if output_format == 'text':
                output = float(open(output_path).read().strip())
            elif output_format == 'json':
                with open(output_path) as f:
                    output = json.load(f)
            elif output_format == 'stdout':
                output = float(result.stdout.strip())
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            return output

        return cli_evaluator

    def _parse_result(self, raw_result, execution_time: float) -> EvaluationResult:
        """
        Parse user's return value based on discovered interface.

        Handles multiple return formats:
        - Single scalar: return 0.5
        - Single dict: return {"objective": 0.5}
        - Tuple (obj, cons): return ({"drag": 0.2}, {"lift": 0.5})
        - Tuple (obj, grad): return (0.5, np.array([...]))
        - NumPy scalar: return np.float64(0.5)
        - List/array (MOO): return [f1, f2] or np.array([f1, f2])

        Args:
            raw_result: Raw return value from user's function
            execution_time: Time taken for evaluation

        Returns:
            EvaluationResult with standardized format
        """
        interface = self.config.get('interface', {}).get('output', {})
        output_format = interface.get('format', 'auto')

        # Auto-detect if not specified
        if output_format == 'auto' or not output_format:
            if isinstance(raw_result, tuple):
                output_format = 'tuple'
            elif isinstance(raw_result, dict):
                output_format = 'dict'
            elif isinstance(raw_result, (int, float, np.number)):
                output_format = 'scalar'
            elif isinstance(raw_result, (list, np.ndarray)):
                output_format = 'array'
            else:
                raise ValueError(f"Cannot auto-detect format for type: {type(raw_result)}")

        # Parse based on format
        if output_format == 'scalar':
            objectives = {'objective': float(raw_result)}
            constraints = {}

        elif output_format == 'dict':
            objectives = raw_result
            constraints = {}

        elif output_format == 'tuple':
            # Could be (objectives, constraints) or (objective, gradient)
            if len(raw_result) == 2:
                first, second = raw_result
                if isinstance(second, dict):
                    # (objectives, constraints)
                    objectives = first if isinstance(first, dict) else {'objective': first}
                    constraints = second
                else:
                    # (objective, gradient) - ignore gradient here
                    objectives = first if isinstance(first, dict) else {'objective': first}
                    constraints = {}
            else:
                raise ValueError(f"Unexpected tuple length: {len(raw_result)}")

        elif output_format == 'array':
            # List or numpy array of objective values (for MOO)
            arr = np.asarray(raw_result)
            if arr.ndim == 0:
                # Scalar wrapped in array
                objectives = {'objective': float(arr)}
            elif arr.ndim == 1:
                if len(arr) == 1:
                    # Single-element array - treat as scalar
                    objectives = {'objective': float(arr[0])}
                else:
                    # Multi-objective array: first value is primary "objective"
                    # Also store indexed values for MOO access
                    objectives = {'objective': float(arr[0])}
                    for i, v in enumerate(arr):
                        objectives[f'f{i}'] = float(v)
            else:
                raise ValueError(f"Expected 1D array, got shape {arr.shape}")
            constraints = {}

        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        # Get cost from performance config
        cost = self.config.get('performance', {}).get('cost_per_eval', 1.0)

        return EvaluationResult(
            objectives=objectives,
            constraints=constraints,
            cost=cost,
            metadata={
                'evaluator_id': self.evaluator_id,
                'execution_time': execution_time,
            }
        )

    def compute_gradient(
        self,
        design: np.ndarray,
        method: str = 'auto'
    ) -> np.ndarray:
        """
        Compute gradient of objective.

        Args:
            design: Design point
            method: Gradient method
                   - "auto": Use user's gradient if available, else finite difference
                   - "user_provided": Use user's gradient function
                   - "finite_difference": Numerical approximation

        Returns:
            Gradient array (same shape as design)
        """
        if method == 'auto':
            if self.config.get('interface', {}).get('gradients', {}).get('available', False):
                method = 'user_provided'
            else:
                method = 'finite_difference'

        if method == 'user_provided':
            # Load gradient function
            source = self.config.get('source', {})
            gradient_callable_name = source.get('gradient_callable')

            if not gradient_callable_name:
                raise ValueError("No gradient function specified in configuration")

            # Import module and get gradient function
            file_path = source.get('file_path')
            spec = importlib.util.spec_from_file_location("user_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            gradient_func = getattr(module, gradient_callable_name)
            return gradient_func(design)

        elif method == 'finite_difference':
            return self._finite_difference_gradient(design)

        else:
            raise ValueError(f"Unknown gradient method: {method}")

    def _finite_difference_gradient(
        self,
        design: np.ndarray,
        step_size: float = 1e-6
    ) -> np.ndarray:
        """
        Compute gradient using finite differences.

        Args:
            design: Design point
            step_size: Step size for finite differences

        Returns:
            Gradient array
        """
        n = len(design)
        gradient = np.zeros(n)

        # Base evaluation
        base_result = self.evaluate(design)
        base_obj = list(base_result.objectives.values())[0]  # First objective

        # Finite difference for each variable
        for i in range(n):
            design_plus = design.copy()
            design_plus[i] += step_size

            result_plus = self.evaluate(design_plus)
            obj_plus = list(result_plus.objectives.values())[0]

            gradient[i] = (obj_plus - base_obj) / step_size

        return gradient

    @property
    def supports_gradients(self) -> bool:
        """Whether gradients are available."""
        # Always true because finite difference is always available
        return True

    @property
    def cost_per_evaluation(self) -> float:
        """Computational cost per evaluation."""
        return self.config.get('performance', {}).get('cost_per_eval', 1.0)

    @property
    def domain(self) -> str:
        """Problem domain."""
        return self.config.get('metadata', {}).get('domain', 'user_defined')

    def __str__(self) -> str:
        """String representation."""
        name = self.config.get('name', 'unknown')
        source_type = self.config.get('source', {}).get('type', 'unknown')
        return f"FoundryEvaluator(name={name}, type={source_type}, id={self.evaluator_id})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"FoundryEvaluator(\n"
            f"  evaluator_id={self.evaluator_id},\n"
            f"  name={self.config.get('name')},\n"
            f"  type={self.config.get('source', {}).get('type')},\n"
            f"  cost={self.cost_per_evaluation}\n"
            f")"
        )
