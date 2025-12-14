"""
Test FoundryEvaluator - Day 1 implementation verification.

Tests:
- Loading Python function evaluators
- Evaluation with PAOLA capabilities
- Observation gates
- Caching
- Gradient computation
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from paola.foundry.evaluator import FoundryEvaluator, InterjectionRequested, EvaluationError
from paola.foundry.capabilities import EvaluationObserver, EvaluationCache


class TestFoundryEvaluatorBasics:
    """Test basic FoundryEvaluator functionality."""

    def test_simple_python_function(self):
        """Test loading and evaluating simple Python function."""

        # Create temporary Python file with evaluator
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import numpy as np

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
""")
            temp_file = f.name

        try:
            # Configuration for this evaluator
            config = {
                'evaluator_id': 'test_rosenbrock',
                'name': 'rosenbrock_test',
                'source': {
                    'type': 'python_function',
                    'file_path': temp_file,
                    'callable_name': 'rosenbrock'
                },
                'interface': {
                    'output': {'format': 'scalar'}
                },
                'capabilities': {
                    'observation_gates': False,  # Disable for simple test
                    'caching': False
                },
                'performance': {
                    'cost_per_eval': 1.0
                }
            }

            # Create evaluator
            evaluator = FoundryEvaluator.from_config(config)

            # Evaluate at known point
            design = np.array([1.0, 1.0])
            result = evaluator.evaluate(design)

            # At (1, 1), Rosenbrock = 0
            assert 'objective' in result.objectives
            assert result.objectives['objective'] == pytest.approx(0.0)

        finally:
            os.unlink(temp_file)

    def test_function_returning_dict(self):
        """Test function that returns dict of objectives."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import numpy as np

def multi_obj(x):
    drag = x[0]**2 + x[1]**2
    lift = 0.3 + 0.1 * x[0]
    return {"drag": drag, "lift": lift}
""")
            temp_file = f.name

        try:
            config = {
                'evaluator_id': 'test_multi',
                'name': 'multi_obj_test',
                'source': {
                    'type': 'python_function',
                    'file_path': temp_file,
                    'callable_name': 'multi_obj'
                },
                'interface': {
                    'output': {'format': 'dict', 'keys': ['drag', 'lift']}
                },
                'capabilities': {
                    'observation_gates': False,
                    'caching': False
                },
                'performance': {'cost_per_eval': 1.0}
            }

            evaluator = FoundryEvaluator.from_config(config)

            design = np.array([1.0, 2.0])
            result = evaluator.evaluate(design)

            # Check both objectives present
            assert 'drag' in result.objectives
            assert 'lift' in result.objectives
            assert result.objectives['drag'] == pytest.approx(5.0)  # 1 + 4
            assert result.objectives['lift'] == pytest.approx(0.4)  # 0.3 + 0.1

        finally:
            os.unlink(temp_file)

    def test_function_returning_tuple(self):
        """Test function returning (objectives, constraints)."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import numpy as np

def constrained_func(x):
    obj = {"cost": x[0]**2 + x[1]**2}
    cons = {"limit": x[0] + x[1] - 1.0}
    return obj, cons
""")
            temp_file = f.name

        try:
            config = {
                'evaluator_id': 'test_constrained',
                'name': 'constrained_test',
                'source': {
                    'type': 'python_function',
                    'file_path': temp_file,
                    'callable_name': 'constrained_func'
                },
                'interface': {
                    'output': {'format': 'tuple'}
                },
                'capabilities': {
                    'observation_gates': False,
                    'caching': False
                },
                'performance': {'cost_per_eval': 1.0}
            }

            evaluator = FoundryEvaluator.from_config(config)

            design = np.array([1.0, 2.0])
            result = evaluator.evaluate(design)

            assert 'cost' in result.objectives
            assert result.objectives['cost'] == pytest.approx(5.0)

            assert 'limit' in result.constraints
            assert result.constraints['limit'] == pytest.approx(2.0)  # 1 + 2 - 1

        finally:
            os.unlink(temp_file)


class TestPAOLACapabilities:
    """Test PAOLA built-in capabilities."""

    def test_observation_gates(self):
        """Test observation gates log evaluations."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def simple_func(x):
    return x[0]**2 + x[1]**2
""")
            temp_file = f.name

        try:
            config = {
                'evaluator_id': 'test_observer',
                'name': 'observer_test',
                'source': {
                    'type': 'python_function',
                    'file_path': temp_file,
                    'callable_name': 'simple_func'
                },
                'interface': {'output': {'format': 'scalar'}},
                'capabilities': {
                    'observation_gates': True,  # Enable
                    'caching': False
                },
                'performance': {'cost_per_eval': 1.0}
            }

            evaluator = FoundryEvaluator.from_config(config)

            # Evaluate multiple times
            for i in range(3):
                design = np.array([float(i), float(i)])
                result = evaluator.evaluate(design)

            # Check observer recorded evaluations
            assert evaluator._observer.evaluation_count == 3

        finally:
            os.unlink(temp_file)

    def test_caching(self):
        """Test evaluation caching."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
call_count = 0

def counting_func(x):
    global call_count
    call_count += 1
    return x[0]**2 + x[1]**2
""")
            temp_file = f.name

        try:
            config = {
                'evaluator_id': 'test_cache',
                'name': 'cache_test',
                'source': {
                    'type': 'python_function',
                    'file_path': temp_file,
                    'callable_name': 'counting_func'
                },
                'interface': {'output': {'format': 'scalar'}},
                'capabilities': {
                    'observation_gates': False,
                    'caching': True  # Enable caching
                },
                'performance': {'cost_per_eval': 1.0}
            }

            evaluator = FoundryEvaluator.from_config(config)

            # Evaluate same design twice
            design = np.array([1.0, 2.0])
            result1 = evaluator.evaluate(design)
            result2 = evaluator.evaluate(design)

            # Both should give same result
            assert result1.objectives == result2.objectives

            # Check cache stats
            assert evaluator._cache.hits == 1
            assert evaluator._cache.misses == 1
            assert evaluator._cache.size == 1

        finally:
            os.unlink(temp_file)

    def test_interjection_callback(self):
        """Test interjection through observer callback."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def simple_func(x):
    return x[0]**2
""")
            temp_file = f.name

        try:
            config = {
                'evaluator_id': 'test_interject',
                'name': 'interject_test',
                'source': {
                    'type': 'python_function',
                    'file_path': temp_file,
                    'callable_name': 'simple_func'
                },
                'interface': {'output': {'format': 'scalar'}},
                'capabilities': {
                    'observation_gates': True,
                    'caching': False
                },
                'performance': {'cost_per_eval': 1.0}
            }

            evaluator = FoundryEvaluator.from_config(config)

            # Set callback that triggers interjection when objective > 1.0
            def should_interject(design, result):
                obj_value = list(result.objectives.values())[0]
                return obj_value > 1.0

            evaluator._observer.set_interjection_callback(should_interject)

            # This should NOT trigger (0.25 < 1.0)
            design1 = np.array([0.5])
            result1 = evaluator.evaluate(design1)  # No exception

            # This SHOULD trigger (4.0 > 1.0)
            design2 = np.array([2.0])
            with pytest.raises(InterjectionRequested):
                evaluator.evaluate(design2)

        finally:
            os.unlink(temp_file)


class TestGradientComputation:
    """Test gradient computation methods."""

    def test_finite_difference_gradient(self):
        """Test finite difference gradient computation."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
def quadratic(x):
    return x[0]**2 + 2*x[1]**2
""")
            temp_file = f.name

        try:
            config = {
                'evaluator_id': 'test_gradient',
                'name': 'gradient_test',
                'source': {
                    'type': 'python_function',
                    'file_path': temp_file,
                    'callable_name': 'quadratic'
                },
                'interface': {'output': {'format': 'scalar'}},
                'capabilities': {
                    'observation_gates': False,
                    'caching': False  # Important: disable for gradient computation
                },
                'performance': {'cost_per_eval': 1.0}
            }

            evaluator = FoundryEvaluator.from_config(config)

            design = np.array([1.0, 2.0])
            gradient = evaluator.compute_gradient(design, method='finite_difference')

            # Analytical gradient: [2*x0, 4*x1] = [2, 8]
            assert gradient[0] == pytest.approx(2.0, abs=1e-4)
            assert gradient[1] == pytest.approx(8.0, abs=1e-4)

        finally:
            os.unlink(temp_file)

    def test_user_provided_gradient(self):
        """Test user-provided gradient function."""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("""
import numpy as np

def quadratic(x):
    return x[0]**2 + 2*x[1]**2

def quadratic_grad(x):
    return np.array([2*x[0], 4*x[1]])
""")
            temp_file = f.name

        try:
            config = {
                'evaluator_id': 'test_user_grad',
                'name': 'user_grad_test',
                'source': {
                    'type': 'python_function',
                    'file_path': temp_file,
                    'callable_name': 'quadratic',
                    'gradient_callable': 'quadratic_grad'
                },
                'interface': {
                    'output': {'format': 'scalar'},
                    'gradients': {'available': True}
                },
                'capabilities': {
                    'observation_gates': False,
                    'caching': False
                },
                'performance': {'cost_per_eval': 1.0}
            }

            evaluator = FoundryEvaluator.from_config(config)

            design = np.array([1.0, 2.0])
            gradient = evaluator.compute_gradient(design, method='user_provided')

            # Exact analytical gradient
            assert gradient[0] == pytest.approx(2.0)
            assert gradient[1] == pytest.approx(8.0)

        finally:
            os.unlink(temp_file)


def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("FOUNDRYEVALUATOR DAY 1 TESTS")
    print("=" * 60)

    print("\n[1/3] Testing Basic Functionality...")
    test_basics = TestFoundryEvaluatorBasics()

    test_basics.test_simple_python_function()
    print("  ✓ Simple Python function")

    test_basics.test_function_returning_dict()
    print("  ✓ Function returning dict")

    test_basics.test_function_returning_tuple()
    print("  ✓ Function returning tuple")

    print("\n[2/3] Testing PAOLA Capabilities...")
    test_caps = TestPAOLACapabilities()

    test_caps.test_observation_gates()
    print("  ✓ Observation gates")

    test_caps.test_caching()
    print("  ✓ Evaluation caching")

    test_caps.test_interjection_callback()
    print("  ✓ Interjection callbacks")

    print("\n[3/3] Testing Gradient Computation...")
    test_grad = TestGradientComputation()

    test_grad.test_finite_difference_gradient()
    print("  ✓ Finite difference gradients")

    test_grad.test_user_provided_gradient()
    print("  ✓ User-provided gradients")

    print("\n" + "=" * 60)
    print("✅ ALL DAY 1 TESTS PASSED!")
    print("=" * 60)

    print("\nDay 1 deliverables:")
    print("  ✓ FoundryEvaluator infrastructure")
    print("  ✓ Observation gates")
    print("  ✓ Evaluation caching")
    print("  ✓ Interjection support")
    print("  ✓ Gradient computation")

    print("\nReady for Day 2: Configuration schema and storage")


if __name__ == "__main__":
    run_tests()
