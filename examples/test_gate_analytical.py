"""
Test OptimizationGate in non-blocking mode (analytical problems).

Demonstrates:
1. Gate wraps objective/gradient for scipy
2. Scipy runs to completion (no blocking)
3. Gate logs all iterations
4. Agent can review trajectory post-run
"""

import numpy as np
import sys
from pathlib import Path
from scipy.optimize import minimize

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aopt.backends import Rosenbrock
from aopt.optimizers.gate import OptimizationGate
from aopt.callbacks import CallbackManager, EventCapture


def test_gate_non_blocking():
    """Test gate in non-blocking mode with scipy SLSQP."""
    print("=" * 70)
    print("Gate Test: Non-Blocking Mode (Analytical Problem)")
    print("=" * 70)

    # 1. Setup problem
    print("\n1. Setting up 2D Rosenbrock...")
    problem = Rosenbrock(dimension=2)
    x_opt, f_opt = problem.get_optimum()
    print(f"   Known optimum: x* = [{x_opt[0]:.1f}, {x_opt[1]:.1f}], f* = {f_opt:.1f}")

    # 2. Create gate with event capture
    print("\n2. Creating gate (non-blocking mode)...")
    callback_mgr = CallbackManager()
    event_capture = EventCapture()
    callback_mgr.register(event_capture)

    gate = OptimizationGate(
        problem_id="rosenbrock_2d",
        blocking=False,  # Non-blocking for analytical problem
        callback_manager=callback_mgr
    )
    print("   ✓ Gate created in non-blocking mode")
    print("   ✓ Scipy will run to completion")
    print("   ✓ Gate logs iterations for post-run analysis")

    # 3. Wrap functions with gate
    print("\n3. Wrapping objective and gradient...")
    objective_func = gate.wrap_objective(problem.evaluate)
    gradient_func = gate.wrap_gradient(problem.gradient)
    print("   ✓ Functions wrapped with observable wrappers")

    # 4. Run scipy optimization
    print("\n4. Running scipy SLSQP...")
    x0 = np.array([-1.0, 1.0])
    bounds = problem.get_bounds()
    bounds_scipy = list(zip(bounds[0], bounds[1]))

    result = minimize(
        fun=objective_func,
        x0=x0,
        method='SLSQP',
        jac=gradient_func,
        bounds=bounds_scipy,
        options={'ftol': 1e-9, 'maxiter': 100}
    )

    print(f"   ✓ Optimization complete")
    print(f"   Success: {result.success}")
    print(f"   Iterations: {result.nit}")
    print(f"   Function calls: {result.nfev}")
    print(f"   Message: {result.message}")

    # 5. Analyze gate history
    print("\n5. Reviewing gate history...")
    history = gate.get_history()
    print(f"   Total calls intercepted: {len(history)}")

    # Show first few and last few iterations
    print("\n   First 3 iterations:")
    for i in range(min(3, len(history))):
        h = history[i]
        if 'objective' in h:
            print(f"     Iter {h['iteration']:2d}: "
                  f"x = [{h['design'][0]:7.4f}, {h['design'][1]:7.4f}], "
                  f"f = {h['objective']:10.6f}")

    if len(history) > 6:
        print("   ...")

    print(f"\n   Last 3 iterations:")
    for i in range(max(0, len(history) - 3), len(history)):
        h = history[i]
        if 'objective' in h:
            print(f"     Iter {h['iteration']:2d}: "
                  f"x = [{h['design'][0]:7.4f}, {h['design'][1]:7.4f}], "
                  f"f = {h['objective']:10.6f}")

    # 6. Check events
    print("\n6. Checking emitted events...")
    iteration_events = event_capture.get_events_by_type('ITERATION_COMPLETE')
    print(f"   ITERATION_COMPLETE events: {len(iteration_events)}")

    # 7. Verify result
    print("\n7. Verifying convergence...")
    final_x = result.x
    final_f = result.fun

    error_x = np.linalg.norm(final_x - x_opt)
    error_f = abs(final_f - f_opt)

    print(f"   Final design:   x = [{final_x[0]:.6f}, {final_x[1]:.6f}]")
    print(f"   Final objective: f = {final_f:.6e}")
    print(f"   Error in x:      ||x - x*|| = {error_x:.2e}")
    print(f"   Error in f:      |f - f*| = {error_f:.2e}")

    # 8. Gate statistics
    print("\n8. Gate statistics...")
    stats = gate.get_statistics()
    print(f"   Total iterations: {stats['total_iterations']}")
    print(f"   Blocking mode: {stats['blocking_mode']}")
    print(f"   Pauses (blocking only): {stats['n_pauses']}")
    print(f"   Total wait time: {stats['total_wait_time']:.3f}s")

    # 9. Success check
    print("\n9. Test result...")
    success = error_x < 1e-3 and error_f < 1e-6
    if success:
        print("   ✓ TEST PASSED - Scipy converged via gate!")
        print("   ✓ Gate logged all iterations")
        print("   ✓ Agent can review trajectory for analysis")
    else:
        print("   ✗ TEST FAILED - Did not converge")

    print("\n" + "=" * 70)
    print("\nKey Observations:")
    print("- Gate in non-blocking mode: scipy runs uninterrupted")
    print("- All iterations logged for post-run agent analysis")
    print("- Events emitted for each iteration")
    print("- Agent sees complete trajectory, can decide if satisfied")
    print("- If not satisfied, agent can restart with different settings")
    print("=" * 70)

    return success


def test_convergence_analysis():
    """Demonstrate agent analyzing convergence from history."""
    print("\n\n" + "=" * 70)
    print("Agent Analysis: Review Convergence Pattern")
    print("=" * 70)

    # Run optimization
    problem = Rosenbrock(dimension=2)
    gate = OptimizationGate(problem_id="rosenbrock_2d", blocking=False)

    result = minimize(
        fun=gate.wrap_objective(problem.evaluate),
        x0=np.array([-1.0, 1.0]),
        method='SLSQP',
        jac=gate.wrap_gradient(problem.gradient),
        bounds=[(-5, 10), (-5, 10)],
        options={'ftol': 1e-9}
    )

    # Agent analyzes trajectory
    print("\n1. Agent reviews optimization trajectory...")
    history = gate.get_history()

    objectives = [h['objective'] for h in history if 'objective' in h]
    gradients = [h.get('gradient_norm', 0) for h in history if 'gradient_norm' in h]

    print(f"   Total iterations: {len(objectives)}")
    print(f"   Initial objective: {objectives[0]:.6f}")
    print(f"   Final objective: {objectives[-1]:.6f}")
    print(f"   Improvement: {objectives[0] - objectives[-1]:.6f}")

    # Check convergence health
    print("\n2. Agent checks convergence health...")
    if len(gradients) > 0:
        final_grad_norm = gradients[-1]
        print(f"   Final gradient norm: {final_grad_norm:.2e}")

        if final_grad_norm < 1e-6:
            print("   ✓ Strong convergence (gradient norm < 1e-6)")
        elif final_grad_norm < 1e-4:
            print("   ✓ Good convergence (gradient norm < 1e-4)")
        else:
            print("   ⚠ Weak convergence (gradient norm > 1e-4)")

    # Check monotonicity
    print("\n3. Agent checks improvement pattern...")
    improvements = [objectives[i] - objectives[i+1]
                   for i in range(len(objectives)-1)]

    n_improvements = sum(1 for imp in improvements if imp > 0)
    improvement_rate = n_improvements / len(improvements) if improvements else 0

    print(f"   Improving iterations: {n_improvements}/{len(improvements)} ({improvement_rate:.1%})")

    if improvement_rate > 0.8:
        print("   ✓ Consistent progress")
    else:
        print("   ⚠ Non-monotonic (line searches or oscillation)")

    # Agent decision
    print("\n4. Agent decision...")
    x_opt, f_opt = problem.get_optimum()
    error_f = abs(objectives[-1] - f_opt)

    if error_f < 1e-6:
        print("   ✓ ACCEPT: Converged to optimum")
        print("   No further action needed")
    else:
        print("   ⚠ NOT SATISFIED: Far from known optimum")
        print("   Agent could:")
        print("   - Restart with tighter tolerance")
        print("   - Try different optimizer (e.g., L-BFGS-B)")
        print("   - Adjust initial guess")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\nRunning gate tests for analytical problems...\n")

    # Test 1: Basic gate functionality
    success = test_gate_non_blocking()

    # Test 2: Agent analysis pattern
    test_convergence_analysis()

    if success:
        print("\n✓ All gate tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Gate tests failed")
        sys.exit(1)
