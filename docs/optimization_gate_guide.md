# Optimization Gate: Iteration-Level Agent Control

## Overview

The **OptimizationGate** enables the agent to observe and control optimization at the iteration level by intercepting scipy's function calls. This solves the fundamental problem that scipy algorithms don't expose iteration-level control.

## The Problem

Scipy optimizers (SLSQP, L-BFGS-B, COBYLA) are black boxes:
```python
# Scipy runs to completion internally - no way to pause and inspect
result = scipy.optimize.minimize(fun=f, x0=x0, method='SLSQP')
# Agent only sees final result - can't observe or control during optimization
```

## The Solution: Gate Pattern

The gate intercepts **function calls** (objective and gradient), which scipy must call repeatedly:

```python
gate = OptimizationGate(problem_id="my_problem", blocking=False)

# Wrap functions with gate
result = scipy.optimize.minimize(
    fun=gate.wrap_objective(objective_func),    # Gate intercepts here
    jac=gate.wrap_gradient(gradient_func),      # And here
    x0=x0,
    method='SLSQP'
)

# Agent can now review complete trajectory
history = gate.get_history()
```

**Key insight**: Scipy waits inside the objective function for our wrapped function to return. The gate controls *when* to return.

## Two Modes: Economics-Driven Design

### Mode 1: Non-Blocking (Analytical Problems)

**When to use:**
- Function evaluation is **cheap** (~$0.01, microseconds)
- Examples: Rosenbrock, Sphere, analytical benchmarks, linear programming
- Optimizer overhead dominates evaluation cost

**Behavior:**
```python
gate = OptimizationGate(blocking=False)  # Non-blocking

result = minimize(
    fun=gate.wrap_objective(f),
    jac=gate.wrap_gradient(g),
    x0=x0,
    method='SLSQP'
)
# Scipy runs to completion (no pauses)
# Gate logs all iterations

# Agent reviews trajectory POST-RUN
history = gate.get_history()
if not agent_satisfied(history):
    # Restart with different settings
    result = minimize(fun=..., x0=better_x0, method='L-BFGS-B')
```

**Economics:**
- Evaluation: $0.01
- Agent overhead: $0.10 per decision
- Blocking at each iteration: **10× cost increase** ❌
- Post-run review: **1× original cost** ✓

**Use cases:**
- Testing optimizer behavior
- Rapid prototyping
- Benchmarking
- Problems with cheap function evaluations

### Mode 2: Blocking (Engineering Problems)

**When to use:**
- Function evaluation is **expensive** (~$500, hours of compute)
- Examples: CFD (SU2), FEA (Abaqus), molecular dynamics
- Single evaluation dominates total cost

**Behavior:**
```python
gate = OptimizationGate(blocking=True)  # Blocking mode

# Start optimization in separate thread/process
optimizer_thread = threading.Thread(
    target=run_optimization,
    args=(gate,)
)
optimizer_thread.start()

# Agent monitors in real-time
while optimizer_running:
    # Gate pauses inside objective function, waiting for agent
    latest = gate.get_latest()

    # Agent observes and decides
    if should_continue(latest):
        gate.agent_continue()       # Resume scipy
    elif should_stop(latest):
        gate.agent_stop("reason")   # Stop immediately
    elif should_restart(latest):
        gate.agent_restart(...)     # Restart with new settings
```

**Economics:**
- Evaluation: $500 (4 hours)
- Agent decision: $0.10 (1 minute)
- Cost to wait for agent: **$0.42** (1 minute of idle cluster time)
- Total overhead per iteration: **$0.52 = 0.1% of evaluation cost** ✓

**Use cases:**
- CFD optimization (airfoil drag minimization)
- FEA optimization (structural compliance)
- Expensive black-box functions
- Long-running simulations where early stopping saves significant cost

## Architecture Details

### How the Gate Works

```python
def observable_objective(x):
    """What scipy calls - doesn't know gate exists."""

    # 1. Evaluate objective (with caching)
    obj_value = objective_func(x)

    # 2. Log iteration
    gate.history.append({'design': x, 'objective': obj_value})

    # 3. Emit event for agent observation
    gate.callback_manager.emit(ITERATION_COMPLETE, data={...})

    # 4. BLOCKING MODE ONLY: Wait for agent decision
    if gate.blocking:
        gate._continue_event.wait(timeout=300)  # Blocks here

        if gate.current_action == STOP:
            raise StopOptimizationSignal()
        elif gate.current_action == RESTART:
            raise RestartOptimizationSignal(...)

    # 5. Return to scipy (or raise exception)
    return obj_value
```

### Control Flow

**Non-blocking:**
```
Scipy → objective(x1) → [gate logs] → return f(x1) → Scipy
     → objective(x2) → [gate logs] → return f(x2) → Scipy
     → ...
     → Complete
Agent reviews trajectory → decides → possibly restarts
```

**Blocking:**
```
Scipy → objective(x1) → [gate logs] → [PAUSE]
                                        ↓
                                   Agent observes
                                        ↓
                                   Agent decides: continue
                                        ↓
                          return f(x1) → Scipy
     → objective(x2) → [gate logs] → [PAUSE]
                                        ↓
                                   Agent observes
                                        ↓
                                   Agent decides: stop
                                        ↓
                          raise StopSignal → Caught by outer loop
```

## Agent Decision Making

### Non-Blocking Pattern

```python
# Run optimization
result = minimize(fun=gate.wrap_objective(f), ...)

# Agent analyzes trajectory
history = gate.get_history()
objectives = [h['objective'] for h in history]

# Decision logic
if converged_to_optimum(objectives):
    return result  # Accept

elif stuck_at_infeasible(history):
    # Restart with tightened constraints
    return restart_with_modified_constraints(...)

elif gradient_noise_detected(history):
    # Switch to derivative-free method
    return minimize(fun=f, method='COBYLA', ...)
```

### Blocking Pattern

```python
# Gate control tools for agent
@tool
def optimizer_continue(gate_id: str):
    """Agent decision: continue optimization."""
    gate = get_gate(gate_id)
    gate.agent_continue()  # Unblocks scipy

@tool
def optimizer_stop(gate_id: str, reason: str):
    """Agent decision: stop optimization."""
    gate = get_gate(gate_id)
    gate.agent_stop(reason)  # Scipy raises exception

@tool
def optimizer_restart(gate_id: str, from_design: list, new_options: dict):
    """Agent decision: restart with new settings."""
    gate = get_gate(gate_id)
    gate.agent_restart(from_design, new_options)
```

## Integration with ReAct Agent

The gate integrates seamlessly with the ReAct loop:

```python
# Agent receives tool: run_optimization
@tool
def run_optimization(problem_id: str, algorithm: str, blocking: bool = False):
    """Run optimization with gate control."""

    # Create gate
    gate = OptimizationGate(
        problem_id=problem_id,
        blocking=blocking,
        callback_manager=agent.callback_manager
    )

    if blocking:
        # Start in background thread
        thread = start_optimization_thread(gate, algorithm)
        return {
            "success": True,
            "gate_id": gate.id,
            "message": "Optimization started. Use optimizer_continue/stop/restart tools."
        }
    else:
        # Run to completion
        result = run_scipy(gate, algorithm)
        history = gate.get_history()
        return {
            "success": True,
            "result": result,
            "history": history,
            "message": "Optimization complete. Review trajectory to decide next action."
        }
```

**Agent's ReAct loop naturally handles both:**
- Non-blocking: Reviews result, reasons, calls restart tool if needed
- Blocking: Receives iteration event, reasons, calls continue/stop/restart

No special logic needed - gate just provides the pause points and control mechanism.

## Performance Considerations

### Caching Integration

The gate integrates with evaluation cache automatically:

```python
def observable_objective(x):
    # Check cache first
    cached = cache_get(design=x, problem_id=gate.problem_id)
    if cached:
        obj_value = cached['objective']
        # Still log and emit event, but no recomputation
    else:
        obj_value = expensive_simulation(x)
        cache_store(design=x, objective=obj_value, ...)

    # Rest of gate logic...
```

This prevents re-evaluating designs during:
- Line searches (scipy tries multiple step sizes)
- Restarts (agent restarts from previously evaluated point)
- Multiple runs (cross-optimization caching)

### Threading Model (Blocking Mode)

```python
# Main thread: ReAct agent loop
agent.run(goal="Minimize drag on airfoil")

# Background thread: Scipy optimization
def optimization_worker():
    result = minimize(fun=gate.wrap_objective(cfd_run), ...)

# Gate bridges the two via threading.Event
# Scipy thread blocks waiting for agent decisions
# Agent thread makes decisions and signals gate
```

## Examples

### Example 1: Analytical Problem

```python
from aopt.optimizers import OptimizationGate
from aopt.backends import Rosenbrock

problem = Rosenbrock(dimension=2)
gate = OptimizationGate(problem_id="rosenbrock", blocking=False)

result = minimize(
    fun=gate.wrap_objective(problem.evaluate),
    jac=gate.wrap_gradient(problem.gradient),
    x0=[-1, 1],
    method='SLSQP'
)

# Agent reviews
history = gate.get_history()
print(f"Converged in {len(history)} iterations")
print(f"Final objective: {result.fun}")
```

### Example 2: Engineering Problem

```python
from aopt.optimizers import OptimizationGate

gate = OptimizationGate(problem_id="airfoil_cfd", blocking=True)

# Start optimization in thread
import threading
result_container = {}

def run_opt():
    try:
        result = minimize(
            fun=gate.wrap_objective(run_su2_cfd),
            jac=gate.wrap_gradient(run_su2_adjoint),
            x0=initial_airfoil,
            method='SLSQP'
        )
        result_container['result'] = result
    except StopOptimizationSignal as e:
        result_container['stopped'] = e.reason

opt_thread = threading.Thread(target=run_opt)
opt_thread.start()

# Agent monitors and controls
while opt_thread.is_alive():
    time.sleep(60)  # Check every minute

    latest = gate.get_latest()
    if latest:
        print(f"Iteration {latest['iteration']}: f = {latest['objective']}")

        # Agent decision logic
        if gradient_exploded(latest):
            gate.agent_stop("Numerical instability detected")
        elif should_continue:
            gate.agent_continue()

opt_thread.join()
```

## Summary

| Aspect | Non-Blocking (Analytical) | Blocking (Engineering) |
|--------|---------------------------|------------------------|
| **Use case** | Cheap evaluations | Expensive evaluations |
| **Economics** | Avoid agent overhead | Agent overhead negligible |
| **Control timing** | Post-run review | Real-time interception |
| **Scipy behavior** | Runs to completion | Pauses at each iteration |
| **Agent pattern** | Review → Restart if needed | Observe → Continue/Stop/Restart |
| **Examples** | Rosenbrock, benchmarks | CFD, FEA, MD simulations |

The gate pattern provides **true iteration-level control** without modifying scipy, adapting to problem economics for optimal efficiency.
