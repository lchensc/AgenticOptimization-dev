# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the design documentation for an **Agentic Optimization Platform** - a next-generation engineering optimization system where an autonomous AI agent controls the optimization process rather than following a fixed, prescribed loop.

### Core Innovation

Unlike traditional optimization platforms (HEEDS, ModeFRONTIER, Dakota, pyOptSparse, FADO) that use fixed control loops with user-configured algorithms, this platform gives full autonomy to an AI agent:

- **Traditional approach**: Platform prescribes the loop, user configures it
- **Agentic approach**: Agent controls everything using tool primitives, composing novel strategies on the fly

The agent continuously observes optimization progress, reasons about numerical health and feasibility patterns, and autonomously adapts the strategy (constraint bounds, gradient methods, exploration control) to achieve reliable convergence.

## Repository Structure

```
AgenticOptimization/
├── docs/
│   ├── agent_controlled_optimization.md  # Detailed technical design
│   └── agentic_optimization_vision.md    # Vision and value proposition
└── CLAUDE.md                             # This file
```

## Key Concepts

### Agent Autonomy vs. Fixed Loops

**Traditional platforms** (all existing software):
```python
# Fixed loop prescribed by platform
for iteration in range(max_iterations):
    design = optimizer.propose()
    result = evaluate(design)
    optimizer.update(result)
    if convergence_criteria_met():
        break
```

**Agentic platform** (this innovation):
```python
# Agent decides everything autonomously
agent = OptimizationAgent(llm="qwen-plus")
goal = "Minimize drag on transonic wing, maintain CL >= 0.5"
tools = platform.get_tools()  # optimizer_*, workflow_*, cache_*, etc.
agent.optimize(goal, tools)
# Agent composes its own strategy, no fixed loop
```

### Tool Primitives Architecture

The platform provides atomic tools that the agent composes into strategies:

**Optimizer Tools**:
- `optimizer_create(algorithm, problem)` - Create optimizer instance
- `optimizer_propose_design()` - Get next design candidate
- `optimizer_update(design, objective, gradient)` - Update optimizer state
- `optimizer_restart_from_checkpoint(new_settings)` - Restart with new configuration
- `optimizer_checkpoint() / restore()` - Save/restore optimizer state

**Evaluation Tools**:
- `workflow_execute(template, design, fidelity)` - Run CFD/FEA simulation
- `workflow_get_template(name)` - Get workflow configuration

**Data Tools**:
- `cache_get(design) / cache_store(design, results)` - Evaluation cache
- `database_query(filter)` - Query optimization history
- `database_find_similar(design)` - Find similar past designs

**Utility Tools**:
- `gradient_compute(design, method)` - Switch between adjoint/finite-difference
- `constraint_adjust_bounds(constraint_id, new_bound)` - Tighten/relax constraints
- `budget_remaining()` - Check remaining computational budget
- `time_elapsed()` - Track optimization time

### Strategic Adaptation Mechanisms

The agent makes informed decisions to adapt strategy based on continuous observation:

1. **Constraint Feasibility Management**
   - Detects repeated constraint violations (e.g., CL = 0.49 when CL >= 0.5 required)
   - Tightens constraints to force optimizer into feasible region
   - Proven pattern for handling optimizer stuck at infeasible boundaries

2. **Gradient Method Switching**
   - Monitors gradient variance/noise levels
   - Switches adjoint ↔ finite-difference when numerical noise detected
   - Proven pattern from AdjointFlow V6 (handles shock oscillations)

3. **Bayesian Exploration Control**
   - Adjusts acquisition functions (EI, UCB, PI) for exploration vs exploitation
   - Uses seed management for deterministic replay
   - Warm-starts surrogate models when changing strategies

4. **Convergence Verification**
   - Automatically runs high-fidelity verification of converged designs
   - Detects false convergence from gradient noise
   - Validates feasibility before terminating

### Evaluation Cache (Critical for Efficiency)

Engineering simulations are 10,000× more expensive than optimizer iterations:
- CFD evaluation: 4-10 CPU hours → $400-$1000
- Gradient (adjoint): 6 CPU hours → $600
- Optimizer iteration: 0.001 hours → $0.10

The cache prevents re-running expensive simulations when the optimizer revisits designs during line search or trust region adjustments.

### Agent Observation Metrics

The agent monitors every iteration:
- **Convergence health**: improvement_rate, gradient_norm, step_size
- **Numerical health**: gradient_variance, condition_number, constraint_activity
- **Optimizer health**: trust_region_size, qp_solver_success, lagrange_multipliers
- **Resource usage**: budget_used, evaluations_count

## Design Documents

### agent_controlled_optimization.md (50KB)

Comprehensive technical design covering:
- Core innovation: Agent autonomy vs prescribed loops
- Comparison with all major existing platforms (HEEDS, ModeFRONTIER, Dakota, pyOptSparse, FADO, Tosca)
- Tool primitives architecture
- Practical adaptation mechanisms and the "replay constraint"
- Agent decision patterns and reasoning examples
- Complete workflow example showing agent in action
- Implementation architecture and phases

**Key sections**:
- Section 4: Detailed comparison with existing software
- Section 7: Tool primitives design
- Section 8: Practical adaptation mechanisms (constraint replay understanding)
- Section 15: Specific agent adaptation strategies with examples
- Section 17: Complete workflow showing agent reasoning and actions

### agentic_optimization_vision.md (7KB)

High-level vision and value propositions:
- 7 core value propositions from first principles analysis
- Paradigm shift from "operator configures loops" to "director sets goals"
- Knowledge accumulation and learning organization concept
- Multi-fidelity intelligence and compositional problem formulation
- Failure mode prevention through predictive analytics

## Implementation Status

**Current state**: Design/documentation phase (no code yet)

**Planned implementation phases**:
1. **Phase 1** (2 months): Core mechanisms - evaluation cache, optimizer wrapper with checkpointing, basic agent with observation
2. **Phase 2** (2 months): Agent intelligence - full observation metrics, constraint management, gradient switching, Bayesian control
3. **Phase 3** (2 months): Validation & extension - real engineering test cases, multiple optimizers, knowledge base, production hardening

## Development Principles

When implementing this platform:

1. **Agent Autonomy First**: Never add a "hook" or "callback" - the agent IS the controller
2. **Tools Not Control Flow**: Platform provides primitives, agent composes strategy
3. **Observable Everything**: Every action must be observable and explainable
4. **Strategic Restarts**: Adaptations are informed restarts from better positions, not unpredictable experiments
5. **Cache Everything**: Simulations are expensive, cache all evaluations
6. **Learn Continuously**: Every optimization adds to the knowledge base

## Key Terminology

- **Fixed loop**: Traditional optimization paradigm where platform controls iteration
- **Tool primitives**: Atomic operations that agent composes into strategies
- **Evaluation cache**: Storage for expensive simulation results (design → objective, gradient, constraints)
- **Strategic restart**: Informed decision to restart optimizer from current best with modified settings
- **Replay constraint**: Limitation that changing optimizer settings (like scaling) changes trajectory, preventing free experimentation
- **Gradient variance**: Metric for detecting numerical noise in gradients
- **Constraint feasibility management**: Agent's ability to detect and fix repeated constraint violations
- **Compositional strategy**: Agent-invented optimization approach combining multiple tools/algorithms

## Architectural Patterns

### Agent-Tool Interaction

```python
# Agent's autonomous reasoning loop (not prescribed)
while not done:
    observation = observe()       # Monitor optimization health
    decision = reason(obs)        # LLM reasoning about what to do
    action = choose_tool(dec)     # Select tool based on decision
    execute(action)               # Use tool primitive
```

### Knowledge Accumulation

```python
# Platform learns patterns across all optimizations
# Example: "Airfoil CL constraints typically undershoot by 2%"
# Later optimization automatically applies this knowledge
# → Tighten CL constraints proactively
```

### Multi-Strategy Composition

```python
# Agent invents strategy never programmed into platform
# Example: Bayesian(10 samples) → SLSQP(2 parallel starts)
#          → Constraint tightening → SLSQP restart → High-fidelity verification
# Total: 45 evaluations, agent decided sequence dynamically
```

## Value Proposition Summary

"The first optimization platform where an AI agent continuously observes optimization progress, detects feasibility and convergence issues, and autonomously adapts strategy (constraint bounds, gradient methods, exploration control) to achieve reliable convergence."

**For Engineers**: Natural language goals instead of algorithm configuration
**For Companies**: 90% success rate (vs 50%), 2-3× faster convergence, knowledge accumulation
**Technical Moat**: Agent autonomy, strategic adaptation, evaluation cache, proven patterns from AdjointFlow
