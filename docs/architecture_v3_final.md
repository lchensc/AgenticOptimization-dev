# AOpt Architecture v3: Complete AI-Centric Platform

**Platform Name**: `aopt` (Agentic Optimization)
**Paper Title**: "Agentic Optimization for Engineering Design"
**Mission**: Build the best AI agent for engineering/science optimization

**Milestone 1 Scope**: Nonlinear + Multi-objective optimization with full ReAct agent loop

**Date**: December 10, 2025

---

## 1. Platform Vision

### Core Innovations

1. **Intelligent Formulation**: Agent converts natural language goals → structured optimization problems
2. **Tool-Based Execution**: Agent composes strategies from primitives (no fixed loops)
3. **Autonomous Adaptation**: Agent observes, reasons, and adapts mid-optimization
4. **Extensible Design**: Built for future problem types (integer, stochastic, etc.)

### What AOpt Provides

```python
class AOpt:
    """
    AI-Centric Optimization Platform

    Components:
    1. Formulation Engine - convert goals to optimization problems
    2. Intelligent Agent - ReAct loop with domain knowledge
    3. Tool Primitives - 15 composable tools
    4. Solver Library - optimizers for different problem types
    """

    def __init__(self):
        # Intelligent Agent (ReAct loop)
        self.agent = ReactAgent(
            llm="claude-sonnet-4-5",
            tools=self.get_all_tools()
        )

        # Formulation + Execution tools
        self.tools = self.get_all_tools()
```

---

## 2. System Architecture (Complete Loop)

```
┌────────────────────────────────────────────────────────┐
│                 USER INTERACTION                       │
│  goal = "Minimize drag on airfoil, maintain CL >= 0.8"│
└────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────┐
│         REACT AGENT (Continuous Loop)                  │
│                                                        │
│  while not done:                                       │
│    ┌─────────────────────────────────────────────┐   │
│    │  1. FORMULATE (if needed)                   │   │
│    │     → formulate_problem(goal)               │   │
│    │     → analyze_problem_structure()           │   │
│    │                                             │   │
│    │  2. STRATEGIZE                              │   │
│    │     → recommend_optimizers()                │   │
│    │     → optimizer_create()                    │   │
│    │                                             │   │
│    │  3. EXECUTE ITERATION                       │   │
│    │     → optimizer_propose()                   │   │
│    │     → evaluate_function()                   │   │
│    │     → optimizer_update()                    │   │
│    │                                             │   │
│    │  4. OBSERVE                                 │   │
│    │     → query_history()                       │   │
│    │     → analyze_convergence()                 │   │
│    │                                             │   │
│    │  5. ADAPT (if needed)                       │   │
│    │     → modify_constraints()                  │   │
│    │     → switch_gradient_method()              │   │
│    │     → optimizer_restart()                   │   │
│    └─────────────────────────────────────────────┘   │
│                                                        │
│  Agent decides when to stop (converged or satisfied)   │
└────────────────────────────────────────────────────────┘
                         ↓ uses
┌────────────────────────────────────────────────────────┐
│              TOOL PRIMITIVES (18 Tools)                │
│                                                        │
│  Formulation (3):      Problem Analysis & Setup       │
│  Optimizer (4):        Create, Propose, Update, Restart│
│  Evaluator (2):        Function, Gradient             │
│  Cache/Provenance (3): Get, Store, Log                │
│  Observer (3):         History, Convergence, Patterns │
│  Adapter (2):          Constraints, Gradient Method   │
│  Resource (1):         Budget                         │
└────────────────────────────────────────────────────────┘
                         ↓
┌────────────────────────────────────────────────────────┐
│              SOLVER BACKENDS                           │
│  • Nonlinear: SLSQP, COBYLA (scipy)                   │
│  • Multi-objective: NSGA-II, MOEA/D (pymoo)           │
│  • Future: Integer (MILP), Stochastic, etc.           │
└────────────────────────────────────────────────────────┘
```

---

## 3. Problem Formulation (Milestone 1)

### 3.1 Problem Schema (Extensible)

```python
from pydantic import BaseModel
from typing import Literal, Optional

class OptimizationProblem(BaseModel):
    """
    Universal problem schema - extensible for future types.
    Milestone 1: Focus on nonlinear + multi-objective.
    """

    # Problem classification
    problem_type: Literal[
        "nonlinear_single",      # Milestone 1 ✓
        "nonlinear_multi",       # Milestone 1 ✓
        "linear",                # Future
        "mixed_integer",         # Future
        "stochastic",            # Future
        "robust"                 # Future
    ]

    # Objectives (1 for single, multiple for multi-objective)
    objectives: list[Objective]
    # [
    #   {"name": "drag", "sense": "minimize"},
    #   {"name": "weight", "sense": "minimize"}  # Multi-objective
    # ]

    # Design variables
    variables: list[Variable]
    # [
    #   {"name": "x1", "type": "continuous", "bounds": [0, 10]},
    #   {"name": "x2", "type": "continuous", "bounds": [-5, 5]}
    # ]

    # Constraints
    constraints: list[Constraint] = []
    # [
    #   {"name": "lift", "type": "inequality", "expr": "CL - 0.8 >= 0"}
    # ]

    # Problem properties (inferred)
    properties: dict = {}
    # {"convex": False, "smooth": True, "expensive": False}


class Objective(BaseModel):
    name: str
    sense: Literal["minimize", "maximize"]


class Variable(BaseModel):
    name: str
    type: Literal["continuous", "integer", "binary"] = "continuous"
    bounds: tuple[float, float]
    initial: Optional[float] = None


class Constraint(BaseModel):
    name: str
    type: Literal["equality", "inequality"]
    expression: str
```

**Key**: Schema is extensible - easy to add integer, stochastic, etc. later.

---

## 4. Complete Tool Set (15 Tools)

### 4.1 Formulation Tools (3 tools)

```python
@tool
def formulate_problem(goal: str, context: dict = {}) -> dict:
    """
    Convert natural language goal to structured problem. May return
    incomplete formulation with clarification questions for user.

    Agent uses LLM reasoning to:
    - Identify problem type (single/multi-objective, nonlinear)
    - Extract objectives (minimize drag, maximize efficiency, etc.)
    - Infer variables (how many? what bounds?)
    - Extract constraints (CL >= 0.8, etc.)

    Returns:
        {
            "problem": OptimizationProblem,  # May be partial
            "confidence": 0.7,  # How confident in formulation
            "questions": [  # Optional: need user input
                "What is acceptable range for weight?",
                "Should I prioritize cost or performance?"
            ]
        }
    """


@tool
def analyze_problem_structure(problem: dict) -> dict:
    """
    Analyze mathematical properties of problem.

    Returns: {
        "complexity": "medium",
        "is_convex": False,
        "is_smooth": True,
        "recommended_optimizers": ["SLSQP", "NSGA-II"]
    }
    """


@tool
def recommend_optimizers(problem: dict) -> list[dict]:
    """
    Recommend optimizers based on problem type.

    Returns: [
        {"name": "SLSQP", "suitability": 0.9, "reason": "..."},
        {"name": "NSGA-II", "suitability": 0.7, "reason": "..."}
    ]
    """
```

### 4.2 Optimizer Tools (4 tools)

```python
@tool
def optimizer_create(
    algorithm: str,
    problem: dict,
    settings: dict = {}
) -> str:
    """Create optimizer. Returns optimizer_id."""


@tool
def optimizer_propose(optimizer_id: str) -> list[float]:
    """Get next design to evaluate."""


@tool
def optimizer_update(
    optimizer_id: str,
    design: list[float],
    objectives: list[float],  # Can be multi-objective!
    gradient: Optional[list[float]] = None,
    constraints: Optional[dict] = None
) -> dict:
    """Update optimizer with results. Returns status."""


@tool
def optimizer_restart(
    optimizer_id: str,
    new_problem: dict,
    restart_from: Literal["best", "current", "custom"] = "best",
    custom_design: Optional[list[float]] = None,
    reuse_cache: bool = True,
    checkpoint_old: bool = True,
    recompute_gradient: bool = True
) -> dict:
    """
    Safely restart optimizer with modified problem.

    Safety mechanisms:
    1. Can restart from best design found so far (not random!)
    2. Reuses evaluation cache (no wasted evaluations)
    3. Checkpoints old optimizer state (enables rollback)
    4. Recomputes gradient at restart point if problem changed

    Returns:
        {
            "new_optimizer_id": "opt_002",
            "restart_design": [0.5, 0.3, ...],
            "restart_objectives": [0.0245],
            "old_checkpoint_id": "ckpt_001",  # For rollback
            "cache_entries_reused": 145
        }
    """
```

### 4.3 Evaluator Tools (2 tools)

```python
@tool
def evaluate_function(
    design: list[float],
    problem: dict,
    objectives: list[str],  # ["drag", "weight"]
    gradient: bool = False,
    constraints: bool = False
) -> dict:
    """
    Evaluate objective(s) and optionally gradient/constraints.

    Returns: {
        "objectives": [0.0245, 12.5],  # Multi-objective values
        "gradient": [...],  # Optional
        "constraints": {...},  # Optional
        "cost": 0.0  # CPU hours
    }
    """


@tool
def compute_gradient(
    design: list[float],
    problem: dict,
    objective: str,
    method: str = "finite_difference"
) -> list[float]:
    """Compute gradient for specific objective."""
```

### 4.4 Cache/Provenance Tools (3 tools) - CRITICAL FOR EFFICIENCY

```python
@tool
def cache_get(
    design: list[float],
    problem_id: str,
    tolerance: float = 1e-9
) -> Optional[dict]:
    """
    Retrieve cached evaluation result for design.

    Prevents re-evaluation during line searches and population duplicates.
    Engineering simulations: 10,000× more expensive than optimizer iterations.

    Returns:
        {
            "objectives": [0.0245],
            "gradient": [...],  # If available
            "constraints": {...},  # If available
            "cost": 0.5,  # CPU hours
            "hit": True
        }

        Returns None if not in cache.
    """


@tool
def cache_store(
    design: list[float],
    problem_id: str,
    objectives: list[float],
    gradient: Optional[list[float]] = None,
    constraints: Optional[dict] = None,
    cost: float = 0.0
) -> dict:
    """
    Store evaluation result in cache.

    Returns:
        {"stored": True, "cache_size": 245, "duplicate": False}
    """


@tool
def run_db_log(
    optimizer_id: str,
    iteration: int,
    design: list[float],
    objectives: list[float],
    action: str,  # "evaluate", "adapt", "restart"
    reasoning: str
) -> dict:
    """
    Log optimization run for provenance and knowledge accumulation.

    Enables:
    - Run replay and debugging
    - Pattern detection across runs
    - Knowledge base learning

    Returns:
        {"logged": True, "run_id": "run_001", "entry_id": 12}
    """
```

### 4.5 Observer Tools (3 tools)

```python
@tool
def query_history(
    optimizer_id: str,
    last_n: Optional[int] = None
) -> list[dict]:
    """Query optimization history."""


@tool
def analyze_convergence(optimizer_id: str) -> dict:
    """
    Analyze convergence health.

    Returns: {
        "gradient_variance": 0.02,
        "improvement_rate": -0.001,
        "converged": False,
        "pareto_front_quality": 0.85  # For multi-objective
    }
    """


@tool
def detect_pattern(
    optimizer_id: str,
    pattern_type: str
) -> Optional[dict]:
    """Detect optimization patterns (violations, noise, etc.)."""
```

### 4.6 Adapter Tools (2 tools)

```python
@tool
def modify_constraints(
    problem: dict,
    constraint_id: str,
    new_bound: float,
    reasoning: str
) -> dict:
    """Modify constraint bounds. Returns updated problem."""


@tool
def switch_gradient_method(
    problem: dict,
    method: str
) -> dict:
    """Switch gradient method. Returns updated problem."""
```

### 4.7 Resource Tool (1 tool)

```python
@tool
def budget_remaining() -> dict:
    """Check remaining computational budget."""
```

**Total: 18 Tools** - Agent composes complete optimization workflows from primitives

---

## 5. ReAct Agent Implementation

### 5.1 Agent Structure (LangGraph)

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    """Agent working memory."""
    messages: Annotated[list, operator.add]  # Conversation
    context: dict  # Current situation
    done: bool  # Agent decides when done


def build_aopt_agent(tools: list):
    """
    Build ReAct agent for optimization.

    Simple continuous loop - agent decides everything.
    """

    workflow = StateGraph(AgentState)

    # Single node: ReAct step
    workflow.add_node("react", create_react_node(tools))

    # Entry point
    workflow.set_entry_point("react")

    # Loop or terminate?
    workflow.add_conditional_edges(
        "react",
        lambda state: "end" if state["done"] else "continue",
        {
            "continue": "react",
            "end": END
        }
    )

    return workflow.compile()


def create_react_node(tools):
    """
    ReAct node: reason → act → observe.
    CRITICAL: Maintains full conversation history for grounding.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, ToolMessage

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    llm_with_tools = llm.bind_tools(tools)

    def react_step(state: AgentState) -> dict:
        """
        Execute one ReAct cycle with full history retention.

        FIXED: Accumulates all messages (user + assistant + tool results)
        to maintain grounding, tool threading, and termination detection.
        """
        context = state["context"]

        # Build prompt with current context
        prompt = build_optimization_prompt(context)

        # FIXED: Preserve full history + add new user prompt
        messages = state["messages"] + [HumanMessage(content=prompt)]

        # Get LLM decision (reasoning + tool calls)
        response = llm_with_tools.invoke(messages)

        # FIXED: Collect all new messages from this turn
        new_messages = [response]

        # Execute tool calls and collect results
        if response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                result = execute_tool(tool_call, tools)
                tool_results.append(result)
                # FIXED: Add tool result message (maintains threading)
                new_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    )
                )

            # Update context with tool results
            new_context = update_context(context, tool_results)

            # FIXED: Append new messages (operator.add accumulates)
            return {
                "messages": new_messages,
                "context": new_context,
                "done": False
            }

        # Check if agent says done
        if "DONE" in response.content.upper() or "CONVERGED" in response.content.upper():
            return {
                "messages": new_messages,  # FIXED: Preserve history
                "context": context,
                "done": True
            }

        # Agent just reasoning, continue
        return {
            "messages": new_messages,  # FIXED: Preserve history
            "context": context,
            "done": False
        }

    return react_step


def build_optimization_prompt(context: dict) -> str:
    """
    Build prompt with current optimization state.
    UPDATED: Now includes budget awareness and cache statistics.
    """
    # Get budget status
    budget_status = context.get('budget_status', {})
    budget_text = f"{budget_status.get('used', 0):.1f} / {budget_status.get('total', 'Unknown')} CPU hours"
    budget_remaining_pct = budget_status.get('remaining_pct', 100)

    # Get cache stats
    cache_stats = context.get('cache_stats', {})
    cache_hit_rate = cache_stats.get('hit_rate', 0.0)

    return f"""
You are an autonomous optimization agent specialized in engineering/science.

**Current Goal:**
{context.get('goal', 'Not set yet')}

**Current Problem Formulation:**
{format_problem(context.get('problem', {}))}

**Optimization Status:**
- Optimizer: {context.get('optimizer_type', 'Not created')}
- Iteration: {context.get('iteration', 0)}
- Current objective(s): {context.get('current_objectives', 'Not evaluated')}
- Best objective(s): {context.get('best_objectives', 'N/A')}

**Resource Status:**
- Budget: {budget_text} ({budget_remaining_pct:.0f}% remaining)
- Cache hit rate: {cache_hit_rate:.1%} (higher = more efficient)
- Total evaluations: {context.get('total_evaluations', 0)}

**Recent History (last 5 iterations):**
{format_history(context.get('history', [])[-5:])}

**Convergence Analysis:**
{format_observations(context.get('observations', {}))}

**Available Tools (18 total):**
{format_tools()}

**Your Task:**
Decide the next action autonomously. You have full control.

Strategy considerations:
1. Use cache_get before expensive evaluations to check for cached results
2. Monitor budget - if low, consider stopping or reducing evaluations
3. Observe convergence regularly (analyze_convergence)
4. Adapt if stuck (modify_constraints, optimizer_restart with safety)

If you haven't formulated the problem yet, start with formulate_problem().
Then create optimizer, execute iterations, observe, adapt as needed.

Stop when:
- Converged (gradient norm < 1e-6, no improvement)
- Budget exhausted
- Agent satisfied with result

Think step-by-step, then use a tool or respond "DONE".
"""
```

---

## 6. Example: Agent Workflow (Nonlinear Single-Objective)

```python
import aopt

agent = aopt.Agent(llm_model="claude-sonnet-4-5")

result = agent.run("""
Minimize the 10-dimensional Rosenbrock function.
Bounds: [-5, 5] for all variables.
""")

# Agent's autonomous execution:

# === FORMULATION PHASE ===
Agent: "I need to formulate this problem first."

Action: formulate_problem(goal="Minimize 10D Rosenbrock, bounds [-5,5]")
→ problem = {
    "problem_type": "nonlinear_single",
    "objectives": [{"name": "rosenbrock", "sense": "minimize"}],
    "variables": [
        {"name": f"x{i}", "type": "continuous", "bounds": [-5, 5]}
        for i in range(10)
    ],
    "constraints": []
}

Agent: "Analyze problem structure..."

Action: analyze_problem_structure(problem)
→ {
    "complexity": "medium",
    "is_convex": False,
    "is_smooth": True,
    "recommended_optimizers": ["SLSQP", "L-BFGS-B"]
}

# === STRATEGIZE PHASE ===
Agent: "Get optimizer recommendations..."

Action: recommend_optimizers(problem)
→ [
    {"name": "SLSQP", "suitability": 0.9},
    {"name": "L-BFGS-B", "suitability": 0.85}
]

Agent: "I'll use SLSQP - best for this problem."

Action: optimizer_create(algorithm="SLSQP", problem=problem)
→ optimizer_id = "opt_001"

# === EXECUTION PHASE ===
Agent: "Get first design..."

Action: optimizer_propose(optimizer_id="opt_001")
→ design_001 = [0, 0, 0, ..., 0]

Agent: "Evaluate with gradient..."

Action: evaluate_function(
    design=design_001,
    problem=problem,
    objectives=["rosenbrock"],
    gradient=True
)
→ {"objectives": [9.0], "gradient": [...], "cost": 0.0}

Agent: "Update optimizer..."

Action: optimizer_update(
    optimizer_id="opt_001",
    design=design_001,
    objectives=[9.0],
    gradient=[...]
)

# ... Iterations 2-44: Similar ...

# === OBSERVATION PHASE (after iteration 20) ===
Agent: "Let me check convergence..."

Action: analyze_convergence(optimizer_id="opt_001")
→ {
    "gradient_variance": 0.01,
    "improvement_rate": -0.05,
    "converged": False
}

Agent: "Good progress. Continue."

# ... Iterations 21-45: Continue ...

# === TERMINATION ===
Agent (iteration 45): "Objective is 1.2e-8, gradient norm 3e-9.
                       Converged to global minimum.
                       DONE."

# Result returned to user
result = {
    "converged": True,
    "final_objectives": [1.2e-8],
    "final_design": [1.0, 1.0, ..., 1.0],
    "iterations": 45,
    "reasoning_log": [...]  # Full agent reasoning
}
```

---

## 7. Example: Multi-Objective Optimization

```python
agent = aopt.Agent(llm_model="claude-sonnet-4-5")

result = agent.run("""
Multi-objective optimization:
Minimize f1(x) = x1^2 + x2^2 (distance from origin)
Minimize f2(x) = (x1-1)^2 + (x2-1)^2 (distance from (1,1))
Bounds: [-2, 2] for both variables.
""")

# Agent's execution:

# === FORMULATION ===
Action: formulate_problem(goal="...")
→ problem = {
    "problem_type": "nonlinear_multi",
    "objectives": [
        {"name": "f1", "sense": "minimize"},
        {"name": "f2", "sense": "minimize"}
    ],
    "variables": [
        {"name": "x1", "type": "continuous", "bounds": [-2, 2]},
        {"name": "x2", "type": "continuous", "bounds": [-2, 2]}
    ],
    "constraints": []
}

Action: analyze_problem_structure(problem)
→ {"complexity": "low", "is_convex": True, "is_multi_objective": True}

# === STRATEGIZE ===
Action: recommend_optimizers(problem)
→ [
    {"name": "NSGA-II", "suitability": 0.95, "reason": "Standard for multi-objective"},
    {"name": "weighted_sum", "suitability": 0.6, "reason": "Simple but limited"}
]

Agent: "I'll use NSGA-II to find Pareto frontier."

Action: optimizer_create(algorithm="NSGA-II", problem=problem, settings={"pop_size": 50})

# === EXECUTION (Population-Based) ===
# NSGA-II proposes population of 50 designs per generation

Agent: "Get population..."
Action: optimizer_propose(optimizer_id="opt_001")
→ population = [design_1, design_2, ..., design_50]

Agent: "Evaluate all..."
for design in population:
    Action: evaluate_function(design=design, objectives=["f1", "f2"])
    → Store results

Agent: "Update optimizer with population results..."
Action: optimizer_update(optimizer_id="opt_001", population_results=[...])

# ... Generations 2-20: Similar ...

# === TERMINATION ===
Agent (generation 20): "Pareto front converged (hypervolume stable).
                        Found 35 non-dominated solutions.
                        DONE."

result = {
    "converged": True,
    "pareto_front": [
        {"design": [0, 0], "objectives": [0.0, 2.0]},
        {"design": [0.5, 0.5], "objectives": [0.5, 0.5]},
        {"design": [1, 1], "objectives": [2.0, 0.0]},
        # ... 32 more points
    ],
    "generations": 20,
    "reasoning_log": [...]
}
```

---

## 8. User Interface & Experience

### 8.1 Design Philosophy

**Layered approach**: Provide clean **programmatic API** first, presentation layer second.

**Rationale**:
- Engineers use: Python scripts, Jupyter notebooks, HPC batch jobs
- TUI is unfamiliar, doesn't work well remotely (SSH, HPC)
- Testing requires headless execution
- Future UI options (Jupyter, web) can layer on top

### 8.2 Milestone 1 UI: API + Rich Console

**Primary interface**: Python API
```python
from aopt import Agent

agent = Agent(llm_model="claude-sonnet-4-5", verbose=True)
result = agent.run("Minimize drag, maintain CL >= 0.8")
```

**Console output**: Rich library (colored logs, progress, tables)
- Works in regular terminals AND log files
- Agent reasoning streams in real-time
- Easy to disable: `verbose=False` for batch jobs
- 80% of TUI benefits, 20% of complexity

**No full TUI** in Milestone 1 - overkill for prototyping, testing

### 8.3 Real-Time Streaming via Callbacks

**Architecture**: Event-driven with optional callbacks

See **`docs/callback_streaming_architecture.md`** for complete specification.

**Key features**:
- Agent emits structured `AgentEvent` instances at key points
- 15+ event types: AGENT_START, TOOL_CALL, CACHE_HIT, CONVERGENCE_CHECK, ADAPTATION_START, etc.
- Multiple callbacks supported simultaneously (console + file + custom)
- Callbacks are **optional** - agent works without them
- Error isolation - callback failures don't break optimization

**User API**:
```python
# Built-in rich console
agent = Agent(verbose=True)  # Auto-registers RichConsoleCallback

# Custom callback
def my_callback(event: AgentEvent):
    if event.event_type == EventType.ITERATION_COMPLETE:
        print(f"Iter {event.iteration}: {event.data}")

agent.register_callback(my_callback)

# Multiple callbacks (console + file + custom)
agent.register_callback(RichConsoleCallback())
agent.register_callback(FileLogger("run.log"))
agent.register_callback(my_metrics_tracker)
```

**Testing**: EventCapture callback for assertions
```python
capture = EventCapture()
agent.register_callback(capture)
agent.run("Minimize Rosenbrock")

assert capture.count(EventType.CACHE_HIT) > 0
```

### 8.4 Future UI Options (Milestone 2+)

**Jupyter integration**:
- Inline plots (convergence, Pareto front)
- Interactive controls (pause, modify constraints)
- `JupyterCallback` for real-time updates

**Web dashboard**:
- For remote HPC monitoring
- Similar to TensorBoard, MLflow
- WebSocket-based event streaming

**CLI wrapper** (optional):
```bash
aopt run --goal "minimize drag" --output results.json
```

---

## 9. Milestone 1: Implementation Plan

### Week 1: Foundation + Critical Fixes + Callbacks
**Goal**: Basic agent + formulation tools + cache + streaming (from review)

- [ ] Setup repository (aopt package structure)
- [ ] Implement OptimizationProblem Pydantic schema
- [ ] **Implement callback architecture** (AgentEvent, EventType, CallbackManager)
- [ ] **Implement ReAct agent with message history + event emission** (CRITICAL FIX)
- [ ] Implement 3 formulation tools
- [ ] **Implement 3 cache/provenance tools** (cache_get, cache_store, run_db_log)
- [ ] **Implement RichConsoleCallback** (beautiful terminal output)
- [ ] **Implement FileLogger and EventCapture callbacks**
- [ ] **Test: Message history retention across multiple turns**
- [ ] **Test: Event emission and capture**
- [ ] Test: Agent can formulate Rosenbrock problem from text

### Week 2: Optimizer Integration + Cache Integration
**Goal**: Single-objective nonlinear working with cache

- [ ] Implement 4 optimizer tools (basic versions)
- [ ] Integrate scipy.optimize (SLSQP, L-BFGS-B, COBYLA)
- [ ] **Implement 2 evaluator tools with automatic cache lookup** (CRITICAL FIX)
- [ ] Implement analytical backend (Rosenbrock, etc.)
- [ ] **Test: Cache prevents re-evaluation** (verify cost=0 on cache hit)
- [ ] Test: Agent solves 2D Rosenbrock end-to-end

### Week 3: Observation & Adaptation + Safe Restarts
**Goal**: Agent can observe and adapt safely

- [ ] Implement 3 observer tools
- [ ] Implement 2 adapter tools (modify_constraints, switch_gradient_method)
- [ ] **Upgrade optimizer_restart with safety** (restart_from, reuse_cache, checkpoint)
- [ ] Implement 1 resource tool (budget_remaining)
- [ ] **Test: Restart reuses cache and starts from best design** (CRITICAL FIX)
- [ ] Test: Agent solves constrained Rosenbrock, adapts constraints

### Week 4: Multi-Objective
**Goal**: Complete Milestone 1

- [ ] Integrate pymoo (NSGA-II)
- [ ] Update optimizer tools for multi-objective
- [ ] Test: Agent solves multi-objective problem
- [ ] Test: Agent handles both single and multi-objective autonomously

**Deliverable**: Agent that formulates and solves:
1. Unconstrained nonlinear (Rosenbrock)
2. Constrained nonlinear (with adaptation)
3. Multi-objective problems (Pareto front)

---

## 10. File Structure

```
aopt/
├── __init__.py
├── agent/
│   ├── __init__.py
│   ├── react_agent.py        # LangGraph ReAct loop with callbacks
│   ├── context.py             # Context management
│   └── prompts.py             # Prompt templates
├── callbacks/
│   ├── __init__.py
│   ├── base.py                # AgentEvent, EventType, CallbackManager
│   ├── rich_console.py        # RichConsoleCallback
│   ├── file_logger.py         # FileLogger
│   └── capture.py             # EventCapture (for testing)
├── formulation/
│   ├── __init__.py
│   ├── schema.py              # OptimizationProblem Pydantic model
│   └── tools.py               # 3 formulation tools
├── tools/
│   ├── __init__.py
│   ├── optimizer_tools.py     # 4 optimizer tools
│   ├── evaluator_tools.py     # 2 evaluator tools
│   ├── cache_tools.py         # 3 cache/provenance tools (CRITICAL)
│   ├── observer_tools.py      # 3 observer tools
│   ├── adapter_tools.py       # 2 adapter tools
│   └── resource_tools.py      # 1 resource tool
├── optimizers/
│   ├── __init__.py
│   ├── base.py                # Base optimizer interface
│   ├── scipy_optimizers.py    # SLSQP, COBYLA, L-BFGS-B
│   └── pymoo_optimizers.py    # NSGA-II
├── backends/
│   ├── __init__.py
│   └── analytical.py          # Rosenbrock, etc.
└── utils/
    ├── __init__.py
    └── formatting.py

tests/
├── test_formulation.py        # Formulation tools
├── test_agent_single.py       # Single-objective
├── test_agent_multi.py        # Multi-objective
└── test_tools.py              # All tools

examples/
├── quickstart.py              # Simple Rosenbrock
├── constrained.py             # Constrained problem
└── multi_objective.py         # Multi-objective
```

---

## 11. Success Criteria

**Milestone 1 Complete When**:

### Core Functionality
- [ ] Agent formulates problem from natural language (nonlinear + multi-obj)
- [ ] Agent solves unconstrained Rosenbrock (10D) → f < 1e-6
- [ ] Agent solves constrained problem → feasible optimum
- [ ] Agent solves multi-objective → Pareto front with 20+ points
- [ ] Agent autonomously chooses optimizer based on problem type
- [ ] Agent observes and adapts (e.g., tighten constraints)

### Critical Fixes (from review)
- [ ] **Message history retention**: Agent maintains full conversation history (user + assistant + tool results)
- [ ] **Evaluation cache**: Cache prevents re-evaluation of identical designs (cache hit = 0 cost)
- [ ] **Restart safety**: optimizer_restart reuses cache, starts from best, checkpoints old state
- [ ] **Budget awareness**: Agent monitors budget and makes budget-aware decisions

### UI/UX & Callbacks
- [ ] **Real-time streaming**: Agent emits events via callback system
- [ ] **Rich console**: Beautiful terminal output with colors, progress, tables
- [ ] **Event capture**: EventCapture for testing, FileLogger for debugging
- [ ] **Multiple callbacks**: Can register console + file + custom simultaneously

### Testing
- [ ] All 18 tools have unit tests (100% coverage)
- [ ] Test: Message history accumulates across multiple ReAct turns
- [ ] Test: Cache eliminates duplicate evaluations (verify cost = 0 on cache hit)
- [ ] Test: Restart reuses cache and starts from best design
- [ ] Test: Event emission and callback execution
- [ ] Agent reasoning is logged and explainable

**Timeline**: 4 weeks

---

## 12. Key Innovations Summary

1. **Intelligent Formulation**: Agent converts goals → structured problems
2. **Full Autonomy**: ReAct loop, no prescribed state machine
3. **Compositional**: Agent composes strategies from 18 tool primitives
4. **Adaptive**: Agent observes, detects patterns, modifies strategy mid-run
5. **Efficient**: Evaluation cache prevents expensive re-computation
6. **Safe**: Optimizer restarts from best design with cache reuse and rollback
7. **Observable**: Real-time event streaming via callbacks (console, file, custom)
8. **Extensible**: Easy to add new problem types, optimizers, tools, UIs
9. **Explainable**: Every decision logged with reasoning

---

**Ready to implement! This is the complete, implementable architecture for Milestone 1.** 🚀
