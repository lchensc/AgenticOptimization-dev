# Architecture v3: High-Severity Fixes for Milestone 1

**Date**: December 10, 2025
**Status**: Critical fixes based on architecture review

---

## Overview

This document addresses the 3 high-severity issues identified in the architecture review:

1. **Conversation state loss in agent loop** - ReAct loop discards message history
2. **Missing safety/rollback for adaptations** - optimizer_restart lacks cache/checkpoint mechanisms
3. **No evaluation cache/provenance** - Critical missing tools for expensive simulations

---

## Fix 1: Conversation State Retention (Critical)

### Problem
**Location**: Section 5.1, `react_step` function (line 443)

```python
# WRONG - discards history!
return {
    "messages": [response],  # ❌ Replaces entire history
    "context": new_context,
    "done": False
}
```

This breaks:
- ReAct grounding (agent can't reference prior reasoning)
- Tool call threading (tool results disconnected from requests)
- Termination detection (can't detect repeated failures)

### Solution
Accumulate **full message history** (user + assistant + tool results):

```python
def create_react_node(tools):
    """
    ReAct node: reason → act → observe.
    Maintains full conversation history for grounding.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    llm_with_tools = llm.bind_tools(tools)

    def react_step(state: AgentState) -> dict:
        """
        Execute one ReAct cycle with full history retention.
        """
        context = state["context"]

        # Build prompt with current context
        prompt = build_optimization_prompt(context)

        # FIXED: Accumulate full conversation history
        # Include all prior messages + new user prompt
        messages = state["messages"] + [HumanMessage(content=prompt)]

        # Get LLM decision (reasoning + tool calls)
        response = llm_with_tools.invoke(messages)

        # FIXED: Collect all new messages (response + tool results)
        new_messages = [response]

        # Execute tool calls and collect results
        if response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                result = execute_tool(tool_call, tools)
                tool_results.append(result)
                # Add tool result message
                new_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    )
                )

            # Update context with tool results
            new_context = update_context(context, tool_results)

            # FIXED: Append all new messages to history
            return {
                "messages": new_messages,  # ✓ Appends to existing (via operator.add)
                "context": new_context,
                "done": False
            }

        # Check if agent says done
        if "DONE" in response.content.upper() or "CONVERGED" in response.content.upper():
            return {
                "messages": new_messages,  # ✓ Preserve history
                "context": context,
                "done": True
            }

        # Agent just reasoning, continue
        return {
            "messages": new_messages,  # ✓ Preserve history
            "context": context,
            "done": False
        }

    return react_step
```

**Key changes**:
1. `messages = state["messages"] + [HumanMessage(...)]` - preserve prior history
2. `new_messages = [response]` - collect all messages from this turn
3. Add `ToolMessage` for each tool result (maintains threading)
4. Return `messages: new_messages` - appends via `operator.add` in AgentState

---

## Fix 2: Evaluation Cache Tools (Critical)

### Problem
**Location**: Section 4 (Tool Set)

No cache tools exist! Without cache:
- Line searches re-evaluate same designs (10× waste)
- Population optimizers evaluate duplicates
- Gradient calls can't reuse objective evaluations
- Knowledge accumulation impossible

### Solution
Add **3 cache/provenance tools** to tool set (raises total to **18 tools**):

```python
# === NEW: Cache/Provenance Tools (3 tools) ===

@tool
def cache_get(
    design: list[float],
    problem_id: str,
    tolerance: float = 1e-9
) -> Optional[dict]:
    """
    Retrieve cached evaluation result for design.

    Args:
        design: Design variables
        problem_id: Problem identifier (for cache isolation)
        tolerance: Similarity tolerance for design matching

    Returns:
        {
            "objectives": [0.0245],
            "gradient": [...],  # If available
            "constraints": {...},  # If available
            "cost": 0.5,  # CPU hours
            "timestamp": "2025-12-10T10:30:00",
            "hit": True
        }

        Returns None if not in cache.
    """
    pass


@tool
def cache_store(
    design: list[float],
    problem_id: str,
    objectives: list[float],
    gradient: Optional[list[float]] = None,
    constraints: Optional[dict] = None,
    cost: float = 0.0,
    metadata: dict = {}
) -> dict:
    """
    Store evaluation result in cache.

    Returns:
        {
            "stored": True,
            "cache_size": 245,
            "duplicate": False  # True if design already cached
        }
    """
    pass


@tool
def run_db_log(
    optimizer_id: str,
    iteration: int,
    design: list[float],
    objectives: list[float],
    action: str,  # "evaluate", "adapt", "restart"
    reasoning: str,
    metadata: dict = {}
) -> dict:
    """
    Log optimization run for provenance and learning.

    Stores:
    - Every evaluation (design → objectives)
    - Every adaptation (reasoning + old/new problem)
    - Every agent decision (action + reasoning)

    This enables:
    - Run replay and debugging
    - Pattern detection across runs
    - Knowledge base accumulation

    Returns:
        {"logged": True, "run_id": "run_001", "entry_id": 12}
    """
    pass
```

### Updated Tool Count
**Total: 18 Tools**
- Formulation (3)
- Optimizer (4)
- Evaluator (2)
- **Cache/Provenance (3)** ← NEW
- Observer (3)
- Adapter (2)
- Resource (1)

### Integration with evaluate_function

```python
@tool
def evaluate_function(
    design: list[float],
    problem: dict,
    objectives: list[str],
    gradient: bool = False,
    constraints: bool = False
) -> dict:
    """
    Evaluate objective(s) with automatic cache lookup.

    UPDATED: Now checks cache first!
    """
    problem_id = problem.get("id", "unknown")

    # 1. Check cache first
    cached = cache_get(design=design, problem_id=problem_id)
    if cached and cached["hit"]:
        # Cache hit - verify it has what we need
        has_gradient = gradient and cached.get("gradient") is not None
        has_constraints = constraints and cached.get("constraints") is not None

        if (not gradient or has_gradient) and (not constraints or has_constraints):
            # Cache has everything we need!
            return {
                "objectives": cached["objectives"],
                "gradient": cached.get("gradient"),
                "constraints": cached.get("constraints"),
                "cost": 0.0,  # No cost - cache hit!
                "cache_hit": True
            }

    # 2. Cache miss or incomplete - evaluate
    result = _evaluate_backend(
        design=design,
        problem=problem,
        objectives=objectives,
        gradient=gradient,
        constraints=constraints
    )

    # 3. Store in cache
    cache_store(
        design=design,
        problem_id=problem_id,
        objectives=result["objectives"],
        gradient=result.get("gradient"),
        constraints=result.get("constraints"),
        cost=result["cost"]
    )

    result["cache_hit"] = False
    return result
```

---

## Fix 3: Safety/Rollback for optimizer_restart (Critical)

### Problem
**Location**: Section 4.2, `optimizer_restart` tool

```python
# WRONG - no safety!
@tool
def optimizer_restart(
    optimizer_id: str,
    new_problem: dict,
    initial_design: list[float]
) -> str:
    """Restart with modified problem."""
```

Issues:
1. No access to **best design so far** (may restart from random point)
2. No **cache reuse** (loses all evaluations)
3. No **checkpoint** of old optimizer state (can't rollback if restart fails)
4. Changing constraints invalidates gradient/penalties - need recomputation

### Solution
Complete restart signature with safety:

```python
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
    1. Can restart from best design found so far
    2. Reuses evaluation cache (no wasted evaluations)
    3. Checkpoints old optimizer state (enables rollback)
    4. Recomputes gradient at restart point if problem changed

    Args:
        optimizer_id: Current optimizer to restart
        new_problem: Modified problem (constraints, bounds, etc.)
        restart_from: Where to restart from
            - "best": Best feasible design found so far
            - "current": Current optimizer position
            - "custom": User-specified design
        custom_design: Required if restart_from="custom"
        reuse_cache: Carry forward evaluation cache
        checkpoint_old: Save old optimizer state before restart
        recompute_gradient: Recompute gradient at restart point
            (required if problem formulation changed)

    Returns:
        {
            "new_optimizer_id": "opt_002",
            "restart_design": [0.5, 0.3, ...],
            "restart_objective": [0.0245],
            "gradient_recomputed": True,
            "old_checkpoint_id": "ckpt_001",  # For rollback
            "cache_entries_reused": 145
        }
    """
    # 1. Checkpoint old optimizer state
    old_checkpoint = None
    if checkpoint_old:
        old_checkpoint = _save_optimizer_checkpoint(optimizer_id)

    # 2. Determine restart design
    if restart_from == "best":
        restart_design = _get_best_design(optimizer_id)
    elif restart_from == "current":
        restart_design = _get_current_design(optimizer_id)
    elif restart_from == "custom":
        if custom_design is None:
            raise ValueError("custom_design required when restart_from='custom'")
        restart_design = custom_design
    else:
        raise ValueError(f"Invalid restart_from: {restart_from}")

    # 3. Recompute gradient at restart point if needed
    gradient = None
    if recompute_gradient:
        gradient = compute_gradient(
            design=restart_design,
            problem=new_problem,
            objective=new_problem["objectives"][0]["name"]
        )

    # 4. Create new optimizer with modified problem
    new_optimizer_id = _create_optimizer_internal(
        algorithm=_get_algorithm(optimizer_id),
        problem=new_problem,
        initial_design=restart_design,
        initial_gradient=gradient
    )

    # 5. Copy cache if requested
    cache_entries = 0
    if reuse_cache:
        cache_entries = _copy_cache(
            from_problem=_get_problem(optimizer_id),
            to_problem=new_problem,
            new_optimizer_id=new_optimizer_id
        )

    # 6. Evaluate at restart point (for new problem formulation)
    restart_result = evaluate_function(
        design=restart_design,
        problem=new_problem,
        objectives=[obj["name"] for obj in new_problem["objectives"]],
        gradient=False,  # Already computed above if needed
        constraints=True
    )

    return {
        "new_optimizer_id": new_optimizer_id,
        "restart_design": restart_design,
        "restart_objectives": restart_result["objectives"],
        "gradient_recomputed": recompute_gradient,
        "old_checkpoint_id": old_checkpoint["id"] if old_checkpoint else None,
        "cache_entries_reused": cache_entries,
        "restart_feasible": _check_feasibility(restart_result, new_problem)
    }
```

### Rollback mechanism

```python
@tool
def optimizer_rollback(
    checkpoint_id: str
) -> dict:
    """
    Rollback to checkpointed optimizer state.

    Use when restart fails or makes things worse.

    Returns:
        {
            "optimizer_id": "opt_001",  # Restored optimizer
            "design": [...],
            "objective": [0.0245]
        }
    """
    pass
```

---

## Fix 4: Updated Agent Prompt (Include Budget + Cache)

### Problem
**Location**: Section 5.1, `build_optimization_prompt`

Current prompt doesn't mention:
- Budget remaining (budget_remaining tool exists but unused)
- Cache hits (agent should know when evaluations are "free")

### Solution

```python
def build_optimization_prompt(context: dict) -> str:
    """
    Build prompt with current optimization state.

    UPDATED: Now includes budget and cache stats.
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
1. Check cache before expensive evaluations (cache_get)
2. Monitor budget - if low, consider stopping or reducing evaluations
3. Observe convergence regularly (analyze_convergence)
4. Adapt if stuck (modify_constraints, optimizer_restart)

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

## Summary of Changes

### 1. Fixed Message History (react_step)
- **Before**: `messages: [response]` - discards history ❌
- **After**: `messages: new_messages` - accumulates with `operator.add` ✓
- **Impact**: Agent maintains grounding, tool threading, termination detection

### 2. Added Cache Tools
- **Before**: 15 tools, no cache ❌
- **After**: 18 tools (added cache_get, cache_store, run_db_log) ✓
- **Impact**: Prevents re-evaluation waste, enables knowledge accumulation

### 3. Enhanced optimizer_restart
- **Before**: 3 parameters, no safety ❌
- **After**: 7 parameters with safety (best design, cache reuse, checkpoint) ✓
- **Impact**: Safe adaptations with rollback capability

### 4. Updated Prompt
- **Before**: No budget/cache awareness ❌
- **After**: Includes budget status and cache hit rate ✓
- **Impact**: Agent makes budget-aware decisions

---

## Updated Architecture Sections

### Section 4: Tool Count
**Total: 18 Tools** (was 15)
- Formulation (3)
- Optimizer (4)
- Evaluator (2) - now with automatic caching
- **Cache/Provenance (3)** ← NEW
- Observer (3)
- Adapter (2)
- Resource (1)

### Section 5.1: ReAct Agent
Replace entire `create_react_node` function with Fix 1 implementation.

### Section 8: Milestone 1 Timeline
**Week 1**: Add cache tools implementation
- [ ] Implement cache_get/cache_store (in-memory dict for Milestone 1)
- [ ] Implement run_db_log (SQLite for Milestone 1)
- [ ] Update evaluate_function to check cache first

**Week 2**: Add optimizer safety
- [ ] Implement optimizer_restart with full safety signature
- [ ] Implement optimizer_rollback
- [ ] Test: Restart from best design with cache reuse

---

## Testing Requirements

### Test 1: Message History Retention
```python
def test_message_history_accumulation():
    """Verify agent retains full conversation history."""
    agent = build_aopt_agent(tools)

    result = agent.invoke({
        "messages": [],
        "context": {"goal": "Minimize Rosenbrock"},
        "done": False
    })

    # After multiple steps, history should contain:
    # - User prompts (HumanMessage)
    # - Agent responses (AIMessage)
    # - Tool results (ToolMessage)
    assert len(result["messages"]) > 10  # Multiple turns
    assert any(isinstance(m, ToolMessage) for m in result["messages"])
```

### Test 2: Cache Prevents Re-evaluation
```python
def test_cache_prevents_reevaluation():
    """Verify cache eliminates duplicate evaluations."""
    design = [1.0, 2.0, 3.0]
    problem = {...}

    # First evaluation - cache miss
    result1 = evaluate_function(design, problem, ["f1"], gradient=True)
    assert result1["cache_hit"] == False
    assert result1["cost"] > 0

    # Second evaluation - cache hit
    result2 = evaluate_function(design, problem, ["f1"], gradient=True)
    assert result2["cache_hit"] == True
    assert result2["cost"] == 0.0  # No cost!
    assert result2["objectives"] == result1["objectives"]
    assert result2["gradient"] == result1["gradient"]
```

### Test 3: Restart Safety
```python
def test_optimizer_restart_safety():
    """Verify restart reuses cache and restarts from best."""
    # Run optimizer for 10 iterations
    for i in range(10):
        design = optimizer_propose(opt_id)
        result = evaluate_function(design, problem, ["f1"])
        optimizer_update(opt_id, design, result["objectives"])

    # Modify problem (tighten constraint)
    new_problem = modify_constraints(problem, "c1", 0.9, "Tighten")

    # Restart safely
    restart_result = optimizer_restart(
        optimizer_id=opt_id,
        new_problem=new_problem,
        restart_from="best",  # ← Starts from best found
        reuse_cache=True,     # ← Reuses 10 evaluations
        checkpoint_old=True   # ← Can rollback if needed
    )

    assert restart_result["cache_entries_reused"] == 10
    assert restart_result["old_checkpoint_id"] is not None
    assert restart_result["gradient_recomputed"] == True
```

---

## Migration from v3 to v3-fixed

1. **Replace** Section 5.1 `react_step` with Fix 1 implementation
2. **Add** Section 4.3.5 "Cache/Provenance Tools" with 3 new tools
3. **Replace** Section 4.2 `optimizer_restart` signature with Fix 3 implementation
4. **Update** Section 5.1 prompt with Fix 4 implementation
5. **Update** Section 10 success criteria to include cache tests

---

**Status**: Ready to update architecture_v3_final.md with these fixes.
