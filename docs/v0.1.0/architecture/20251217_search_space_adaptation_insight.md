# Search Space Adaptation for Gradient-Based Optimization

**Date**: 2025-12-17
**Status**: Key architectural insight for future development
**Impact**: High - enables capabilities no existing optimizer has

---

## The Problem

All existing gradient-based NLP optimizers (IPOPT, SciPy, SNOPT, NLopt) share a fundamental limitation:

```python
# User specifies bounds ONCE
result = minimize(objective, x0, bounds=[(0, 100)] * n_vars)

# Optimizer works within these bounds
# Cannot adapt bounds based on what it learns
```

**Consequences:**
1. Poor scaling if bounds are too wide
2. Stuck in local minima - can't escape by exploring other regions
3. No progressive refinement strategy
4. User must guess good bounds upfront

---

## The Insight

Paola's graph architecture naturally enables **search space adaptation**:

```
┌─────────────────────────────────────────────────────────┐
│                    GRAPH NODE 1                          │
│  Wide bounds: [0, 100]                                  │
│  Purpose: Exploration, find promising region            │
│  Result: x* ≈ [42, 58, 31, 89, ...]                    │
└─────────────────────────────────────────────────────────┘
                         │
                         │ Agent analyzes results
                         │ "Solution in [30-60] range for most vars"
                         │ "x4=89 near upper bound - keep wide"
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    GRAPH NODE 2                          │
│  Adapted bounds: [25, 75] for x1,x2,x3                 │
│                  [50, 100] for x4 (was near bound)      │
│  Purpose: Refined search with better conditioning       │
│  Result: Better solution, faster convergence            │
└─────────────────────────────────────────────────────────┘
                         │
                         │ Agent: "Try alternative region?"
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    GRAPH NODE 3                          │
│  Alternative bounds: [60, 100] for exploration          │
│  Purpose: Check if better basin exists                  │
│  Result: Found different local minimum                  │
└─────────────────────────────────────────────────────────┘
```

---

## Why This Is Novel

| Capability | Traditional Optimizer | Paola with Graph |
|------------|----------------------|------------------|
| Adapt bounds mid-optimization | ❌ Impossible | ✅ Between nodes |
| Detect bound-hitting solutions | ❌ Silent | ✅ Agent analyzes |
| Multi-start with intelligence | ❌ Random restarts | ✅ Informed regions |
| Progressive refinement | ❌ User must do manually | ✅ Agent decides |
| Escape local minima | ❌ Stuck | ✅ Branch to new region |

**No existing optimization platform does this automatically.**

---

## Concrete Scenarios

### Scenario 1: Poor Initial Bounds

```
Problem: User specifies bounds [0, 1000] for all variables
         But optimal region is actually [45, 55]

Traditional: Optimizer struggles with poor scaling
             Takes 500+ iterations

Paola Graph:
  n1: Run with [0, 1000], converges to ~[48, 52, 51, ...]
  n2: Agent narrows to [40, 60], much better conditioning
      Converges in 50 iterations with higher accuracy
```

### Scenario 2: Solution Hits Bounds

```
Problem: Optimal x3* = -5, but user set bounds [0, 100]

Traditional: Returns x3 = 0 (bound), user doesn't know why

Paola Graph:
  n1: Result shows x3 = 0 (at lower bound)
  Agent: "x3 hit lower bound - optimal might be outside"
  Agent asks user: "Should I extend x3 bounds to [-50, 100]?"
  n2: With extended bounds, finds true optimum x3* = -5
```

### Scenario 3: Multi-Modal Landscape

```
Problem: Multiple local minima at different regions

Traditional: Stuck in whichever basin x0 falls into

Paola Graph:
  n1: [0, 100] → finds minimum at ~30
  n2: [50, 100] (branch) → finds minimum at ~75
  n3: [0, 50] (branch) → confirms ~30 is local
  Agent: "Found two local minima. x=75 has lower objective."
```

---

## Implementation Considerations

### What the Agent Needs to Detect

1. **Bound-hitting**: Solution components at or near bounds
2. **Clustering**: Most solution values in narrow sub-region
3. **Slow convergence**: Might indicate poor conditioning
4. **Multiple restarts converging to same point**: Likely global in that region

### What the Agent Can Decide

1. **Narrow bounds**: When solution clusters away from bounds
2. **Extend bounds**: When solution hits bounds
3. **Branch to alternative region**: For multi-modal exploration
4. **Tighten tolerances**: After narrowing bounds (better conditioning)

### Graph Edge Types for Search Space Adaptation

| Edge Type | Meaning | Bounds Change |
|-----------|---------|---------------|
| `refine` | Narrow to promising region | Tighter |
| `extend` | Solution hit bounds | Wider for some vars |
| `branch` | Explore alternative region | Different region |
| `restart` | Try fresh with same bounds | Same |

---

## Future Development Tasks

1. **Agent reasoning prompts**: How to analyze solution w.r.t. bounds
2. **Automatic bound detection**: Identify bound-hitting variables
3. **Conditioning metrics**: Detect when bounds are too wide
4. **Multi-modal detection**: Recognize when problem has multiple basins
5. **User interaction**: When to ask user vs decide automatically

---

## Relationship to Other Paola Features

- **Graph storage**: Enables memory of past solutions and bounds
- **Skills**: Optimizer-specific guidance on bound handling
- **Cross-graph learning**: "For similar problems, this bound strategy worked"
- **The Paola Principle**: User doesn't configure bounds adaptation - Paola does

---

## Summary

**Search space adaptation** is a capability that:
- No existing gradient-based optimizer has
- Naturally emerges from Paola's graph architecture
- Provides concrete value (better conditioning, escape local minima)
- Aligns with "optimization complexity is Paola intelligence"

This should be a **core feature** of Paola's value proposition for gradient-based optimization.
