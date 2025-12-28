# Paola Current State and Roadmap

**Date**: 2025-12-17
**Version**: v0.3.1

## Executive Summary

Paola (Package for Agentic Optimization with Learning and Analysis) has reached a stable v0.3.1 milestone with:
- **Graph-based architecture** for multi-node optimization tracking
- **Skills infrastructure** for progressive-disclosure optimizer expertise
- **Clean codebase** with all legacy session-based code removed

The agent can now run optimizations, track them in graphs, and access optimizer expertise via Skills. The next priorities are completing Skills content and real-world testing.

---

## Current Architecture

### Core Concepts

```
User Request
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                    PAOLA AGENT (LLM)                         │
│  Conversational interface, reasons about optimization        │
└─────────────────────────────────────────────────────────────┘
    │                    │                    │
    ▼                    ▼                    ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ Graph Tools  │  │ Optimization │  │ Skill Tools  │
│              │  │ Tools        │  │              │
│ start_graph  │  │ run_optim    │  │ list_skills  │
│ get_state    │  │ get_prob_info│  │ load_skill   │
│ finalize     │  │ list_optim   │  │ query_skills │
│ query_past   │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
    │                    │                    │
    ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                      FOUNDRY (Data Layer)                    │
│  Two-tier storage: GraphRecord (LLM) + GraphDetail (debug)  │
└─────────────────────────────────────────────────────────────┘
```

### Graph Model

- **Graph**: Complete optimization task (may involve multiple optimizer runs)
- **Node**: Single optimizer execution within a graph
- **Edge**: Relationship between nodes (warm_start, restart, refine, branch, explore)
- **Pattern**: Graph structure (single, multistart, chain, tree, dag)

### Two-Tier Storage (v0.3.1)

| Tier | Content | Size | Purpose |
|------|---------|------|---------|
| **Tier 1: GraphRecord** | Problem signature, strategy pattern, node summaries with config, outcomes | ~1KB | LLM learning, cross-graph queries |
| **Tier 2: GraphDetail** | Convergence history, x trajectories, full solutions | 10-100KB | Visualization, debugging |

### Skills Infrastructure

Progressive-disclosure optimizer expertise:
```
list_skills()           → Discover available skills
load_skill(name)        → Get overview
load_skill(name, "options") → Get full option reference
load_skill(name, "options.warm_start") → Get specific section
```

---

## What's Working (v0.3.1)

### Fully Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| CLI with conversational interface | ✅ | `python -m paola.cli` |
| Graph management tools | ✅ | start, get_state, finalize, query_past |
| Multiple optimizer backends | ✅ | SciPy, IPOPT, Optuna |
| Skills infrastructure | ✅ | loader, index, tools |
| IPOPT skill content | ✅ | Overview + 250 options in YAML |
| Evaluator registration | ✅ | File-based storage |
| Two-tier graph storage | ✅ | Cross-graph learning enabled |
| Token tracking | ✅ | Cost display in CLI |

### Optimizer Backends

| Backend | Status | Methods/Samplers |
|---------|--------|------------------|
| SciPy | ✅ | SLSQP, L-BFGS-B, trust-constr, COBYLA, Nelder-Mead, Powell, BFGS, CG |
| IPOPT | ✅ | Interior-point (requires cyipopt) |
| Optuna | ✅ | TPE, CMA-ES, Random, Grid, QMC |

### CLI Commands

```
/graphs              - List all optimization graphs
/graph show <id>     - Show graph details
/graph plot <id>     - Plot convergence
/graph compare       - Compare graphs
/graph query         - Query past graphs
/skills              - List optimizer skills
/skill <name>        - Show skill details
/evals               - List evaluators
/analyze <id>        - AI-powered analysis
```

---

## Recent Changes (This Session)

### Removed (Cleanup)

| Removed | Reason |
|---------|--------|
| Session-based architecture (v0.2.0) | Replaced by graphs |
| `config_tools` (config_scipy, config_ipopt, etc.) | Replaced by Skills |
| `get_optimizer_options` tool | Replaced by Skills |
| `key_options` from backend info | Redundant with Skills |
| `supports_*` flags from backend info | Misleading (method-level varies) |
| Hardcoded `recommendation` in list_optimizers | Redundant with Skills |

### Improved

| Change | Why |
|--------|-----|
| Added skill guidance to system prompt | Balance: use skills for non-default config |
| Updated skill tool descriptions | Situational triggers (when to use/not use) |
| Simplified `list_optimizers` output | Minimal info + pointer to Skills |
| Updated CLAUDE.md | Reflects v0.3.1 with Skills |

---

## What's Missing / In Progress

### Skills Content (Priority)

| Skill | Status | Notes |
|-------|--------|-------|
| IPOPT | ✅ Done | 250+ options in YAML |
| SciPy | ❌ TODO | 8 methods, each with different options |
| Optuna | ❌ TODO | Samplers, pruners, distributions |
| NLopt | ❌ TODO | No backend yet |

### Features Not Yet Implemented

| Feature | Status | Blocked By |
|---------|--------|------------|
| Knowledge base with RAG | Skeleton | Need vector DB integration |
| Learned skills (auto-generated) | Not started | Need successful optimization data |
| Multi-run analysis | Not started | Need more graph data |
| Domain skills (aerodynamics, etc.) | Not started | Need domain content |

---

## Code Statistics

```
paola/
├── agent/          # 3 files, ~800 lines
├── tools/          # 12 files, ~2500 lines
├── skills/         # 5 files, ~600 lines
├── foundry/        # 8 files, ~1500 lines
├── optimizers/     # 2 files, ~500 lines
├── cli/            # 4 files, ~700 lines
└── analysis/       # 3 files, ~400 lines

Total: ~7000 lines Python
```

---

## Open Questions

### Design Decisions Needed

1. **SciPy skill structure**: One skill with 8 method sections, or separate skills per method?
2. **Skill versioning**: How to handle skill updates? Version field in YAML?
3. **Configurations**: Should skills include pre-built configs (e.g., "warm_start_from_parent")?
4. **Testing strategy**: How to test agent behavior with Skills systematically?

### Technical Debt

1. `config_tools.py` still exists (commented out) - delete after testing confirms Skills work
2. Some tests may reference removed tools - need test cleanup
3. Backward compatibility alias `list_available_optimizers` - keep or remove?

---

## Roadmap Options

### Option A: Complete Skills Content First

Focus on making the existing infrastructure production-ready:

1. **Phase 2a**: SciPy skill (all 8 methods)
2. **Phase 2b**: Optuna skill (samplers, pruners)
3. **Phase 2c**: Pre-built configurations
4. **Phase 3**: Real-world testing with engineering problems
5. **Phase 4**: Knowledge base activation

**Pros**: Solid foundation, agent can handle any optimizer
**Cons**: Delayed novel features

### Option B: Vertical Slice with Learning

Build end-to-end learning loop with minimal content:

1. **Phase 2**: Basic SciPy/Optuna skills (overview only)
2. **Phase 3**: Cross-graph learning (query_past_graphs → inform decisions)
3. **Phase 4**: Auto-generated learned skills from successful graphs
4. **Phase 5**: Complete skill content

**Pros**: Demonstrates learning value proposition early
**Cons**: Incomplete optimizer coverage

### Option C: Engineering Focus (Direction A from brainstorm)

Pursue high-value per-iterate control for expensive simulations:

1. **Phase 2**: Basic skill content
2. **Phase 3**: Per-evaluation graph nodes (not per-optimizer-run)
3. **Phase 4**: Surrogate model integration
4. **Phase 5**: Cost-aware agent decisions

**Pros**: High differentiation, direct cost savings
**Cons**: Significant schema changes, research risk

---

## Recommended Next Steps

### Immediate (This Week)

1. **Delete `config_tools.py`** if testing confirms Skills work
2. **Create SciPy skill** (overview + method capabilities)
3. **Create Optuna skill** (overview + sampler info)
4. **Test agent end-to-end** with a real optimization problem

### Short-term (Next 2 Weeks)

1. **Complete skill options** for SciPy and Optuna
2. **Add pre-built configurations** to IPOPT skill
3. **Document skill authoring** for future contributors
4. **Clean up test suite** for removed legacy code

### Medium-term

1. **Activate cross-graph learning** - agent uses query_past_graphs
2. **Learned skills prototype** - auto-generate from successful graphs
3. **Evaluate Direction A** (per-iterate graphs) feasibility

---

## Files Reference

### Key Files

| File | Purpose |
|------|---------|
| `paola/agent/prompts/optimization.py` | System prompt (instruction #4 for skills) |
| `paola/skills/tools.py` | Skill tools (list, load, query) |
| `paola/skills/registry.yaml` | Global skill index |
| `paola/skills/optimizers/ipopt/` | IPOPT skill content |
| `paola/tools/optimization_tools.py` | run_optimization, list_optimizers |
| `paola/optimizers/backends.py` | SciPy, IPOPT, Optuna backends |
| `CLAUDE.md` | Project guidance for AI assistants |

### Disabled (Pending Deletion)

| File | Status |
|------|--------|
| `paola/tools/config_tools.py` | Commented out, replaced by Skills |

---

## Summary

Paola v0.3.1 has a clean, working foundation:
- Graph-based architecture for tracking optimization
- Skills infrastructure for optimizer expertise
- Minimal, focused tools that delegate details to Skills

The main gaps are:
- Incomplete skill content (SciPy, Optuna)
- No real-world testing yet
- Learning features not activated

Recommended priority: **Complete Skills content** (Option A) to make the agent production-ready, then activate learning features.
