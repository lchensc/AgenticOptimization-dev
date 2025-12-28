# PAOLA Documentation

**Package for Agentic Optimization with Learning and Analysis**

This directory contains comprehensive documentation for the PAOLA project, organized by topic and purpose.

---

## 📖 Documentation Structure

```
docs/
├── README.md                    # This file - documentation index
├── architecture/                # Core architecture and design
├── planning/                    # Development plans and vision
├── implementation/              # Technical implementation guides
├── progress/                    # Phase completion reports
├── decisions/                   # Design decisions and analysis
└── archive/                     # Historical/legacy documents
```

---

## 🏗️ Architecture (Core Design)

**Location**: `architecture/`

The foundation of PAOLA's design - start here to understand the system.

### Current Architecture (v0.2.0 - Universal Vision)

| Document | Description | Status |
|----------|-------------|--------|
| **[universal_architecture.md](architecture/universal_architecture.md)** | **Comprehensive universal design** - Full architecture for universal agentic optimization platform | ✅ **Current** (Dec 14, 2025) |
| [architecture_crystallized.md](architecture/architecture_crystallized.md) | Previous architecture (v0.1.0) - Engineering-focused design | ✅ Superseded by universal_architecture.md |

### Design Philosophy & Vision

| Document | Description |
|----------|-------------|
| [agentic_optimization_vision.md](architecture/agentic_optimization_vision.md) | High-level vision and value propositions (7 core innovations) |
| [agent_controlled_optimization.md](architecture/agent_controlled_optimization.md) | Detailed technical design (50KB) - Agent autonomy, tool primitives, comparison with existing platforms |

**Quick navigation**:
- **New to PAOLA?** Start with `universal_architecture.md`
- **Understanding the vision?** Read `agentic_optimization_vision.md`
- **Deep technical details?** See `agent_controlled_optimization.md`

---

## 📋 Planning (Development Roadmap)

**Location**: `planning/`

Strategic plans and future development roadmap.

| Document | Description | Status |
|----------|-------------|--------|
| [next_phase_development_plan.md](planning/next_phase_development_plan.md) | **Phases 6-8 development plan** - From current state (Phases 1-5) to universal platform | ✅ Current plan |
| [PAOLA_vision.md](planning/PAOLA_vision.md) | Original vision document | Reference |

**Timeline**:
- **Phase 6** (4-5 weeks): Universal Adaptation - Flexible formulation, multi-optimizer integration, intelligent adaptation
- **Phase 7** (4-5 weeks): Universal Backends - Analytical, CFD, ML backends with multi-fidelity
- **Phase 8** (4-5 weeks): Universal Knowledge - Cross-domain learning with 50+ runs

---

## 🔧 Implementation (Technical Guides)

**Location**: `implementation/`

Detailed implementation guides for specific components.

### Component Architectures

| Document | Topic | Description |
|----------|-------|-------------|
| [callback_streaming_architecture.md](implementation/callback_streaming_architecture.md) | Event System | Event streaming, callback management, observability |
| [cli_architecture.md](implementation/cli_architecture.md) | CLI/REPL | Interactive command-line interface design |
| [run_architecture.md](implementation/run_architecture.md) | Run Management | Run lifecycle, storage, persistence |
| [optimization_gate_guide.md](implementation/optimization_gate_guide.md) | Gate Control | Continuation decisions, budget management |

### Implementation Guides

| Document | Topic | Description |
|----------|-------|-------------|
| [refactoring_blueprint.md](implementation/refactoring_blueprint.md) | Refactoring | Platform → Foundry refactoring blueprint |
| [prompt_caching_guide.md](implementation/prompt_caching_guide.md) | LLM Optimization | Prompt caching for 90% cost savings |

**Use cases**:
- **Implementing new callbacks?** See `callback_streaming_architecture.md`
- **Adding CLI commands?** See `cli_architecture.md`
- **Working with runs?** See `run_architecture.md`
- **Optimizing LLM costs?** See `prompt_caching_guide.md`

---

## ✅ Progress (Completion Reports)

**Location**: `progress/`

Phase-by-phase implementation progress and completion reports.

| Phase | Document | Deliverable | Status |
|-------|----------|-------------|--------|
| **Phase 1** | [phase1_completion_report.md](progress/phase1_completion_report.md) | Data Foundry (single source of truth) | ✅ Complete |
| **Phase 2** | [phase2_completion_report.md](progress/phase2_completion_report.md) | Analysis (deterministic metrics + AI) | ✅ Complete |
| **Phase 3** | [phase3_completion_report.md](progress/phase3_completion_report.md) | Knowledge (skeleton interface) | ✅ Complete |
| **Phase 4** | [phase4_completion_report.md](progress/phase4_completion_report.md) | Agent polish (prompts, tools) | ✅ Complete |
| **Phase 5** | [phase5_summary.md](progress/phase5_summary.md) | End-to-end integration | ✅ Complete |

**Current status**: **v0.1.0 complete** - All 5 phases delivered, 30+ tests passing

**Next phases**: See `planning/next_phase_development_plan.md` for Phases 6-8

---

## 🎯 Decisions (Design Rationale)

**Location**: `decisions/`

Analysis and decisions on specific design choices.

| Document | Topic | Decision |
|----------|-------|----------|
| [matplotlib_plotting_analysis.md](decisions/matplotlib_plotting_analysis.md) | Plotting | Matplotlib analysis |
| [normalized_terminal_plotting.md](decisions/normalized_terminal_plotting.md) | Terminal UI | Terminal-based plotting approach |

**Purpose**: These documents capture the reasoning behind key design decisions, useful for:
- Understanding trade-offs
- Revisiting decisions if requirements change
- Onboarding new contributors

---

## 📦 Archive (Legacy Documents)

**Location**: `archive/`

Historical and superseded documents - kept for reference.

### Legacy Architecture Versions

| Document | Version | Notes |
|----------|---------|-------|
| architecture_v3_final.md | v0.0.3 | Early architecture |
| architecture_v3_final_review.md | v0.0.3 | Architecture review |
| architecture_v3_high_severity_fixes.md | v0.0.3 | Critical fixes |
| current_architecture.md | v0.0.x | Intermediate version |
| architecture_diagram.txt | v0.0.x | ASCII diagram |

### Legacy Plans

| Document | Phase | Notes |
|----------|-------|-------|
| phase2_complete.md | Phase 2 | Early completion report (superseded by phase2_completion_report.md) |
| phase3_skeleton_plan.md | Phase 3 | Planning document (superseded by phase3_completion_report.md) |

### Other Archive

| Document | Notes |
|----------|-------|
| architect_feedback.md | Early architectural feedback |

**Purpose**: Preserved for historical reference, not for active use.

---

## 🗺️ Document Relationships

```
┌────────────────────────────────────────────────────────┐
│                  ARCHITECTURE                          │
│  - universal_architecture.md (CURRENT)                 │
│  - agentic_optimization_vision.md (vision)             │
│  - agent_controlled_optimization.md (deep dive)        │
└───────────────────────┬────────────────────────────────┘
                        │
            ┌───────────┴───────────┐
            │                       │
            ▼                       ▼
┌───────────────────────┐  ┌───────────────────────┐
│      PLANNING         │  │   IMPLEMENTATION      │
│  - next_phase_dev...  │  │  - callback_streaming │
│  - PAOLA_vision       │  │  - cli_architecture   │
└───────────────────────┘  │  - run_architecture   │
                           │  - prompt_caching     │
                           └───────────┬───────────┘
                                       │
                                       ▼
                           ┌───────────────────────┐
                           │      PROGRESS         │
                           │  - phase1_complete    │
                           │  - phase2_complete    │
                           │  - phase3_complete    │
                           │  - phase4_complete    │
                           │  - phase5_summary     │
                           └───────────────────────┘
```

---

## 📚 Reading Paths

### For New Contributors

1. **Start**: `architecture/universal_architecture.md` - Understand the full system
2. **Vision**: `architecture/agentic_optimization_vision.md` - Understand the "why"
3. **Current state**: `progress/phase5_summary.md` - See what's implemented
4. **Next steps**: `planning/next_phase_development_plan.md` - See what's coming

### For Implementers

1. **Architecture**: `architecture/universal_architecture.md` - System overview
2. **Component guide**: Choose from `implementation/`:
   - Events: `callback_streaming_architecture.md`
   - CLI: `cli_architecture.md`
   - Runs: `run_architecture.md`
3. **Implementation plan**: `planning/next_phase_development_plan.md` - Detailed tasks

### For Researchers / External Audience

1. **Vision**: `architecture/agentic_optimization_vision.md` - Core innovation
2. **Technical design**: `architecture/agent_controlled_optimization.md` - Deep dive
3. **Universal architecture**: `architecture/universal_architecture.md` - Full system

---

## 🔍 Search Guide

**Looking for...**

- **"What is PAOLA?"** → `architecture/universal_architecture.md` (Executive Summary)
- **"Why autonomous agent?"** → `architecture/agentic_optimization_vision.md`
- **"How does the agent work?"** → `architecture/agent_controlled_optimization.md` (Section 15)
- **"What's implemented?"** → `progress/phase5_summary.md`
- **"What's next?"** → `planning/next_phase_development_plan.md`
- **"How to implement X?"** → `implementation/` folder
- **"Event system?"** → `implementation/callback_streaming_architecture.md`
- **"CLI commands?"** → `implementation/cli_architecture.md`
- **"Storage/runs?"** → `implementation/run_architecture.md`
- **"LLM optimization?"** → `implementation/prompt_caching_guide.md`
- **"Tool primitives?"** → `architecture/agent_controlled_optimization.md` (Section 7)
- **"Comparison with HEEDS/ModeFRONTIER?"** → `architecture/agent_controlled_optimization.md` (Section 4)
- **"Knowledge base design?"** → `architecture/universal_architecture.md` (Part 3.5)
- **"Optimizer integration?"** → `architecture/universal_architecture.md` (Part 3.3)
- **"Backend interface?"** → `architecture/universal_architecture.md` (Part 3.2)

---

## 📊 Documentation Stats

**Total documents**: 25 (excluding archives)

| Category | Count | Purpose |
|----------|-------|---------|
| Architecture | 4 | System design and vision |
| Planning | 2 | Development roadmap |
| Implementation | 6 | Technical guides |
| Progress | 5 | Completion reports |
| Decisions | 2 | Design rationale |
| Archive | 8 | Historical reference |

**Lines of documentation**: ~120,000+ (excluding code examples)

**Last updated**: December 14, 2025

---

## 🚀 Quick Links

**Most important documents**:
1. [**Universal Architecture**](architecture/universal_architecture.md) - Complete system design
2. [**Next Phase Plan**](planning/next_phase_development_plan.md) - Development roadmap
3. [**Phase 5 Summary**](progress/phase5_summary.md) - Current implementation status

**For daily development**:
- Implementation guides in `implementation/`
- Current phase plan in `planning/`
- Recent progress in `progress/`

---

**Version**: 0.2.0 (Universal Vision)
**Status**: Production-ready (Phases 1-5), Planning complete (Phases 6-8)
**Next milestone**: Phase 6 Week 1 - Flexible problem formulation
