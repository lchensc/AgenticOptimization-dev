# Knowledge Module (Skeleton Implementation)

**Status**: Skeleton - Interfaces defined, minimal implementation
**Purpose**: Enable organizational learning across optimization runs

## Current Status

This is a **skeleton implementation** with:
- ✅ Interfaces fully defined and documented
- ✅ In-memory storage backend (for testing)
- ✅ Agent tools (placeholders)
- ✅ CLI commands (skeleton)
- ⏳ Real implementation pending

## Why Skeleton?

The knowledge module cannot be properly designed without real optimization data. We need to observe:

1. **Problem signatures** - What features actually discriminate problem classes?
   - Is dimensionality enough? Or do we need physics type, constraint structure, etc.?
   - What makes two problems "similar" for warm-starting?

2. **Insight schemas** - What information is valuable to store?
   - Which algorithm settings matter most?
   - What adaptations are worth recording?
   - How to represent "what worked"?

3. **Retrieval strategies** - How to find relevant insights?
   - Simple feature matching? Embedding-based similarity?
   - What embedding model captures optimization problem similarity?

4. **Agent usage patterns** - How does the agent actually use knowledge?
   - When does it query? At problem start? During adaptation?
   - How many insights does it need? Top-1? Top-5?
   - Does it blend multiple strategies or pick the best?

These questions can only be answered by:
- Running many real optimizations (50-100+)
- Observing what the agent learns
- Seeing what patterns emerge
- Measuring what actually helps

## Vision: RAG-Based Learning

**Long-term goal**: Retrieval-Augmented Generation for optimization

```
┌─────────────────┐
│ New Problem     │
│ - dimensions:10 │
│ - constraints:2 │
│ - nonlinear     │
└────────┬────────┘
         │
         v
┌─────────────────────────────┐
│ Embedding Model             │
│ problem → vector            │
└────────┬────────────────────┘
         │
         v
┌─────────────────────────────┐
│ Vector Store                │
│ Similarity Search           │
└────────┬────────────────────┘
         │
         v
┌─────────────────────────────┐
│ Top-K Similar Problems      │
│ 1. Rosenbrock 10D (0.95)   │
│    ✓ SLSQP, 45 iters       │
│ 2. Ackley 12D (0.87)       │
│    ✓ BFGS, 67 iters        │
│ 3. ...                      │
└────────┬────────────────────┘
         │
         v
┌─────────────────────────────┐
│ Agent uses insights to:     │
│ - Warm-start with SLSQP     │
│ - Set ftol=1e-6 (worked)    │
│ - Expect ~50 iterations     │
└─────────────────────────────┘
```

## Architecture (Planned)

### Data Model

**Insight Record**:
```python
{
    "insight_id": "uuid",
    "problem_signature": {
        "dimensions": int,
        "constraints_count": int,
        "problem_type": str,       # "linear", "nonlinear", "mixed"
        "physics": Optional[str],   # "fluid", "structural", etc.
        "objective_type": str,      # "smooth", "noisy", "discontinuous"
        "constraint_types": [str],  # ["equality", "inequality", "bounds"]
    },
    "strategy": {
        "algorithm": str,
        "settings": {
            "ftol": float,
            "maxiter": int,
            # ... algorithm-specific
        },
        "adaptations": [
            {
                "iteration": int,
                "action": str,           # "constraint_tighten", "gradient_switch"
                "reason": str,
                "outcome": str,
            }
        ],
    },
    "outcome": {
        "success": bool,
        "iterations": int,
        "evaluations": int,
        "final_objective": float,
        "convergence_rate": float,
        "constraint_satisfaction": bool,
    },
    "metadata": {
        "timestamp": str,
        "user": Optional[str],
        "tags": [str],
        "notes": str,
    }
}
```

### Embedding Strategy (Future)

**Hand-crafted features** (Phase 3.2):
```python
def problem_to_features(signature):
    """Convert problem to feature vector."""
    return [
        signature["dimensions"],
        signature["constraints_count"],
        1.0 if signature["problem_type"] == "nonlinear" else 0.0,
        # ... more hand-crafted features
    ]
```

**Learned embeddings** (Phase 3.3):
```python
from sentence_transformers import SentenceTransformer

def problem_to_embedding(signature):
    """Convert problem to semantic embedding."""
    text = f"Optimization problem: {signature['problem_type']} "
    text += f"with {signature['dimensions']} dimensions "
    text += f"and {signature['constraints_count']} constraints"

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = model.encode(text)
    return embedding
```

### Storage Backend (Future)

**File-based** (Phase 3.2):
```
.paola/knowledge/
├── index.json              # Metadata index
├── insights/
│   ├── insight_001.json
│   ├── insight_002.json
│   └── ...
└── embeddings/             # Optional cached embeddings
    └── embeddings.npy
```

**Vector store** (Phase 3.3):
```python
from chromadb import Client

client = Client()
collection = client.create_collection("optimization_insights")

# Store
collection.add(
    ids=[insight_id],
    embeddings=[embedding],
    metadatas=[insight]
)

# Retrieve
results = collection.query(
    query_embeddings=[problem_embedding],
    n_results=5
)
```

## Current API

See `knowledge_base.py` for full interface documentation.

### KnowledgeBase

```python
from paola.knowledge import KnowledgeBase

kb = KnowledgeBase()

# Store insight
insight_id = kb.store_insight(
    problem_signature={...},
    strategy={...},
    outcome={...}
)

# Retrieve insights (currently returns empty)
insights = kb.retrieve_insights(
    problem_signature={...},
    top_k=5
)

# Inspect
all_insights = kb.get_all_insights()
count = kb.count()
```

### Agent Tools

```python
from paola.tools.knowledge_tools import (
    store_optimization_insight,
    retrieve_optimization_knowledge
)

# Agent after optimization
result = store_optimization_insight.invoke({
    "problem_type": "rosenbrock",
    "dimensions": 10,
    "algorithm": "SLSQP",
    "success": True,
    "iterations": 45,
    "final_objective": 0.001,
    "notes": "Converged well with ftol=1e-6"
})
# Returns: {"status": "not_implemented", ...}

# Agent before optimization
insights = retrieve_optimization_knowledge.invoke({
    "problem_type": "rosenbrock",
    "dimensions": 10,
    "top_k": 3
})
# Returns: []
```

### CLI Commands

```bash
paola> /knowledge
# Shows skeleton status message

paola> /knowledge show <id>
# Shows "not implemented" message
```

## Implementation Roadmap

### Phase 3.1: Skeleton (Current - Complete)
**Time**: 2-3 hours
**Status**: ✅ Done

- Interfaces defined
- MemoryKnowledgeStorage works
- Tools callable (return placeholders)
- CLI commands exist

### Phase 3.2: Basic Implementation
**Time**: 2-3 days
**Prerequisites**: 20-50 optimization runs with real data

Tasks:
1. Analyze real optimization data to determine:
   - Which problem features discriminate
   - Which strategy details matter
   - What outcomes to track

2. Implement file-based storage:
   - JSON files for insights
   - Index for fast lookup
   - Backup/restore functionality

3. Simple retrieval (no embeddings yet):
   - Exact match on problem_type
   - Filter by dimensions (± tolerance)
   - Rank by outcome quality

4. Agent integration:
   - Automatic insight storage after run
   - Manual retrieval prompting
   - CLI for inspection

### Phase 3.3: Production Features
**Time**: 2-3 weeks
**Prerequisites**: 100+ optimization runs, validated patterns

Tasks:
1. Embedding-based retrieval:
   - Choose embedding model (hand-crafted vs learned)
   - Benchmark on real data
   - Tune similarity thresholds

2. Vector store integration:
   - ChromaDB or FAISS
   - Efficient similarity search
   - Scaling to 1000+ insights

3. Insight extraction:
   - AI-powered pattern detection
   - Multi-run analysis
   - Strategy recommendation

4. Advanced features:
   - Knowledge evolution (insights improve over time)
   - Confidence scores (how reliable is insight?)
   - Context-aware retrieval (user preferences, domain)

## Success Metrics (Future)

When fully implemented, success looks like:

**Immediate impact**:
- 30% faster convergence on similar problems (warm-start)
- 50% reduction in strategy trial-and-error
- Automatic reuse of proven approaches

**Organizational learning**:
- Knowledge base grows with every run
- New team members benefit from past optimizations
- Best practices codified and retrievable

**Metrics to track**:
- Retrieval precision: Are retrieved insights actually helpful?
- Coverage: What % of new problems match existing knowledge?
- Improvement: How much faster are warm-started optimizations?

## Development Notes

### Testing Strategy

**Unit tests** (Phase 3.1 - now):
```python
# Test interfaces work
kb = KnowledgeBase()
insight_id = kb.store_insight(...)
assert insight_id is not None
```

**Integration tests** (Phase 3.2):
```python
# Test with real optimization data
run = platform.load_run(1)
insight_id = kb.store_insight_from_run(run)
insights = kb.retrieve_insights(run.problem_signature)
assert len(insights) > 0
```

**End-to-end tests** (Phase 3.3):
```python
# Test agent uses knowledge
agent.optimize(problem_1)  # Agent stores insight
agent.optimize(problem_2)  # Agent retrieves + uses
assert problem_2.iterations < problem_1.iterations  # Warm-start helped
```

### Data Requirements

Need real optimization runs with:
- Diverse problems (10+ types)
- Multiple algorithms (3+ per problem)
- Success and failure cases
- Detailed iteration history
- Adaptation events recorded

Minimum dataset: 50 runs
Recommended: 100-200 runs
Production: 1000+ runs

## Related Documentation

- `../platform/` - Run storage (what knowledge learns from)
- `../analysis/` - Metrics computation (what knowledge extracts)
- `../agent/` - Agent implementation (who uses knowledge)
- `docs/refactoring_blueprint.md` - Overall architecture plan

## Questions to Answer (With Real Data)

1. **Problem signatures**:
   - What's the minimal discriminative feature set?
   - Do we need physics type or is math structure enough?
   - How important are constraint types vs count?

2. **Strategy representation**:
   - Which algorithm settings actually matter?
   - Are adaptations generalizable across problems?
   - How to represent "explore then exploit" patterns?

3. **Similarity metrics**:
   - Is Euclidean distance on features good enough?
   - Do we need domain-specific similarity?
   - What similarity threshold indicates "retrievable"?

4. **Retrieval strategy**:
   - Top-K vs threshold-based?
   - Blend insights or pick best?
   - Recency bias (newer insights better)?

These will be answered iteratively as we collect data.

## Contact & Iteration

This module will evolve based on:
- Real optimization data
- Agent usage patterns
- User feedback
- Performance metrics

Expected iteration cycles:
1. Run 20 optimizations → Analyze → Update schema
2. Run 50 more → Implement retrieval → Measure impact
3. Run 100 more → Add embeddings → Production ready

**Next milestone**: Collect first 20 optimization runs with diverse problems.
