# Phase 3: Knowledge Module (Skeleton Implementation)

**Status**: Planned
**Approach**: Skeleton only - interfaces defined, minimal implementation
**Rationale**: Knowledge module is highly data-driven and needs agent iteration

## Why Skeleton?

The knowledge module cannot be properly designed without:
1. **Real optimization data** - What insights are actually valuable?
2. **Agent usage patterns** - How does the agent want to use knowledge?
3. **Problem diversity** - What problem signatures are discriminative?
4. **Success patterns** - What strategies actually work?

Therefore: Define interfaces now, implement iteratively with real data later.

## What Gets Built

### 1. Module Structure
```
paola/knowledge/
├── __init__.py              # Public API exports
├── knowledge_base.py        # KnowledgeBase class (skeleton)
├── storage.py               # Storage backends (minimal)
└── README.md                # Design intent + future work
```

### 2. KnowledgeBase Class (Interface Only)

**File**: `paola/knowledge/knowledge_base.py`

```python
class KnowledgeBase:
    """
    Knowledge accumulation from past optimizations.

    CURRENT STATUS: Skeleton implementation
    - Interfaces defined and documented
    - Minimal in-memory storage
    - Ready for iteration with real data

    FUTURE WORK:
    - Embedding-based retrieval (RAG)
    - Vector store integration
    - Insight extraction prompts
    - Multi-run pattern analysis
    """

    def __init__(self, storage=None):
        """Initialize knowledge base with storage backend."""
        self.storage = storage or MemoryKnowledgeStorage()

    def store_insight(
        self,
        problem_signature: Dict[str, Any],
        strategy: Dict[str, Any],
        outcome: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store optimization insight.

        Args:
            problem_signature: Problem characteristics
                - dimensions: int
                - constraints_count: int
                - problem_type: str (e.g., "nonlinear", "constrained")
                - physics: Optional[str] (e.g., "fluid", "structural")
            strategy: What was done
                - algorithm: str
                - settings: Dict
                - adaptations: List[Dict]
            outcome: What happened
                - success: bool
                - iterations: int
                - final_objective: float
                - convergence_quality: str
            metadata: Optional context
                - timestamp, user, tags, etc.

        Returns:
            insight_id: Unique identifier for stored insight

        CURRENT: Stores in memory dict
        FUTURE: Embed problem_signature, store in vector DB
        """
        # TODO: Real implementation
        pass

    def retrieve_insights(
        self,
        problem_signature: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant insights for a problem.

        Args:
            problem_signature: Current problem characteristics
            top_k: Number of insights to return

        Returns:
            List of insights ordered by relevance:
            [
                {
                    "insight_id": str,
                    "similarity": float,
                    "problem_signature": Dict,
                    "strategy": Dict,
                    "outcome": Dict,
                    "metadata": Dict,
                }
            ]

        CURRENT: Returns empty list
        FUTURE: Embedding-based similarity search
        """
        # TODO: Real implementation
        return []

    def get_all_insights(self) -> List[Dict[str, Any]]:
        """
        Get all stored insights.

        Returns:
            List of all insights

        CURRENT: Returns from memory dict
        FUTURE: Query from storage backend
        """
        # TODO: Real implementation
        return []

    def clear(self) -> None:
        """Clear all insights (for testing)."""
        # TODO: Real implementation
        pass
```

### 3. Storage Backends (Minimal)

**File**: `paola/knowledge/storage.py`

```python
class KnowledgeStorage(ABC):
    """Abstract storage backend for knowledge."""

    @abstractmethod
    def store(self, insight_id: str, insight: Dict[str, Any]) -> None:
        """Store insight."""
        pass

    @abstractmethod
    def retrieve(self, insight_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve insight by ID."""
        pass

    @abstractmethod
    def list_all(self) -> List[Dict[str, Any]]:
        """List all insights."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all insights."""
        pass


class MemoryKnowledgeStorage(KnowledgeStorage):
    """In-memory knowledge storage (for development/testing)."""

    def __init__(self):
        self._insights: Dict[str, Dict[str, Any]] = {}

    def store(self, insight_id: str, insight: Dict[str, Any]) -> None:
        self._insights[insight_id] = insight

    def retrieve(self, insight_id: str) -> Optional[Dict[str, Any]]:
        return self._insights.get(insight_id)

    def list_all(self) -> List[Dict[str, Any]]:
        return list(self._insights.values())

    def clear(self) -> None:
        self._insights.clear()


class FileKnowledgeStorage(KnowledgeStorage):
    """
    File-based knowledge storage.

    CURRENT: Skeleton only
    FUTURE: JSON file per insight, index file for retrieval
    """

    def __init__(self, base_dir: str = ".paola/knowledge"):
        self.base_dir = Path(base_dir)
        # TODO: Initialize storage

    # TODO: Implement methods
```

### 4. Agent Tools (Minimal)

**File**: `paola/tools/knowledge_tools.py`

```python
@tool
def store_optimization_insight(
    problem_type: str,
    algorithm: str,
    success: bool,
    iterations: int,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Store insight from completed optimization.

    CURRENT: Placeholder - stores minimal data in memory
    FUTURE: Rich problem signatures, strategy details, embeddings

    Use this when:
    - Optimization completed successfully
    - You learned something valuable about the problem/algorithm
    - You want future optimizations to benefit from this experience

    Args:
        problem_type: Type of problem (e.g., "rosenbrock_10d")
        algorithm: Algorithm used (e.g., "SLSQP")
        success: Whether optimization succeeded
        iterations: Number of iterations taken
        notes: Free-form notes about what worked/didn't work

    Returns:
        {"insight_id": str, "status": "stored"}
    """
    # TODO: Real implementation
    return {"insight_id": "placeholder", "status": "not_implemented"}


@tool
def retrieve_optimization_knowledge(
    problem_type: str,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant insights for a problem.

    CURRENT: Placeholder - returns empty list
    FUTURE: Embedding-based similarity search

    Use this when:
    - Starting a new optimization
    - Want to learn from similar past problems
    - Need to warm-start with proven strategies

    Args:
        problem_type: Type of problem to find insights for
        top_k: Number of insights to retrieve

    Returns:
        List of relevant insights
    """
    # TODO: Real implementation
    return []
```

### 5. CLI Commands (Skeleton)

**Add to** `paola/cli/commands.py`:

```python
def handle_knowledge_list(self):
    """List all stored insights."""
    # TODO: Implement
    self.console.print("[yellow]Knowledge module not yet implemented[/yellow]")
    self.console.print("[dim]This is a skeleton - will be built with real optimization data[/dim]")

def handle_knowledge_show(self, insight_id: str):
    """Show detailed insight."""
    # TODO: Implement
    self.console.print("[yellow]Knowledge module not yet implemented[/yellow]")
```

**Add to** `paola/cli/repl.py`:

```python
elif cmd == '/knowledge':
    if len(cmd_parts) == 1:
        self.command_handler.handle_knowledge_list()
    elif cmd_parts[1] == 'show' and len(cmd_parts) > 2:
        self.command_handler.handle_knowledge_show(cmd_parts[2])
    else:
        self.console.print("[red]Usage: /knowledge OR /knowledge show <id>[/red]")
```

### 6. Documentation

**File**: `paola/knowledge/README.md`

```markdown
# Knowledge Module (Skeleton)

## Current Status

This is a **skeleton implementation** - interfaces are defined but not fully implemented.

## Why Skeleton?

The knowledge module is highly data-driven and needs real optimization data to be designed properly:

1. **Problem signatures** - What features discriminate problem classes?
2. **Insight schema** - What information is valuable to store?
3. **Retrieval strategy** - What makes insights "similar"?
4. **Agent patterns** - How does the agent actually use knowledge?

These questions can only be answered by:
- Running many real optimizations
- Observing what the agent learns
- Seeing what patterns emerge

## Design Intent

**Vision**: RAG-based knowledge accumulation
- Agent stores insights from successful optimizations
- Agent retrieves insights when starting similar problems
- Knowledge improves over time (organizational learning)

**Architecture** (planned):
```
Problem → Embedding → Vector Store → Similarity Search → Insights
```

## Implementation Plan

**Phase 3.1** (Current): Skeleton
- Interfaces defined
- Minimal in-memory storage
- Placeholder tools

**Phase 3.2** (After agent iteration):
- Real problem signatures (from observed data)
- Insight extraction (what patterns matter?)
- Simple retrieval (keyword/feature matching)

**Phase 3.3** (Production):
- Embedding models for similarity
- Vector store integration
- Advanced retrieval strategies
- Multi-run pattern analysis

## Current API

See `knowledge_base.py` for interface definitions.

All methods are documented but return placeholders.
```

## Implementation Tasks

**Estimated time**: 2-3 hours (skeleton only)

1. Create `paola/knowledge/` directory
2. Write `knowledge_base.py` with interface definitions
3. Write `storage.py` with MemoryKnowledgeStorage
4. Write `tools/knowledge_tools.py` with placeholder tools
5. Add skeleton CLI commands
6. Write `knowledge/README.md` documenting design intent
7. Update `paola/knowledge/__init__.py` with exports
8. Create basic test showing interfaces work

## Success Criteria

- ✅ Module structure exists
- ✅ All interfaces documented with clear docstrings
- ✅ MemoryKnowledgeStorage works (for testing)
- ✅ Tools can be called (return placeholders)
- ✅ CLI commands exist (show "not implemented" message)
- ✅ README explains why skeleton and what's next
- ✅ Basic test passes (interfaces callable)

## What This Enables

With skeleton in place:
1. **Phase 4** (Agent Polish) can proceed - agent tools exist
2. **Agent experimentation** - Can start thinking about knowledge usage
3. **Iterative development** - Easy to fill in implementation with real data
4. **Interface stability** - Contract defined, implementation flexible

## What NOT to Build

DO NOT implement in Phase 3:
- ❌ Embedding models
- ❌ Vector stores
- ❌ Sophisticated retrieval algorithms
- ❌ Insight extraction prompts
- ❌ Multi-run analysis

These require real data and agent iteration to design properly.

## Next Steps (After Phase 4 & 5)

Once we have:
- Agent working well (Phase 4 polish)
- Multiple optimization runs (Phase 5 integration)
- Real usage patterns

Then return to knowledge module with:
- Observed problem signatures from real runs
- Insight schemas based on what agent learned
- Retrieval strategies based on usage patterns
- Real data to validate embeddings/similarity
