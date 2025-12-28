# Paola Recording Infrastructure: Professional Implementation Design

**Date**: 2025-12-28
**Status**: Design Document
**Purpose**: Core recording infrastructure for Paola v0.2.0 - the "PyTorch-style" automatic recording system

---

## Research Summary

### Relevant Prior Art

| System | Approach | Key Learning |
|--------|----------|--------------|
| [PyTorch Autograd](https://docs.pytorch.org/docs/stable/notes/autograd.html) | Tensors record operations via DAG of Function objects | Tracing at DATA level, not code level |
| [JAX Tracing](https://docs.jax.dev/en/latest/tracing.html) | Tracer objects wrap arguments and record operations | Jaxpr intermediate representation |
| [OpenTelemetry](https://opentelemetry.io/docs/languages/python/instrumentation/) | Monkey patching + BaseInstrumentor pattern | Per-library instrumentors, entry points |
| [wrapt](https://wrapt.readthedocs.io/) | Universal wrapper protocol (wrapped, instance, args, kwargs) | Safe monkey patching, preserves introspection |
| [joblib Memory](https://joblib.readthedocs.io/en/latest/memory.html) | Cryptographic hashing for numpy arrays | Disk-backed caching for large data |

### Key Insights

1. **PyTorch/JAX**: The recording happens at the DATA level (tensors/tracers), not by instrumenting user code
2. **OpenTelemetry**: Professional instrumentation uses BaseInstrumentor pattern with per-library packages
3. **wrapt**: Safe monkey patching requires careful handling of bound methods, signatures, and introspection
4. **scipy callbacks**: Only called per-iteration, not per-evaluation - insufficient for our needs
5. **numpy caching**: Use `tobytes()` + shape or joblib-style cryptographic hashing

---

## Why Not Use OpenTelemetry Directly?

[OpenTelemetry](https://opentelemetry.io/docs/concepts/signals/traces/) is an observability framework for **distributed systems** (HTTP requests, database queries, microservices). We learn from its architecture but cannot use it directly.

### What OpenTelemetry Does

```
Trace: "User login request"
├── Span: "HTTP POST /login" (200ms)
│   ├── Span: "Validate credentials" (50ms)
│   └── Span: "Query database" (100ms)
│       └── Attribute: sql="SELECT * FROM users WHERE..."
```

**Data model**: Traces → Spans → Attributes (key-value strings/numbers)

### Why It Doesn't Fit Paola

| Requirement | OpenTelemetry | Paola Needs |
|-------------|---------------|-------------|
| **Data type** | Strings, numbers, booleans | **Numpy arrays** (50-10000 floats) |
| **Recording** | Request/response timing | **Every function evaluation** (x → f(x)) |
| **Caching** | Not designed for this | **Critical** - evaluations cost $100-$10000 |
| **Domain** | Distributed systems | **Numerical optimization** |
| **Output** | Send to Jaeger/Zipkin | **Store in Foundry for learning** |
| **Semantics** | "How long did X take?" | **"What optimizer config achieved what result?"** |

### What We Learn From OpenTelemetry

| OTel Pattern | Our Adaptation |
|--------------|----------------|
| `BaseInstrumentor` class | Our `BaseInstrumentor` for scipy/optuna/pymoo |
| Per-library packages | `ScipyInstrumentor`, `OptunaInstrumentor`, etc. |
| Monkey-patching approach | Same - patch `minimize()`, `Study.optimize()` |
| `instrument()` / `uninstrument()` | Same lifecycle management |

### The Right Analogy

| System | Domain | What We Learn |
|--------|--------|---------------|
| **PyTorch autograd** | Deep learning | Recording at data level (tensors → objectives) |
| **OpenTelemetry** | Distributed systems | Instrumentation architecture patterns |
| **Paola** | Numerical optimization | Combines both approaches |

**Paola is essentially "OpenTelemetry for numerical optimization"** - same professional instrumentation patterns, completely different domain and data model.

---

## Dual Recording Architecture

A critical design decision: we record optimizer information through **two complementary channels**.

### The Fundamental Question

Recording serves two purposes:
1. **Evaluations** (x, f(x), gradient) - Essential for caching and learning
2. **Optimizer calls** (package, method, config) - For querying and verification

For #2, we could either:
- **Instrumentation**: Automatically capture what actually executed
- **Script storage**: Save the code LLM wrote, extract metadata later

**Answer: We need BOTH.**

### Why Both? Different Truths

| Source | What It Captures | Nature |
|--------|------------------|--------|
| **Script** | What LLM *intended* to do | Intention |
| **Instrumentation** | What *actually* executed | Reality |

These can diverge:

```python
# LLM writes (with a bug):
result = minimize(f, x0, method='SLSQP', options={'maxiter': 1000})

# LLM self-reports (incorrectly):
paola.complete(f, optimizer='scipy:L-BFGS-B')  # Wrong method!
```

- **Script-only**: We'd store wrong metadata if LLM misreports
- **Instrumentation**: Catches the discrepancy - SLSQP actually ran

### Complementary Value

| Instrumentation Provides | Script Provides |
|-------------------------|-----------------|
| **Verification** - catches LLM mistakes | **Context** - why decisions were made |
| **Efficiency** - O(1) structured queries | **Reproducibility** - re-run exactly |
| **Actual execution** - not just intention | **Fallback** - when instrumentation fails |
| **Human summaries** - instant understanding | **Flexibility** - works with any optimizer |

### The Complete Record

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE NODE RECORD                              │
│                                                                         │
│  1. EVALUATIONS: [(x, f, grad), ...]                                   │
│     └─ Essential, no debate                                            │
│                                                                         │
│  2. INSTRUMENTED DATA: {package, method, config}                       │
│     └─ What ACTUALLY ran (verification)                                │
│     └─ Efficient querying at scale                                     │
│     └─ Human-readable summaries                                        │
│                                                                         │
│  3. SCRIPT: "<Python source>"                                          │
│     └─ WHY decisions were made (context)                               │
│     └─ Reproducibility (re-run exactly)                                │
│     └─ Fallback when instrumentation fails                             │
│                                                                         │
│  4. LLM REASONING: "Chose SLSQP because..."                            │
│     └─ High-level strategy explanation                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Verification Mode

When both instrumentation AND LLM self-report exist, we cross-check:

```python
def complete(
    obj: RecordingObjective,
    script: Optional[str] = None,
    optimizer: Optional[str] = None,  # LLM's claim
    reasoning: Optional[str] = None,
) -> Dict[str, Any]:
    """Finalize with verification."""

    instrumented = obj._optimizer_calls  # What actually ran
    reported = optimizer                  # What LLM claims

    # Cross-check
    if instrumented and reported:
        actual_method = instrumented[0].method
        claimed_method = reported.split(':')[1] if ':' in reported else reported

        if actual_method.lower() != claimed_method.lower():
            warnings.warn(
                f"Discrepancy detected!\n"
                f"  Instrumented: {actual_method}\n"
                f"  LLM reported: {claimed_method}\n"
                f"Using instrumented value (ground truth)."
            )

    # Trust hierarchy: instrumented > reported > inferred
    if instrumented:
        final_optimizer = f"{instrumented[0].package}:{instrumented[0].method}"
        final_config = instrumented[0].config
        source = 'instrumented'
    elif reported:
        final_optimizer = reported
        final_config = {}
        source = 'reported'
    else:
        final_optimizer = 'unknown'
        final_config = {}
        source = 'none'

    # Store everything
    node = {
        'optimizer': final_optimizer,
        'optimizer_source': source,  # Transparency about data origin
        'config': final_config,
        'script': script,            # Full source for reproduction
        'reasoning': reasoning,
        'instrumented_calls': [vars(c) for c in instrumented],
        'llm_reported': reported,
        # ... evaluations, best_x, etc.
    }
```

### Failure Modes and Handling

| Scenario | What Happens |
|----------|--------------|
| Instrumentation works, LLM reports correctly | ✅ Verified data, high confidence |
| Instrumentation works, LLM reports wrongly | ⚠️ Warning logged, use instrumented |
| Instrumentation fails, LLM reports | 📝 Use reported, note lower confidence |
| Instrumentation fails, no report | 🔄 Extract from script via LLM, or mark unknown |

### Efficiency Comparison

**For learning across 1000s of runs:**

Without structured data (script-only):
```
For each past optimization:
    1. Load script
    2. LLM call to parse/extract optimizer info
    3. Aggregate

Cost: O(n) LLM calls, slow, expensive
```

With instrumented data:
```sql
SELECT optimizer, AVG(best_f), COUNT(*)
FROM nodes
WHERE n_dim BETWEEN 40 AND 60
GROUP BY optimizer

Cost: O(1), instant
```

### Analogies

| Domain | Intention | Reality | Why Both? |
|--------|-----------|---------|-----------|
| **Databases** | Query submitted | Write-ahead log | Recovery, audit |
| **Aviation** | Flight plan | Black box | Investigation |
| **Finance** | Transaction request | Audit trail | Compliance |
| **Paola** | Script (LLM code) | Instrumentation | Verification, efficiency |

### Design Principle

**"Trust but verify"**

- **Script** is the intention and context
- **Instrumentation** is the verification layer
- **Together** they provide complete, verified, reproducible records

When instrumentation works: We have verified, efficient, structured data.
When instrumentation fails: We fall back to script + LLM extraction.

This is defense in depth for data integrity.

---

## Architecture Overview

The architecture implements **dual recording**: instrumentation captures reality, scripts capture intention.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LLM-GENERATED CODE                                │
│                                                                             │
│   f = paola.objective(problem_id=7)                                        │
│   result = scipy.optimize.minimize(f, x0, method='SLSQP')                  │
│   paola.complete(f, script=__source__, reasoning="...")                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
         │                              │                              │
         │ evaluations                  │ optimizer calls              │ script + reasoning
         │ (x, f, grad)                 │ (auto-captured)              │ (LLM-provided)
         ▼                              ▼                              ▼
┌──────────────────────────────────┐  ┌──────────────────────────┐  ┌──────────────────┐
│      RECORDING OBJECTIVE         │  │  INSTRUMENTATION LAYER   │  │  SCRIPT STORAGE  │
│                                  │  │                          │  │                  │
│  RecordingObjective              │  │  ScipyInstrumentor       │  │  source: str     │
│    ├─ __call__(x) → records      │  │  OptunaInstrumentor      │  │  reasoning: str  │
│    ├─ history: List[EvalRecord]  │  │  PymooInstrumentor       │  │  llm_reported:   │
│    ├─ optimizer_calls: List      │◄─│  CyipoptInstrumentor     │  │    optimizer     │
│    └─ cache integration          │  │                          │  │                  │
└──────────────────────────────────┘  └──────────────────────────┘  └──────────────────┘
         │                                       │                              │
         │                                       │                              │
         └───────────────────────────────────────┴──────────────────────────────┘
                                                 │
                                                 ▼
                              ┌──────────────────────────────────────────────────┐
                              │              VERIFICATION LAYER                   │
                              │                                                  │
                              │  Cross-check: instrumented vs. LLM-reported      │
                              │  Trust hierarchy: instrumented > reported > none │
                              │  Flag discrepancies for debugging                │
                              │                                                  │
                              └──────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                              ┌──────────────────────────────────────────────────┐
                              │              CACHE LAYER                          │
                              │                                                  │
                              │  ArrayHasher: hash(x) → stable key               │
                              │  EvaluationCache: memory + disk                  │
                              │                                                  │
                              └──────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                              ┌──────────────────────────────────────────────────┐
                              │              FOUNDRY LAYER                        │
                              │                                                  │
                              │  Node Record:                                    │
                              │    ├─ evaluations: [(x, f, grad), ...]          │
                              │    ├─ optimizer: "scipy:SLSQP" (verified)       │
                              │    ├─ optimizer_source: "instrumented"          │
                              │    ├─ config: {ftol: 1e-6, ...}                 │
                              │    ├─ script: "<full source>"                   │
                              │    ├─ reasoning: "Chose SLSQP because..."       │
                              │    └─ best_x, best_f, n_evals, wall_time        │
                              │                                                  │
                              └──────────────────────────────────────────────────┘
```

### Data Flow Summary

| Data | Source | Purpose |
|------|--------|---------|
| `evaluations` | RecordingObjective.__call__ | Learning, caching |
| `optimizer`, `config` | Instrumentation (primary) | Verified structured queries |
| `script` | LLM provides | Reproduction, context, fallback |
| `reasoning` | LLM provides | High-level strategy understanding |
| `optimizer_source` | System | Transparency about data origin |

---

## Layer 1: Core Recording

### 1.1 Data Structures

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import time

@dataclass
class EvaluationRecord:
    """Single evaluation of the objective function."""
    x: np.ndarray                    # Input vector (copy)
    f: float                         # Objective value
    timestamp: float                 # Unix timestamp
    eval_number: int                 # Sequential evaluation number
    cached: bool                     # Was this a cache hit?
    cost: float                      # Wall time for this evaluation (0 if cached)
    gradient: Optional[np.ndarray] = None  # If gradient was computed

    def to_dict(self) -> Dict[str, Any]:
        return {
            'x': self.x.tolist(),
            'f': self.f,
            'timestamp': self.timestamp,
            'eval_number': self.eval_number,
            'cached': self.cached,
            'cost': self.cost,
            'gradient': self.gradient.tolist() if self.gradient is not None else None
        }


@dataclass
class OptimizerCall:
    """Record of an optimizer invocation (captured by instrumentation)."""
    package: str                     # 'scipy', 'optuna', 'pymoo', 'cyipopt'
    method: str                      # 'SLSQP', 'TPESampler', 'NSGA2', etc.
    config: Dict[str, Any]           # Options passed to optimizer
    bounds: Optional[Any] = None     # Bounds if provided
    constraints: Optional[Any] = None # Constraints if provided
    eval_start: int = 0              # Evaluation number when this call started
    timestamp: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecordingSession:
    """Complete recording of an optimization run."""
    problem_id: int
    graph_id: int
    evaluations: List[EvaluationRecord] = field(default_factory=list)
    optimizer_calls: List[OptimizerCall] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    reasoning: Optional[str] = None
    code: Optional[str] = None       # Generated code (for reproduction)
```

### 1.2 RecordingObjective

```python
class RecordingObjective:
    """
    The core recording mechanism - like a PyTorch tensor with requires_grad=True.

    Records all evaluations and optimizer calls automatically.
    """

    def __init__(
        self,
        evaluator: Callable[[np.ndarray], float],
        problem_id: int,
        graph_id: int,
        cache: 'EvaluationCache',
        hasher: 'ArrayHasher',
    ):
        self._evaluator = evaluator
        self._problem_id = problem_id
        self._graph_id = graph_id
        self._cache = cache
        self._hasher = hasher

        # Recording state
        self._history: List[EvaluationRecord] = []
        self._optimizer_calls: List[OptimizerCall] = []
        self._start_time = time.time()
        self._finalized = False

        # Thread safety (for parallel evaluations)
        self._lock = threading.Lock()

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate objective - ALL calls go through here.

        This is the key recording point - like tensor operations in PyTorch.
        """
        if self._finalized:
            raise RuntimeError("RecordingObjective already finalized")

        x = np.asarray(x, dtype=np.float64)
        x_copy = x.copy()  # Ensure we have our own copy

        # Create record
        record = EvaluationRecord(
            x=x_copy,
            f=0.0,  # Will be filled
            timestamp=time.time(),
            eval_number=len(self._history),
            cached=False,
            cost=0.0
        )

        # Check cache
        cache_key = self._hasher.hash(x_copy)
        cached_result = self._cache.get(self._problem_id, cache_key)

        if cached_result is not None:
            record.f = cached_result
            record.cached = True
            record.cost = 0.0
        else:
            # Evaluate (expensive!)
            t0 = time.perf_counter()
            record.f = self._evaluator(x)
            record.cost = time.perf_counter() - t0

            # Store in cache
            self._cache.store(self._problem_id, cache_key, record.f)

        # Thread-safe append
        with self._lock:
            self._history.append(record)

        return record.f

    def _record_optimizer_call(
        self,
        package: str,
        method: str,
        config: Dict[str, Any],
        **extra
    ) -> None:
        """
        Called by instrumentation patches when an optimizer is invoked.

        The user never calls this - it's called automatically by patched functions.
        """
        call = OptimizerCall(
            package=package,
            method=method,
            config=config,
            eval_start=len(self._history),
            **extra
        )

        with self._lock:
            self._optimizer_calls.append(call)

    def get_best(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Return best (x, f) from history."""
        if not self._history:
            return None, None

        best_idx = np.argmin([h.f for h in self._history])
        best = self._history[best_idx]
        return best.x.copy(), best.f

    def get_stats(self) -> Dict[str, Any]:
        """Return execution statistics."""
        if not self._history:
            return {}

        n_cached = sum(1 for h in self._history if h.cached)
        total_cost = sum(h.cost for h in self._history)

        return {
            'n_evaluations': len(self._history),
            'n_cached': n_cached,
            'n_actual': len(self._history) - n_cached,
            'cache_hit_rate': n_cached / len(self._history),
            'total_cost': total_cost,
            'wall_time': time.time() - self._start_time,
            'cost_saved': n_cached * (total_cost / max(1, len(self._history) - n_cached))
        }

    @property
    def history(self) -> List[EvaluationRecord]:
        """Read-only access to evaluation history."""
        return self._history.copy()

    @property
    def optimizer_calls(self) -> List[OptimizerCall]:
        """Read-only access to optimizer call history."""
        return self._optimizer_calls.copy()
```

---

## Layer 2: Caching

### 2.1 Array Hashing

```python
import hashlib
from typing import Union

class ArrayHasher:
    """
    Efficient and stable hashing for numpy arrays.

    Based on joblib's approach but simplified for our use case.
    """

    def __init__(self, tolerance: float = 1e-10):
        self._tolerance = tolerance

    def hash(self, x: np.ndarray) -> str:
        """
        Create a stable hash key for a numpy array.

        Uses:
        1. Shape for structural identity
        2. tobytes() for content
        3. Quantization for numerical stability
        """
        x = np.asarray(x, dtype=np.float64)

        # Quantize to avoid floating-point noise
        if self._tolerance > 0:
            x = np.round(x / self._tolerance) * self._tolerance

        # Create hash from shape + bytes
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(str(x.shape).encode())
        hasher.update(x.tobytes())

        return hasher.hexdigest()

    def hash_with_metadata(
        self,
        x: np.ndarray,
        problem_id: int
    ) -> str:
        """Hash including problem context."""
        hasher = hashlib.blake2b(digest_size=16)
        hasher.update(str(problem_id).encode())
        hasher.update(str(x.shape).encode())
        hasher.update(np.asarray(x, dtype=np.float64).tobytes())
        return hasher.hexdigest()
```

### 2.2 Evaluation Cache

```python
from abc import ABC, abstractmethod
import json
from pathlib import Path

class CacheBackend(ABC):
    """Abstract cache backend interface."""

    @abstractmethod
    def get(self, key: str) -> Optional[float]:
        pass

    @abstractmethod
    def store(self, key: str, value: float) -> None:
        pass

    @abstractmethod
    def contains(self, key: str) -> bool:
        pass


class MemoryCache(CacheBackend):
    """In-memory cache for fast access."""

    def __init__(self, max_size: int = 100000):
        self._cache: Dict[str, float] = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[float]:
        return self._cache.get(key)

    def store(self, key: str, value: float) -> None:
        if len(self._cache) >= self._max_size:
            # Simple eviction: remove oldest 10%
            keys_to_remove = list(self._cache.keys())[:self._max_size // 10]
            for k in keys_to_remove:
                del self._cache[k]
        self._cache[key] = value

    def contains(self, key: str) -> bool:
        return key in self._cache


class DiskCache(CacheBackend):
    """Disk-backed cache for persistence across sessions."""

    def __init__(self, cache_dir: Path):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self._cache_dir / 'index.json'
        self._index = self._load_index()

    def _load_index(self) -> Dict[str, str]:
        if self._index_file.exists():
            return json.loads(self._index_file.read_text())
        return {}

    def _save_index(self) -> None:
        self._index_file.write_text(json.dumps(self._index))

    def get(self, key: str) -> Optional[float]:
        if key not in self._index:
            return None
        cache_file = self._cache_dir / self._index[key]
        if cache_file.exists():
            return float(cache_file.read_text())
        return None

    def store(self, key: str, value: float) -> None:
        filename = f"{key}.txt"
        cache_file = self._cache_dir / filename
        cache_file.write_text(str(value))
        self._index[key] = filename
        self._save_index()

    def contains(self, key: str) -> bool:
        return key in self._index


class EvaluationCache:
    """
    Two-tier cache: memory (fast) + disk (persistent).

    Evaluation results are cached by (problem_id, x_hash).
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        memory_size: int = 100000
    ):
        self._memory = MemoryCache(max_size=memory_size)
        self._disk = DiskCache(cache_dir) if cache_dir else None

        # Statistics
        self._hits = 0
        self._misses = 0

    def get(self, problem_id: int, x_hash: str) -> Optional[float]:
        """Get cached evaluation result."""
        key = f"{problem_id}:{x_hash}"

        # Try memory first
        result = self._memory.get(key)
        if result is not None:
            self._hits += 1
            return result

        # Try disk
        if self._disk:
            result = self._disk.get(key)
            if result is not None:
                self._memory.store(key, result)  # Promote to memory
                self._hits += 1
                return result

        self._misses += 1
        return None

    def store(self, problem_id: int, x_hash: str, value: float) -> None:
        """Store evaluation result in cache."""
        key = f"{problem_id}:{x_hash}"
        self._memory.store(key, value)
        if self._disk:
            self._disk.store(key, value)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
```

---

## Layer 3: Instrumentation

### 3.1 Base Instrumentor (OpenTelemetry-style)

```python
from abc import ABC, abstractmethod
import wrapt

class BaseInstrumentor(ABC):
    """
    Base class for library instrumentation.

    Following OpenTelemetry's pattern for professional instrumentation.
    """

    _is_instrumented: bool = False

    @abstractmethod
    def instrumentation_dependencies(self) -> List[str]:
        """Return list of packages this instrumentor patches."""
        pass

    @abstractmethod
    def _instrument(self) -> None:
        """Apply patches. Called by instrument()."""
        pass

    @abstractmethod
    def _uninstrument(self) -> None:
        """Remove patches. Called by uninstrument()."""
        pass

    def instrument(self) -> None:
        """Instrument the library (apply patches)."""
        if self._is_instrumented:
            return

        # Check dependencies are available
        for dep in self.instrumentation_dependencies():
            try:
                __import__(dep)
            except ImportError:
                return  # Silently skip if library not installed

        self._instrument()
        self._is_instrumented = True

    def uninstrument(self) -> None:
        """Remove instrumentation (unapply patches)."""
        if not self._is_instrumented:
            return
        self._uninstrument()
        self._is_instrumented = False
```

### 3.2 SciPy Instrumentor

```python
class ScipyInstrumentor(BaseInstrumentor):
    """Instrumentation for scipy.optimize."""

    _original_minimize = None
    _original_differential_evolution = None

    def instrumentation_dependencies(self) -> List[str]:
        return ['scipy']

    def _instrument(self) -> None:
        import scipy.optimize

        # Store originals
        self._original_minimize = scipy.optimize.minimize
        self._original_differential_evolution = scipy.optimize.differential_evolution

        # Apply patches
        scipy.optimize.minimize = self._patched_minimize
        scipy.optimize.differential_evolution = self._patched_differential_evolution

    def _uninstrument(self) -> None:
        import scipy.optimize

        if self._original_minimize:
            scipy.optimize.minimize = self._original_minimize
        if self._original_differential_evolution:
            scipy.optimize.differential_evolution = self._original_differential_evolution

    def _patched_minimize(self, fun, x0, args=(), method=None, **kwargs):
        """Patched scipy.optimize.minimize that records optimizer calls."""

        # Check if fun is a RecordingObjective
        if isinstance(fun, RecordingObjective):
            # Extract configuration
            config = {}
            if 'options' in kwargs:
                config.update(kwargs['options'])
            if 'tol' in kwargs:
                config['tol'] = kwargs['tol']

            # Record the optimizer call
            fun._record_optimizer_call(
                package='scipy',
                method=method or 'BFGS',  # Default method
                config=config,
                bounds=kwargs.get('bounds'),
                constraints=kwargs.get('constraints'),
                jac=kwargs.get('jac'),
                hess=kwargs.get('hess'),
            )

        # Call original
        return self._original_minimize(fun, x0, args=args, method=method, **kwargs)

    def _patched_differential_evolution(self, func, bounds, **kwargs):
        """Patched scipy.optimize.differential_evolution."""

        if isinstance(func, RecordingObjective):
            config = {
                'strategy': kwargs.get('strategy', 'best1bin'),
                'maxiter': kwargs.get('maxiter', 1000),
                'popsize': kwargs.get('popsize', 15),
                'tol': kwargs.get('tol', 0.01),
                'mutation': kwargs.get('mutation', (0.5, 1)),
                'recombination': kwargs.get('recombination', 0.7),
            }

            func._record_optimizer_call(
                package='scipy',
                method='differential_evolution',
                config=config,
                bounds=bounds,
            )

        return self._original_differential_evolution(func, bounds, **kwargs)
```

### 3.3 Optuna Instrumentor

```python
class OptunaInstrumentor(BaseInstrumentor):
    """Instrumentation for Optuna."""

    _original_optimize = None

    def instrumentation_dependencies(self) -> List[str]:
        return ['optuna']

    def _instrument(self) -> None:
        import optuna

        self._original_optimize = optuna.study.Study.optimize
        optuna.study.Study.optimize = self._patched_optimize

    def _uninstrument(self) -> None:
        import optuna

        if self._original_optimize:
            optuna.study.Study.optimize = self._original_optimize

    def _patched_optimize(self, func, n_trials=None, timeout=None, **kwargs):
        """Patched Study.optimize that records optimizer calls."""

        # func might be a wrapper around RecordingObjective
        # We need to detect the RecordingObjective
        recording_obj = self._extract_recording_objective(func)

        if recording_obj is not None:
            # Get sampler info
            sampler = self.sampler
            sampler_name = sampler.__class__.__name__

            # Extract sampler config
            config = {
                'n_trials': n_trials,
                'timeout': timeout,
                'sampler': sampler_name,
            }

            # Try to get sampler-specific params
            if hasattr(sampler, '_n_startup_trials'):
                config['n_startup_trials'] = sampler._n_startup_trials

            recording_obj._record_optimizer_call(
                package='optuna',
                method=sampler_name.replace('Sampler', ''),  # 'TPESampler' -> 'TPE'
                config=config,
            )

        return self._original_optimize(self, func, n_trials=n_trials, timeout=timeout, **kwargs)

    def _extract_recording_objective(self, func):
        """Extract RecordingObjective from various wrapper patterns."""
        if isinstance(func, RecordingObjective):
            return func

        # Check if it's a closure that contains a RecordingObjective
        if hasattr(func, '__closure__') and func.__closure__:
            for cell in func.__closure__:
                if isinstance(cell.cell_contents, RecordingObjective):
                    return cell.cell_contents

        return None
```

### 3.4 Pymoo Instrumentor

```python
class PymooInstrumentor(BaseInstrumentor):
    """Instrumentation for pymoo."""

    _original_minimize = None

    def instrumentation_dependencies(self) -> List[str]:
        return ['pymoo']

    def _instrument(self) -> None:
        from pymoo.optimize import minimize as pymoo_minimize
        import pymoo.optimize

        self._original_minimize = pymoo_minimize
        pymoo.optimize.minimize = self._patched_minimize

    def _uninstrument(self) -> None:
        import pymoo.optimize

        if self._original_minimize:
            pymoo.optimize.minimize = self._original_minimize

    def _patched_minimize(self, problem, algorithm, termination=None, **kwargs):
        """Patched pymoo.optimize.minimize."""

        # Check if problem uses a RecordingObjective
        recording_obj = self._extract_recording_objective(problem)

        if recording_obj is not None:
            algo_name = algorithm.__class__.__name__

            config = {
                'algorithm': algo_name,
            }

            # Extract algorithm-specific params
            if hasattr(algorithm, 'pop_size'):
                config['pop_size'] = algorithm.pop_size
            if termination is not None:
                config['termination'] = str(termination)

            recording_obj._record_optimizer_call(
                package='pymoo',
                method=algo_name,
                config=config,
            )

        return self._original_minimize(problem, algorithm, termination=termination, **kwargs)
```

### 3.5 Instrumentation Manager

```python
class InstrumentationManager:
    """
    Central manager for all instrumentation.

    Handles registration, activation, and lifecycle of instrumentors.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._instrumentors = {}
            cls._instance._is_active = False
        return cls._instance

    def register(self, name: str, instrumentor: BaseInstrumentor) -> None:
        """Register an instrumentor."""
        self._instrumentors[name] = instrumentor

    def instrument_all(self) -> None:
        """Activate all registered instrumentors."""
        if self._is_active:
            return

        for name, inst in self._instrumentors.items():
            try:
                inst.instrument()
            except Exception as e:
                # Log but don't fail - some libraries may not be installed
                pass

        self._is_active = True

    def uninstrument_all(self) -> None:
        """Deactivate all instrumentors."""
        for name, inst in self._instrumentors.items():
            try:
                inst.uninstrument()
            except Exception:
                pass

        self._is_active = False

    @property
    def is_active(self) -> bool:
        return self._is_active


# Global instance
_manager = InstrumentationManager()

def get_instrumentation_manager() -> InstrumentationManager:
    return _manager
```

---

## Layer 4: Public API

### 4.1 Main API Functions

```python
# paola/__init__.py

from .recording import RecordingObjective
from .cache import EvaluationCache, ArrayHasher
from .instrumentation import get_instrumentation_manager
from .foundry import get_foundry

# Module-level state
_cache: Optional[EvaluationCache] = None
_hasher: Optional[ArrayHasher] = None
_initialized: bool = False


def _ensure_initialized() -> None:
    """Lazy initialization of Paola infrastructure."""
    global _cache, _hasher, _initialized

    if _initialized:
        return

    # Initialize cache
    foundry = get_foundry()
    _cache = EvaluationCache(
        cache_dir=foundry.cache_dir,
        memory_size=100000
    )
    _hasher = ArrayHasher(tolerance=1e-10)

    # Activate instrumentation
    manager = get_instrumentation_manager()
    manager.register('scipy', ScipyInstrumentor())
    manager.register('optuna', OptunaInstrumentor())
    manager.register('pymoo', PymooInstrumentor())
    manager.register('cyipopt', CyipoptInstrumentor())
    manager.instrument_all()

    _initialized = True


def objective(
    problem_id: int,
    graph_id: Optional[int] = None,
) -> RecordingObjective:
    """
    Create a recording objective for optimization.

    This is the main entry point - like creating a tensor with requires_grad=True.

    Args:
        problem_id: The problem to optimize
        graph_id: Graph to record to (auto-created if None)

    Returns:
        RecordingObjective that records all evaluations and optimizer calls

    Example:
        f = paola.objective(problem_id=7)
        result = scipy.optimize.minimize(f, x0, method='SLSQP')
        paola.complete(f)
    """
    _ensure_initialized()

    foundry = get_foundry()

    # Create graph if not provided
    if graph_id is None:
        graph_id = foundry.create_graph(problem_id)

    # Get evaluator
    evaluator = foundry.get_evaluator(problem_id)

    return RecordingObjective(
        evaluator=evaluator,
        problem_id=problem_id,
        graph_id=graph_id,
        cache=_cache,
        hasher=_hasher,
    )


def complete(
    obj: RecordingObjective,
    script: Optional[str] = None,
    optimizer: Optional[str] = None,  # LLM's self-report (for verification)
    reasoning: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Finalize a recording objective and store results.

    Implements DUAL RECORDING:
    - Instrumentation: What actually executed (verified)
    - Script: What LLM wrote (reproducible, context)

    Args:
        obj: The RecordingObjective to finalize
        script: The optimization code LLM wrote (for reproduction)
        optimizer: LLM's claim of what optimizer was used (for verification)
        reasoning: Explanation of strategy choice

    Returns:
        Summary of the optimization run

    Example:
        f = paola.objective(problem_id=7)
        result = scipy.optimize.minimize(f, x0, method='SLSQP')
        summary = paola.complete(f,
            script=optimization_code,
            optimizer='scipy:SLSQP',
            reasoning="SLSQP chosen for smooth constrained problem"
        )
    """
    if obj._finalized:
        raise RuntimeError("RecordingObjective already finalized")

    obj._finalized = True

    # Extract results
    best_x, best_f = obj.get_best()
    stats = obj.get_stats()

    # DUAL RECORDING: Determine optimizer with verification
    instrumented = obj._optimizer_calls
    reported = optimizer

    # Cross-check instrumented vs reported
    if instrumented and reported:
        actual = instrumented[0].method
        claimed = reported.split(':')[1] if ':' in reported else reported
        if actual.lower() != claimed.lower():
            warnings.warn(
                f"Optimizer discrepancy detected!\n"
                f"  Instrumented: {actual}\n"
                f"  LLM reported: {claimed}\n"
                f"Using instrumented value (ground truth)."
            )

    # Trust hierarchy: instrumented > reported > unknown
    if instrumented:
        primary = instrumented[0]
        final_optimizer = f"{primary.package}:{primary.method}"
        final_config = primary.config
        optimizer_source = 'instrumented'
    elif reported:
        final_optimizer = reported
        final_config = {}
        optimizer_source = 'reported'
    else:
        final_optimizer = 'unknown'
        final_config = {}
        optimizer_source = 'none'
        warnings.warn(
            "No optimizer detected. Consider using explicit API or checking instrumentation."
        )

    # Create node record with BOTH instrumented data AND script
    node = {
        # Verified optimizer info
        'optimizer': final_optimizer,
        'optimizer_source': optimizer_source,
        'config': final_config,

        # LLM-provided context
        'script': script,
        'reasoning': reasoning or '',
        'llm_reported_optimizer': reported,

        # Evaluation data
        'evaluations': [e.to_dict() for e in obj._history],
        'instrumented_calls': [vars(c) for c in instrumented],

        # Summary stats
        'best_x': best_x.tolist() if best_x is not None else None,
        'best_f': best_f,
        'n_evaluations': stats['n_evaluations'],
        'n_cached': stats['n_cached'],
        'wall_time': stats['wall_time'],
    }

    # Store in foundry
    foundry = get_foundry()
    foundry.add_node(obj._graph_id, node)

    return {
        'graph_id': obj._graph_id,
        'optimizer': final_optimizer,
        'optimizer_source': optimizer_source,
        'best_x': best_x,
        'best_f': best_f,
        **stats
    }
```

### 4.2 Context Manager API (Optional)

```python
from contextlib import contextmanager

@contextmanager
def optimize(
    problem_id: int,
    graph_id: Optional[int] = None,
    script: Optional[str] = None,
    reasoning: Optional[str] = None,
):
    """
    Context manager for optimization with automatic finalization.

    Supports dual recording - script provided for reproduction.

    Example:
        with paola.optimize(problem_id=7, reasoning="Testing SLSQP") as f:
            result = scipy.optimize.minimize(f, x0, method='SLSQP')
        # Automatically finalized on exit with instrumented data
    """
    obj = objective(problem_id=problem_id, graph_id=graph_id)

    try:
        yield obj
    except Exception as e:
        # Record the error but re-raise
        obj._finalized = True
        foundry = get_foundry()
        foundry.add_node(obj._graph_id, {
            'status': 'error',
            'error': str(e),
            'script': script,
            'evaluations': [ev.to_dict() for ev in obj._history],
            'instrumented_calls': [vars(c) for c in obj._optimizer_calls],
        })
        raise
    else:
        complete(obj, script=script, reasoning=reasoning)
```

---

## Usage Examples

**Design Principle**: One LLM turn = One node. The LLM inspects checkpoint results and decides what comes next.

### Multi-Turn Optimization (The Core Pattern)

```python
# ════════════════════════════════════════════════════════════════
# TURN 1: LLM explores with Optuna TPE
# ════════════════════════════════════════════════════════════════

SCRIPT_T1 = '''
import paola
import optuna
import numpy as np

f = paola.objective(problem_id=7, goal="Minimize drag coefficient")

def optuna_objective(trial):
    x = [trial.suggest_float(f'x{i}', -5, 5) for i in range(50)]
    return f(np.array(x))

study = optuna.create_study()
study.optimize(optuna_objective, n_trials=100)
'''

# Execute the script
f = paola.objective(problem_id=7, goal="Minimize drag coefficient")

def optuna_objective(trial):
    x = [trial.suggest_float(f'x{i}', -5, 5) for i in range(50)]
    return f(np.array(x))

study = optuna.create_study()
study.optimize(optuna_objective, n_trials=100)

# Checkpoint returns summary for LLM inspection
summary = paola.checkpoint(f,
    script=SCRIPT_T1,
    reasoning="Global exploration with TPE to find promising region"
)
# Returns: {
#   'graph_id': 42, 'node_id': 'n1',
#   'best_x': [...], 'best_f': 0.523,
#   'n_evaluations': 100, 'optimizer': 'optuna:TPE',
#   'optimizer_source': 'instrumented'
# }

# ════════════════════════════════════════════════════════════════
# LLM INSPECTS: best_f=0.523, 100 evaluations
# LLM DECIDES: "Found promising region, switch to gradient method"
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# TURN 2: LLM refines with L-BFGS-B
# ════════════════════════════════════════════════════════════════

SCRIPT_T2 = '''
import paola
import numpy as np
from scipy.optimize import minimize

f = paola.continue_graph(42, parent_node="n1", edge_type="warm_start")
# Uses best_x from n1 as x0
result = minimize(f, f.get_warm_start(), method='L-BFGS-B')
'''

# Continue from node n1
f = paola.continue_graph(42, parent_node="n1", edge_type="warm_start")

# Warm start from parent's best solution
x0 = f.get_warm_start()  # Returns parent node's best_x
result = minimize(f, x0, method='L-BFGS-B')

summary = paola.checkpoint(f,
    script=SCRIPT_T2,
    reasoning="Gradient refinement from TPE's best solution"
)
# Returns: {
#   'graph_id': 42, 'node_id': 'n2',
#   'best_x': [...], 'best_f': 0.312,
#   'n_evaluations': 47, 'optimizer': 'scipy:L-BFGS-B',
#   'optimizer_source': 'instrumented'
# }

# ════════════════════════════════════════════════════════════════
# LLM INSPECTS: best_f=0.312, improvement from 0.523
# LLM DECIDES: "Good convergence, finalize"
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# TURN 3: Finalize
# ════════════════════════════════════════════════════════════════

result = paola.finalize_graph(42)
# Final graph: n1 (Optuna TPE) → n2 (L-BFGS-B) [chain pattern]
```

### Simple One-Node Optimization

For simple problems, one turn may suffice - use `paola.complete()` as shorthand:

```python
SCRIPT = '''
import paola
import numpy as np
from scipy.optimize import minimize

f = paola.objective(problem_id=7)
x0 = np.zeros(50)
result = minimize(f, x0, method='SLSQP', options={'ftol': 1e-6})
'''

f = paola.objective(problem_id=7)
x0 = np.zeros(50)
result = minimize(f, x0, method='SLSQP', options={'ftol': 1e-6})

# complete() = checkpoint() + finalize_graph()
summary = paola.complete(f,
    script=SCRIPT,
    reasoning="Direct gradient solve for smooth convex problem"
)
```

### Branching Strategy (Multiple Paths)

```python
# ════════════════════════════════════════════════════════════════
# TURN 1: Global exploration
# ════════════════════════════════════════════════════════════════

f = paola.objective(problem_id=7)
study = optuna.create_study()
study.optimize(optuna_objective, n_trials=100)
summary1 = paola.checkpoint(f, script=SCRIPT_EXPLORE, reasoning="Initial exploration")
# graph_id=42, node_id='n1', best_f=0.45

# ════════════════════════════════════════════════════════════════
# LLM DECIDES: Try two different refinement strategies in parallel
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# TURN 2a: Branch A - Gradient refinement
# ════════════════════════════════════════════════════════════════

f_a = paola.continue_graph(42, parent_node="n1", edge_type="branch")
result_a = minimize(f_a, f_a.get_warm_start(), method='L-BFGS-B')
summary_a = paola.checkpoint(f_a, script=SCRIPT_LBFGS, reasoning="Branch A: gradient")
# node_id='n2', best_f=0.32

# ════════════════════════════════════════════════════════════════
# TURN 2b: Branch B - CMA-ES refinement
# ════════════════════════════════════════════════════════════════

f_b = paola.continue_graph(42, parent_node="n1", edge_type="branch")
es = cma.CMAEvolutionStrategy(f_b.get_warm_start(), 0.5)
es.optimize(f_b, maxfevals=500)
summary_b = paola.checkpoint(f_b, script=SCRIPT_CMAES, reasoning="Branch B: CMA-ES")
# node_id='n3', best_f=0.28

# ════════════════════════════════════════════════════════════════
# LLM INSPECTS: Branch B (CMA-ES) achieved better result
# LLM DECIDES: Finalize with n3 as winner
# ════════════════════════════════════════════════════════════════

result = paola.finalize_graph(42)
# Graph: n1 → n2 (branch), n1 → n3 (branch) [tree pattern]
```

### Fallback When Instrumentation Fails

```python
from some_new_optimizer import fancy_optimize  # Not instrumented

f = paola.objective(problem_id=7)
result = fancy_optimize(f, x0, method='NewMethod')

# Instrumentation won't capture this optimizer
# LLM's self-report provides the metadata as fallback
summary = paola.checkpoint(f,
    script=optimization_script,
    optimizer='some_new_optimizer:NewMethod',  # Reported source
    reasoning="Testing new optimizer library"
)
# summary['optimizer_source'] will be 'reported' (not 'instrumented')
```

---

## Error Handling (Critical for Physics Engines)

**Problem**: External physics engines (CFD, FEA) crash on infeasible designs (negative thickness, invalid geometry). This is common in engineering optimization and MUST be recorded for LLM to reformulate the problem.

**Key Principle**: Every evaluation is valuable data. Never lose it on crash.

### Design: Incremental Persistence

```
┌─────────────────────────────────────────────────────────────────────────┐
│  paola.objective(7)                                                      │
│  ├── Creates graph in foundry (status="in_progress")                    │
│  └── Opens evaluation log: cache/graph_42/evaluations.jsonl             │
│                                                                          │
│  f(x1) → Immediately logs {"x": x1, "f": 0.5, "status": "ok"}           │
│  f(x2) → Immediately logs {"x": x2, "f": 0.3, "status": "ok"}           │
│  ...                                                                     │
│  f(x46) → CFD CRASH!                                                     │
│           Logs {"x": x46, "status": "crash", "error": "negative thick"}  │
│           Script terminates                                              │
│                                                                          │
│  Result: 45 successful evals + crash info preserved on disk             │
└─────────────────────────────────────────────────────────────────────────┘
```

**Implementation:**
- `RecordingObjective.__call__()` writes to JSONL immediately (append mode)
- Use file locking for concurrent safety (fcntl on Unix)
- `checkpoint()` summarizes and updates graph status
- `get_graph_state()` can read partial data from crashed runs

### Recovery Flow

```
Turn 1: Script crashes during optimization
════════════════════════════════════════════
Agent executes script → subprocess crashes
Agent sees stderr: "Mesh generation failed: thickness=-0.5"
Agent queries: get_graph_state(42)
  → Returns: {
      status: "in_progress",
      n_evaluations: 45,
      last_attempted: {"x": [..., -0.5, ...], "status": "crash"},
      crash_reason: "negative thickness"
    }

Agent analyzes: "x[3]=-0.5 caused crash. Need constraint x[3] > 0."

Turn 2: Agent reformulates problem
════════════════════════════════════════════
Option A: Tighter bounds
  derive_problem(7, bounds={"x3": [0.1, 5.0]})  # Force positive thickness

Option B: Add constraint
  derive_problem(7, constraints=[{"expr": "x[3] > 0.1", "type": "ineq"}])

Option C: Different optimizer with trust region
  Use optimizer with smaller step size to avoid jumping to infeasible regions

Agent writes new script with reformulated problem
Agent continues: paola.continue_graph(42, parent_node="n1", edge_type="restart")
```

### Error Types and Agent Response

| Error Type | Persisted Data | Agent Action |
|------------|---------------|--------------|
| Syntax/Import error | Nothing (script didn't start) | Fix code, retry |
| Crash during eval | All evals before crash | Analyze failed design, reformulate, continue |
| Optimizer didn't converge | All evals, status="incomplete" | Try different optimizer/params |
| Timeout | All evals before timeout | Longer timeout or fewer evals |
| Invalid problem_id | Nothing | Fix ID, retry |

### Crash Detection in RecordingObjective

```python
class RecordingObjective:
    def __call__(self, x: np.ndarray) -> float:
        # Log BEFORE calling evaluator (so we know what was attempted)
        attempt = {"x": x.tolist(), "status": "pending", "timestamp": time.time()}
        self._append_to_log(attempt)

        try:
            f = self.evaluator(x)
            # Update log entry to success
            self._update_log_entry(attempt, {"status": "ok", "f": f})
            self.evaluations.append({"x": x, "f": f})
            return f
        except Exception as e:
            # Update log entry to crash
            self._update_log_entry(attempt, {
                "status": "crash",
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise  # Re-raise so optimizer knows eval failed

    def _append_to_log(self, entry: dict):
        """Append to JSONL with file locking."""
        with open(self.log_path, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(json.dumps(entry) + '\n')
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)
```

### Why Crash Data is Valuable

1. **Infeasible Region Mapping**: Crashed designs tell LLM where the infeasible region is
2. **Bound Refinement**: LLM can tighten bounds to exclude crash-causing regions
3. **Constraint Discovery**: LLM can propose new constraints (e.g., `thickness > 0`)
4. **Optimizer Selection**: Some optimizers handle infeasible regions better (trust region)
5. **Learning for Future**: Similar problems can benefit from knowing which regions crash

---

## Testing Strategy

### Unit Tests

1. **RecordingObjective**
   - Records all evaluations correctly
   - Cache hits recorded
   - Thread-safe under parallel evaluation
   - get_best() works correctly

2. **ArrayHasher**
   - Stable hashes for same arrays
   - Different hashes for different arrays
   - Handles tolerance correctly
   - Works with various shapes

3. **EvaluationCache**
   - Memory cache works
   - Disk cache persists
   - Two-tier promotion works
   - Hit rate tracking accurate

4. **Instrumentors**
   - Patches apply correctly
   - Patches remove cleanly
   - Original functions still work
   - RecordingObjective detected correctly

### Integration Tests

1. **Full scipy workflow**
2. **Full optuna workflow**
3. **Full pymoo workflow**
4. **Multi-stage optimization**
5. **Concurrent optimization**

---

## Performance Considerations

1. **Array hashing**: O(n) where n is array size - acceptable
2. **Cache lookup**: O(1) for memory, O(1) amortized for disk
3. **Recording overhead**: Minimal - just append to list
4. **Patch overhead**: One isinstance() check per optimizer call

### Benchmarks to Implement

- Overhead per evaluation (target: <1ms)
- Cache lookup time (target: <0.1ms)
- Memory usage per 1M evaluations
- Disk cache performance

---

## Robustness and Version Compatibility

**Critical Question**: How robust is instrumentation when optimizer libraries change their API/syntax?

This is the Achilles' heel of any instrumentation approach. We must design for resilience.

### Risk Spectrum

```
LOW RISK                                              HIGH RISK
─────────────────────────────────────────────────────────────────
│ Function renamed │ Signature changes │ Semantic changes │ Complete rewrite │
│ (easy to detect) │ (can adapt)       │ (silent failures)│ (must rewrite)   │
```

### Concrete Scenarios

**1. Signature Changes (Medium Risk)**
```python
# SciPy 1.x
minimize(fun, x0, method='SLSQP', options={...})

# Hypothetical SciPy 2.x
minimize(fun, x0, method='SLSQP', config=OptimizerConfig(...))  # 'options' → 'config'
```
Our patch would silently fail to capture `config` if we only look for `options`.

**2. Function Renamed/Moved (Low Risk - Detectable)**
```python
# Before
scipy.optimize.minimize

# After
scipy.optimize.minimizers.minimize  # Moved to submodule
```
Patch fails at import time → we know immediately.

**3. Semantic Changes (High Risk - Silent)**
```python
# Before: 'maxiter' means max function evaluations
# After: 'maxiter' means max optimizer iterations

# Our recording would have wrong interpretation
```

### Mitigation Strategies

#### Strategy 1: Version-Aware Instrumentors

```python
# paola/recording/instrumentors/scipy.py

from packaging.version import Version
import scipy

class SciPyInstrumentor(BaseInstrumentor):
    SUPPORTED_VERSIONS = {
        "1.7": SciPy17Adapter,
        "1.8": SciPy18Adapter,
        "1.9": SciPy19Adapter,
        "1.10": SciPy110Adapter,
        "1.11": SciPy111Adapter,
        "1.12": SciPy112Adapter,
        "1.14": SciPy114Adapter,
    }

    def __init__(self):
        self.version = Version(scipy.__version__)
        self.adapter = self._select_adapter()

    def _select_adapter(self):
        # Find closest supported version
        for v, adapter in self.SUPPORTED_VERSIONS.items():
            if self.version >= Version(v):
                return adapter

        # Unknown version → warn and use latest adapter
        warnings.warn(
            f"SciPy {self.version} not tested. Using latest adapter. "
            "Recording may be incomplete."
        )
        return list(self.SUPPORTED_VERSIONS.values())[-1]
```

#### Strategy 2: Signature Introspection (wrapt approach)

```python
import wrapt
import inspect

def create_safe_wrapper(original_func, capture_func):
    """Create wrapper that adapts to any signature."""

    sig = inspect.signature(original_func)

    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        # Bind arguments to parameter names
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()

        # Extract what we care about (by name, not position)
        method = bound.arguments.get('method')
        options = bound.arguments.get('options') or bound.arguments.get('config') or {}

        # Capture
        capture_func(method=method, options=options)

        # Call original with original args
        return wrapped(*args, **kwargs)

    return wrapper(original_func)
```

#### Strategy 3: Graceful Degradation

```python
class RecordingObjective:
    def __init__(self, problem_id, strict=False):
        self.strict = strict
        self._optimizer_recorded = False

    def complete(self):
        if not self._optimizer_recorded:
            if self.strict:
                raise RecordingError(
                    "No optimizer call detected. Instrumentation may have failed."
                )
            else:
                warnings.warn(
                    "No optimizer call recorded. Using fallback: analyzing call patterns. "
                    "Consider using explicit API: paola.record_optimizer('scipy', 'SLSQP', {...})"
                )
                self._infer_optimizer_from_evaluations()

    def _infer_optimizer_from_evaluations(self):
        """Fallback: guess optimizer from evaluation patterns."""
        n_evals = len(self._evaluations)
        eval_pattern = self._analyze_x_progression()

        # Heuristics
        if eval_pattern == 'gradient_descent':
            self._optimizer_info = {'package': 'unknown', 'method': 'gradient-based'}
        elif eval_pattern == 'random_sampling':
            self._optimizer_info = {'package': 'unknown', 'method': 'derivative-free'}
```

#### Strategy 4: Explicit API (The Escape Hatch)

```python
# If instrumentation fails, user can always be explicit
f = paola.objective(problem_id=7)
result = some_new_optimizer_we_dont_support(f, x0)

# Explicit recording still works
paola.record_optimizer(f,
    package='new_optimizer_lib',
    method='NewMethod',
    config={'param': 'value'}
)
paola.complete(f)
```

### Testing Strategy

```python
# tests/test_instrumentation_versions.py

import pytest
from packaging.version import Version

# Test matrix
SCIPY_VERSIONS = ['1.7.0', '1.8.0', '1.9.0', '1.10.0', '1.11.0', '1.12.0', '1.14.0']
OPTUNA_VERSIONS = ['3.0.0', '3.1.0', '3.2.0', '3.3.0', '3.4.0', '3.5.0']

@pytest.mark.parametrize('scipy_version', SCIPY_VERSIONS)
def test_scipy_instrumentation(scipy_version):
    """Test instrumentation against specific SciPy version."""
    # This would run in CI with different scipy versions installed
    ...

def test_unknown_version_graceful():
    """Test that unknown versions don't crash, just warn."""
    with mock.patch('scipy.__version__', '99.0.0'):
        instrumentor = SciPyInstrumentor()
        assert instrumentor.adapter is not None  # Uses fallback
```

### OpenTelemetry Model (For Reference)

OpenTelemetry maintains **separate packages** for each instrumented library:
```
opentelemetry-instrumentation-requests==0.45b0
opentelemetry-instrumentation-flask==0.45b0
opentelemetry-instrumentation-django==0.45b0
```

Each can be versioned and updated independently. We could adopt similar:
```
paola-instrumentation-scipy>=1.0.0
paola-instrumentation-optuna>=1.0.0
paola-instrumentation-ipopt>=1.0.0
```
But this may be overkill for our use case (fewer libraries to support).

### Recommended Approach Summary

| Strategy | Complexity | Robustness | Recommendation |
|----------|------------|------------|----------------|
| Version-aware adapters | Medium | High | ✅ Essential |
| Signature introspection | Low | Medium | ✅ Use wrapt |
| Graceful degradation | Low | High | ✅ Always have fallback |
| Separate packages | High | Very High | ❌ Later if needed |
| CI version matrix | Medium | High | ✅ Essential for confidence |

### Defense in Depth Principle

1. **First line**: Version-aware adapters handle known changes
2. **Second line**: Signature introspection adapts to minor changes
3. **Third line**: Graceful degradation warns but doesn't crash
4. **Fourth line**: Explicit API always available as escape hatch

---

## Execution Pattern: Multi-Turn Only

**Design Decision**: Paola uses **multi-turn execution only** (Pattern B). Single-turn pattern is withdrawn.

### Why Multi-Turn Only?

For true dynamic decision-making, LLM must:
1. Execute some optimization
2. Receive results back
3. Think/reason about results
4. Decide what to do next
5. Execute next step

This requires **multiple LLM turns**, not a single script execution.

### One Turn = One Node

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-TURN EXECUTION FLOW                                 │
│                                                                              │
│  TURN 1: Start & Explore                                                    │
│  ════════════════════════                                                    │
│  LLM queries: get_problem_info(7), query_past_graphs(...)                   │
│  LLM writes & executes:                                                     │
│      f = paola.objective(problem_id=7, goal="Minimize drag")                │
│      study.optimize(lambda trial: f([...]), n_trials=100)                   │
│      summary = paola.checkpoint(f, script=SCRIPT, reasoning="Explore")      │
│                                                                              │
│  System returns: {graph_id: 98, node_id: "n1", best_f: 0.5, ...}            │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TURN 2: Inspect & Refine                                                   │
│  ════════════════════════                                                    │
│  LLM sees: best_f=0.5, 100 evaluations                                      │
│  LLM thinks: "Promising region found. Refine with gradient method."         │
│  LLM writes & executes:                                                     │
│      f = paola.continue_graph(98, parent_node="n1", edge_type="warm_start") │
│      result = minimize(f, summary['best_x'], method='SLSQP')                │
│      summary = paola.checkpoint(f, script=SCRIPT, reasoning="Refine")       │
│                                                                              │
│  System returns: {graph_id: 98, node_id: "n2", best_f: 0.32, ...}           │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TURN 3: Finalize                                                           │
│  ════════════════                                                            │
│  LLM calls: paola.finalize_graph(98)                                        │
│                                                                              │
│  Final Graph 98:                                                            │
│      pattern: "chain"                                                       │
│      n1: Optuna TPE (100 evals) → 0.5                                       │
│      n2: scipy SLSQP (45 evals) → 0.32  [warm_start from n1]               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Node Boundary

- `checkpoint()` ends a node and returns summary for inspection
- All optimizer calls between `objective()`/`continue_graph()` and `checkpoint()` are ONE node
- LLM must inspect checkpoint results before writing next script

---

## Finalized Recording API

```python
# ═══════════════════════════════════════════════════════════════════════════
#                        PAOLA v0.2.0 RECORDING API
# ═══════════════════════════════════════════════════════════════════════════

paola.objective(problem_id: int, goal: str = None) -> RecordingObjective
    """
    Start new graph, return recording objective.

    Creates new graph in Foundry, initializes per-graph cache.
    """

paola.checkpoint(
    f: RecordingObjective,
    script: str,
    reasoning: str = None
) -> dict
    """
    End current node, persist to Foundry, return summary.

    LLM MUST inspect this before deciding next action.

    Returns:
        {
            'graph_id': int,
            'node_id': str,           # "n1", "n2", ...
            'best_x': List[float],
            'best_f': float,
            'n_evaluations': int,
            'n_cached': int,
            'optimizer': str,         # Verified from instrumentation
            'optimizer_source': str,  # "instrumented" | "reported"
            'instrumented_calls': [...],
            'wall_time': float,
        }
    """

paola.continue_graph(
    graph_id: int,
    parent_node: str,
    edge_type: str = "warm_start"
) -> RecordingObjective
    """
    Resume existing graph from parent node.

    Creates new node with edge to parent.
    Returns fresh RecordingObjective for next turn's code.

    Edge types: "warm_start", "refine", "branch", "explore"
    """

paola.complete(
    f: RecordingObjective,
    script: str,
    reasoning: str = None
) -> dict
    """
    Shorthand for final node: checkpoint() + finalize_graph().

    Use when LLM decides current node is final.
    """

paola.finalize_graph(graph_id: int) -> GraphRecord
    """
    Mark graph as complete. No more nodes can be added.
    """
```

---

## Foundry Integration

### Storage Structure (Updated)

```
.paola_foundry/
├── graphs/                    # Tier 1: GraphRecord (~1KB)
│   └── graph_XXXX.json
│
├── details/                   # Tier 2: GraphDetail (10-100KB)
│   └── graph_XXXX_detail.json
│
├── scripts/                   # Tier 3: Node Scripts (NEW)
│   └── graph_XXXX/
│       ├── n1.py              # Script for node 1
│       ├── n2.py              # Script for node 2
│       └── ...
│
├── cache/                     # Evaluation Cache (NEW)
│   └── graph_XXXX/            # Per-graph isolation
│       ├── index.json         # x_hash → f(x)
│       └── evaluations/       # Optional: detailed records
│
├── problems/
│   ├── index.json
│   └── problem_XXXX.json
│
├── evaluators/
│   └── ...
│
└── metadata.json
```

### Per-Graph Cache

**Design Decision**: Cache is scoped per-graph, not per-problem.

**Rationale**:
- Each graph = one optimization "attempt"
- Problem formulation can be mutated (derived problems)
- Cache from graph A might give WRONG results for graph B if problem changed
- Within a graph, all nodes share cache (warm-start benefits)

```python
# Cache is created when graph starts
cache_dir = f".paola_foundry/cache/graph_{graph_id}/"

# All nodes in graph share the cache
# Node n2 benefits from n1's cached evaluations if warm-starting
```

### Scripts Per Node

**Design Decision**: Each node has its own script file.

```
scripts/graph_0098/
├── n1.py    # "Optuna exploration with TPE sampler"
├── n2.py    # "SLSQP refinement from best point"
└── n3.py    # "CMA-ES escape from local minimum"
```

**Benefits**:
- Perfect reproducibility per node
- Clear audit trail
- Easy to re-run specific nodes

---

## Tool Layer Changes

### Tools to Keep

| Tool | Purpose |
|------|---------|
| `get_problem_info(problem_id)` | Problem characteristics for reasoning |
| `list_optimizers()` | Available optimizers (for LLM to know imports) |
| `query_past_graphs(...)` | Learning from history (CRITICAL) |
| `get_graph_state(graph_id)` | Current graph state for multi-turn |

### Tools to Remove

| Tool | Replacement |
|------|-------------|
| `run_optimization(...)` | LLM writes `minimize(f, ...)` directly |
| `start_graph(...)` | `paola.objective()` |
| `finalize_graph(...)` (tool) | `paola.finalize_graph()` (API) |

### Skills (Keep All)

All optimizer skills remain essential:
- `ipopt.yaml` - How to configure IPOPT correctly
- `scipy.yaml` - SciPy minimize options
- `optuna.yaml` - Sampler configuration
- `pymoo.yaml` - Multi-objective algorithms

LLM reads skills to write correct optimization code.

---

## Schema Updates (NodeSummary)

```python
@dataclass
class NodeSummary:
    # Existing fields
    node_id: str
    optimizer: str
    optimizer_family: str
    config: Dict[str, Any]
    init_strategy: str
    parent_node: Optional[str]
    edge_type: Optional[str]
    status: str
    n_evaluations: int
    wall_time: float
    start_objective: Optional[float]
    best_objective: Optional[float]

    # NEW: Dual Recording fields
    optimizer_source: str              # "instrumented" | "reported" | "none"
    instrumented_calls: List[Dict]     # Raw instrumentation data
    llm_reported_optimizer: Optional[str]  # What LLM claimed
    script_ref: str                    # "scripts/graph_42/n1.py"
    reasoning: str                     # LLM's strategy explanation
```

---

## Future Enhancements

1. **More instrumentors**: NLopt, scipy.optimize.basinhopping, etc.
2. **Gradient recording**: Track gradient evaluations separately
3. **Constraint recording**: Track constraint evaluations
4. **Distributed caching**: Redis/memcached backend
5. **Async evaluation support**: For parallel optimization

---

## Summary

This design provides:

1. **Zero-boilerplate recording** - User writes normal optimization code
2. **Dual recording architecture** - Instrumentation (reality) + Script (intention)
3. **Verification layer** - Cross-check LLM reports against actual execution
4. **Efficient caching** - Two-tier with numpy array support
5. **Professional quality** - Following OpenTelemetry/wrapt patterns
6. **Graceful degradation** - Falls back to script when instrumentation fails
7. **Extensible** - Easy to add new optimizer instrumentors

### Core Insights

1. **The objective is to optimization what tensors are to autodiff.**
   - RecordingObjective captures all evaluations automatically

2. **Instrumentation is the "mirror of truth".**
   - Captures what actually executed, not just what was intended
   - Enables efficient O(1) queries across thousands of runs
   - Provides instant human-readable summaries

3. **Scripts provide context and fallback.**
   - WHY decisions were made (reasoning, comments)
   - HOW strategies connect (control flow)
   - Perfect reproducibility (re-run exactly)
   - Fallback when instrumentation fails

4. **"Trust but verify" principle.**
   - LLM provides script and self-report
   - Instrumentation verifies what actually ran
   - Discrepancies are flagged for debugging
   - Trust hierarchy: instrumented > reported > inferred

### The Complete Record

Each optimization produces:
```
Node Record:
├─ evaluations: [(x, f, grad), ...]     # Essential data
├─ optimizer: "scipy:SLSQP"             # Verified by instrumentation
├─ optimizer_source: "instrumented"     # Transparency
├─ config: {ftol: 1e-6, ...}            # Actual config used
├─ script: "<Python source>"            # Reproducibility
├─ reasoning: "Chose SLSQP because..."  # Context
├─ llm_reported_optimizer: "..."        # For verification
└─ best_x, best_f, n_evals, wall_time   # Summary stats
```

This dual approach provides **verified, efficient, reproducible, and contextualized** optimization records.

---

## Agent Prompt Design (v0.2.0)

**Philosophy**: Trust LLM intelligence. Minimal guidance. One example.

### Draft Prompt (~250 words)

```python
OPTIMIZATION_PROMPT = """You are Paola, an optimization expert.

You write Python optimization code directly. Execute → Inspect → Decide → Repeat.

## Recording API

```python
import paola

# Start new graph
f = paola.objective(problem_id=7, goal="Minimize drag")

# Run any optimizer
from scipy.optimize import minimize
result = minimize(f, x0, method='SLSQP')

# Checkpoint and inspect
summary = paola.checkpoint(f, script=SCRIPT, reasoning="Initial attempt")
print(json.dumps(summary))  # Returns: {graph_id, node_id, best_f, ...}

# Continue from checkpoint (next turn)
f = paola.continue_graph(42, parent_node="n1", edge_type="warm_start")

# Finalize when done
paola.finalize_graph(42)
```

## Multi-Turn Pattern

One turn = one optimization node. Inspect checkpoint results. Decide next action.
- `warm_start`: Refine from best solution
- `restart`: Fresh start with knowledge
- `branch`: Explore alternative strategy

## Information Tools

- `get_problem_info(id)`: Problem characteristics
- `list_optimizers()`: Available optimizers
- `query_past_graphs(...)`: Learn from past successes
- `get_graph_state(id)`: Current graph status
- `load_skill(name)`: Optimizer configuration details

## Key Principles

1. Every f(x) is logged automatically - crashes inform reformulation
2. Use skills for optimizer-specific guidance (IPOPT options, Optuna samplers)
3. Past graphs are your knowledge base - query before starting

Current task: {goal}
"""
```

### Prompt Structure

| Section | Words | Purpose |
|---------|-------|---------|
| Identity | 10 | "You are Paola, an optimization expert" |
| Mode | 15 | "Write code directly. Execute → Inspect → Decide" |
| API Example | 100 | One complete example with comments |
| Multi-turn | 50 | Edge types and pattern |
| Tools | 50 | List available information tools |
| Principles | 30 | Key hints (crashes, skills, past graphs) |
| **Total** | ~250 | Minimal but complete |

### What's NOT in the Prompt

Following minimalist philosophy, we exclude:
- ❌ Step-by-step workflow instructions
- ❌ "If X, then Y" decision trees
- ❌ Multiple examples (available via skills)
- ❌ Detailed error handling instructions
- ❌ IMPORTANT/CRITICAL/WARNING markers

### Skills as Extended Documentation

For detailed optimizer guidance, LLM uses `load_skill()`:
- `load_skill("scipy")` → SLSQP, L-BFGS-B, trust-constr details
- `load_skill("ipopt")` → 250+ options organized by category
- `load_skill("optuna")` → TPE, CMA-ES, samplers, pruners
- `load_skill("cyipopt")` → Interior-point for large-scale NLP

---

## Implementation Phases (MVP First)

**Strategy**: Minimal viable version first. Instrumentation deferred.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     MVP IMPLEMENTATION ORDER                             │
│                                                                          │
│  Phase 1: RecordingObjective (no instrumentation)                       │
│      │                                                                   │
│      ▼                                                                   │
│  Phase 2: Recording API (objective, checkpoint, continue, finalize)     │
│      │                                                                   │
│      ├───────────────┬───────────────┐                                  │
│      ▼               ▼               ▼                                   │
│  Phase 3:        Phase 4:        Phase 5:                               │
│  Foundry         Tool Layer      Agent Prompt                           │
│  Integration     Update          Update                                  │
│                                                                          │
│  ════════════════════════════════════════════════════════════════════   │
│                          MVP COMPLETE                                    │
│  ════════════════════════════════════════════════════════════════════   │
│                                                                          │
│  Phase 6 (Post-MVP): Instrumentation                                    │
│      - ScipyInstrumentor                                                │
│      - OptunaInstrumentor                                               │
│      - CyipoptInstrumentor                                              │
│      - Verification logic                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: Core RecordingObjective (MVP)

**Files to create**:
- `paola/recording/__init__.py`
- `paola/recording/objective.py` - RecordingObjective class
- `paola/recording/cache.py` - ArrayHasher, EvaluationCache (per-graph)

**NOT included in MVP**:
- ~~instrumentation/~~ (deferred to Phase 6)

**RecordingObjective MVP Features**:
- `__call__(x)` → logs to JSONL immediately, returns f(x)
- `get_best()` → returns best (x, f) seen
- `get_warm_start()` → returns parent node's best_x (for continue_graph)
- Incremental persistence (crash-safe)
- Per-graph cache directory

### Phase 2: Recording API (MVP)

**Files to create/modify**:
- `paola/api.py` - Public API
- `paola/__init__.py` - Export public API

**API Functions**:
```python
paola.objective(problem_id, goal=None) -> RecordingObjective
paola.checkpoint(f, script, reasoning=None, optimizer=None) -> dict
paola.continue_graph(graph_id, parent_node, edge_type) -> RecordingObjective
paola.complete(f, script, reasoning=None, optimizer=None) -> dict
paola.finalize_graph(graph_id) -> GraphRecord
```

**MVP Behavior**:
- `optimizer` parameter is LLM-reported (no instrumentation verification)
- `optimizer_source` = "reported" always in MVP

### Phase 3: Foundry Integration (MVP)

**Files to modify**:
- `paola/foundry/storage/file_storage.py` - Add scripts/, cache/ directories
- `paola/foundry/schema/graph_record.py` - Add dual recording fields (script_ref, reasoning)
- `paola/foundry/foundry.py` - Integration with Recording API

**New storage structure**:
```
.paola_foundry/
├── scripts/graph_XXXX/n1.py    # NEW
├── cache/graph_XXXX/           # NEW
│   └── evaluations.jsonl
└── ... existing ...
```

### Phase 4: Tool Layer Update (MVP)

**Files to modify**:
- `paola/tools/graph.py` - Keep get_graph_state, remove start_graph
- `paola/tools/optimizer.py` - Remove run_optimization (or keep as legacy)
- `paola/tools/__init__.py` - Update exports

**Keep**:
- `get_problem_info()`
- `list_optimizers()`
- `query_past_graphs()`
- `get_graph_state()`
- `finalize_graph()` (tool wrapper for API)

**Remove/Deprecate**:
- `run_optimization()` → replaced by LLM writing code
- `start_graph()` → replaced by `paola.objective()`

### Phase 5: Agent Prompt Update (MVP)

**Files to modify**:
- `paola/agent/prompts/optimization.py` - New prompt (see Agent Prompt Design section above)

### Phase 6: Instrumentation (Post-MVP)

**Files to create** (deferred):
- `paola/recording/instrumentation/__init__.py`
- `paola/recording/instrumentation/base.py` - BaseInstrumentor
- `paola/recording/instrumentation/scipy.py` - ScipyInstrumentor
- `paola/recording/instrumentation/optuna.py` - OptunaInstrumentor
- `paola/recording/instrumentation/cyipopt.py` - CyipoptInstrumentor

**Post-MVP features**:
- Automatic optimizer detection
- Verification: `optimizer_source` = "instrumented" when detected
- Cross-check instrumented vs LLM-reported

---

## Key Design Decisions Summary

1. **Multi-turn only** - Each LLM turn = one node. Enables dynamic decision-making.
2. **Per-graph cache** - Problem formulation can be mutated (derived problems).
3. **Scripts per node** - Each node has its own script file for reproducibility.
4. **Dual recording** - Instrumentation verifies, scripts provide context.
5. **No backward compatibility** - Clean break for v0.2.0 redesign.
6. **Subprocess execution** - Script runs in subprocess, returns JSON via stdout.
7. **File-based state** - Foundry files enable multi-turn without IPC.
8. **Incremental persistence** - Every evaluation logged immediately (JSONL). Never lose data on crash.
9. **Crash = learning opportunity** - Failed evals inform problem reformulation (bounds, constraints).
10. **Minimal prompting** - Trust LLM intelligence. One example, minimal hints.
