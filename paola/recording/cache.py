"""
Evaluation caching for expensive function evaluations.

Provides:
- ArrayHasher: Stable hashing for numpy arrays with tolerance
- EvaluationCache: Per-graph cache with memory and disk tiers
"""

import hashlib
import json
import fcntl
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np


class ArrayHasher:
    """
    Stable hashing for numpy arrays with configurable tolerance.

    Uses quantization to handle floating-point comparison issues.
    Two arrays that are "close enough" will hash to the same value.

    Example:
        hasher = ArrayHasher(tolerance=1e-8)
        h1 = hasher.hash(np.array([1.0, 2.0, 3.0]))
        h2 = hasher.hash(np.array([1.0 + 1e-10, 2.0, 3.0]))
        assert h1 == h2  # Same hash due to tolerance
    """

    def __init__(self, tolerance: float = 1e-10):
        """
        Initialize hasher with tolerance.

        Args:
            tolerance: Values within this tolerance hash the same.
                       Set to 0 for exact matching.
        """
        self.tolerance = tolerance

    def hash(self, x: np.ndarray) -> str:
        """
        Compute stable hash for numpy array.

        Args:
            x: Input array (any shape)

        Returns:
            Hex string hash (64 characters)
        """
        # Flatten and convert to float64 for consistency
        flat = np.asarray(x, dtype=np.float64).ravel()

        if self.tolerance > 0:
            # Quantize to tolerance grid
            quantized = np.round(flat / self.tolerance) * self.tolerance
        else:
            quantized = flat

        # Use tobytes for stable representation
        data = quantized.tobytes()

        # SHA-256 for collision resistance
        return hashlib.sha256(data).hexdigest()

    def arrays_equal(self, x1: np.ndarray, x2: np.ndarray) -> bool:
        """Check if two arrays are equal within tolerance."""
        return self.hash(x1) == self.hash(x2)


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    x_hash: str
    x: np.ndarray
    f: float
    timestamp: float
    eval_time: Optional[float] = None
    gradient: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        d = {
            "x_hash": self.x_hash,
            "x": self.x.tolist(),
            "f": self.f,
            "timestamp": self.timestamp,
        }
        if self.eval_time is not None:
            d["eval_time"] = self.eval_time
        if self.gradient is not None:
            d["gradient"] = self.gradient.tolist()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CacheEntry":
        """Create from dict."""
        return cls(
            x_hash=d["x_hash"],
            x=np.array(d["x"]),
            f=d["f"],
            timestamp=d["timestamp"],
            eval_time=d.get("eval_time"),
            gradient=np.array(d["gradient"]) if d.get("gradient") else None,
        )


class EvaluationCache:
    """
    Per-graph evaluation cache with two-tier storage.

    Tier 1: In-memory dict for fast lookup
    Tier 2: JSONL file for persistence (crash-safe)

    Features:
    - Immediate disk persistence (append-only JSONL)
    - File locking for concurrent safety
    - Automatic memory loading from disk on init
    - Hit rate tracking

    Example:
        cache = EvaluationCache(cache_dir=Path(".paola_foundry/cache/graph_42"))

        # Check cache
        result = cache.get(x)
        if result is None:
            f = expensive_function(x)
            cache.put(x, f)
    """

    def __init__(
        self,
        cache_dir: Path,
        hasher: Optional[ArrayHasher] = None,
        load_existing: bool = True,
    ):
        """
        Initialize cache for a specific graph.

        Args:
            cache_dir: Directory for this graph's cache (e.g., cache/graph_42/)
            hasher: ArrayHasher instance (default: tolerance=1e-10)
            load_existing: Whether to load existing cache from disk
        """
        self.cache_dir = Path(cache_dir)
        self.hasher = hasher or ArrayHasher()

        # In-memory cache: hash -> CacheEntry
        self._memory: Dict[str, CacheEntry] = {}

        # Stats
        self._hits = 0
        self._misses = 0

        # Ensure directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache file path
        self._cache_file = self.cache_dir / "evaluations.jsonl"

        # Load existing cache if requested
        if load_existing and self._cache_file.exists():
            self._load_from_disk()

    @property
    def cache_file(self) -> Path:
        """Path to the cache JSONL file."""
        return self._cache_file

    def _load_from_disk(self) -> None:
        """Load existing cache entries from JSONL file."""
        try:
            with open(self._cache_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = CacheEntry.from_dict(json.loads(line))
                            self._memory[entry.x_hash] = entry
                        except (json.JSONDecodeError, KeyError) as e:
                            # Skip malformed entries
                            pass
        except FileNotFoundError:
            pass

    def _append_to_disk(self, entry: CacheEntry) -> None:
        """Append entry to JSONL file with file locking."""
        with open(self._cache_file, 'a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry.to_dict()) + '\n')
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def get(self, x: np.ndarray) -> Optional[Tuple[float, Optional[np.ndarray]]]:
        """
        Look up cached evaluation.

        Args:
            x: Design point

        Returns:
            (f, gradient) if cached, None if not found.
            gradient may be None even if f is cached.
        """
        x_hash = self.hasher.hash(x)
        entry = self._memory.get(x_hash)

        if entry is not None:
            self._hits += 1
            return (entry.f, entry.gradient)

        self._misses += 1
        return None

    def put(
        self,
        x: np.ndarray,
        f: float,
        gradient: Optional[np.ndarray] = None,
        eval_time: Optional[float] = None,
    ) -> str:
        """
        Store evaluation result.

        Args:
            x: Design point
            f: Objective value
            gradient: Optional gradient
            eval_time: Optional evaluation time in seconds

        Returns:
            Hash of the design point
        """
        x_hash = self.hasher.hash(x)

        entry = CacheEntry(
            x_hash=x_hash,
            x=np.asarray(x).copy(),
            f=f,
            timestamp=time.time(),
            eval_time=eval_time,
            gradient=np.asarray(gradient).copy() if gradient is not None else None,
        )

        # Store in memory
        self._memory[x_hash] = entry

        # Persist to disk immediately (crash-safe)
        self._append_to_disk(entry)

        return x_hash

    def contains(self, x: np.ndarray) -> bool:
        """Check if design point is cached."""
        return self.hasher.hash(x) in self._memory

    @property
    def size(self) -> int:
        """Number of cached evaluations."""
        return len(self._memory)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        return {
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }

    def get_all_entries(self) -> list[CacheEntry]:
        """Get all cached entries (for inspection/debugging)."""
        return list(self._memory.values())

    def clear_memory(self) -> None:
        """Clear in-memory cache (disk cache remains)."""
        self._memory.clear()
        self._hits = 0
        self._misses = 0
