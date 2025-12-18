"""
Token usage tracking and cost calculation for LLM calls.

Provides:
- Per-call token tracking with cost calculation
- Session-level statistics accumulation
- Budget enforcement
- LangChain callback integration
- Claude Code-style usage display

Usage:
    # Standalone usage
    tracker = TokenTracker()
    stats = tracker.track_call("qwen-plus", input_tokens=150, output_tokens=80)
    print(f"Cost: ${stats.total_cost_usd:.4f}")

    # With LangChain agent
    callback = TokenTrackingCallback(tracker)
    agent = build_optimization_agent(..., callback_manager)
    callback_manager.register(callback)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import threading
import logging

logger = logging.getLogger(__name__)


# Model configurations (updated 2025-12-16)
# All costs are per 1M tokens in USD
# Qwen prices from: https://help.aliyun.com/zh/model-studio/billing-for-model-studio
MODEL_CONFIGS = {
    # Qwen models (via DashScope)
    "qwen-flash": {
        "full_name": "qwen-flash",
        "provider": "qwen",
        "cost_input": 0.05,
        "cost_output": 0.40,
        "cost_cache_write": 0.05,
        "cost_cache_read": 0.005,
        "api_key_env": "DASHSCOPE_API_KEY"
    },
    "qwen-turbo": {
        "full_name": "qwen-turbo",
        "provider": "qwen",
        "cost_input": 0.30,
        "cost_output": 0.60,
        "cost_cache_write": 0.30,
        "cost_cache_read": 0.03,
        "api_key_env": "DASHSCOPE_API_KEY"
    },
    "qwen-plus": {
        "full_name": "qwen-plus",
        "provider": "qwen",
        "cost_input": 0.40,
        "cost_output": 1.20,
        "cost_cache_write": 0.40,
        "cost_cache_read": 0.04,
        "api_key_env": "DASHSCOPE_API_KEY"
    },
    "qwen-max": {
        "full_name": "qwen-max",
        "provider": "qwen",
        "cost_input": 1.2,       # $1.2-3 depending on context length
        "cost_output": 6.0,      # $6-15 depending on context length
        "cost_cache_write": 1.2,
        "cost_cache_read": 0.12,
        "api_key_env": "DASHSCOPE_API_KEY"
    },
    "qwq-32b-preview": {
        "full_name": "qwq-32b-preview",
        "provider": "qwen",
        "cost_input": 0.28,      # ¥0.002/1K (Qwen 2.5-32B pricing)
        "cost_output": 0.84,     # ¥0.006/1K
        "cost_cache_write": 0.28,
        "cost_cache_read": 0.028,
        "api_key_env": "DASHSCOPE_API_KEY"
    },

    # Anthropic Claude models - https://www.anthropic.com/pricing
    "claude-sonnet-4": {
        "full_name": "claude-sonnet-4",
        "provider": "anthropic",
        "cost_input": 3.0,
        "cost_output": 15.0,
        "cost_cache_write": 3.75,   # 1.25x input price
        "cost_cache_read": 0.30,    # 90% discount
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "claude-sonnet-4-5": {
        "full_name": "claude-sonnet-4-5-20250929",
        "provider": "anthropic",
        "cost_input": 3.0,
        "cost_output": 15.0,
        "cost_cache_write": 3.75,
        "cost_cache_read": 0.30,
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "claude-haiku-4-5": {
        "full_name": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "cost_input": 1.0,
        "cost_output": 5.0,
        "cost_cache_write": 1.25,
        "cost_cache_read": 0.10,
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "claude-3-5-haiku-latest": {
        "full_name": "claude-3-5-haiku-latest",
        "provider": "anthropic",
        "cost_input": 0.80,   # $0.80 per 1M tokens
        "cost_output": 4.0,   # $4.00 per 1M tokens
        "cost_cache_write": 1.0,
        "cost_cache_read": 0.08,
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "claude-3-5-haiku-20241022": {
        "full_name": "claude-3-5-haiku-20241022",
        "provider": "anthropic",
        "cost_input": 0.80,   # $0.80 per 1M tokens
        "cost_output": 4.0,   # $4.00 per 1M tokens
        "cost_cache_write": 1.0,
        "cost_cache_read": 0.08,
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "claude-3-haiku-20240307": {
        "full_name": "claude-3-haiku-20240307",
        "provider": "anthropic",
        "cost_input": 0.25,   # $0.25 per 1M tokens
        "cost_output": 1.25,  # $1.25 per 1M tokens
        "cost_cache_write": 0.30,
        "cost_cache_read": 0.03,
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "claude-3-5-sonnet-20241022": {
        "full_name": "claude-3-5-sonnet-20241022",
        "provider": "anthropic",
        "cost_input": 3.0,
        "cost_output": 15.0,
        "cost_cache_write": 3.75,
        "cost_cache_read": 0.3,
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "claude-sonnet-4-20250514": {
        "full_name": "claude-sonnet-4-20250514",
        "provider": "anthropic",
        "cost_input": 3.0,
        "cost_output": 15.0,
        "cost_cache_write": 3.75,
        "cost_cache_read": 0.3,
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "claude-opus-4": {
        "full_name": "claude-opus-4",
        "provider": "anthropic",
        "cost_input": 15.0,
        "cost_output": 75.0,
        "cost_cache_write": 18.75,
        "cost_cache_read": 1.5,
        "api_key_env": "ANTHROPIC_API_KEY"
    },

    # OpenAI models - https://openai.com/api/pricing/
    "gpt-4": {
        "full_name": "gpt-4",
        "provider": "openai",
        "cost_input": 30.0,
        "cost_output": 60.0,
        "cost_cache_write": 30.0,
        "cost_cache_read": 30.0,  # OpenAI doesn't have cache pricing
        "api_key_env": "OPENAI_API_KEY"
    },
    "gpt-4-turbo": {
        "full_name": "gpt-4-turbo",
        "provider": "openai",
        "cost_input": 10.0,
        "cost_output": 30.0,
        "cost_cache_write": 10.0,
        "cost_cache_read": 10.0,
        "api_key_env": "OPENAI_API_KEY"
    },
    "gpt-3.5-turbo": {
        "full_name": "gpt-3.5-turbo",
        "provider": "openai",
        "cost_input": 0.5,
        "cost_output": 1.5,
        "cost_cache_write": 0.5,
        "cost_cache_read": 0.5,
        "api_key_env": "OPENAI_API_KEY"
    },
    "gpt-4o": {
        "full_name": "gpt-4o",
        "provider": "openai",
        "cost_input": 2.5,
        "cost_output": 10.0,
        "cost_cache_write": 2.5,
        "cost_cache_read": 2.5,
        "api_key_env": "OPENAI_API_KEY"
    },
    "gpt-4o-mini": {
        "full_name": "gpt-4o-mini",
        "provider": "openai",
        "cost_input": 0.15,
        "cost_output": 0.6,
        "cost_cache_write": 0.15,
        "cost_cache_read": 0.15,
        "api_key_env": "OPENAI_API_KEY"
    },

    # Local models via Ollama (FREE - runs on your hardware)
    # Usage: --model ollama:devstral or --model ollama:qwen2.5-coder:7b
    "ollama:devstral": {
        "full_name": "devstral (via Ollama)",
        "provider": "ollama",
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cache_write": 0.0,
        "cost_cache_read": 0.0,
        "api_key_env": None  # No API key needed
    },
    "ollama:qwen2.5-coder": {
        "full_name": "qwen2.5-coder (via Ollama)",
        "provider": "ollama",
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cache_write": 0.0,
        "cost_cache_read": 0.0,
        "api_key_env": None
    },
    "ollama:qwen2.5-coder:7b": {
        "full_name": "qwen2.5-coder:7b (via Ollama)",
        "provider": "ollama",
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cache_write": 0.0,
        "cost_cache_read": 0.0,
        "api_key_env": None
    },
    "ollama:qwen2.5-coder:32b": {
        "full_name": "qwen2.5-coder:32b (via Ollama)",
        "provider": "ollama",
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cache_write": 0.0,
        "cost_cache_read": 0.0,
        "api_key_env": None
    },
    "ollama:deepseek-r1:8b": {
        "full_name": "deepseek-r1:8b (via Ollama)",
        "provider": "ollama",
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cache_write": 0.0,
        "cost_cache_read": 0.0,
        "api_key_env": None
    },
    "ollama:llama3.1:8b": {
        "full_name": "llama3.1:8b (via Ollama)",
        "provider": "ollama",
        "cost_input": 0.0,
        "cost_output": 0.0,
        "cost_cache_write": 0.0,
        "cost_cache_read": 0.0,
        "api_key_env": None
    },
}

# Backward compatibility: Create simple pricing dict
MODEL_PRICING = {
    model: {
        "input": config["cost_input"],
        "output": config["cost_output"],
        "cache_write": config.get("cost_cache_write", config["cost_input"]),
        "cache_read": config.get("cost_cache_read", config["cost_input"])
    }
    for model, config in MODEL_CONFIGS.items()
}


@dataclass
class UsageStats:
    """Statistics for a single LLM call."""

    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0  # Anthropic prompt caching (write to cache)
    cache_read_tokens: int = 0      # Anthropic prompt caching (read from cache)
    total_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    cache_write_cost_usd: float = 0.0
    cache_read_cost_usd: float = 0.0
    cache_savings_usd: float = 0.0
    total_cost_usd: float = 0.0
    call_id: str = ""

    def __post_init__(self):
        """Calculate derived fields."""
        # Note: Don't compute total_tokens here!
        # It should be set explicitly when creating UsageStats
        # because Qwen and Anthropic have different semantics:
        # - Qwen: prompt_tokens INCLUDES cached_tokens (breakdown)
        # - Anthropic: input_tokens EXCLUDES cache (must add cache_read)
        pass


@dataclass
class SessionStats:
    """Accumulated statistics for an optimization session."""

    call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_tokens: int = 0
    total_input_cost_usd: float = 0.0
    total_output_cost_usd: float = 0.0
    total_cache_write_cost_usd: float = 0.0
    total_cache_read_cost_usd: float = 0.0
    total_cache_savings_usd: float = 0.0
    total_cost_usd: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0.0
    models_used: Dict[str, int] = field(default_factory=dict)  # model -> call count
    cost_by_model: Dict[str, float] = field(default_factory=dict)  # model -> total cost

    def update_duration(self):
        """Update duration based on current time."""
        self.duration_seconds = (datetime.now() - self.start_time).total_seconds()

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate as percentage of input tokens."""
        total_input = self.total_input_tokens + self.total_cache_creation_tokens + self.total_cache_read_tokens
        if total_input == 0:
            return 0.0
        return (self.total_cache_read_tokens / total_input) * 100.0


@dataclass
class BudgetStatus:
    """Budget tracking status."""

    budget_usd: float
    used_usd: float
    remaining_usd: float
    usage_percent: float
    exceeded: bool
    warning: bool  # True if > 80% of budget used


class TokenTracker:
    """
    Thread-safe token usage tracker with cost calculation.

    Tracks all LLM calls in a session, calculates costs, and provides
    statistics for monitoring and budget enforcement.

    Architecture:
    - Standalone module (no dependencies on agent code)
    - Thread-safe for concurrent usage
    - Minimal memory footprint (stores per-call stats)
    - Extensible pricing table

    Example:
        >>> tracker = TokenTracker()
        >>> stats = tracker.track_call("qwen-plus", input_tokens=150, output_tokens=80)
        >>> print(f"Cost: ${stats.total_cost_usd:.4f}")
        Cost: $0.0028

        >>> session = tracker.get_session_stats()
        >>> print(f"Total: {session.total_tokens} tokens, ${session.total_cost_usd:.4f}")
        Total: 230 tokens, $0.0028
    """

    def __init__(self):
        """Initialize token tracker."""
        self._lock = threading.Lock()
        self._call_history: List[UsageStats] = []
        self._session_stats = SessionStats()
        self._call_counter = 0

    def track_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        timestamp: Optional[datetime] = None
    ) -> UsageStats:
        """
        Record token usage from an LLM call.

        Args:
            model: Model name (e.g., "qwen-plus", "claude-sonnet-4")
            input_tokens: Input/prompt tokens
            output_tokens: Output/completion tokens
            total_tokens: Total tokens from API (if 0, will compute from input+output)
            cache_creation_tokens: Tokens written to cache (Anthropic only)
            cache_read_tokens: Tokens read from cache (Anthropic/Qwen)
            timestamp: Call timestamp (defaults to now)

        Returns:
            UsageStats with calculated costs
        """
        with self._lock:
            self._call_counter += 1
            call_id = f"call_{self._call_counter}"
            ts = timestamp or datetime.now()

            # Use API's total_tokens if provided, otherwise compute
            # For Qwen: prompt_tokens already includes cached_tokens (breakdown)
            # For Anthropic: input_tokens excludes cache (need to add)
            if total_tokens == 0:
                total_tokens = input_tokens + output_tokens

            # Calculate costs
            costs = self._calculate_costs(
                model,
                input_tokens,
                output_tokens,
                cache_creation_tokens,
                cache_read_tokens
            )

            # Create stats
            stats = UsageStats(
                timestamp=ts,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
                input_cost_usd=costs["input"],
                output_cost_usd=costs["output"],
                cache_write_cost_usd=costs["cache_write"],
                cache_read_cost_usd=costs["cache_read"],
                cache_savings_usd=costs["cache_savings"],
                total_cost_usd=costs["total"],
                call_id=call_id
            )

            # Store in history
            self._call_history.append(stats)

            # Update session stats
            self._update_session_stats(stats)

            return stats

    def _calculate_costs(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation_tokens: int,
        cache_read_tokens: int
    ) -> Dict[str, float]:
        """
        Calculate costs for token usage.

        Returns:
            Dict with input, output, cache_write, cache_read, cache_savings, total costs in USD
        """
        # Get pricing (default to zeros if model not found)
        pricing = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})

        # Standard costs (per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing.get("input", 0.0)
        output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0.0)

        # Cache costs (Anthropic only)
        cache_write_cost = 0.0
        cache_read_cost = 0.0
        cache_savings = 0.0

        if cache_creation_tokens > 0 and "cache_write" in pricing:
            cache_write_cost = (cache_creation_tokens / 1_000_000) * pricing["cache_write"]

        if cache_read_tokens > 0 and "cache_read" in pricing:
            cache_read_cost = (cache_read_tokens / 1_000_000) * pricing["cache_read"]
            # Savings = what it would have cost without caching
            full_cost = (cache_read_tokens / 1_000_000) * pricing["input"]
            cache_savings = full_cost - cache_read_cost

        total_cost = input_cost + output_cost + cache_write_cost + cache_read_cost

        return {
            "input": input_cost,
            "output": output_cost,
            "cache_write": cache_write_cost,
            "cache_read": cache_read_cost,
            "cache_savings": cache_savings,
            "total": total_cost
        }

    def _update_session_stats(self, stats: UsageStats):
        """Update session statistics with new call stats."""
        self._session_stats.call_count += 1
        self._session_stats.total_input_tokens += stats.input_tokens
        self._session_stats.total_output_tokens += stats.output_tokens
        self._session_stats.total_cache_creation_tokens += stats.cache_creation_tokens
        self._session_stats.total_cache_read_tokens += stats.cache_read_tokens
        self._session_stats.total_tokens += stats.total_tokens
        self._session_stats.total_input_cost_usd += stats.input_cost_usd
        self._session_stats.total_output_cost_usd += stats.output_cost_usd
        self._session_stats.total_cache_write_cost_usd += stats.cache_write_cost_usd
        self._session_stats.total_cache_read_cost_usd += stats.cache_read_cost_usd
        self._session_stats.total_cache_savings_usd += stats.cache_savings_usd
        self._session_stats.total_cost_usd += stats.total_cost_usd
        self._session_stats.update_duration()

        # Track per-model usage
        model = stats.model
        self._session_stats.models_used[model] = self._session_stats.models_used.get(model, 0) + 1
        self._session_stats.cost_by_model[model] = self._session_stats.cost_by_model.get(model, 0.0) + stats.total_cost_usd

    def get_session_stats(self) -> SessionStats:
        """
        Get accumulated session statistics.

        Returns:
            SessionStats with totals for current session
        """
        with self._lock:
            # Update duration before returning
            self._session_stats.update_duration()
            return self._session_stats

    def get_call_history(self) -> List[UsageStats]:
        """
        Get per-call usage history.

        Returns:
            List of UsageStats for all calls in session
        """
        with self._lock:
            return list(self._call_history)

    def check_budget(self, budget_usd: float) -> BudgetStatus:
        """
        Check if current usage exceeds budget.

        Args:
            budget_usd: Budget limit in USD

        Returns:
            BudgetStatus with usage details
        """
        with self._lock:
            used = self._session_stats.total_cost_usd
            remaining = budget_usd - used
            usage_percent = (used / budget_usd) * 100.0 if budget_usd > 0 else 0.0

            return BudgetStatus(
                budget_usd=budget_usd,
                used_usd=used,
                remaining_usd=remaining,
                usage_percent=usage_percent,
                exceeded=used > budget_usd,
                warning=usage_percent > 80.0
            )

    def reset(self):
        """Reset tracker for new session."""
        with self._lock:
            self._call_history.clear()
            self._session_stats = SessionStats()
            self._call_counter = 0
            logger.info("TokenTracker reset")


class TokenTrackingCallback:
    """
    LangChain callback for automatic token tracking.

    Integrates with LangGraph/LangChain to automatically track token usage
    from LLM responses and display usage like Claude Code.

    Architecture:
    - Non-intrusive: Uses LangChain's callback system
    - Automatic: Tracks all LLM calls without code changes
    - Flexible: Can be enabled/disabled per agent

    Note: This is a simple callable wrapper that works with our CallbackManager.
    For direct LangChain integration, use LangChainTokenCallback instead.

    Usage:
        tracker = TokenTracker()
        callback = TokenTrackingCallback(tracker, verbose=True)
        callback_manager.register(callback)
    """

    def __init__(self, tracker: TokenTracker, verbose: bool = True):
        """
        Initialize callback.

        Args:
            tracker: TokenTracker instance to record usage
            verbose: If True, display usage after each call
        """
        self.tracker = tracker
        self.verbose = verbose

    def __call__(self, event: dict):
        """
        Handle callback event from LangGraph/LangChain.

        Args:
            event: Event dictionary with type and data
        """
        # Check if this is an LLM completion event
        # LangGraph emits events with different structures depending on version
        # We need to handle both the raw response and the event wrapper

        # Try to extract usage from event
        usage = self._extract_usage(event)

        if usage:
            # Track the call
            stats = self.tracker.track_call(
                model=usage.get("model", "unknown"),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                cache_creation_tokens=usage.get("cache_creation_tokens", 0),
                cache_read_tokens=usage.get("cache_read_tokens", 0)
            )

            # Display usage if verbose
            if self.verbose:
                self._display_usage(stats)

    def _extract_usage(self, event: dict) -> Optional[Dict]:
        """
        Extract token usage from LangChain/LangGraph event.

        Handles multiple event formats from different LangChain versions.

        Args:
            event: Event dictionary

        Returns:
            Dict with model, input_tokens, output_tokens, cache tokens or None
        """
        # Try direct access to usage_metadata (LangChain 0.1+)
        if isinstance(event, dict) and "usage_metadata" in event:
            metadata = event["usage_metadata"]
            return {
                "model": event.get("response_metadata", {}).get("model_name", "unknown"),
                "input_tokens": metadata.get("input_tokens", 0),
                "output_tokens": metadata.get("output_tokens", 0),
                "cache_creation_tokens": metadata.get("cache_creation_input_tokens", 0),
                "cache_read_tokens": metadata.get("cache_read_input_tokens", 0)
            }

        # Try response_metadata format (older LangChain)
        if isinstance(event, dict) and "response_metadata" in event:
            response_meta = event["response_metadata"]
            usage = response_meta.get("usage", {})

            if usage:
                return {
                    "model": response_meta.get("model_name", "unknown"),
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                    "cache_creation_tokens": usage.get("cache_creation_input_tokens", 0),
                    "cache_read_tokens": usage.get("cache_read_input_tokens", 0)
                }

        return None

    def _display_usage(self, stats: UsageStats):
        """
        Display token usage like Claude Code.

        Format:
            💰 Token Usage: 2,847 tokens (in: 2,150, out: 697) | Cost: $0.0086
               ⚡ Cache Read: 1,800 tokens (saved $0.0054)

        Args:
            stats: UsageStats to display
        """
        # Main usage line
        usage_line = (
            f"💰 Token Usage: {stats.total_tokens:,} tokens "
            f"(in: {stats.input_tokens:,}, out: {stats.output_tokens:,}) "
            f"| Cost: ${stats.total_cost_usd:.4f}"
        )

        print(usage_line)

        # Cache line if cache was used
        if stats.cache_read_tokens > 0:
            cache_line = (
                f"   ⚡ Cache Read: {stats.cache_read_tokens:,} tokens "
                f"(saved ${stats.cache_savings_usd:.4f})"
            )
            print(cache_line)


def format_session_stats(stats: SessionStats) -> str:
    """
    Format session statistics for display.

    Args:
        stats: SessionStats to format

    Returns:
        Formatted string with session summary
    """
    # Calculate duration
    minutes = int(stats.duration_seconds // 60)
    seconds = int(stats.duration_seconds % 60)
    duration_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

    # Build output
    lines = [
        "",
        "📊 Session Statistics",
        "━" * 60,
        f"Calls:               {stats.call_count}",
        f"Total Tokens:        {stats.total_tokens:,}",
        f"  • Input:           {stats.total_input_tokens:,}",
        f"  • Output:          {stats.total_output_tokens:,}",
    ]

    # Add cache stats if cache was used
    if stats.total_cache_creation_tokens > 0:
        lines.append(f"  • Cache Write:     {stats.total_cache_creation_tokens:,}")

    if stats.total_cache_read_tokens > 0:
        lines.append(
            f"  • Cache Read:      {stats.total_cache_read_tokens:,} "
            f"({stats.cache_hit_rate:.1f}% hit rate)"
        )

    lines.append("")

    # Show models used with pricing info
    if stats.models_used:
        lines.append("Models Used:")
        for model, call_count in sorted(stats.models_used.items(), key=lambda x: -x[1]):
            model_cost = stats.cost_by_model.get(model, 0.0)

            # Get model config for pricing display
            config = MODEL_CONFIGS.get(model)
            if config:
                pricing_info = (
                    f"${config['cost_input']:.2f}/${config['cost_output']:.2f} per 1M"
                )
            else:
                pricing_info = "pricing unknown"

            lines.append(
                f"  • {model:<25} {call_count:>2} call{'s' if call_count != 1 else ' ':<2} "
                f"${model_cost:>8.4f}  ({pricing_info})"
            )
        lines.append("")

    lines.extend([
        "Cost Breakdown:",
        f"  • Input:           ${stats.total_input_cost_usd:.4f}",
        f"  • Output:          ${stats.total_output_cost_usd:.4f}",
    ])

    if stats.total_cache_write_cost_usd > 0:
        lines.append(f"  • Cache Write:     ${stats.total_cache_write_cost_usd:.4f}")

    if stats.total_cache_read_cost_usd > 0:
        lines.append(f"  • Cache Read:      ${stats.total_cache_read_cost_usd:.4f}")

    if stats.total_cache_savings_usd > 0:
        lines.append(f"  • Cache Savings:   -${stats.total_cache_savings_usd:.4f}")

    lines.extend([
        f"  • Total:           ${stats.total_cost_usd:.4f}",
        "",
        f"Duration:            {duration_str}",
        "━" * 60,
        ""
    ])

    return "\n".join(lines)


# LangChain BaseCallbackHandler for direct integration
try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult

    class LangChainTokenCallback(BaseCallbackHandler):
        """
        LangChain BaseCallbackHandler for token tracking.

        This integrates directly with LangChain/LangGraph by inheriting from
        BaseCallbackHandler and implementing on_llm_end.

        Usage:
            tracker = TokenTracker()
            callback = LangChainTokenCallback(tracker, verbose=True)

            # Pass to LLM
            llm = ChatQwen(model="qwen-plus", callbacks=[callback])

            # Or pass to agent invoke
            agent.invoke(state, config={"callbacks": [callback]})
        """

        def __init__(self, tracker: TokenTracker, verbose: bool = True):
            """
            Initialize callback.

            Args:
                tracker: TokenTracker instance
                verbose: If True, display usage after each call
            """
            super().__init__()
            self.tracker = tracker
            self.verbose = verbose

        def on_llm_end(self, response: LLMResult, **kwargs) -> None:
            """
            Called when LLM completes.

            Args:
                response: LLM response with usage metadata
                **kwargs: Additional arguments
            """
            # Extract usage metadata from response
            if not response.llm_output:
                return

            # Try to get usage from llm_output
            usage_data = response.llm_output.get("token_usage") or response.llm_output.get("usage")

            if not usage_data:
                return

            # Extract model name
            model = response.llm_output.get("model_name", "unknown")

            # Debug: Log usage_data structure for troubleshooting cache token extraction
            logger.debug(f"Usage data for {model}: {usage_data}")

            # Extract token counts (handle different formats)
            input_tokens = usage_data.get("prompt_tokens") or usage_data.get("input_tokens") or 0
            output_tokens = usage_data.get("completion_tokens") or usage_data.get("output_tokens") or 0

            # Get total_tokens from API (preferred) or compute fallback
            # Most APIs provide this field directly
            total_tokens = usage_data.get("total_tokens", 0)

            # Extract cache tokens (handle both Anthropic and Qwen formats)
            # Anthropic: flat structure with cache_creation_input_tokens, cache_read_input_tokens
            # Qwen: nested under prompt_tokens_details with cached_tokens, cache_creation_input_tokens

            cache_creation = 0
            cache_read = 0

            # Try Anthropic format first (flat structure)
            cache_creation = usage_data.get("cache_creation_input_tokens", 0)
            cache_read = usage_data.get("cache_read_input_tokens", 0)

            # Try Qwen format (nested prompt_tokens_details)
            prompt_details = usage_data.get("prompt_tokens_details")
            if prompt_details:
                # Qwen uses "cached_tokens" for cache read
                cache_read = prompt_details.get("cached_tokens", cache_read)
                # Qwen also has cache_creation_input_tokens in nested structure
                cache_creation = prompt_details.get("cache_creation_input_tokens", cache_creation)

            # Track the call
            stats = self.tracker.track_call(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cache_creation_tokens=cache_creation,
                cache_read_tokens=cache_read
            )

            # Display if verbose
            if self.verbose:
                self._display_usage(stats)

        def _display_usage(self, stats: UsageStats):
            """Display token usage like Claude Code (minimal format)."""
            # Calculate NEW tokens processed (excluding cached tokens)
            # Cached tokens were already processed in a previous call
            new_input_tokens = stats.input_tokens - stats.cache_read_tokens
            new_tokens = new_input_tokens + stats.output_tokens

            # Claude Code style: minimal display
            # Use sys.stdout to avoid interfering with Rich console output
            import sys
            sys.stdout.write(f"({new_tokens} tokens)\n")
            sys.stdout.flush()

except ImportError:
    # LangChain not available, skip LangChain callback
    LangChainTokenCallback = None
    logger.warning("langchain_core not available - LangChainTokenCallback disabled")
