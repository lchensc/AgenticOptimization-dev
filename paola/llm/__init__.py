"""
LLM utilities - token tracking, caching, and cost management.
"""

from .token_tracker import (
    TokenTracker,
    UsageStats,
    SessionStats,
    BudgetStatus,
    TokenTrackingCallback,
    format_session_stats,
    MODEL_CONFIGS
)

# Try to import LangChain callback (may not be available)
try:
    from .token_tracker import LangChainTokenCallback
    __all__ = [
        "TokenTracker",
        "UsageStats",
        "SessionStats",
        "BudgetStatus",
        "TokenTrackingCallback",
        "LangChainTokenCallback",
        "format_session_stats",
        "MODEL_CONFIGS"
    ]
except ImportError:
    LangChainTokenCallback = None
    __all__ = [
        "TokenTracker",
        "UsageStats",
        "SessionStats",
        "BudgetStatus",
        "TokenTrackingCallback",
        "format_session_stats",
        "MODEL_CONFIGS"
    ]
