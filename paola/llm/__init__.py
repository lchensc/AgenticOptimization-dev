"""
LLM utilities - model initialization, token tracking, and cost management.
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

from .models import (
    initialize_llm,
    detect_provider,
    QWEN_AVAILABLE,
    ANTHROPIC_AVAILABLE,
    OPENAI_AVAILABLE,
    OLLAMA_AVAILABLE
)

# Try to import LangChain callback (may not be available)
try:
    from .token_tracker import LangChainTokenCallback
    __all__ = [
        # Model initialization
        "initialize_llm",
        "detect_provider",
        "QWEN_AVAILABLE",
        "ANTHROPIC_AVAILABLE",
        "OPENAI_AVAILABLE",
        "OLLAMA_AVAILABLE",
        # Token tracking
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
        # Model initialization
        "initialize_llm",
        "detect_provider",
        "QWEN_AVAILABLE",
        "ANTHROPIC_AVAILABLE",
        "OPENAI_AVAILABLE",
        "OLLAMA_AVAILABLE",
        # Token tracking
        "TokenTracker",
        "UsageStats",
        "SessionStats",
        "BudgetStatus",
        "TokenTrackingCallback",
        "format_session_stats",
        "MODEL_CONFIGS"
    ]
