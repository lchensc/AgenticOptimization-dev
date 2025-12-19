"""
LLM model initialization utilities.

Shared module for initializing LLM backends across all agents.
Supports multiple providers: Ollama (local), vLLM (local), Qwen, Anthropic, OpenAI.
"""

import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Import LangChain providers with graceful fallbacks
try:
    from langchain_qwq import ChatQwen
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    logger.debug("langchain-qwq not available. Install: pip install langchain-qwq")

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.debug("langchain-anthropic not available")

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.debug("langchain-openai not available")

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.debug("langchain-ollama not available. Install: pip install langchain-ollama")


def initialize_llm(
    llm_model: str,
    temperature: float = 0.0,
    enable_thinking: bool = False
):
    """
    Initialize LLM based on model name.

    Supports:
    - Ollama models (local, prefix with "ollama:")
    - vLLM models (local, prefix with "vllm:")
    - Qwen models (via DASHSCOPE_API_KEY)
    - Anthropic models (via ANTHROPIC_API_KEY)
    - OpenAI models (via OPENAI_API_KEY)

    Args:
        llm_model: Model name (e.g., "ollama:devstral", "vllm:deepseek-r1", "qwen-plus", "claude-sonnet-4", "gpt-4")
        temperature: 0.0 = deterministic, 1.0 = creative
        enable_thinking: Enable Qwen deep thinking mode

    Returns:
        LLM instance
    """
    # Detect provider
    is_ollama = llm_model.lower().startswith("ollama:")
    is_vllm = llm_model.lower().startswith("vllm:")
    is_qwen = any(m in llm_model.lower() for m in ["qwen", "qwq"])
    is_openai = any(m in llm_model.lower() for m in ["gpt", "openai"])

    # Ollama (local models)
    if is_ollama:
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama requires langchain-ollama. Install: pip install langchain-ollama"
            )

        # Extract model name after "ollama:" prefix
        model_name = llm_model.split(":", 1)[1] if ":" in llm_model else llm_model

        # Get Ollama base URL from environment (default: localhost)
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

        logger.info(f"Initialized Ollama model: {model_name} at {base_url}")
        return ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
        )

    # vLLM (local models via OpenAI-compatible API)
    if is_vllm:
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "vLLM integration requires langchain-openai. Install: pip install langchain-openai"
            )

        # Extract model name after "vllm:" prefix
        model_name = llm_model.split(":", 1)[1] if ":" in llm_model else "default"

        # Get vLLM base URL from environment (default: localhost:8000)
        base_url = os.environ.get("VLLM_API_BASE", "http://localhost:8000/v1")

        logger.info(f"Initialized vLLM model: {model_name} at {base_url}")
        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key="dummy",  # vLLM doesn't require a real API key for local use
            temperature=temperature,
            max_tokens=4096,
        )

    if is_qwen:
        if not QWEN_AVAILABLE:
            raise ImportError(
                "Qwen requires langchain-qwq. Install: pip install langchain-qwq"
            )

        if not os.environ.get("DASHSCOPE_API_KEY"):
            raise ValueError(
                "DASHSCOPE_API_KEY not found. Either:\n"
                "1. Set DASHSCOPE_API_KEY in .env file\n"
                "2. Set DASHSCOPE_API_KEY environment variable\n"
                "Get key at: https://dashscope.console.aliyun.com/"
            )

        # Configure Qwen
        qwen_kwargs = {
            "model": llm_model,
            "temperature": temperature
        }

        # Add thinking mode if enabled
        if enable_thinking:
            qwen_kwargs["model_kwargs"] = {"extra_body": {"enable_thinking": True}}

        logger.info(f"Initialized Qwen model: {llm_model}")
        return ChatQwen(**qwen_kwargs)

    elif is_openai:
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI requires langchain-openai. Install: pip install langchain-openai"
            )

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment")

        logger.info(f"Initialized OpenAI model: {llm_model}")
        return ChatOpenAI(model=llm_model, temperature=temperature, max_tokens=4096)

    else:  # Anthropic
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic requires langchain-anthropic. Install: pip install langchain-anthropic"
            )

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY not found in environment")

        logger.info(f"Initialized Anthropic model: {llm_model}")
        return ChatAnthropic(model=llm_model, temperature=temperature, max_tokens=4096)


def detect_provider(llm_model: str) -> dict:
    """
    Detect LLM provider and capabilities from model name.

    Args:
        llm_model: Model name

    Returns:
        Dict with provider info and capabilities:
        - provider: "ollama", "vllm", "qwen", "anthropic", "openai"
        - supports_cache_control: bool
        - is_local: bool
    """
    is_ollama = llm_model.lower().startswith("ollama:")
    is_vllm = llm_model.lower().startswith("vllm:")
    is_qwen = any(m in llm_model.lower() for m in ["qwen", "qwq"])
    is_anthropic = "claude" in llm_model.lower()
    is_openai = any(m in llm_model.lower() for m in ["gpt", "openai"])

    if is_ollama:
        return {
            "provider": "ollama",
            "supports_cache_control": False,
            "is_local": True
        }
    elif is_vllm:
        return {
            "provider": "vllm",
            "supports_cache_control": False,
            "is_local": True
        }
    elif is_qwen:
        return {
            "provider": "qwen",
            "supports_cache_control": True,
            "is_local": False
        }
    elif is_anthropic:
        return {
            "provider": "anthropic",
            "supports_cache_control": True,
            "is_local": False
        }
    elif is_openai:
        return {
            "provider": "openai",
            "supports_cache_control": False,
            "is_local": False
        }
    else:
        # Default to Anthropic
        return {
            "provider": "anthropic",
            "supports_cache_control": True,
            "is_local": False
        }
