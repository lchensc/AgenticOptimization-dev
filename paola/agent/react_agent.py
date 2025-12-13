"""
ReAct agent implementation using LangGraph.

Implements the core agent loop with:
- Message history retention (CRITICAL FIX from architecture review)
- Real-time event emission via callbacks
- Tool execution with error handling
- Full autonomy (agent decides when to stop)
"""

from typing import TypedDict, Annotated, Optional, Any
import operator
import time
import logging
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END

from ..callbacks import CallbackManager, EventType, create_event
from .prompts import build_optimization_prompt

logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()

# Import LangChain providers
try:
    from langchain_qwq import ChatQwen
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False
    logger.warning("langchain-qwq not available. Install: pip install langchain-qwq")

try:
    from langchain_anthropic import ChatAnthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("langchain-anthropic not available")

try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("langchain-openai not available")


class AgentState(TypedDict):
    """
    Agent working memory.

    CRITICAL: messages uses operator.add to accumulate full history,
    not replace it. This maintains grounding across ReAct turns.
    """
    messages: Annotated[list, operator.add]  # Conversation history
    context: dict  # Current optimization state
    done: bool  # Agent decides when done
    iteration: int  # Current iteration number
    callback_manager: Optional[CallbackManager]  # For event emission


def initialize_llm(
    llm_model: str,
    temperature: float = 0.0,
    enable_thinking: bool = False
):
    """
    Initialize LLM based on model name.

    Supports:
    - Qwen models (via DASHSCOPE_API_KEY)
    - Anthropic models (via ANTHROPIC_API_KEY)
    - OpenAI models (via OPENAI_API_KEY)

    Args:
        llm_model: Model name (e.g., "qwen-plus", "claude-sonnet-4", "gpt-4")
        temperature: 0.0 = deterministic, 1.0 = creative
        enable_thinking: Enable Qwen deep thinking mode

    Returns:
        LLM instance
    """
    # Detect provider
    is_qwen = any(m in llm_model.lower() for m in ["qwen", "qwq"])
    is_openai = any(m in llm_model.lower() for m in ["gpt", "openai"])

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


def build_optimization_agent(
    tools: list,
    llm_model: str,
    callback_manager: Optional[CallbackManager] = None,
    temperature: float = 0.0
):
    """
    Build ReAct agent for optimization.

    Simple continuous loop - agent decides everything.
    No prescribed state machine, just: observe → reason → act → repeat.

    Args:
        tools: List of available tools
        llm_model: LLM model name (e.g., "qwen-plus", "claude-sonnet-4")
        callback_manager: Optional callback manager for events
        temperature: LLM temperature (0.0 = deterministic)

    Returns:
        Compiled LangGraph workflow
    """
    workflow = StateGraph(AgentState)

    # Single node: ReAct step
    workflow.add_node("react", create_react_node(tools, llm_model, temperature))

    # Entry point
    workflow.set_entry_point("react")

    # Loop or terminate?
    workflow.add_conditional_edges(
        "react",
        lambda state: "end" if state["done"] else "continue",
        {
            "continue": "react",
            "end": END
        }
    )

    return workflow.compile()


def create_react_node(tools: list, llm_model: str, temperature: float = 0.0):
    """
    ReAct node: reason → act → observe.

    CRITICAL: Maintains full conversation history for grounding.
    Emits events at key points for real-time monitoring.
    """
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

    # Initialize LLM based on model name
    llm = initialize_llm(llm_model, temperature)
    llm_with_tools = llm.bind_tools(tools)

    # Detect provider for caching support
    is_anthropic = "claude" in llm_model.lower()
    is_qwen = any(m in llm_model.lower() for m in ["qwen", "qwq"])
    supports_cache_control = is_anthropic or is_qwen  # Both support explicit caching!

    def react_step(state: AgentState) -> dict:
        """
        Execute one ReAct cycle with full history retention.

        Message format must be:
        - HumanMessage (prompt)
        - AIMessage (possibly with tool_calls)
        - ToolMessage (for each tool_call)
        - Repeat...

        The key is that after an AIMessage with tool_calls, we MUST have
        ToolMessages before the next HumanMessage.
        """
        context = state["context"]
        iteration = state.get("iteration", 0) + 1
        callback_manager = state.get("callback_manager")

        # EMIT: Agent step start
        if callback_manager:
            callback_manager.emit(create_event(
                event_type=EventType.AGENT_STEP,
                iteration=iteration,
                data={"step": iteration}
            ))

        # Build prompt with current context and actual available tools
        prompt = build_optimization_prompt(context, tools)

        # For first iteration, start fresh with just the prompt
        # For subsequent iterations, we continue from where we left off
        if iteration == 1:
            # First iteration: start with system context + user goal
            # Use cache_control if provider supports it (Anthropic, Qwen)
            if supports_cache_control:
                # Explicit caching format (works for both Anthropic and Qwen)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    }
                ]
                provider_name = "Anthropic" if is_anthropic else "Qwen"
                logger.info(f"First message with {provider_name} cache_control enabled")
            else:
                # Standard format for OpenAI, etc.
                messages = [HumanMessage(content=prompt)]
        else:
            # Subsequent iterations: use existing history
            # Don't add another HumanMessage - let the agent continue
            messages = list(state["messages"])

            # If the last message was from assistant without tool calls,
            # add a follow-up prompt
            if messages and not (hasattr(messages[-1], 'tool_calls') and messages[-1].tool_calls):
                messages.append(HumanMessage(content=f"Continue. Current status:\n{prompt}"))

        # Get LLM decision (reasoning + tool calls)
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            if callback_manager:
                callback_manager.emit(create_event(
                    event_type=EventType.TOOL_ERROR,
                    iteration=iteration,
                    data={"error": str(e), "tool_name": "llm"}
                ))
            raise

        # Collect all new messages from this turn
        # Start with the prompt we added (if first iteration)
        if iteration == 1:
            # For first iteration, add the user message to history
            # Convert dict format back to HumanMessage for consistency
            new_messages = [HumanMessage(content=prompt), response]
        else:
            # Check if we added a follow-up prompt
            if messages and isinstance(messages[-1], HumanMessage) and messages[-1] not in state["messages"]:
                new_messages = [messages[-1], response]
            else:
                new_messages = [response]

        # EMIT: Reasoning (if agent provided text)
        if callback_manager and hasattr(response, 'content') and response.content:
            callback_manager.emit(create_event(
                event_type=EventType.REASONING,
                iteration=iteration,
                data={"reasoning": response.content}
            ))

        # Handle invalid_tool_calls - send error feedback to LLM for self-correction
        # This is the standard LangChain pattern: don't try to fix, let LLM retry
        if hasattr(response, 'invalid_tool_calls') and response.invalid_tool_calls:
            for inv in response.invalid_tool_calls:
                # Extract error information
                tool_name = inv.get('name', 'unknown')
                error_msg = inv.get('error', 'Failed to parse tool call')
                tool_id = inv.get('id', 'unknown')

                logger.warning(f"Invalid tool call: {tool_name} - {error_msg}")

                # EMIT: Tool error for invalid call
                if callback_manager:
                    callback_manager.emit(create_event(
                        event_type=EventType.TOOL_ERROR,
                        iteration=iteration,
                        data={
                            "tool_name": tool_name,
                            "error": f"Invalid tool call: {error_msg}",
                            "invalid": True
                        }
                    ))

                # Send error back to LLM via ToolMessage for self-correction
                new_messages.append(ToolMessage(
                    content=f"Error: {error_msg}",
                    tool_call_id=tool_id,
                    status="error"
                ))

        # Execute tool calls and collect results
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []

            for tool_call in response.tool_calls:
                # EMIT: Tool call start
                if callback_manager:
                    callback_manager.emit(create_event(
                        event_type=EventType.TOOL_CALL,
                        iteration=iteration,
                        data={
                            "tool_name": tool_call["name"],
                            "arguments": tool_call["args"]
                        }
                    ))

                # Execute tool
                start_time = time.time()
                try:
                    result = execute_tool(tool_call, tools)
                    duration = time.time() - start_time

                    # EMIT: Tool result
                    if callback_manager:
                        callback_manager.emit(create_event(
                            event_type=EventType.TOOL_RESULT,
                            iteration=iteration,
                            data={
                                "tool_name": tool_call["name"],
                                "result": result,
                                "duration": duration
                            }
                        ))

                    # EMIT: Special events for specific tools
                    if callback_manager:
                        emit_tool_specific_events(
                            tool_call["name"],
                            result,
                            iteration,
                            callback_manager
                        )

                    tool_results.append(result)

                    # FIXED: Add tool result message (maintains threading)
                    new_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        )
                    )

                except Exception as e:
                    logger.error(f"Tool {tool_call['name']} failed: {e}")

                    # EMIT: Tool error
                    if callback_manager:
                        callback_manager.emit(create_event(
                            event_type=EventType.TOOL_ERROR,
                            iteration=iteration,
                            data={
                                "tool_name": tool_call["name"],
                                "error": str(e)
                            }
                        ))

                    # Add error message to history
                    new_messages.append(
                        ToolMessage(
                            content=f"Error: {str(e)}",
                            tool_call_id=tool_call["id"]
                        )
                    )

            # Update context with tool results
            new_context = update_context(context, tool_results)

            # FIXED: Append new messages (operator.add accumulates)
            return {
                "messages": new_messages,
                "context": new_context,
                "done": False,
                "iteration": iteration
            }

        # Check if agent says done
        content = response.content if hasattr(response, 'content') else ""
        if "DONE" in content.upper() or "CONVERGED" in content.upper():
            # EMIT: Agent done
            if callback_manager:
                callback_manager.emit(create_event(
                    event_type=EventType.AGENT_DONE,
                    iteration=iteration,
                    data={
                        "reason": "converged",
                        "final_context": context
                    }
                ))

            return {
                "messages": new_messages,  # FIXED: Preserve history
                "context": context,
                "done": True,
                "iteration": iteration
            }

        # Agent just reasoning, continue
        return {
            "messages": new_messages,  # FIXED: Preserve history
            "context": context,
            "done": False,
            "iteration": iteration
        }

    return react_step


def execute_tool(tool_call: dict, tools: list) -> Any:
    """
    Execute a single tool call.

    Args:
        tool_call: Tool call from LLM
        tools: List of available tools

    Returns:
        Tool result

    Raises:
        Exception: If tool not found or execution fails
    """
    tool_name = tool_call["name"]
    tool_args = tool_call.get("args", {})

    # Find tool
    tool = None
    for t in tools:
        if hasattr(t, 'name') and t.name == tool_name:
            tool = t
            break

    if tool is None:
        raise ValueError(f"Tool not found: {tool_name}")

    # Execute tool
    return tool.invoke(tool_args)


def emit_tool_specific_events(
    tool_name: str,
    result: dict,
    iteration: int,
    callback_manager: CallbackManager
) -> None:
    """
    Emit specialized events based on tool results.

    Examples:
    - evaluate_function with cache_hit=True → CACHE_HIT event
    - analyze_convergence → CONVERGENCE_CHECK event
    - detect_pattern with pattern found → PATTERN_DETECTED event
    """
    if not isinstance(result, dict):
        return

    # Cache hit detection
    if tool_name == "evaluate_function" and result.get("cache_hit"):
        callback_manager.emit(create_event(
            event_type=EventType.CACHE_HIT,
            iteration=iteration,
            data={
                "design": result.get("design"),
                "saved_cost": result.get("cost", 0)
            }
        ))

    # Convergence check
    if tool_name == "analyze_convergence":
        callback_manager.emit(create_event(
            event_type=EventType.CONVERGENCE_CHECK,
            iteration=iteration,
            data=result
        ))

    # Pattern detection
    if tool_name == "detect_pattern" and result:
        callback_manager.emit(create_event(
            event_type=EventType.PATTERN_DETECTED,
            iteration=iteration,
            data=result
        ))

    # Optimizer restart
    if tool_name == "optimizer_restart":
        callback_manager.emit(create_event(
            event_type=EventType.RESTART,
            iteration=iteration,
            data=result
        ))


def update_context(context: dict, tool_results: list) -> dict:
    """
    Update agent context with tool results.

    Context updates are mostly handled by tools themselves (via run management).
    This function maintains agent working memory for decision-making.

    Args:
        context: Current context
        tool_results: Results from tool executions

    Returns:
        Updated context
    """
    new_context = context.copy()

    # Tool results are already persisted by the platform
    # Context here is just for agent's working memory
    # Most updates happen in tools (start_optimization_run, run_scipy_optimization, etc.)

    return new_context
