"""
Conversational agent implementation (like Claude Code).

Key differences from ReAct agent:
- No autonomous loop that keeps going until "done"
- One user message → one agent response → STOP
- Agent can use multiple tools in a single response (ReAct cycles within response)
- But always stops after answering the specific user request
- User controls interaction pace

This still uses ReAct (Reasoning + Acting) but scoped to the user's request,
not an entire imagined workflow.
"""

from typing import List, Optional, Any, Dict
import logging
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from ..callbacks import CallbackManager, EventType, create_event
from .prompts import build_optimization_prompt
from .react_agent import initialize_llm

logger = logging.getLogger(__name__)


def repair_message_history(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Validate and repair message history for LLM API compatibility.

    Qwen API requires that every AIMessage with tool_calls MUST be followed by
    ToolMessages for each tool_call_id. This function ensures this invariant.

    Args:
        messages: List of messages

    Returns:
        Repaired message list
    """
    repaired = []
    pending_tool_ids = set()  # tool_call_ids that need ToolMessages

    for msg in messages:
        # Check if previous AIMessage had tool_calls that need responses
        if pending_tool_ids and isinstance(msg, ToolMessage):
            # This ToolMessage responds to a pending tool_call
            tool_id = getattr(msg, 'tool_call_id', None)
            if tool_id in pending_tool_ids:
                pending_tool_ids.discard(tool_id)
        elif pending_tool_ids and not isinstance(msg, ToolMessage):
            # Non-ToolMessage but we have pending tool_calls - add error responses
            for tool_id in pending_tool_ids:
                logger.warning(f"Adding missing ToolMessage for tool_call_id: {tool_id}")
                repaired.append(ToolMessage(
                    content="Error: Tool call was not processed.",
                    tool_call_id=tool_id,
                    name="unknown"
                ))
            pending_tool_ids.clear()

        repaired.append(msg)

        # Track new tool_calls from AIMessages
        if isinstance(msg, AIMessage):
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_id = tc.get('id')
                    if tool_id:
                        pending_tool_ids.add(tool_id)
            # Also track invalid_tool_calls
            if hasattr(msg, 'invalid_tool_calls') and msg.invalid_tool_calls:
                for tc in msg.invalid_tool_calls:
                    tool_id = tc.get('id')
                    if tool_id:
                        pending_tool_ids.add(tool_id)

    # Handle any remaining pending tool_calls at the end
    for tool_id in pending_tool_ids:
        logger.warning(f"Adding missing ToolMessage for tool_call_id at end: {tool_id}")
        repaired.append(ToolMessage(
            content="Error: Tool call was not processed.",
            tool_call_id=tool_id,
            name="unknown"
        ))

    return repaired


class ConversationalAgentExecutor:
    """
    Wrapper for conversational agent function to provide LangGraph-compatible API.

    This allows the conversational agent to be used with the same interface as
    the LangGraph-based ReAct agent (which has .invoke() method).
    """

    def __init__(self, agent_func, llm_model: str = ""):
        """
        Initialize executor.

        Args:
            agent_func: The conversational agent function
            llm_model: Model name (used for callback compatibility detection)
        """
        self.agent_func = agent_func
        self.llm_model = llm_model

    def invoke(self, state: dict, config: dict = None) -> dict:
        """
        Invoke the agent (LangGraph-compatible API).

        Args:
            state: Agent state dict
            config: Optional config dict (for callbacks, etc.)

        Returns:
            Updated state dict
        """
        # Determine if callbacks should be passed to LLM
        # Qwen models have issues with callbacks during tool calling
        is_qwen = any(m in self.llm_model.lower() for m in ["qwen", "qwq"])

        # Pass config through to agent function for token tracking
        # Include model info so agent can decide on callback passing
        if config is None:
            config = {}
        config["_skip_llm_callbacks"] = is_qwen

        return self.agent_func(state, config)


def build_conversational_agent(
    tools: list,
    llm_model: str = "qwen-plus",
    callback_manager: Optional[CallbackManager] = None,
    temperature: float = 0.0
):
    """
    Build conversational optimization agent (like Claude Code).

    This agent:
    - Responds to ONE user message at a time
    - Can use multiple tools in the response
    - Always stops after responding
    - Waits for next user input

    Args:
        tools: List of available tools
        llm_model: LLM model name
        callback_manager: Optional callback manager
        temperature: LLM temperature

    Returns:
        Callable agent function
    """
    # Initialize LLM
    llm = initialize_llm(llm_model, temperature)
    llm_with_tools = llm.bind_tools(tools)

    def invoke_agent(state: dict, config: dict = None) -> dict:
        """
        Process one user message with ReAct cycles.

        Flow:
        1. Get user message from state
        2. Build prompt with context (first time only)
        3. LLM responds (may use tools)
        4. Execute any tool calls
        5. LLM responds to tool results
        6. Repeat until LLM gives final answer (no tool calls)
        7. Return and STOP

        Args:
            state: {
                "messages": conversation history,
                "context": optimization state,
                "callback_manager": optional,
                "iteration": current iteration (for continuing)
            }
            config: Optional config with callbacks for token tracking

        Returns:
            Updated state with new messages
        """
        messages = list(state.get("messages", []))  # Copy to avoid mutation
        context = state.get("context", {})
        callback_mgr = state.get("callback_manager", callback_manager)

        # Extract LLM callbacks from config (for token tracking)
        # Skip for Qwen models as callbacks interfere with tool calling
        skip_llm_callbacks = config.get("_skip_llm_callbacks", False) if config else False
        llm_callbacks = None
        if not skip_llm_callbacks and config:
            llm_callbacks = config.get("callbacks", [])

        # Count how many messages are already in history to detect first call
        # First call: just user message. Subsequent: user + AI + tools + ...
        num_ai_messages = sum(1 for m in messages if isinstance(m, AIMessage))
        is_first_invocation = (num_ai_messages == 0)

        # Build prompt with context (only on first invocation for this user message)
        if is_first_invocation:
            # Update context with user's goal
            context["goal"] = messages[-1].content if messages else "Not set"
            prompt = build_optimization_prompt(context, tools)
            # Replace user message with full prompt (same format as ReAct agent)
            messages[-1] = HumanMessage(content=prompt)

        # ReAct cycles: keep processing until we get a final text response
        max_iterations = 15  # Safety limit
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Emit agent step event
            if callback_mgr:
                callback_mgr.emit(create_event(
                    event_type=EventType.AGENT_STEP,
                    iteration=iteration,
                    data={"step": iteration}
                ))

            # LLM responds (Reasoning + Action decision)
            try:
                # Repair message history to ensure tool_call/ToolMessage pairing
                # This is critical for Qwen API which strictly validates message ordering
                messages = repair_message_history(messages)

                # Invoke LLM with optional callbacks for token tracking
                # (callbacks skipped for Qwen models as they interfere with tool calling)
                if llm_callbacks:
                    response = llm_with_tools.invoke(messages, config={"callbacks": llm_callbacks})
                else:
                    response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                # Return error message
                error_msg = AIMessage(content=f"Error: {str(e)}")
                return {
                    "messages": messages + [error_msg],
                    "context": context,
                    "done": True,
                    "iteration": iteration
                }

            # Check for tool calls (valid or invalid)
            has_tool_calls = bool(response.tool_calls)
            has_invalid_tool_calls = hasattr(response, 'invalid_tool_calls') and bool(response.invalid_tool_calls)

            if not has_tool_calls and not has_invalid_tool_calls:
                # Check if this is a real completion (has content) or empty response (abnormal)
                # Handle both string and list content formats (Claude may return list of blocks)
                content = response.content
                if isinstance(content, list):
                    content = ' '.join(
                        block.get('text', '') if isinstance(block, dict) else str(block)
                        for block in content
                    )
                has_content = content and content.strip() if isinstance(content, str) else bool(content)

                if has_content:
                    # Has content but no tool calls → final text response → STOP
                    messages.append(response)
                    # Don't emit REASONING here - REPL will display the final response
                    logger.info(f"Agent completed request in {iteration} ReAct cycles")
                    return {
                        "messages": messages,
                        "context": context,
                        "done": True,
                        "iteration": iteration
                    }
                else:
                    # Empty response (no content, no tool_calls) - abnormal
                    # Don't add to history, add continuation prompt and retry
                    logger.warning(f"Empty response from LLM, adding continuation prompt")
                    messages.append(HumanMessage(content="Continue with the task. Use tools to make progress."))
                    continue  # Retry

            # Has tool calls (valid or invalid) - add response to messages
            messages.append(response)

            # Emit reasoning if there's content before tool calls
            if callback_mgr and response.content:
                callback_mgr.emit(create_event(
                    event_type=EventType.REASONING,
                    iteration=iteration,
                    data={"reasoning": response.content}
                ))

            # Handle invalid tool calls first - send error feedback for self-correction
            # This is critical for Qwen which may produce malformed tool calls
            if has_invalid_tool_calls:
                for inv in response.invalid_tool_calls:
                    tool_name = inv.get('name', 'unknown')
                    error_msg = inv.get('error', 'Failed to parse tool call')
                    tool_id = inv.get('id', f'invalid_{tool_name}')

                    logger.warning(f"Invalid tool call: {tool_name} - {error_msg}")

                    # Emit tool error event
                    if callback_mgr:
                        callback_mgr.emit(create_event(
                            event_type=EventType.TOOL_ERROR,
                            iteration=iteration,
                            data={
                                "tool_name": tool_name,
                                "error": f"Invalid tool call: {error_msg}",
                                "invalid": True
                            }
                        ))

                    # Send error back to LLM via ToolMessage for self-correction
                    # This is REQUIRED - Qwen API demands ToolMessage for every tool_call_id
                    messages.append(ToolMessage(
                        content=f"Error: {error_msg}. Please fix the tool call format.",
                        tool_call_id=tool_id,
                        name=tool_name
                    ))

            # Execute valid tool calls (Acting phase)
            for tool_call in (response.tool_calls or []):
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})

                # Emit tool call event
                if callback_mgr:
                    callback_mgr.emit(create_event(
                        event_type=EventType.TOOL_CALL,
                        iteration=iteration,
                        data={"tool_name": tool_name, "args": tool_args}
                    ))

                # Execute tool
                try:
                    tool_result = execute_tool(tools, tool_call)
                    success = True
                except Exception as e:
                    logger.error(f"Tool '{tool_name}' execution failed: {e}")
                    tool_result = {
                        "error": str(e),
                        "success": False
                    }
                    success = False

                # Create tool message (Observation phase)
                tool_msg = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                    name=tool_name
                )
                messages.append(tool_msg)

                # Emit tool result event
                if callback_mgr:
                    callback_mgr.emit(create_event(
                        event_type=EventType.TOOL_RESULT,
                        iteration=iteration,
                        data={"tool_name": tool_name, "result": tool_result, "success": success}
                    ))

            # Add continuation prompt after tool execution (like ReAct agent)
            # This is crucial - tells the LLM to continue working
            prompt = build_optimization_prompt(context, tools)
            messages.append(HumanMessage(content=f"Continue. Current status:\n{prompt}"))

            # Continue loop - LLM will reason about tool results (next ReAct cycle)

        # Safety: max iterations reached
        logger.warning(f"Max iterations ({max_iterations}) reached without final response")
        final_msg = AIMessage(
            content=f"I've completed the requested actions but reached the iteration limit. "
                    f"Tools were used {iteration} times."
        )
        return {
            "messages": messages + [final_msg],
            "context": context,
            "done": True,
            "iteration": iteration
        }

    return ConversationalAgentExecutor(invoke_agent, llm_model=llm_model)


def execute_tool(tools: list, tool_call: dict) -> Any:
    """
    Execute a tool call.

    Args:
        tools: List of available tools
        tool_call: Tool call dict with name and args

    Returns:
        Tool result

    Raises:
        ValueError: If tool not found
    """
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # Find tool
    tool = None
    for t in tools:
        if hasattr(t, 'name') and t.name == tool_name:
            tool = t
            break

    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found")

    # Execute tool
    return tool.invoke(tool_args)
