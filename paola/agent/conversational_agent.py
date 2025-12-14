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


class ConversationalAgentExecutor:
    """
    Wrapper for conversational agent function to provide LangGraph-compatible API.

    This allows the conversational agent to be used with the same interface as
    the LangGraph-based ReAct agent (which has .invoke() method).
    """

    def __init__(self, agent_func):
        """
        Initialize executor.

        Args:
            agent_func: The conversational agent function
        """
        self.agent_func = agent_func

    def invoke(self, state: dict, config: dict = None) -> dict:
        """
        Invoke the agent (LangGraph-compatible API).

        Args:
            state: Agent state dict
            config: Optional config dict (for callbacks, etc.)

        Returns:
            Updated state dict
        """
        # Note: config contains callbacks for token tracking, but we ignore it here
        # because the agent_func already has callback_manager in closure
        return self.agent_func(state)


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

    def invoke_agent(state: dict) -> dict:
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

        Returns:
            Updated state with new messages
        """
        messages = list(state.get("messages", []))  # Copy to avoid mutation
        context = state.get("context", {})
        callback_mgr = state.get("callback_manager", callback_manager)

        # Count how many messages are already in history to detect first call
        # First call: just user message. Subsequent: user + AI + tools + ...
        num_ai_messages = sum(1 for m in messages if isinstance(m, AIMessage))
        is_first_invocation = (num_ai_messages == 0)

        # Build prompt with context (only on first invocation for this user message)
        if is_first_invocation:
            prompt = build_optimization_prompt(context, tools)
            # Get the last user message (the new request)
            last_user_msg = messages[-1].content if messages else ""
            # Replace with system context + user message
            messages[-1] = HumanMessage(content=f"{prompt}\n\nUser request: {last_user_msg}")

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

            # Add AI response to messages
            messages.append(response)

            # Check if response has tool calls
            if not response.tool_calls:
                # No tool calls → final text response → STOP
                logger.info(f"Agent completed request in {iteration} ReAct cycles")
                return {
                    "messages": messages,
                    "context": context,
                    "done": True,  # Always done after giving final response
                    "iteration": iteration
                }

            # Execute tool calls (Acting phase)
            for tool_call in response.tool_calls:
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

    return ConversationalAgentExecutor(invoke_agent)


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
