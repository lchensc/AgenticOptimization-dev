"""
Minimal ReAct agent - token-efficient, flexible, autonomous.

Key design:
- No hardcoded instructions - just tool schemas from LangChain
- Minimal system prompt - just the goal
- Proper message format for Qwen API
- Handles invalid_tool_calls by sending error feedback to LLM
"""

import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

logger = logging.getLogger(__name__)


def create_minimal_react_agent(llm, tools: list, max_iterations: int = 20):
    """
    Create a minimal ReAct agent.

    The agent receives:
    - Tool schemas (automatic from bind_tools)
    - User goal (provided at runtime)

    No hardcoded instructions - the LLM must figure out tool usage from schemas.
    Invalid tool calls are sent back to LLM as error messages for self-correction.
    """
    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    def run_agent(goal: str, verbose: bool = True):
        """Run the agent with minimal prompting."""
        # Add a system message to encourage reasoning
        system_prompt = (
            "You are an autonomous optimization agent. For each iteration:\n"
            "1. First, explain your reasoning about what to do next and why\n"
            "2. Then, use the appropriate tool(s) to take action\n"
            "3. After receiving tool results, analyze them before deciding next steps\n\n"
            "Always show your thought process before making tool calls."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=goal)
        ]
        iteration = 0

        if verbose:
            print(f"\n{'='*60}")
            print(f"Goal: {goal}")
            print(f"Tools: {list(tool_map.keys())}")
            print(f"{'='*60}")

        while iteration < max_iterations:
            iteration += 1

            if verbose:
                print(f"\n{'─'*70}")
                print(f"Iteration {iteration}")
                print(f"{'─'*70}")

            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                if verbose:
                    print(f"❌ LLM ERROR: {e}")
                return {"success": False, "error": str(e), "iterations": iteration}

            messages.append(response)

            # Print agent reasoning/content if any
            if verbose and response.content:
                print(f"\n💭 Agent Reasoning:")
                # Format multi-line content nicely
                lines = response.content.split('\n')
                for line in lines[:10]:  # Show first 10 lines
                    print(f"   {line}")
                if len(lines) > 10:
                    print(f"   ... ({len(lines) - 10} more lines)")

            # Process tool calls (valid ones)
            tool_calls = getattr(response, 'tool_calls', None) or []

            # Handle invalid_tool_calls - send error feedback to LLM for self-correction
            # This is the standard LangChain pattern: don't try to fix, let LLM retry
            invalid_calls = getattr(response, 'invalid_tool_calls', None) or []
            if invalid_calls and verbose:
                print(f"\n⚠️  Invalid Tool Calls:")

            for inv in invalid_calls:
                # Extract error information
                tool_name = inv.get('name', 'unknown')
                error_msg = inv.get('error', 'Failed to parse tool call')
                tool_id = inv.get('id', 'unknown')

                if verbose:
                    print(f"\n   Tool: {tool_name}")
                    print(f"   Error: {error_msg[:150]}...")
                    # Show the actual invalid args if available
                    if 'args' in inv and isinstance(inv['args'], str):
                        args_preview = inv['args'][:100] + "..." if len(inv['args']) > 100 else inv['args']
                        print(f"   Invalid JSON: {args_preview}")

                # Send the error message exactly as LangChain provides it
                # LangChain already includes: the invalid JSON, the JSONDecodeError, and position
                # The LLM will learn from this and self-correct
                messages.append(ToolMessage(
                    content=error_msg,  # Use the full error message from LangChain
                    tool_call_id=tool_id,
                    status="error"
                ))

            if tool_calls:
                if verbose:
                    print(f"\n🔧 Tool Calls:")

                for tc in tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc.get("args", {})
                    tool_id = tc["id"]

                    if verbose:
                        print(f"\n   📤 Calling: {tool_name}")
                        formatted_args = _format_args(tool_args)
                        print(f"      Args: {formatted_args}")

                    if tool_name in tool_map:
                        try:
                            result = tool_map[tool_name].invoke(tool_args)
                            result_str = str(result)[:500]
                            if verbose:
                                summary = _summarize_result(result)
                                print(f"   📥 Result: {summary}")
                        except Exception as e:
                            result_str = f"Error: {e}"
                            if verbose:
                                print(f"   ❌ Error: {e}")
                    else:
                        result_str = f"Unknown tool: {tool_name}"
                        if verbose:
                            print(f"   ❌ {result_str}")

                    messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))
            else:
                # No tool calls - check if done
                if verbose and not invalid_calls:
                    print(f"\n⏸️  No tool calls made")

                content = (response.content or "").upper()
                done_signals = ["DONE", "COMPLETE", "CONVERGED", "SUCCESSFULLY"]
                if any(sig in content for sig in done_signals):
                    if verbose:
                        print(f"\n{'='*70}")
                        print("✅ Agent Completed Successfully!")
                        print(f"{'='*70}")
                    return {
                        "success": True,
                        "iterations": iteration,
                        "final_message": response.content,
                    }

                # No tool call and no completion signal - nudge
                if iteration < max_iterations - 1:
                    if verbose:
                        print(f"\n💬 Prompting agent to continue...")
                    messages.append(HumanMessage(content="Continue or say DONE."))

        if verbose:
            print(f"\n{'='*70}")
            print("⚠️  Max Iterations Reached (Agent did not complete)")
            print(f"{'='*70}")

        return {
            "success": False,
            "iterations": iteration,
            "reason": "max_iterations",
        }

    return run_agent


def _format_args(args: dict) -> str:
    """Format args for display."""
    items = []
    for k, v in args.items():
        if isinstance(v, list) and len(v) > 3:
            items.append(f"{k}=[...{len(v)} items]")
        elif isinstance(v, str) and len(v) > 50:
            items.append(f"{k}='{v[:47]}...'")
        else:
            items.append(f"{k}={v}")
    return ", ".join(items)


def _summarize_result(result) -> str:
    """Summarize tool result."""
    if isinstance(result, dict):
        if "success" in result:
            msg = f"success={result['success']}"
            if "final_objective" in result:
                msg += f", f={result['final_objective']:.2e}"
            if "message" in result:
                msg += f" - {result['message'][:50]}"
            return msg
        return str(result)[:100]
    return str(result)[:100]
