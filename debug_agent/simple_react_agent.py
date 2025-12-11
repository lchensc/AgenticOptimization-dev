"""
Simple ReAct agent for debugging.

This is a minimal implementation that properly handles:
1. Tool calls and responses
2. Message history format (crucial for Qwen API)
3. Error handling
"""

import time
import logging
from typing import Optional, Any

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

logger = logging.getLogger(__name__)


def create_simple_react_agent(llm, tools: list, max_iterations: int = 20):
    """
    Create a simple ReAct agent that properly handles message format.

    Args:
        llm: LLM instance (already initialized)
        tools: List of LangChain tools
        max_iterations: Maximum number of iterations

    Returns:
        A function that runs the agent
    """

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create tool lookup
    tool_map = {t.name: t for t in tools}

    def run_agent(goal: str, context: Optional[dict] = None, verbose: bool = True):
        """
        Run the ReAct agent.

        Args:
            goal: The user's goal
            context: Optional context dict
            verbose: Print progress

        Returns:
            Final result dict
        """
        messages = []
        iteration = 0

        # Initial prompt
        system_prompt = f"""You are an optimization agent. Your goal is:
{goal}

Available tools:
{chr(10).join(f'- {t.name}: {t.description[:100]}...' for t in tools)}

Instructions:
1. Use optimizer_create to create an SLSQP optimizer for the problem
2. The problem is already registered as "rosenbrock_10d"
3. Bounds are [[-5, 10]] for each of the 10 variables
4. Use evaluate_function and compute_gradient to evaluate designs
5. Use analyze_convergence to check if converged
6. Say "DONE" when optimization is complete

IMPORTANT: When calling tools, use proper Python list syntax for arrays.
For 10 variables: bounds=[[−5,10],[−5,10],[−5,10],[−5,10],[−5,10],[−5,10],[−5,10],[−5,10],[−5,10],[−5,10]]
"""

        messages.append(HumanMessage(content=system_prompt))

        if verbose:
            print(f"\n{'='*60}")
            print("Starting ReAct Agent")
            print(f"{'='*60}")
            print(f"Goal: {goal}")

        while iteration < max_iterations:
            iteration += 1

            if verbose:
                print(f"\n[Iteration {iteration}]")

            # Call LLM
            try:
                response = llm_with_tools.invoke(messages)
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                if verbose:
                    print(f"  ERROR: {e}")
                return {"success": False, "error": str(e), "iterations": iteration}

            # Add AI response to messages
            messages.append(response)

            # Print reasoning
            if verbose and response.content:
                content = response.content
                if len(content) > 300:
                    content = content[:300] + "..."
                print(f"  Agent: {content}")

            # Check for tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call["id"]

                    if verbose:
                        print(f"  -> Tool: {tool_name}")
                        # print(f"     Args: {tool_args}")

                    # Execute tool
                    if tool_name in tool_map:
                        try:
                            result = tool_map[tool_name].invoke(tool_args)
                            result_str = str(result)
                            if len(result_str) > 500:
                                result_str = result_str[:500] + "..."

                            if verbose:
                                msg = result.get('message', result_str[:100]) if isinstance(result, dict) else result_str[:100]
                                print(f"  <- Result: {msg}")

                        except Exception as e:
                            result_str = f"Error: {str(e)}"
                            if verbose:
                                print(f"  <- Error: {e}")
                    else:
                        result_str = f"Error: Tool '{tool_name}' not found. Available: {list(tool_map.keys())}"
                        if verbose:
                            print(f"  <- {result_str}")

                    # Add tool result message - CRITICAL for API format
                    messages.append(ToolMessage(
                        content=result_str,
                        tool_call_id=tool_id
                    ))

            else:
                # No tool calls - check if done
                content = response.content.upper() if response.content else ""
                if "DONE" in content or "CONVERGED" in content:
                    if verbose:
                        print(f"\n{'='*60}")
                        print("Agent completed!")
                        print(f"{'='*60}")

                    return {
                        "success": True,
                        "iterations": iteration,
                        "final_message": response.content,
                        "messages": messages
                    }

                # Add a nudge to continue
                if iteration < max_iterations - 1:
                    messages.append(HumanMessage(
                        content="Please continue with the optimization. Use the tools to make progress, or say DONE if finished."
                    ))

        if verbose:
            print(f"\n{'='*60}")
            print("Max iterations reached")
            print(f"{'='*60}")

        return {
            "success": False,
            "iterations": iteration,
            "reason": "max_iterations",
            "messages": messages
        }

    return run_agent


def test_simple_agent():
    """Test the simple ReAct agent."""
    import os
    import sys
    from pathlib import Path

    # Add project root
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from dotenv import load_dotenv
    load_dotenv()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY not set!")
        return

    # Setup problem
    from aopt.backends.analytical import Rosenbrock
    from aopt.tools import (
        register_problem, clear_problem_registry, clear_optimizer_registry, cache_clear,
        evaluate_function, compute_gradient,
        optimizer_create, optimizer_propose, optimizer_update, optimizer_restart,
        analyze_convergence, detect_pattern, get_gradient_quality,
    )

    clear_problem_registry()
    clear_optimizer_registry()
    cache_clear()

    problem = Rosenbrock(dimension=10)
    register_problem("rosenbrock_10d", problem)

    # Initialize LLM
    from aopt.agent.react_agent import initialize_llm

    print("Initializing Qwen-flash...")
    llm = initialize_llm("qwen-flash", temperature=0.0)

    # Get tools
    tools = [
        evaluate_function,
        compute_gradient,
        optimizer_create,
        optimizer_propose,
        optimizer_update,
        optimizer_restart,
        analyze_convergence,
        detect_pattern,
        get_gradient_quality,
    ]

    # Create and run agent
    agent = create_simple_react_agent(llm, tools, max_iterations=50)

    result = agent(
        goal="""Solve a 10d Rosenbrock optimization problem with SLSQP.

Problem details:
- Problem ID: "rosenbrock_10d" (already registered)
- Number of variables: 10
- Bounds: [-5, 10] for all variables
- Initial design: [0,0,0,0,0,0,0,0,0,0]
- Known optimum: x*=[1,1,...,1], f*=0

Strategy:
1. Create optimizer with optimizer_create
2. Loop: propose -> evaluate -> gradient -> update
3. After 10+ iterations, use analyze_convergence to check progress
4. Stop when gradient_norm < 1e-5 or objective < 1e-6
5. Say "DONE" with final objective value when finished""",
        verbose=True
    )

    print(f"\nFinal result: success={result['success']}, iterations={result['iterations']}")


if __name__ == "__main__":
    test_simple_agent()
