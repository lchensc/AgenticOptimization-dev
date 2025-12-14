"""
Test script for conversational agent behavior.

This verifies that the conversational agent:
1. Executes only what user asks
2. Stops after completing the specific request
3. Does NOT execute entire workflows autonomously
"""

import sys
from paola.agent.conversational_agent import build_conversational_agent
from paola.tools.evaluator_tools import create_benchmark_problem, create_nlp_problem
from paola.tools.registration_tools import foundry_list_evaluators, foundry_get_evaluator
from paola.tools.optimizer_tools import run_scipy_optimization
from paola.tools.run_tools import start_optimization_run, finalize_optimization_run, get_active_runs, set_foundry
from paola.foundry import OptimizationFoundry, FileStorage
from langchain_core.messages import HumanMessage

# Set up foundry
storage = FileStorage(base_dir=".paola_data")
foundry = OptimizationFoundry(storage=storage)
set_foundry(foundry)

# Tools
tools = [
    create_benchmark_problem,
    create_nlp_problem,
    foundry_list_evaluators,
    foundry_get_evaluator,
    start_optimization_run,
    finalize_optimization_run,
    get_active_runs,
    run_scipy_optimization,
]

# Build conversational agent
print("Building conversational agent...")
agent = build_conversational_agent(
    tools=tools,
    llm_model="qwen-flash",  # Use fast/cheap model for testing
    temperature=0.0
)

# Test 1: Simple request - should ONLY list evaluators
print("\n" + "="*80)
print("TEST 1: List registered evaluators")
print("="*80)
print("User request: 'list registered evaluators'\n")

state = {
    "messages": [HumanMessage(content="list registered evaluators")],
    "context": {"goal": "list registered evaluators"},
    "iteration": 0
}

result = agent.invoke(state)

print(f"\nAgent completed in {result['iteration']} iterations")
print(f"Tool calls made: {len([m for m in result['messages'] if hasattr(m, 'tool_calls') and m.tool_calls])}")
print(f"Final response:\n{result['messages'][-1].content}\n")

# Count tool calls
tool_call_count = 0
for msg in result['messages']:
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        tool_call_count += len(msg.tool_calls)
        for tc in msg.tool_calls:
            print(f"  - Tool used: {tc.get('name')}")

print(f"\nTotal tool calls: {tool_call_count}")

# Verify it stopped after listing (should be 1 tool call max)
if tool_call_count <= 1:
    print("✅ PASS: Agent executed only what was asked")
else:
    print(f"❌ FAIL: Agent made {tool_call_count} tool calls (expected 1)")

# Test 2: Create problem request - should NOT optimize
print("\n" + "="*80)
print("TEST 2: Create problem (should NOT optimize)")
print("="*80)
print("User request: 'create a 2D rosenbrock problem from registered evaluators'\n")

state2 = {
    "messages": [HumanMessage(content="create a 2D rosenbrock problem from registered evaluators")],
    "context": {"goal": "create a 2D rosenbrock problem from registered evaluators"},
    "iteration": 0
}

result2 = agent.invoke(state2)

print(f"\nAgent completed in {result2['iteration']} iterations")

# Count tool calls and identify them
tool_calls_made = []
for msg in result2['messages']:
    if hasattr(msg, 'tool_calls') and msg.tool_calls:
        for tc in msg.tool_calls:
            tool_name = tc.get('name')
            tool_calls_made.append(tool_name)
            print(f"  - Tool used: {tool_name}")

print(f"\nTotal tool calls: {len(tool_calls_made)}")
print(f"Final response:\n{result2['messages'][-1].content}\n")

# Verify it did NOT run optimization
optimization_tools = ['run_scipy_optimization', 'start_optimization_run', 'finalize_optimization_run']
used_optimization = any(tool in tool_calls_made for tool in optimization_tools)

if not used_optimization:
    print("✅ PASS: Agent did NOT run optimization (only created problem as requested)")
else:
    print(f"❌ FAIL: Agent ran optimization tools when not asked: {[t for t in tool_calls_made if t in optimization_tools]}")

# Verify it created the problem
if 'create_nlp_problem' in tool_calls_made or 'create_benchmark_problem' in tool_calls_made:
    print("✅ PASS: Agent created the problem")
else:
    print("❌ FAIL: Agent did not create the problem")

print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("The conversational agent should:")
print("✓ Execute only what user requests")
print("✓ Stop after completing the specific task")
print("✓ NOT execute entire workflows autonomously")
print("✓ Wait for user's next instruction")
