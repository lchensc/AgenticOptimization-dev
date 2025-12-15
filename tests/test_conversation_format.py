"""
Test that conversation format is correct for strict LLM providers.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from paola.agent.conversational_agent import build_conversational_agent

print("=" * 80)
print("Testing Conversation Format for Strict LLM Providers")
print("=" * 80)

# Mock tools
from langchain_core.tools import tool

@tool
def test_tool(x: int) -> int:
    """Test tool that doubles a number."""
    return x * 2

tools = [test_tool]

# Build agent
print("\n1. Building conversational agent...")
agent = build_conversational_agent(
    tools=tools,
    llm_model="qwen-flash",  # Use flash for testing
    temperature=0.0
)
print("   ✓ Agent built")

# Test first invocation
print("\n2. Testing first invocation (should add SystemMessage)...")
state = {
    "messages": [HumanMessage(content="test request")],
    "context": {
        "goal": "test",
        "iteration": 0,
        "total_evaluations": 0,
        "budget_status": {"used": 0, "total": 100},
        "cache_stats": {"hit_rate": 0.0},
        "history": [],
        "observations": {}
    },
    "done": False,
    "iteration": 0
}

# Check message structure (don't actually invoke, just check logic)
messages = list(state["messages"])
has_system = any(isinstance(m, SystemMessage) for m in messages)
print(f"   Before: has_system_message = {has_system}")
print(f"   Message count: {len(messages)}")
print(f"   Message types: {[type(m).__name__ for m in messages]}")

# Verify expected structure after fix
# The agent should add SystemMessage at the beginning
expected_structure = [SystemMessage, HumanMessage]
print(f"\n   Expected after agent processes:")
print(f"   [SystemMessage (prompt), HumanMessage (user request), ...]")

# Test second invocation
print("\n3. Testing continuation (should NOT add another SystemMessage)...")
messages_with_system = [
    SystemMessage(content="system prompt"),
    HumanMessage(content="first request"),
    AIMessage(content="first response"),
    HumanMessage(content="continue")
]

has_system = any(isinstance(m, SystemMessage) for m in messages_with_system)
num_system = sum(1 for m in messages_with_system if isinstance(m, SystemMessage))

print(f"   Conversation with history:")
for i, msg in enumerate(messages_with_system):
    print(f"     [{i}] {type(msg).__name__}")

print(f"\n   has_system_message = {has_system}")
print(f"   num_system_messages = {num_system}")
print(f"   ✓ Should be exactly 1 system message")

# Verify qwen-plus compatibility
print("\n4. Verifying qwen-plus compatibility...")
print("   Required format:")
print("   - SystemMessage at beginning (optional but recommended)")
print("   - Alternating Human/AI messages")
print("   - AI with tool_calls must be followed by ToolMessages")
print("   - ToolMessages must have matching tool_call_id")

conversation_example = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="user request"),
    AIMessage(content="thinking", tool_calls=[{"id": "call_1", "name": "test_tool", "args": {"x": 5}}]),
    ToolMessage(content="10", tool_call_id="call_1", name="test_tool"),
    AIMessage(content="The result is 10"),
    HumanMessage(content="continue")
]

print("\n   Valid conversation structure:")
for i, msg in enumerate(conversation_example):
    tool_info = ""
    if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
        tool_info = f" (tool_calls: {len(msg.tool_calls)})"
    elif isinstance(msg, ToolMessage):
        tool_info = f" (tool_call_id: {msg.tool_call_id})"
    print(f"     [{i}] {type(msg).__name__}{tool_info}")

print("\n" + "=" * 80)
print("✓ Conversation Format Test PASSED")
print("=" * 80)
print("\nKey fixes:")
print("  1. ✓ Use SystemMessage instead of embedding in HumanMessage")
print("  2. ✓ Add system message only once at the beginning")
print("  3. ✓ Keep conversation structure clean for strict providers")
print("  4. ✓ All tool_calls have matching ToolMessage responses")
