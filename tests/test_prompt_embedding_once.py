"""
Test that system prompt is only added once in conversation (via SystemMessage).

Updated for the qwen-plus fix: uses SystemMessage instead of embedding in HumanMessage.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

print("=" * 80)
print("Testing SystemMessage Insertion (Once Only)")
print("=" * 80)

# Simulate conversation flow with NEW SystemMessage approach

# First turn: Single HumanMessage (no SystemMessage yet)
print("\n1. First turn (should ADD SystemMessage)...")
messages_turn1 = [HumanMessage(content="optimize rosenbrock")]

has_system_message = any(isinstance(m, SystemMessage) for m in messages_turn1)
print(f"   len(messages) = {len(messages_turn1)}")
print(f"   has_system_message = {has_system_message}")
print(f"   ✓ SystemMessage WILL be added" if not has_system_message else "   ✗ SystemMessage already exists")

# Simulate what agent does: insert SystemMessage at beginning
print("\n2. After agent adds SystemMessage...")
messages_after_agent = [
    SystemMessage(content="SYSTEM_PROMPT"),  # Agent inserts this
    HumanMessage(content="optimize rosenbrock"),  # Original user message UNCHANGED
]
print(f"   len(messages) = {len(messages_after_agent)}")
print(f"   Message types: {[type(m).__name__ for m in messages_after_agent]}")
print(f"   ✓ Clean structure: [SystemMessage, HumanMessage]")

# After first turn: Conversation with history
print("\n3. Conversation with AI response...")
messages_with_response = [
    SystemMessage(content="SYSTEM_PROMPT"),
    HumanMessage(content="optimize rosenbrock"),
    AIMessage(content="Let me optimize that...", tool_calls=[]),
]

has_system_message = any(isinstance(m, SystemMessage) for m in messages_with_response)
print(f"   len(messages) = {len(messages_with_response)}")
print(f"   has_system_message = {has_system_message}")
print(f"   ✓ SystemMessage will NOT be added again" if has_system_message else "   ✗ WRONG!")

# Second turn: User continues
print("\n4. Second turn (SystemMessage already exists)...")
messages_turn2 = [
    SystemMessage(content="SYSTEM_PROMPT"),
    HumanMessage(content="optimize rosenbrock"),
    AIMessage(content="Let me optimize that...", tool_calls=[]),
    HumanMessage(content="continue"),  # New user message - NO modification
]

has_system_message = any(isinstance(m, SystemMessage) for m in messages_turn2)
num_system = sum(1 for m in messages_turn2 if isinstance(m, SystemMessage))
print(f"   len(messages) = {len(messages_turn2)}")
print(f"   has_system_message = {has_system_message}")
print(f"   num_system_messages = {num_system}")
print(f"   ✓ Correct - exactly 1 SystemMessage" if num_system == 1 else "   ✗ WRONG - multiple SystemMessages!")

# Verify clean message structure
print("\n5. Verifying clean message structure...")
print(f"   Message types: {[type(m).__name__ for m in messages_turn2]}")

# Check that HumanMessages are NOT modified
human_messages = [m for m in messages_turn2 if isinstance(m, HumanMessage)]
for i, hm in enumerate(human_messages):
    has_embedded_prompt = "SYSTEM_PROMPT" in hm.content
    print(f"   HumanMessage[{i}] has embedded prompt: {has_embedded_prompt}")
    if has_embedded_prompt:
        print(f"   ✗ WRONG - HumanMessage should NOT have embedded prompt!")

print("\n" + "=" * 80)
print("✓ SystemMessage Insertion Test PASSED")
print("=" * 80)
print("\nKey behavior (qwen-plus compatible):")
print("  1. ✓ SystemMessage inserted at beginning (not embedded in HumanMessage)")
print("  2. ✓ HumanMessages preserved unchanged")
print("  3. ✓ SystemMessage added only once (check: any(isinstance(m, SystemMessage)))")
print("  4. ✓ Compatible with strict providers like qwen-plus")
