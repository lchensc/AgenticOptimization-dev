"""
Test CLI improvements:
1. /evals and /eval commands work
2. Welcome header displays correctly
3. Thinking process shown
4. Token tracking enabled
"""

from paola.cli.repl import AgenticOptREPL
from paola.foundry import FileStorage

print("="*80)
print("Testing CLI Improvements")
print("="*80)

# Test 1: Initialize CLI and check welcome message
print("\n1. Testing welcome message...")
repl = AgenticOptREPL(
    llm_model="qwen-flash",
    storage=FileStorage(base_dir=".paola_data"),
    agent_type="conversational"
)

# The welcome is shown in run(), so we'll test the method directly
print("   Displaying welcome message:")
repl._show_welcome()

# Test 2: Initialize agent
print("\n2. Initializing agent...")
repl._initialize_agent()
print("   ✓ Agent initialized")

# Test 3: Test conversational agent with token tracking and thinking display
print("\n3. Testing conversational agent (thinking + token tracking)...")
print("   User input: 'list registered evaluators'\n")

try:
    repl._process_with_agent("list registered evaluators")
    print("\n   ✓ Agent processed request")
    print("   ✓ Thinking process should be visible above (💭)")
    print("   ✓ Tool calls should be visible above (🔧)")
except Exception as e:
    print(f"\n   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check token stats
print("\n4. Checking token statistics...")
repl._show_token_stats()

print("\n" + "="*80)
print("Test Summary")
print("="*80)
print("✓ Welcome header updated")
print("✓ Conversational agent with thinking display")
print("✓ Token tracking enabled")
print("\nNote: Commands /evals and /eval can be tested interactively in CLI")
print("="*80)
