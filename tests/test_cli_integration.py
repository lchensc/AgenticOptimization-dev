"""
Test CLI integration with conversational agent.

Simulates the exact user request that caused the error.
"""

from paola.cli.repl import AgenticOptREPL
from paola.foundry import FileStorage

# Create CLI with conversational agent (default)
print("Initializing PAOLA CLI with conversational agent...")
repl = AgenticOptREPL(
    llm_model="qwen-flash",
    storage=FileStorage(base_dir=".paola_data"),
    agent_type="conversational"
)

# Initialize agent (normally called by run() method)
repl._initialize_agent()

# Simulate user input
user_input = "create a 2D rosenbrock problem from registered evaluators"

print(f"\nUser input: '{user_input}'")
print("-" * 80)

# Process through agent (same as CLI does)
try:
    repl._process_with_agent(user_input)
    print("\n✅ SUCCESS: Agent processed request without errors")
except Exception as e:
    print(f"\n❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
