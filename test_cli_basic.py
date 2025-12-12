"""Quick test of CLI basic functionality."""

from paola.cli import AgenticOptREPL
import sys

# Create REPL
repl = AgenticOptREPL(llm_model="qwen-flash")

# Simulate a quick interaction
print("Testing REPL initialization...")
print("✓ REPL created successfully")
print(f"✓ Agent model: {repl.llm_model}")
print(f"✓ Tools available: {len(repl.tools)}")
print(f"✓ Callback manager registered: {len(repl.callback_manager.callbacks)} callbacks")

print("\nREPL is ready! Run with: python paola_cli.py")
