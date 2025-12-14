#!/usr/bin/env python
"""PAOLA CLI entry point."""

import sys
from dotenv import load_dotenv
from paola.cli import AgenticOptREPL

# Load environment variables
load_dotenv()

def main():
    """Main entry point."""
    # Parse command line args
    llm_model = "qwen-flash"  # Default to cheap model
    agent_type = "conversational"  # Default agent type

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            llm_model = args[i + 1]
            i += 2
        elif args[i] == "--react":
            agent_type = "react"
            i += 1
        else:
            i += 1

    # Create and run REPL
    repl = AgenticOptREPL(llm_model=llm_model, agent_type=agent_type)
    try:
        repl.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
