#!/usr/bin/env python
"""PAOLA CLI entry point."""

import sys
from dotenv import load_dotenv
from aopt.cli import AgenticOptREPL  # TODO: Rename package aopt -> paola

# Load environment variables
load_dotenv()

def main():
    """Main entry point."""
    # Parse command line args for model selection
    llm_model = "qwen-flash"  # Default to cheap model
    if len(sys.argv) > 1:
        if sys.argv[1] == "--model" and len(sys.argv) > 2:
            llm_model = sys.argv[2]

    # Create and run REPL
    repl = AgenticOptREPL(llm_model=llm_model)
    try:
        repl.run()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
