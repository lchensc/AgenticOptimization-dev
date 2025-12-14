"""
Entry point for running PAOLA CLI as a module.

Usage:
    python -m paola.cli
    python -m paola.cli --react
    python -m paola.cli --model qwen-plus
"""

import sys
from .repl import AgenticOptREPL


def main():
    """Main entry point for CLI."""
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

    repl = AgenticOptREPL(llm_model=llm_model, agent_type=agent_type)
    repl.run()


if __name__ == "__main__":
    main()
