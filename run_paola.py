#!/usr/bin/env python
"""
Launch PAOLA CLI.

Usage:
    python run_paola.py

    or

    python -m paola.cli
"""

from paola.cli.repl import AgenticOptREPL


def main():
    """Main entry point."""
    repl = AgenticOptREPL(llm_model="qwen-flash")
    repl.run()


if __name__ == "__main__":
    main()
