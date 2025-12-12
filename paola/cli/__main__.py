"""
Entry point for running PAOLA CLI as a module.

Usage:
    python -m paola.cli
"""

from .repl import AgenticOptREPL


def main():
    """Main entry point for CLI."""
    repl = AgenticOptREPL(llm_model="qwen-flash")
    repl.run()


if __name__ == "__main__":
    main()
