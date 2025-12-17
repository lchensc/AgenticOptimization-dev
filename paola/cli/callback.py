"""CLI callback for streaming agent output."""

import re
import json
from rich.console import Console
from rich.syntax import Syntax
from ..callbacks import AgentEvent, EventType


def strip_numbered_prefix(text: str) -> str:
    """
    Strip numbered list prefixes from text.

    Examples:
        "1. First, I'll check..." -> "First, I'll check..."
        "2. Create NLP Problem:" -> "Create NLP Problem:"
    """
    # Pattern: start of string, digits, period, whitespace
    return re.sub(r'^\d+\.\s+', '', text.strip())


class CLICallback:
    """
    Real-time streaming display for CLI.

    Displays agent reasoning, tool calls, and results as they happen.
    Supports developer mode for verbose debugging output.
    """

    def __init__(self, developer_mode: bool = False):
        self.console = Console()
        self.developer_mode = developer_mode

    def __call__(self, event: AgentEvent):
        """Handle and display event."""

        if event.event_type == EventType.REASONING:
            # Agent thinking - always show reasoning regardless of developer mode
            reasoning = event.data.get('reasoning', '')
            # Handle both string and list formats (Claude returns list of content blocks)
            if isinstance(reasoning, list):
                # Extract text from content blocks
                reasoning = ' '.join(
                    block.get('text', '') if isinstance(block, dict) else str(block)
                    for block in reasoning
                )
            if reasoning and isinstance(reasoning, str):
                # Strip numbered prefixes (e.g., "1. First..." -> "First...")
                reasoning = strip_numbered_prefix(reasoning)
                if reasoning:
                    # Flush stdout to ensure proper ordering with print() statements
                    import sys
                    sys.stdout.flush()
                    self.console.print(f"💭 {reasoning}", style="dim")

        elif event.event_type == EventType.TOOL_CALL:
            # Tool invocation - yellow
            tool_name = event.data.get('tool_name', 'unknown')
            args = event.data.get('args', {})
            self.console.print(f"🔧 {tool_name}...", style="yellow")

            # In developer mode, show tool arguments
            if self.developer_mode and args:
                try:
                    args_json = json.dumps(args, indent=2, default=str)
                    # Truncate very long arguments
                    if len(args_json) > 500:
                        args_json = args_json[:500] + "\n  ... (truncated)"
                    self.console.print(f"   args: {args_json}", style="dim cyan")
                except Exception:
                    self.console.print(f"   args: {args}", style="dim cyan")

        elif event.event_type == EventType.TOOL_RESULT:
            # Tool completion - green
            tool_name = event.data.get('tool_name', 'unknown')
            duration = event.data.get('duration', 0)
            result = event.data.get('result', {})
            self.console.print(f"✓ {tool_name} completed ({duration:.2f}s)", style="green")

            # In developer mode, show tool result
            if self.developer_mode and result:
                try:
                    result_json = json.dumps(result, indent=2, default=str)
                    # Truncate very long results
                    if len(result_json) > 1000:
                        result_json = result_json[:1000] + "\n  ... (truncated)"
                    self.console.print(f"   result: {result_json}", style="dim green")
                except Exception:
                    result_str = str(result)
                    if len(result_str) > 1000:
                        result_str = result_str[:1000] + "... (truncated)"
                    self.console.print(f"   result: {result_str}", style="dim green")

        elif event.event_type == EventType.TOOL_ERROR:
            # Tool error - red
            tool_name = event.data.get('tool_name', 'unknown')
            error = event.data.get('error', 'Unknown error')
            self.console.print(f"✗ {tool_name} failed: {error}", style="bold red")

        elif event.event_type == EventType.AGENT_DONE:
            # Agent finished
            reason = event.data.get('reason', 'complete')
            if reason == 'converged':
                self.console.print()  # Add spacing
