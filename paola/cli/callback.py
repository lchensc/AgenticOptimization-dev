"""CLI callback for streaming agent output."""

import re
from rich.console import Console
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
    """

    def __init__(self):
        self.console = Console()

    def __call__(self, event: AgentEvent):
        """Handle and display event."""

        if event.event_type == EventType.REASONING:
            # Agent thinking - dim style
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
                    self.console.print(f"💭 {reasoning}", style="dim")

        elif event.event_type == EventType.TOOL_CALL:
            # Tool invocation - yellow
            tool_name = event.data.get('tool_name', 'unknown')
            self.console.print(f"🔧 {tool_name}...", style="yellow")

        elif event.event_type == EventType.TOOL_RESULT:
            # Tool completion - green
            tool_name = event.data.get('tool_name', 'unknown')
            duration = event.data.get('duration', 0)
            self.console.print(f"✓ {tool_name} completed ({duration:.2f}s)", style="green")

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
