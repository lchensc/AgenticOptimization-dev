"""CLI callback for streaming agent output."""

from rich.console import Console
from ..callbacks import AgentEvent, EventType


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
            reasoning = event.data.get('reasoning', '').strip()
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
