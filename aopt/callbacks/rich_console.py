"""
Rich console callback for beautiful terminal output.

Uses the rich library for colored output, progress bars, tables, and panels.
"""

from .base import AgentEvent, EventType, CallbackFunction
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
from typing import Optional


class RichConsoleCallback:
    """
    Beautiful terminal output using rich library.

    Features:
    - Colored output based on event type
    - Tables for convergence metrics
    - Panels for important events (adaptations, convergence)
    - Progress tracking

    Example:
        >>> callback = RichConsoleCallback()
        >>> agent.register_callback(callback)
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize rich console callback.

        Args:
            verbose: If False, only show important events
        """
        self.console = Console()
        self.verbose = verbose
        self.progress: Optional[Progress] = None

    def __call__(self, event: AgentEvent) -> None:
        """Handle event and render to console."""

        if event.event_type == EventType.AGENT_START:
            self._handle_agent_start(event)

        elif event.event_type == EventType.REASONING:
            if self.verbose:
                self._handle_reasoning(event)

        elif event.event_type == EventType.TOOL_CALL:
            if self.verbose:
                self._handle_tool_call(event)

        elif event.event_type == EventType.TOOL_RESULT:
            if self.verbose:
                self._handle_tool_result(event)

        elif event.event_type == EventType.CACHE_HIT:
            self._handle_cache_hit(event)

        elif event.event_type == EventType.ITERATION_COMPLETE:
            self._handle_iteration_complete(event)

        elif event.event_type == EventType.CONVERGENCE_CHECK:
            self._handle_convergence_check(event)

        elif event.event_type == EventType.PATTERN_DETECTED:
            self._handle_pattern_detected(event)

        elif event.event_type == EventType.ADAPTATION_START:
            self._handle_adaptation(event)

        elif event.event_type == EventType.AGENT_DONE:
            self._handle_agent_done(event)

        elif event.event_type == EventType.TOOL_ERROR:
            self._handle_tool_error(event)

    def _handle_agent_start(self, event: AgentEvent) -> None:
        """Display agent start banner."""
        self.console.print(Panel(
            f"[bold cyan]Optimization Started[/bold cyan]\n"
            f"Goal: {event.data.get('goal', 'N/A')}\n"
            f"Budget: {event.data.get('budget', 'Unlimited')} CPU hours",
            title="AOpt Agent",
            border_style="cyan"
        ))

    def _handle_reasoning(self, event: AgentEvent) -> None:
        """Display agent reasoning."""
        reasoning = event.data.get('reasoning', '')
        if reasoning:
            self.console.print(f"[dim]💭 {reasoning.strip()}[/dim]")

    def _handle_tool_call(self, event: AgentEvent) -> None:
        """Display tool call."""
        tool_name = event.data.get('tool_name', 'unknown')
        self.console.print(f"[yellow]🔧 Calling {tool_name}...[/yellow]")

    def _handle_tool_result(self, event: AgentEvent) -> None:
        """Display tool result."""
        tool_name = event.data.get('tool_name', 'unknown')
        duration = event.data.get('duration', 0)
        self.console.print(
            f"[green]✓ {tool_name} completed ({duration:.2f}s)[/green]\n"
        )

    def _handle_cache_hit(self, event: AgentEvent) -> None:
        """Display cache hit (important for efficiency)."""
        saved = event.data.get('saved_cost', 0)
        self.console.print(
            f"[bright_green]⚡ Cache hit! Saved {saved:.2f} CPU hours[/bright_green]"
        )

    def _handle_iteration_complete(self, event: AgentEvent) -> None:
        """Display iteration completion."""
        iter_num = event.iteration
        objectives = event.data.get('objectives', [])

        if isinstance(objectives, list) and objectives:
            obj_str = ", ".join([f"{obj:.6e}" for obj in objectives])
        else:
            obj_str = str(objectives)

        self.console.print(
            f"[blue]Iteration {iter_num}: objectives = [{obj_str}][/blue]"
        )

    def _handle_convergence_check(self, event: AgentEvent) -> None:
        """Display convergence analysis as table."""
        table = Table(title=f"Convergence Analysis (Iter {event.iteration})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        for key, value in event.data.items():
            if isinstance(value, float):
                table.add_row(key, f"{value:.6e}")
            else:
                table.add_row(key, str(value))

        self.console.print(table)

    def _handle_pattern_detected(self, event: AgentEvent) -> None:
        """Display pattern detection warning."""
        self.console.print(Panel(
            f"[bold yellow]⚠️  Pattern Detected[/bold yellow]\n"
            f"Type: {event.data.get('pattern_type', 'Unknown')}\n"
            f"Recommendation: {event.data.get('recommendation', 'N/A')}",
            border_style="yellow"
        ))

    def _handle_adaptation(self, event: AgentEvent) -> None:
        """Display adaptation action."""
        self.console.print(Panel(
            f"[bold magenta]🔄 Adaptation[/bold magenta]\n"
            f"Type: {event.data.get('adaptation_type', 'Unknown')}\n"
            f"Reasoning: {event.data.get('reasoning', 'N/A')}",
            border_style="magenta"
        ))

    def _handle_agent_done(self, event: AgentEvent) -> None:
        """Display completion banner."""
        self.console.print(Panel(
            f"[bold green]✅ Optimization Complete[/bold green]\n"
            f"Iterations: {event.iteration}\n"
            f"Reason: {event.data.get('reason', 'Unknown')}",
            title="Success",
            border_style="green"
        ))

    def _handle_tool_error(self, event: AgentEvent) -> None:
        """Display tool error."""
        tool_name = event.data.get('tool_name', 'unknown')
        error = event.data.get('error', 'Unknown error')
        self.console.print(
            f"[bold red]❌ {tool_name} failed: {error}[/bold red]"
        )
