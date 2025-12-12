"""Command handlers for CLI - reads from storage and displays."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ..storage import StorageBackend


class CommandHandler:
    """
    Handles deterministic /commands.
    Pure presentation logic - reads from storage.
    """

    def __init__(self, storage: StorageBackend, console: Console):
        self.storage = storage
        self.console = console

    def handle_runs(self):
        """Display all runs in table format."""
        runs = self.storage.load_all_runs()

        if not runs:
            self.console.print("\n[dim]No optimization runs yet[/dim]\n")
            return

        table = Table(title="Optimization Runs")
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Problem")
        table.add_column("Algorithm", style="yellow")
        table.add_column("Status")
        table.add_column("Best Value", justify="right")
        table.add_column("Evals", justify="right")
        table.add_column("Time", justify="right")

        for run in runs:
            status = "✓" if run.success else "✗"
            status_style = "green" if run.success else "red"

            table.add_row(
                str(run.run_id),
                run.problem_name,
                run.algorithm,
                f"[{status_style}]{status}[/{status_style}]",
                f"{run.objective_value:.6f}",
                str(run.n_evaluations),
                f"{run.duration:.1f}s"
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def handle_show(self, run_id: int):
        """Show detailed run information."""
        run = self.storage.load_run(run_id)

        if not run:
            self.console.print(f"\n[red]Run #{run_id} not found[/red]\n")
            return

        # Build detailed panel
        status_text = "✓ Complete" if run.success else "✗ Failed"
        status_style = "green" if run.success else "red"

        info = f"""[bold]Run #{run.run_id}: {run.algorithm} on {run.problem_name}[/bold]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[cyan]Status:[/cyan]       [{status_style}]{status_text}[/{status_style}]
[cyan]Objective:[/cyan]    {run.objective_value:.6f}
[cyan]Evaluations:[/cyan]  {run.n_evaluations}
[cyan]Time:[/cyan]         {run.duration:.2f}s
[cyan]Message:[/cyan]      {run.result_data.get('message', 'N/A')}

[bold]Final Solution (first 5 dimensions):[/bold]"""

        # Show first 5 dimensions of solution
        x = run.result_data.get('x', [])
        for i in range(min(5, len(x))):
            info += f"\n  x[{i}] = {x[i]:.6f}"

        if len(x) > 5:
            info += f"\n  ... ({len(x) - 5} more dimensions)"

        self.console.print()
        self.console.print(Panel(info, border_style="cyan", padding=(1, 2)))
        self.console.print()

    def handle_plot(self, run_id: int):
        """Plot convergence history (ASCII in terminal)."""
        run = self.storage.load_run(run_id)

        if not run:
            self.console.print(f"\n[red]Run #{run_id} not found[/red]\n")
            return

        # For now, show basic info - convergence history plotting will be added
        # when we capture iteration-by-iteration data
        info = f"""[bold]Convergence Plot - Run #{run.run_id}[/bold]

[cyan]Problem:[/cyan]     {run.problem_name}
[cyan]Algorithm:[/cyan]   {run.algorithm}
[cyan]Final Value:[/cyan] {run.objective_value:.6f}
[cyan]Evaluations:[/cyan] {run.n_evaluations}

[yellow]Note:[/yellow] Detailed iteration-by-iteration plotting coming soon.
For now, use the analyze_convergence tool for detailed analysis."""

        self.console.print()
        self.console.print(Panel(info, border_style="yellow", padding=(1, 2)))
        self.console.print()

    def handle_best(self):
        """Show best solution across all runs."""
        runs = self.storage.load_all_runs()

        if not runs:
            self.console.print("\n[dim]No optimization runs yet[/dim]\n")
            return

        # Find best run (minimum objective value)
        best_run = min(runs, key=lambda r: r.objective_value)

        info = f"""[bold green]Best Solution Across All Runs[/bold green]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[cyan]Run ID:[/cyan]      #{best_run.run_id}
[cyan]Problem:[/cyan]     {best_run.problem_name}
[cyan]Algorithm:[/cyan]   {best_run.algorithm}
[cyan]Objective:[/cyan]   [bold green]{best_run.objective_value:.6f}[/bold green] ✓
[cyan]Evaluations:[/cyan] {best_run.n_evaluations}
[cyan]Time:[/cyan]        {best_run.duration:.2f}s"""

        self.console.print()
        self.console.print(Panel(info, border_style="green", padding=(1, 2)))
        self.console.print()
