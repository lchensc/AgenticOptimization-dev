"""Command handlers for CLI - reads from platform and displays."""

from typing import List, Optional, Dict, Any
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import asciichartpy as asciichart

from ..platform import OptimizationPlatform
from ..analysis import compute_metrics, ai_analyze


class CommandHandler:
    """
    Handles deterministic /commands.
    Pure presentation logic - reads from platform storage.
    """

    def __init__(self, platform: OptimizationPlatform, console: Console):
        self.platform = platform
        self.console = console

    def handle_runs(self):
        """Display all runs in table format."""
        runs = self.platform.load_all_runs()

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
        """Show detailed run information with metrics."""
        run = self.platform.load_run(run_id)

        if not run:
            self.console.print(f"\n[red]Run #{run_id} not found[/red]\n")
            return

        # Compute metrics
        metrics = compute_metrics(run)

        # Build detailed panel
        status_text = "✓ Complete" if run.success else "✗ Failed"
        status_style = "green" if run.success else "red"

        # Convergence status
        conv_status = "⚠ STALLED" if metrics["convergence"]["is_stalled"] else "✓ Converging"
        conv_style = "yellow" if metrics["convergence"]["is_stalled"] else "green"

        info = f"""[bold]Run #{run.run_id}: {run.algorithm} on {run.problem_name}[/bold]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[cyan]Status:[/cyan]       [{status_style}]{status_text}[/{status_style}]
[cyan]Objective:[/cyan]    {run.objective_value:.6f}
[cyan]Evaluations:[/cyan]  {run.n_evaluations}
[cyan]Time:[/cyan]         {run.duration:.2f}s
[cyan]Message:[/cyan]      {run.result_data.get('message', 'N/A')}

[bold]Metrics:[/bold]
[cyan]Convergence:[/cyan]  [{conv_style}]{conv_status}[/{conv_style}]
  - Rate: {metrics['convergence']['rate']:.4f}
  - Improvement (last 10): {metrics['convergence']['improvement_last_10']:.6f}
  - Total iterations: {metrics['convergence']['iterations_total']}

[cyan]Efficiency:[/cyan]
  - Improvement per eval: {metrics['efficiency']['improvement_per_eval']:.6f}

[cyan]Gradient:[/cyan]      Quality: {metrics['gradient']['quality']}
  - Norm: {metrics['gradient']['norm']:.6e}
  - Variance: {metrics['gradient']['variance']:.6e}

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
        run = self.platform.load_run(run_id)

        if not run:
            self.console.print(f"\n[red]Run #{run_id} not found[/red]\n")
            return

        # Extract iterations from result_data
        iterations = run.result_data.get('iterations', [])

        if not iterations:
            self.console.print(f"\n[yellow]No iteration data available for run #{run_id}[/yellow]\n")
            return

        # Extract objective values
        objectives = [it['objective'] for it in iterations]

        # Normalize to fixed chart width for terminal display
        max_chart_width = 60  # Fits in 80-char terminal with Y-axis labels
        original_length = len(objectives)

        if len(objectives) > max_chart_width:
            # Downsample to max_chart_width points for clean terminal display
            step = len(objectives) / max_chart_width
            objectives_to_plot = []
            for i in range(max_chart_width):
                idx = int(i * step)
                objectives_to_plot.append(objectives[idx])
        else:
            objectives_to_plot = objectives

        # Create ASCII plot
        plot_config = {
            'height': 15,
            'format': '{:8.2e}',
        }

        try:
            chart = asciichart.plot(objectives_to_plot, plot_config)
        except Exception as e:
            # Fallback if plotting fails
            self.console.print(f"\n[red]Error creating plot: {e}[/red]\n")
            return

        # Create x-axis labels aligned with actual chart width
        # asciichartpy makes chart width = number of data points
        chart_width = len(objectives_to_plot)
        num_ticks = 5

        # Build x-axis string with labels at exact positions
        x_axis_line = " " * 10  # Space for Y-axis labels (asciichartpy uses ~10 chars)
        x_axis_chars = [' '] * chart_width

        for i in range(num_ticks):
            tick_pos = int(i * (chart_width - 1) / (num_ticks - 1))
            # Use original_length for actual iteration number
            tick_label = str(int(i * (original_length - 1) / (num_ticks - 1)))
            # Center the label at tick position (shift left by half label length)
            label_start = tick_pos - len(tick_label) // 2
            # Ensure label doesn't go out of bounds
            if label_start < 0:
                label_start = 0
            if label_start + len(tick_label) > chart_width:
                label_start = chart_width - len(tick_label)
            # Place label characters
            for j, char in enumerate(tick_label):
                char_pos = label_start + j
                if 0 <= char_pos < chart_width:
                    x_axis_chars[char_pos] = char

        x_axis_line += ''.join(x_axis_chars)

        # Build info panel
        downsampled_note = f" [dim](chart shows {len(objectives_to_plot)} sampled points)[/dim]" if len(objectives_to_plot) < original_length else ""

        info = f"""[bold cyan]Convergence History - Run #{run.run_id}[/bold cyan]

[bold]{run.algorithm} on {run.problem_name}[/bold]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[cyan]Initial Value:[/cyan]  {objectives[0]:.6e}
[cyan]Final Value:[/cyan]    {objectives[-1]:.6e}
[cyan]Improvement:[/cyan]    {objectives[0] - objectives[-1]:.6e}
[cyan]Evaluations:[/cyan]    {original_length}{downsampled_note}

[bold]Objective Value vs Iterations:[/bold]

{chart}
          {'─' * chart_width}
[white]{x_axis_line}[/white]
Iteration"""

        self.console.print()
        self.console.print(Panel(info, border_style="cyan", padding=(1, 2)))
        self.console.print()

    def handle_best(self):
        """Show best solution across all runs."""
        runs = self.platform.load_all_runs()

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

    def handle_compare(self, run_ids: List[int]):
        """Compare multiple runs side-by-side."""
        if len(run_ids) < 2:
            self.console.print("\n[red]Need at least 2 runs to compare[/red]\n")
            return

        # Load all runs
        runs = []
        for run_id in run_ids:
            run = self.platform.load_run(run_id)
            if run is None:
                self.console.print(f"\n[red]Run #{run_id} not found[/red]\n")
                return
            runs.append(run)

        # Create comparison table
        table = Table(title=f"Comparison: {' vs '.join(f'Run #{r.run_id}' for r in runs)}")
        table.add_column("Metric", style="cyan")

        for run in runs:
            status_icon = "✓" if run.success else "✗"
            table.add_column(f"#{run.run_id} ({run.algorithm})", justify="right")

        # Add rows for each metric
        metrics = [
            ("Problem", [r.problem_name for r in runs]),
            ("Objective", [f"{r.objective_value:.6e}" for r in runs]),
            ("Evaluations", [str(r.n_evaluations) for r in runs]),
            ("Time (s)", [f"{r.duration:.2f}" for r in runs]),
            ("Success", ["✓" if r.success else "✗" for r in runs]),
        ]

        for metric_name, values in metrics:
            # Highlight best value for numeric metrics
            if metric_name in ["Objective", "Evaluations", "Time (s)"]:
                # Find best (minimum for these metrics)
                numeric_values = []
                for v in values:
                    try:
                        if metric_name == "Success":
                            numeric_values.append(0)
                        else:
                            numeric_values.append(float(v.replace('✓', '0').replace('✗', '1')))
                    except:
                        numeric_values.append(float('inf'))

                best_idx = numeric_values.index(min(numeric_values))
                styled_values = []
                for i, v in enumerate(values):
                    if i == best_idx and metric_name != "Success":
                        styled_values.append(f"[bold green]{v} ✓[/bold green]")
                    else:
                        styled_values.append(v)
                table.add_row(metric_name, *styled_values)
            else:
                table.add_row(metric_name, *values)

        self.console.print()
        self.console.print(table)
        self.console.print()

    def handle_plot_compare(self, run_ids: List[int]):
        """Plot convergence comparison for multiple runs."""
        if len(run_ids) < 2:
            self.console.print("\n[red]Need at least 2 runs to compare[/red]\n")
            return

        # Load all runs and extract convergence data
        runs_data = []
        for run_id in run_ids:
            run = self.platform.load_run(run_id)
            if run is None:
                self.console.print(f"\n[red]Run #{run_id} not found[/red]\n")
                return

            iterations = run.result_data.get('iterations', [])
            if not iterations:
                self.console.print(f"\n[yellow]No iteration data for run #{run_id}[/yellow]\n")
                return

            objectives = [it['objective'] for it in iterations]
            runs_data.append({
                'run': run,
                'objectives': objectives
            })

        # Find max length and pad shorter sequences with their final value
        max_len = max(len(rd['objectives']) for rd in runs_data)

        series = []
        labels = []
        for rd in runs_data:
            run = rd['run']
            obj = rd['objectives']

            # Pad with final value if needed
            if len(obj) < max_len:
                obj = obj + [obj[-1]] * (max_len - len(obj))

            series.append(obj)
            labels.append(f"#{run.run_id} ({run.algorithm})")

        # Downsample if too many points (to fit terminal width)
        max_chart_width = 60  # Fits in 80-char terminal with Y-axis labels
        original_max_len = max_len
        if max_len > max_chart_width:
            # Downsample all series to max_chart_width points
            step = max_len / max_chart_width
            downsampled_series = []
            for s in series:
                downsampled = []
                for i in range(max_chart_width):
                    idx = int(i * step)
                    downsampled.append(s[idx])
                downsampled_series.append(downsampled)
            series = downsampled_series
            max_len = max_chart_width

        # Create multi-line ASCII plot with colors
        try:
            plot_config = {
                'height': 15,
                'format': '{:8.2e}',
                'colors': [
                    asciichart.blue,
                    asciichart.red,
                    asciichart.green,
                    asciichart.yellow,
                    asciichart.magenta,
                ][:len(series)]
            }
            chart = asciichart.plot(series, plot_config)
        except Exception as e:
            self.console.print(f"\n[red]Error creating comparison plot: {e}[/red]\n")
            return

        # Build legend with colored markers
        color_styles = ['blue', 'red', 'green', 'yellow', 'magenta']
        legend_lines = []
        for i in range(len(runs_data)):
            color = color_styles[i % len(color_styles)]
            legend_lines.append(
                f"  [{color}]●[/{color}] {labels[i]}: {runs_data[i]['run'].problem_name} → {runs_data[i]['objectives'][-1]:.6e}"
            )
        legend = "\n".join(legend_lines)

        # Build header with legend
        downsampled_note = f"\n[dim](Chart shows {max_len} sampled points from {original_max_len} total iterations)[/dim]" if max_len < original_max_len else ""

        header = f"""[bold cyan]Convergence Comparison[/bold cyan]

[bold]Legend:[/bold]
{legend}{downsampled_note}

[bold]Objective Value vs Iterations:[/bold]"""

        # Create x-axis labels aligned with actual chart width
        # asciichartpy makes chart width = number of data points
        chart_width = max_len
        num_ticks = 5

        # Build x-axis string with labels at exact positions
        x_axis_line = " " * 10  # Space for Y-axis labels (asciichartpy uses ~10 chars)
        x_axis_chars = [' '] * chart_width

        for i in range(num_ticks):
            tick_pos = int(i * (chart_width - 1) / (num_ticks - 1))
            # Use original_max_len for actual iteration number
            tick_label = str(int(i * (original_max_len - 1) / (num_ticks - 1)))
            # Center the label at tick position (shift left by half label length)
            label_start = tick_pos - len(tick_label) // 2
            # Ensure label doesn't go out of bounds
            if label_start < 0:
                label_start = 0
            if label_start + len(tick_label) > chart_width:
                label_start = chart_width - len(tick_label)
            # Place label characters
            for j, char in enumerate(tick_label):
                char_pos = label_start + j
                if 0 <= char_pos < chart_width:
                    x_axis_chars[char_pos] = char

        x_axis_line += ''.join(x_axis_chars)

        # Build complete panel content including chart and x-axis
        separator_line = " " * 10 + "─" * chart_width

        # Create renderable with chart (ANSI colors) and x-axis
        panel_content = Group(
            Text.from_markup(header),
            Text(""),
            Text.from_ansi(chart),
            Text(""),
            Text(separator_line),
            Text.from_markup(f"[white]{x_axis_line}[/white]"),
            Text("Iteration")
        )

        self.console.print()
        self.console.print(Panel(panel_content, border_style="cyan", padding=(1, 2)))
        self.console.print()


    def handle_analyze(self, run_id: int, focus: str = 'overall'):
        """AI-powered analysis of optimization run (costs money)."""
        run = self.platform.load_run(run_id)

        if not run:
            self.console.print(f"\n[red]Run #{run_id} not found[/red]\n")
            return

        # Show deterministic metrics first (instant preview)
        self.console.print("\n[cyan]Computing metrics...[/cyan]")
        metrics = compute_metrics(run)

        # Display summary
        self.console.print(f"[dim]Convergence: {metrics['convergence']['rate']:.4f}, Stalled: {metrics['convergence']['is_stalled']}[/dim]")
        self.console.print(f"[dim]Gradient quality: {metrics['gradient']['quality']}[/dim]")

        # Check if AI insights already cached
        if run.ai_insights:
            from datetime import datetime
            cached_time = datetime.fromisoformat(run.ai_insights.get('metadata', {}).get('timestamp', '2000-01-01'))
            age_minutes = (datetime.now() - cached_time).total_seconds() / 60
            self.console.print(f"\n[dim]Found cached AI analysis ({age_minutes:.1f} minutes old)[/dim]")
            user_input = input("Use cached analysis? (y/n, default: y): ").strip().lower()
            if user_input in ['', 'y', 'yes']:
                self._display_ai_insights(run.ai_insights)
                return

        # Confirm cost
        self.console.print(f"\n[yellow]⚠ AI analysis costs ~$0.02-0.05. Continue? (y/n)[/yellow]")
        user_input = input("> ").strip().lower()
        if user_input not in ['y', 'yes']:
            self.console.print("[dim]Cancelled[/dim]\n")
            return

        # Run AI analysis
        self.console.print("\n[dim]Analyzing with AI (this may take 5-10 seconds)...[/dim]")

        try:
            insights = ai_analyze(run, metrics, focus=focus)

            # Display insights
            self._display_ai_insights(insights)

        except Exception as e:
            self.console.print(f"\n[red]AI analysis failed: {e}[/red]\n")

    def _display_ai_insights(self, insights: Dict[str, Any]):
        """Display AI analysis results."""

        # Diagnosis
        self.console.print(f"\n[bold cyan]Diagnosis:[/bold cyan]")
        self.console.print(f"{insights.get('diagnosis', 'N/A')}\n")

        # Root cause
        self.console.print(f"[bold cyan]Root Cause:[/bold cyan]")
        self.console.print(f"{insights.get('root_cause', 'N/A')}\n")

        # Confidence
        confidence = insights.get('confidence', 'unknown')
        conf_style = {'high': 'green', 'medium': 'yellow', 'low': 'red'}.get(confidence, 'dim')
        self.console.print(f"[bold cyan]Confidence:[/bold cyan] [{conf_style}]{confidence.upper()}[/{conf_style}]\n")

        # Evidence
        evidence = insights.get('evidence', [])
        if evidence:
            self.console.print(f"[bold cyan]Evidence:[/bold cyan]")
            for e in evidence:
                self.console.print(f"  • {e}")
            self.console.print()

        # Recommendations
        recommendations = insights.get('recommendations', [])
        if recommendations:
            self.console.print(f"[bold cyan]Recommendations:[/bold cyan]")
            for i, rec in enumerate(recommendations, 1):
                self.console.print(f"\n{i}. [yellow]{rec.get('action', 'unknown')}[/yellow]")
                self.console.print(f"   {rec.get('rationale', 'No rationale provided')}")
                if rec.get('args'):
                    self.console.print(f"   [dim]Args: {rec['args']}[/dim]")
                if rec.get('expected_impact'):
                    self.console.print(f"   [dim]Expected: {rec['expected_impact']}[/dim]")

        # Metadata
        metadata = insights.get('metadata', {})
        if metadata:
            self.console.print(f"\n[dim]Model: {metadata.get('model', 'unknown')}, Cost: ~${metadata.get('cost_estimate', 0):.4f}[/dim]")

        self.console.print()

    def handle_knowledge_list(self):
        """List all stored insights (SKELETON - not implemented)."""
        self.console.print()
        self.console.print(Panel(
            Text.from_markup(
                "[bold yellow]Knowledge Module: Skeleton Only[/bold yellow]\n\n"
                "The knowledge module interfaces are defined but not yet implemented.\n\n"
                "[dim]Why skeleton?[/dim]\n"
                "Knowledge accumulation is highly data-driven and needs real optimization\n"
                "runs to determine:\n"
                "  • What problem signatures are discriminative\n"
                "  • What insights are valuable to store\n"
                "  • How the agent actually uses knowledge\n\n"
                "[dim]Current status:[/dim]\n"
                "  ✓ Interfaces defined (KnowledgeBase class)\n"
                "  ✓ Storage backends (MemoryKnowledgeStorage)\n"
                "  ✓ Agent tools created (placeholders)\n"
                "  ⏳ Implementation pending real data\n\n"
                "[cyan]See paola/knowledge/README.md for design intent[/cyan]"
            ),
            border_style="yellow",
            padding=(1, 2)
        ))
        self.console.print()

    def handle_knowledge_show(self, insight_id: str):
        """Show detailed insight (SKELETON - not implemented)."""
        self.console.print()
        self.console.print(f"[yellow]Knowledge module not yet implemented[/yellow]")
        self.console.print(f"[dim]Requested insight ID: {insight_id}[/dim]")
        self.console.print()
        self.console.print("[dim]This command will show detailed insight when implemented[/dim]")
        self.console.print()

