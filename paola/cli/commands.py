"""Command handlers for CLI - reads from foundry and displays.

v0.3.0: Updated for graph-based architecture.
- /graphs shows all optimization graphs
- Graph = complete optimization task (may contain multiple nodes)
- Node = single optimizer execution within a graph

v0.2.0: Session-based architecture (legacy).
- /sessions shows all optimization sessions
- Session = complete optimization task (may contain multiple runs)
"""

from typing import List, Dict, Any
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import asciichartpy as asciichart

from ..foundry import OptimizationFoundry


class CommandHandler:
    """
    Handles deterministic /commands.
    Pure presentation logic - reads from foundry storage.
    """

    def __init__(self, foundry: OptimizationFoundry, console: Console):
        self.foundry = foundry
        self.console = console

    # =========================================================================
    # Graph Commands (v0.3.0+)
    # =========================================================================

    def handle_graphs(self):
        """Display all optimization graphs in table format."""
        graphs = self.foundry.load_all_graphs()

        if not graphs:
            self.console.print("\n[dim]No optimization graphs yet[/dim]\n")
            return

        table = Table(title="Optimization Graphs")
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Problem")
        table.add_column("Nodes", justify="right")
        table.add_column("Pattern")
        table.add_column("Status")
        table.add_column("Best Value", justify="right")
        table.add_column("Evals", justify="right")
        table.add_column("Time", justify="right")

        for graph in graphs:
            status = "✓" if graph.success else "✗"
            status_style = "green" if graph.success else "red"

            # Get optimizers used
            optimizers = set()
            for node in list(graph.nodes.values())[:3]:
                opt_name = node.optimizer.split(":")[0]
                optimizers.add(opt_name)
            opt_str = ", ".join(optimizers)
            if len(graph.nodes) > 3:
                opt_str += "..."

            table.add_row(
                str(graph.graph_id),
                graph.problem_id,
                str(len(graph.nodes)),
                graph.detect_pattern(),
                f"[{status_style}]{status}[/{status_style}]",
                f"{graph.final_objective:.6f}" if graph.final_objective else "N/A",
                str(graph.total_evaluations),
                f"{graph.total_wall_time:.1f}s"
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def handle_graph_show(self, graph_id: int):
        """Show detailed graph information."""
        graph = self.foundry.load_graph(graph_id)

        if not graph:
            self.console.print(f"\n[red]Graph #{graph_id} not found[/red]\n")
            return

        # Build detailed panel
        status_text = "✓ Complete" if graph.success else "✗ Failed"
        status_style = "green" if graph.success else "red"

        # List nodes in graph
        nodes_info = ""
        for node in graph.nodes.values():
            node_status = "✓" if node.status == "completed" else "✗"
            node_style = "green" if node.status == "completed" else "red"

            # Check if node has parent
            parent_str = ""
            parent = graph.get_parent_node(node.node_id)
            if parent:
                edge = [e for e in graph.edges if e.target == node.node_id][0]
                parent_str = f" (from {parent} via {edge.edge_type})"

            nodes_info += f"\n  [{node_style}]{node_status}[/{node_style}] {node.node_id}: {node.optimizer} → {node.best_objective:.6e}{parent_str}"

        # Get best node
        best_node = graph.get_best_node()
        best_info = ""
        if best_node:
            best_info = f"\n\n[bold]Best Node:[/bold] {best_node.node_id} ({best_node.optimizer}) = {best_node.best_objective:.6e}"

        info = f"""[bold]Graph #{graph.graph_id}: {graph.problem_id}[/bold]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[cyan]Status:[/cyan]       [{status_style}]{status_text}[/{status_style}]
[cyan]Pattern:[/cyan]      {graph.detect_pattern()}
[cyan]Best Objective:[/cyan] {graph.final_objective:.6e}
[cyan]Total Evaluations:[/cyan] {graph.total_evaluations}
[cyan]Total Time:[/cyan]   {graph.total_wall_time:.2f}s
[cyan]Created:[/cyan]      {graph.created_at}
[cyan]Goal:[/cyan]         {graph.goal or 'N/A'}

[bold]Nodes ({len(graph.nodes)}):[/bold]{nodes_info}{best_info}

[bold]Final Solution (first 5 dimensions):[/bold]"""

        # Show first 5 dimensions of solution
        x = graph.final_design
        if x:
            for i in range(min(5, len(x))):
                info += f"\n  x[{i}] = {x[i]:.6f}"

            if len(x) > 5:
                info += f"\n  ... ({len(x) - 5} more dimensions)"
        else:
            info += "\n  [dim]No solution available[/dim]"

        # Show decisions if any
        if graph.decisions:
            info += "\n\n[bold]Decisions:[/bold]"
            for d in graph.decisions[-3:]:  # Last 3 decisions
                info += f"\n  • {d.decision_type}: {d.reasoning[:50]}..."

        self.console.print()
        self.console.print(Panel(info, border_style="cyan", padding=(1, 2)))
        self.console.print()

    def handle_graph_plot(self, graph_id: int):
        """Plot convergence history for a graph (ASCII in terminal)."""
        graph = self.foundry.load_graph(graph_id)

        if not graph:
            self.console.print(f"\n[red]Graph #{graph_id} not found[/red]\n")
            return

        # Collect objectives from all nodes in topological order
        objectives = []
        node_boundaries = []  # Track where each node starts

        for node_id in graph.get_topological_sort():
            node = graph.nodes[node_id]
            node_start = len(objectives)

            # Get progress data
            if hasattr(node, 'progress') and node.progress:
                progress = node.progress
                if hasattr(progress, 'iterations'):
                    for it in progress.iterations:
                        objectives.append(it.objective)
                elif hasattr(progress, 'trials'):
                    for trial in progress.trials:
                        objectives.append(trial.objective)

            if len(objectives) > node_start:
                node_boundaries.append((node_id, node_start, len(objectives)))

        if not objectives:
            self.console.print(f"\n[yellow]No iteration data available for graph #{graph_id}[/yellow]\n")
            return

        # Normalize to fixed chart width
        max_chart_width = 60
        original_length = len(objectives)

        if len(objectives) > max_chart_width:
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
            self.console.print(f"\n[red]Error creating plot: {e}[/red]\n")
            return

        # Build x-axis
        chart_width = len(objectives_to_plot)
        num_ticks = 5

        x_axis_line = " " * 10
        x_axis_chars = [' '] * chart_width

        for i in range(num_ticks):
            tick_pos = int(i * (chart_width - 1) / (num_ticks - 1))
            tick_label = str(int(i * (original_length - 1) / (num_ticks - 1)))
            label_start = tick_pos - len(tick_label) // 2
            if label_start < 0:
                label_start = 0
            if label_start + len(tick_label) > chart_width:
                label_start = chart_width - len(tick_label)
            for j, char in enumerate(tick_label):
                char_pos = label_start + j
                if 0 <= char_pos < chart_width:
                    x_axis_chars[char_pos] = char

        x_axis_line += ''.join(x_axis_chars)

        # Build node sequence string
        node_sequence = " → ".join(graph.nodes[nid].optimizer for nid, _, _ in node_boundaries) if node_boundaries else "N/A"

        downsampled_note = f" [dim](chart shows {len(objectives_to_plot)} sampled points)[/dim]" if len(objectives_to_plot) < original_length else ""

        info = f"""[bold cyan]Convergence History - Graph #{graph.graph_id}[/bold cyan]

[bold]{node_sequence} on {graph.problem_id}[/bold]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[cyan]Pattern:[/cyan]      {graph.detect_pattern()}
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

    def handle_graph_best(self):
        """Show best solution across all graphs."""
        graphs = self.foundry.load_all_graphs()

        if not graphs:
            self.console.print("\n[dim]No optimization graphs yet[/dim]\n")
            return

        # Find best graph (minimum objective value)
        valid_graphs = [g for g in graphs if g.final_objective is not None]
        if not valid_graphs:
            self.console.print("\n[dim]No completed optimization graphs[/dim]\n")
            return

        best_graph = min(valid_graphs, key=lambda g: g.final_objective)

        # Get optimizers used
        optimizers = ", ".join(set(n.optimizer for n in best_graph.nodes.values()))

        info = f"""[bold green]Best Solution Across All Graphs[/bold green]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[cyan]Graph ID:[/cyan]    #{best_graph.graph_id}
[cyan]Problem:[/cyan]     {best_graph.problem_id}
[cyan]Pattern:[/cyan]     {best_graph.detect_pattern()}
[cyan]Optimizers:[/cyan]  {optimizers}
[cyan]Objective:[/cyan]   [bold green]{best_graph.final_objective:.6e}[/bold green] ✓
[cyan]Evaluations:[/cyan] {best_graph.total_evaluations}
[cyan]Time:[/cyan]        {best_graph.total_wall_time:.2f}s"""

        self.console.print()
        self.console.print(Panel(info, border_style="green", padding=(1, 2)))
        self.console.print()

    def handle_graph_compare(self, graph_ids: List[int]):
        """Compare multiple graphs side-by-side."""
        if len(graph_ids) < 2:
            self.console.print("\n[red]Need at least 2 graphs to compare[/red]\n")
            return

        # Load all graphs
        graphs = []
        for graph_id in graph_ids:
            graph = self.foundry.load_graph(graph_id)
            if graph is None:
                self.console.print(f"\n[red]Graph #{graph_id} not found[/red]\n")
                return
            graphs.append(graph)

        # Create comparison table
        table = Table(title=f"Comparison: {' vs '.join(f'Graph #{g.graph_id}' for g in graphs)}")
        table.add_column("Metric", style="cyan")

        for graph in graphs:
            optimizers = ", ".join(set(n.optimizer.split(":")[0] for n in list(graph.nodes.values())[:2]))
            if len(graph.nodes) > 2:
                optimizers += "..."
            table.add_column(f"#{graph.graph_id} ({optimizers})", justify="right")

        # Add rows for each metric
        metrics = [
            ("Problem", [g.problem_id for g in graphs]),
            ("Pattern", [g.detect_pattern() for g in graphs]),
            ("Objective", [f"{g.final_objective:.6e}" if g.final_objective else "N/A" for g in graphs]),
            ("Nodes", [str(len(g.nodes)) for g in graphs]),
            ("Evaluations", [str(g.total_evaluations) for g in graphs]),
            ("Time (s)", [f"{g.total_wall_time:.2f}" for g in graphs]),
            ("Success", ["✓" if g.success else "✗" for g in graphs]),
        ]

        for metric_name, values in metrics:
            if metric_name in ["Objective", "Evaluations", "Time (s)"]:
                numeric_values = []
                for v in values:
                    try:
                        if v == "N/A":
                            numeric_values.append(float('inf'))
                        else:
                            numeric_values.append(float(v.replace('✓', '0').replace('✗', '1')))
                    except:
                        numeric_values.append(float('inf'))

                best_idx = numeric_values.index(min(numeric_values))
                styled_values = []
                for i, v in enumerate(values):
                    if i == best_idx and v != "N/A":
                        styled_values.append(f"[bold green]{v} ✓[/bold green]")
                    else:
                        styled_values.append(v)
                table.add_row(metric_name, *styled_values)
            else:
                table.add_row(metric_name, *values)

        self.console.print()
        self.console.print(table)
        self.console.print()

    # =========================================================================
    # Session Commands (v0.2.0 Legacy)
    # =========================================================================

    def handle_sessions(self):
        """Display all sessions in table format."""
        sessions = self.foundry.load_all_sessions()

        if not sessions:
            self.console.print("\n[dim]No optimization sessions yet[/dim]\n")
            return

        table = Table(title="Optimization Sessions")
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Problem")
        table.add_column("Runs", justify="right")
        table.add_column("Status")
        table.add_column("Best Value", justify="right")
        table.add_column("Evals", justify="right")
        table.add_column("Time", justify="right")

        for session in sessions:
            status = "✓" if session.success else "✗"
            status_style = "green" if session.success else "red"

            # Get best optimizer from runs
            optimizers = ", ".join(set(r.optimizer.split(":")[0] for r in session.runs[:3]))
            if len(session.runs) > 3:
                optimizers += "..."

            table.add_row(
                str(session.session_id),
                session.problem_id,
                str(len(session.runs)),
                f"[{status_style}]{status}[/{status_style}]",
                f"{session.final_objective:.6f}",
                str(session.total_evaluations),
                f"{session.total_wall_time:.1f}s"
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def handle_show(self, session_id: int):
        """Show detailed session information."""
        session = self.foundry.load_session(session_id)

        if not session:
            self.console.print(f"\n[red]Session #{session_id} not found[/red]\n")
            return

        # Build detailed panel
        status_text = "✓ Complete" if session.success else "✗ Failed"
        status_style = "green" if session.success else "red"

        # List runs in session
        runs_info = ""
        for run in session.runs:
            run_status = "✓" if run.run_success else "✗"
            run_style = "green" if run.run_success else "red"
            warm_start = f" (warm from #{run.warm_start_from})" if run.warm_start_from else ""
            runs_info += f"\n  [{run_style}]{run_status}[/{run_style}] Run #{run.run_id}: {run.optimizer} → {run.best_objective:.6e}{warm_start}"

        info = f"""[bold]Session #{session.session_id}: {session.problem_id}[/bold]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[cyan]Status:[/cyan]       [{status_style}]{status_text}[/{status_style}]
[cyan]Best Objective:[/cyan] {session.final_objective:.6e}
[cyan]Total Evaluations:[/cyan] {session.total_evaluations}
[cyan]Total Time:[/cyan]   {session.total_wall_time:.2f}s
[cyan]Created:[/cyan]      {session.created_at}

[bold]Runs ({len(session.runs)}):[/bold]{runs_info}

[bold]Final Solution (first 5 dimensions):[/bold]"""

        # Show first 5 dimensions of solution
        x = session.final_design
        for i in range(min(5, len(x))):
            info += f"\n  x[{i}] = {x[i]:.6f}"

        if len(x) > 5:
            info += f"\n  ... ({len(x) - 5} more dimensions)"

        # Show decisions if any
        if session.decisions:
            info += "\n\n[bold]Decisions:[/bold]"
            for d in session.decisions[-3:]:  # Last 3 decisions
                info += f"\n  • {d.decision_type}: {d.reasoning[:50]}..."

        self.console.print()
        self.console.print(Panel(info, border_style="cyan", padding=(1, 2)))
        self.console.print()

    def handle_plot(self, session_id: int):
        """Plot convergence history (ASCII in terminal)."""
        session = self.foundry.load_session(session_id)

        if not session:
            self.console.print(f"\n[red]Session #{session_id} not found[/red]\n")
            return

        # Collect objectives from all runs in session
        objectives = []
        for run in session.runs:
            # Get progress data (iterations/trials)
            progress = run.progress
            if hasattr(progress, 'iterations'):
                for it in progress.iterations:
                    objectives.append(it.objective)
            elif hasattr(progress, 'trials'):
                for trial in progress.trials:
                    objectives.append(trial.objective)

        if not objectives:
            self.console.print(f"\n[yellow]No iteration data available for session #{session_id}[/yellow]\n")
            return

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
        chart_width = len(objectives_to_plot)
        num_ticks = 5

        # Build x-axis string with labels at exact positions
        x_axis_line = " " * 10  # Space for Y-axis labels
        x_axis_chars = [' '] * chart_width

        for i in range(num_ticks):
            tick_pos = int(i * (chart_width - 1) / (num_ticks - 1))
            tick_label = str(int(i * (original_length - 1) / (num_ticks - 1)))
            label_start = tick_pos - len(tick_label) // 2
            if label_start < 0:
                label_start = 0
            if label_start + len(tick_label) > chart_width:
                label_start = chart_width - len(tick_label)
            for j, char in enumerate(tick_label):
                char_pos = label_start + j
                if 0 <= char_pos < chart_width:
                    x_axis_chars[char_pos] = char

        x_axis_line += ''.join(x_axis_chars)

        # Build info panel
        downsampled_note = f" [dim](chart shows {len(objectives_to_plot)} sampled points)[/dim]" if len(objectives_to_plot) < original_length else ""

        optimizers = " → ".join(r.optimizer for r in session.runs)

        info = f"""[bold cyan]Convergence History - Session #{session.session_id}[/bold cyan]

[bold]{optimizers} on {session.problem_id}[/bold]
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
        """Show best solution across all sessions."""
        sessions = self.foundry.load_all_sessions()

        if not sessions:
            self.console.print("\n[dim]No optimization sessions yet[/dim]\n")
            return

        # Find best session (minimum objective value)
        best_session = min(sessions, key=lambda s: s.final_objective)

        optimizers = ", ".join(r.optimizer for r in best_session.runs)

        info = f"""[bold green]Best Solution Across All Sessions[/bold green]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[cyan]Session ID:[/cyan]  #{best_session.session_id}
[cyan]Problem:[/cyan]     {best_session.problem_id}
[cyan]Optimizers:[/cyan]  {optimizers}
[cyan]Objective:[/cyan]   [bold green]{best_session.final_objective:.6e}[/bold green] ✓
[cyan]Evaluations:[/cyan] {best_session.total_evaluations}
[cyan]Time:[/cyan]        {best_session.total_wall_time:.2f}s"""

        self.console.print()
        self.console.print(Panel(info, border_style="green", padding=(1, 2)))
        self.console.print()

    def handle_compare(self, session_ids: List[int]):
        """Compare multiple sessions side-by-side."""
        if len(session_ids) < 2:
            self.console.print("\n[red]Need at least 2 sessions to compare[/red]\n")
            return

        # Load all sessions
        sessions = []
        for session_id in session_ids:
            session = self.foundry.load_session(session_id)
            if session is None:
                self.console.print(f"\n[red]Session #{session_id} not found[/red]\n")
                return
            sessions.append(session)

        # Create comparison table
        table = Table(title=f"Comparison: {' vs '.join(f'Session #{s.session_id}' for s in sessions)}")
        table.add_column("Metric", style="cyan")

        for session in sessions:
            # Get optimizers used in session
            optimizers = ", ".join(set(r.optimizer.split(":")[0] for r in session.runs[:2]))
            if len(session.runs) > 2:
                optimizers += "..."
            table.add_column(f"#{session.session_id} ({optimizers})", justify="right")

        # Add rows for each metric
        metrics = [
            ("Problem", [s.problem_id for s in sessions]),
            ("Objective", [f"{s.final_objective:.6e}" for s in sessions]),
            ("Runs", [str(len(s.runs)) for s in sessions]),
            ("Evaluations", [str(s.total_evaluations) for s in sessions]),
            ("Time (s)", [f"{s.total_wall_time:.2f}" for s in sessions]),
            ("Success", ["✓" if s.success else "✗" for s in sessions]),
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

    def handle_plot_compare(self, session_ids: List[int]):
        """Plot convergence comparison for multiple sessions."""
        if len(session_ids) < 2:
            self.console.print("\n[red]Need at least 2 sessions to compare[/red]\n")
            return

        # Load all sessions and extract convergence data
        sessions_data = []
        for session_id in session_ids:
            session = self.foundry.load_session(session_id)
            if session is None:
                self.console.print(f"\n[red]Session #{session_id} not found[/red]\n")
                return

            # Collect objectives from all runs in session
            objectives = []
            for run in session.runs:
                progress = run.progress
                if hasattr(progress, 'iterations'):
                    for it in progress.iterations:
                        objectives.append(it.objective)
                elif hasattr(progress, 'trials'):
                    for trial in progress.trials:
                        objectives.append(trial.objective)

            if not objectives:
                self.console.print(f"\n[yellow]No iteration data for session #{session_id}[/yellow]\n")
                return

            sessions_data.append({
                'session': session,
                'objectives': objectives
            })

        # Find max length and pad shorter sequences with their final value
        max_len = max(len(sd['objectives']) for sd in sessions_data)

        series = []
        labels = []
        for sd in sessions_data:
            session = sd['session']
            obj = sd['objectives']

            # Pad with final value if needed
            if len(obj) < max_len:
                obj = obj + [obj[-1]] * (max_len - len(obj))

            series.append(obj)
            # Get optimizers used
            optimizers = ", ".join(set(r.optimizer.split(":")[0] for r in session.runs[:2]))
            labels.append(f"#{session.session_id} ({optimizers})")

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
        for i in range(len(sessions_data)):
            color = color_styles[i % len(color_styles)]
            legend_lines.append(
                f"  [{color}]●[/{color}] {labels[i]}: {sessions_data[i]['session'].problem_id} → {sessions_data[i]['objectives'][-1]:.6e}"
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


    def handle_analyze(self, session_id: int, focus: str = 'overall'):
        """AI-powered analysis of optimization session (costs money)."""
        session = self.foundry.load_session(session_id)

        if not session:
            self.console.print(f"\n[red]Session #{session_id} not found[/red]\n")
            return

        # Show basic session metrics
        self.console.print("\n[cyan]Computing metrics...[/cyan]")

        # Compute basic metrics from session
        objectives = []
        for run in session.runs:
            progress = run.progress
            if hasattr(progress, 'iterations'):
                for it in progress.iterations:
                    objectives.append(it.objective)
            elif hasattr(progress, 'trials'):
                for trial in progress.trials:
                    objectives.append(trial.objective)

        if not objectives:
            self.console.print(f"\n[yellow]No iteration data available for session #{session_id}[/yellow]\n")
            return

        # Basic convergence metrics
        n_iters = len(objectives)
        improvement = objectives[0] - objectives[-1] if n_iters >= 2 else 0.0
        is_stalled = False
        if n_iters >= 5:
            last_5 = objectives[-5:]
            is_stalled = abs(max(last_5) - min(last_5)) < 1e-6

        self.console.print(f"[dim]Iterations: {n_iters}, Improvement: {improvement:.6e}, Stalled: {is_stalled}[/dim]")
        self.console.print(f"[dim]Final objective: {session.final_objective:.6e}[/dim]")

        # Confirm cost
        self.console.print(f"\n[yellow]⚠ AI analysis costs ~$0.02-0.05. Continue? (y/n)[/yellow]")
        user_input = input("> ").strip().lower()
        if user_input not in ['y', 'yes']:
            self.console.print("[dim]Cancelled[/dim]\n")
            return

        # Run AI analysis
        self.console.print("\n[dim]Analyzing with AI (this may take 5-10 seconds)...[/dim]")

        # Build metrics dict for AI analysis
        metrics = {
            "convergence": {
                "iterations_total": n_iters,
                "improvement": improvement,
                "is_stalled": is_stalled,
            },
            "objective": {
                "initial": objectives[0] if objectives else 0.0,
                "final": objectives[-1] if objectives else 0.0,
                "best": min(objectives) if objectives else 0.0,
            },
            "session": {
                "n_runs": len(session.runs),
                "total_evaluations": session.total_evaluations,
                "total_wall_time": session.total_wall_time,
                "success": session.success,
            }
        }

        try:
            # Note: ai_analyze needs to be updated for sessions
            # For now, show a placeholder message
            self.console.print("\n[yellow]Note: Full AI analysis support for sessions coming soon.[/yellow]")
            self.console.print(f"\n[bold cyan]Session #{session_id} Summary:[/bold cyan]")
            self.console.print(f"  Problem: {session.problem_id}")
            self.console.print(f"  Runs: {len(session.runs)}")
            self.console.print(f"  Best objective: {session.final_objective:.6e}")
            self.console.print(f"  Evaluations: {session.total_evaluations}")
            self.console.print(f"  Status: {'✓ Success' if session.success else '✗ Failed'}")
            self.console.print()

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

    def handle_register(self, file_path: str):
        """
        Register an evaluator function.

        Interactive registration flow using agent.

        Args:
            file_path: Path to Python file containing evaluator
        """
        from pathlib import Path

        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            self.console.print(f"\n[red]Error: File not found: {file_path}[/red]\n")
            return

        if not path.suffix == '.py':
            self.console.print(f"\n[red]Error: File must be a Python file (.py)[/red]\n")
            return

        self.console.print(f"\n[cyan]Registering evaluator from:[/cyan] {file_path}")
        self.console.print("[dim]Reading file...[/dim]")

        # Import registration tools
        from ..tools.registration_tools import (
            read_file,
            execute_python,
            foundry_store_evaluator
        )

        # Read file
        result = read_file.func(file_path=str(path))

        if not result["success"]:
            self.console.print(f"\n[red]Error reading file: {result['error']}[/red]\n")
            return

        # Display file contents
        self.console.print("\n[bold]File contents:[/bold]")
        self.console.print(Panel(result["contents"], border_style="dim", padding=(1, 2)))

        # Interactive questions
        self.console.print("\n[cyan]Please provide the following information:[/cyan]\n")

        function_name = input("  Function name: ").strip()
        if not function_name:
            self.console.print("[red]Function name is required[/red]")
            return

        evaluator_name = input("  Evaluator name (default: same as function): ").strip()
        if not evaluator_name:
            evaluator_name = function_name

        evaluator_id = input(f"  Evaluator ID (default: {evaluator_name}_eval): ").strip()
        if not evaluator_id:
            evaluator_id = f"{evaluator_name}_eval"

        # Build configuration
        config = {
            "evaluator_id": evaluator_id,
            "name": evaluator_name,
            "source": {
                "type": "python_function",
                "file_path": str(path.absolute()),
                "callable_name": function_name
            },
            "interface": {
                "output": {"format": "auto"}
            },
            "capabilities": {
                "observation_gates": True,
                "caching": True
            },
            "performance": {
                "cost_per_eval": 1.0
            }
        }

        # Test configuration
        self.console.print("\n[dim]Testing configuration...[/dim]")

        test_code = f"""
from paola.foundry import FoundryEvaluator
import numpy as np

config = {config}
evaluator = FoundryEvaluator.from_config(config)
result = evaluator.evaluate(np.array([1.0, 1.0]))
print(f"Test result: {{result.objectives}}")
print("SUCCESS")
"""

        test_result = execute_python.func(code=test_code, timeout=10)

        if not test_result["success"]:
            self.console.print(f"\n[red]Configuration test failed:[/red]")
            self.console.print(f"[dim]stdout:[/dim] {test_result.get('stdout', '')}")
            self.console.print(f"[red]stderr:[/red] {test_result.get('stderr', '')}")
            self.console.print("\n[yellow]Registration aborted[/yellow]\n")
            return

        if "SUCCESS" not in test_result["stdout"]:
            self.console.print(f"\n[yellow]Warning: Test did not complete successfully[/yellow]")
            self.console.print(f"[dim]Output:[/dim] {test_result['stdout']}")
            confirm = input("Continue with registration anyway? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes']:
                self.console.print("[dim]Registration cancelled[/dim]\n")
                return

        # Store in Foundry
        self.console.print("[dim]Storing in Foundry...[/dim]")

        store_result = foundry_store_evaluator.func(
            config=config,
            test_result=test_result
        )

        if not store_result["success"]:
            self.console.print(f"\n[red]Failed to store evaluator: {store_result['error']}[/red]\n")
            return

        # Success
        info = f"""[bold green]✓ Evaluator Registered Successfully[/bold green]

[cyan]Evaluator ID:[/cyan]  {store_result['evaluator_id']}
[cyan]Name:[/cyan]          {config['name']}
[cyan]Source:[/cyan]        {config['source']['file_path']}
[cyan]Function:[/cyan]      {config['source']['callable_name']}

[dim]You can now use this evaluator in optimizations[/dim]"""

        self.console.print()
        self.console.print(Panel(info, border_style="green", padding=(1, 2)))
        self.console.print()

    def handle_evaluators(self):
        """List all registered evaluators."""
        from ..tools.registration_tools import foundry_list_evaluators

        result = foundry_list_evaluators.invoke({})

        if not result["success"]:
            self.console.print(f"\n[red]Error: {result['error']}[/red]\n")
            return

        evaluators = result["evaluators"]

        if not evaluators:
            self.console.print("\n[dim]No evaluators registered yet[/dim]\n")
            self.console.print("[dim]Use /register <file.py> to register an evaluator[/dim]\n")
            return

        # Create table
        table = Table(title="Registered Evaluators")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Type", style="yellow")
        table.add_column("Status")

        for ev in evaluators:
            status_icon = "✓" if ev.get("status") == "active" else "●"
            status_style = "green" if ev.get("status") == "active" else "yellow"

            table.add_row(
                ev["evaluator_id"],
                ev["name"],
                ev["type"],
                f"[{status_style}]{status_icon}[/{status_style}]"
            )

        self.console.print()
        self.console.print(table)
        self.console.print()
        self.console.print(f"[dim]Total: {result['count']} evaluators[/dim]")
        self.console.print()

    def handle_evaluator_show(self, evaluator_id: str):
        """Show detailed evaluator configuration."""
        from ..tools.registration_tools import foundry_get_evaluator

        result = foundry_get_evaluator.invoke({"evaluator_id": evaluator_id})

        if not result["success"]:
            self.console.print(f"\n[red]Error: {result['error']}[/red]\n")
            return

        config = result["config"]

        # Build info panel
        info = f"""[bold cyan]Evaluator: {config['name']}[/bold cyan]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[cyan]ID:[/cyan]           {config['evaluator_id']}
[cyan]Status:[/cyan]       {config.get('status', 'unknown')}
[cyan]Type:[/cyan]         {config['source']['type']}
[cyan]Source:[/cyan]       {config['source']['file_path']}
[cyan]Function:[/cyan]     {config['source']['callable_name']}

[bold]Capabilities:[/bold]
  • Observation gates: {config.get('capabilities', {}).get('observation_gates', False)}
  • Caching: {config.get('capabilities', {}).get('caching', False)}

[bold]Performance:[/bold]
  • Cost per eval: {config.get('performance', {}).get('cost_per_eval', 'N/A')}
  • Total evaluations: {config.get('performance', {}).get('total_evaluations', 0)}
  • Success rate: {config.get('performance', {}).get('success_rate', 'N/A')}"""

        self.console.print()
        self.console.print(Panel(info, border_style="cyan", padding=(1, 2)))
        self.console.print()

