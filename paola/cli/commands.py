"""Command handlers for CLI - reads from foundry and displays.

v0.3.1: Two-tier graph storage for cross-graph learning.
- GraphRecord (Tier 1): Compact ~1KB, strategy-focused for LLM queries
- GraphDetail (Tier 2): Full trajectories 10-100KB, for visualization
- /graph query command for cross-graph learning

v0.3.0: Graph-based architecture.
- /graphs shows all optimization graphs
- Graph = complete optimization task (may contain multiple nodes)
- Node = single optimizer execution within a graph
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
        # Use two-tier storage (GraphRecords - Tier 1)
        records = self.foundry.load_all_graph_records()

        if not records:
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

        for record in records:
            status = "✓" if record.success else "✗"
            status_style = "green" if record.success else "red"

            # Get optimizers used
            optimizers = set()
            for node in list(record.nodes.values())[:3]:
                opt_name = node.optimizer.split(":")[0]
                optimizers.add(opt_name)
            opt_str = ", ".join(optimizers)
            if len(record.nodes) > 3:
                opt_str += "..."

            table.add_row(
                str(record.graph_id),
                str(record.problem_id),
                str(len(record.nodes)),
                record.pattern,
                f"[{status_style}]{status}[/{status_style}]",
                f"{record.final_objective:.6f}" if record.final_objective else "N/A",
                str(record.total_evaluations),
                f"{record.total_wall_time:.1f}s"
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def handle_graph_show(self, graph_id: int):
        """Show detailed graph information."""
        # Use two-tier storage
        record = self.foundry.load_graph_record(graph_id)
        detail = self.foundry.load_graph_detail(graph_id)

        if not record:
            self.console.print(f"\n[red]Graph #{graph_id} not found[/red]\n")
            return

        # Status
        status_text = "✓ Success" if record.success else "✗ Failed"
        status_style = "green" if record.success else "red"

        # Find best node
        best_node_id = None
        best_objective = float('inf')
        for node_id, node in record.nodes.items():
            if node.best_objective is not None and node.best_objective < best_objective:
                best_objective = node.best_objective
                best_node_id = node_id

        # Build ASCII graph structure
        graph_art = self._build_graph_ascii_from_record(record)

        # Header
        info = f"""[bold]Graph #{record.graph_id}[/bold]  {record.problem_id}  [{status_style}]{status_text}[/{status_style}]
[dim]{record.pattern} pattern • {len(record.nodes)} nodes • {record.total_evaluations} evals • {record.total_wall_time:.1f}s[/dim]

[bold]Structure:[/bold]
{graph_art}

[bold]Nodes:[/bold]"""

        # Simple node list
        for node_id, node in record.nodes.items():
            status_icon = "✓" if node.status == "completed" else "✗"
            style = "green" if node.status == "completed" else "red"
            is_best = node_id == best_node_id
            best_marker = " [bold green]★ best[/bold green]" if is_best else ""
            obj_str = f"{node.best_objective:.4e}" if node.best_objective is not None else "N/A"
            info += f"\n  [{style}]{status_icon}[/{style}] [cyan]{node_id}[/cyan]: {node.optimizer} → {obj_str} ({node.n_evaluations} evals){best_marker}"

        # Edges (if any)
        if record.edges:
            info += "\n\n[bold]Edges:[/bold]"
            for edge in record.edges:
                info += f"\n  {edge.source} → {edge.target} [dim]({edge.edge_type})[/dim]"

        # Best solution preview (from GraphDetail - Tier 2)
        if detail and best_node_id and best_node_id in detail.nodes:
            best_x = detail.nodes[best_node_id].best_x
            if best_x:
                x_str = ", ".join(f"{xi:.3f}" for xi in best_x[:4])
                if len(best_x) > 4:
                    x_str += f", ... ({len(best_x)} dims)"
                info += f"\n\n[bold]Best x:[/bold] [{x_str}]"
        elif record.final_x:
            # Fallback to record's final_x
            x = record.final_x
            x_str = ", ".join(f"{xi:.3f}" for xi in x[:4])
            if len(x) > 4:
                x_str += f", ... ({len(x)} dims)"
            info += f"\n\n[bold]Best x:[/bold] [{x_str}]"

        self.console.print()
        self.console.print(Panel(info, border_style="cyan", padding=(1, 2)))
        self.console.print()

    def _build_graph_ascii(self, graph) -> str:
        """Build ASCII art representation of graph structure (vertical tree)."""
        if not graph.nodes:
            return "  (empty)"

        pattern = graph.detect_pattern()
        nodes = list(graph.nodes.keys())

        if pattern == "single":
            return "      (n1)"

        if pattern == "multistart":
            return "  " + "   ".join(f"({n})" for n in nodes)

        # Use horizontal tree style (cleaner, shows connections properly)
        # This renders like:
        #   (n1)
        #    └─(n2)
        #       ├─(n3)
        #       └─(n4)

        lines = []
        root_nodes = [n for n in nodes if not graph.get_predecessors(n)]

        def render_node(node_id, prefix="", is_last=True):
            """Recursively render a node and its children."""
            # Draw connector and node
            if prefix == "":
                # Root node
                lines.append(f"  ({node_id})")
            else:
                connector = "└─" if is_last else "├─"
                lines.append(f"{prefix}{connector}({node_id})")

            # Get children and render them
            children = graph.get_successors(node_id)
            for i, child in enumerate(children):
                is_child_last = (i == len(children) - 1)
                if prefix == "":
                    child_prefix = "   "
                else:
                    child_prefix = prefix + ("  " if is_last else "│ ")
                render_node(child, child_prefix, is_child_last)

        for i, root in enumerate(root_nodes):
            if i > 0:
                lines.append("")  # Blank line between trees
            render_node(root)

        return "\n".join(lines)

    def _build_graph_ascii_from_record(self, record) -> str:
        """Build ASCII art representation of graph structure from GraphRecord."""
        if not record.nodes:
            return "  (empty)"

        nodes = list(record.nodes.keys())

        # Single node with no edges
        if len(nodes) == 1 and not record.edges:
            return "      (n1)"

        # No edges = multistart pattern
        if not record.edges:
            return "  " + "   ".join(f"({n})" for n in nodes)

        # Build parent→children mapping from edges
        children_map = {}  # node_id → list of children
        parents_set = set()  # nodes that have parents
        for edge in record.edges:
            if edge.source not in children_map:
                children_map[edge.source] = []
            children_map[edge.source].append(edge.target)
            parents_set.add(edge.target)

        # Root nodes are those without parents
        root_nodes = [n for n in nodes if n not in parents_set]

        lines = []

        def render_node(node_id, prefix="", is_last=True):
            """Recursively render a node and its children."""
            if prefix == "":
                lines.append(f"  ({node_id})")
            else:
                connector = "└─" if is_last else "├─"
                lines.append(f"{prefix}{connector}({node_id})")

            children = children_map.get(node_id, [])
            for i, child in enumerate(children):
                is_child_last = (i == len(children) - 1)
                if prefix == "":
                    child_prefix = "   "
                else:
                    child_prefix = prefix + ("  " if is_last else "│ ")
                render_node(child, child_prefix, is_child_last)

        for i, root in enumerate(root_nodes):
            if i > 0:
                lines.append("")
            render_node(root)

        return "\n".join(lines) if lines else "  (empty)"

    def handle_graph_plot(self, graph_id: int):
        """Plot convergence history for a graph (ASCII in terminal)."""
        # Try to load GraphRecord (Tier 1) and GraphDetail (Tier 2)
        record = self.foundry.load_graph_record(graph_id)
        detail = self.foundry.load_graph_detail(graph_id)

        # If no detail file (old format or missing), fall back to legacy
        if not detail:
            graph = self.foundry.load_graph(graph_id)
            if graph:
                # Use legacy data extraction (has full progress data)
                return self._plot_legacy_graph(graph)
            elif not record:
                self.console.print(f"\n[red]Graph #{graph_id} not found[/red]\n")
                return
            else:
                # Record exists but no detail and no legacy - can't plot
                self.console.print(f"\n[yellow]No convergence data available for graph #{graph_id}[/yellow]")
                self.console.print(f"[dim](Graph was saved without trajectory data)[/dim]\n")
                return

        if not record:
            self.console.print(f"\n[red]Graph #{graph_id} not found[/red]\n")
            return

        # Collect objectives from GraphDetail (Tier 2)
        objectives = []
        node_boundaries = []  # Track where each node starts
        node_optimizers = {}  # Map node_id to optimizer name

        # Get optimizer names from record
        for node_id, node_summary in record.nodes.items():
            node_optimizers[node_id] = node_summary.optimizer

        # Get convergence data from detail
        if detail:
            # Sort nodes by node_id (n1, n2, n3...)
            for node_id in sorted(detail.nodes.keys()):
                node_detail = detail.nodes[node_id]
                node_start = len(objectives)

                # Get convergence history
                for point in node_detail.convergence_history:
                    objectives.append(point.objective)

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
        node_sequence = " → ".join(node_optimizers.get(nid, nid) for nid, _, _ in node_boundaries) if node_boundaries else "N/A"

        downsampled_note = f" [dim](chart shows {len(objectives_to_plot)} sampled points)[/dim]" if len(objectives_to_plot) < original_length else ""

        info = f"""[bold cyan]Convergence History - Graph #{record.graph_id}[/bold cyan]

[bold]{node_sequence} on {record.problem_id}[/bold]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[cyan]Pattern:[/cyan]      {record.pattern}
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

    def _plot_legacy_graph(self, graph):
        """Plot convergence for legacy OptimizationGraph format."""
        # Collect objectives from all nodes in topological order
        objectives = []
        node_boundaries = []

        for node_id in graph.topological_sort():
            node = graph.nodes[node_id]
            node_start = len(objectives)

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
            self.console.print(f"\n[yellow]No iteration data available for graph #{graph.graph_id}[/yellow]\n")
            return

        # Normalize to fixed chart width
        max_chart_width = 60
        original_length = len(objectives)

        if len(objectives) > max_chart_width:
            step = len(objectives) / max_chart_width
            objectives_to_plot = [objectives[int(i * step)] for i in range(max_chart_width)]
        else:
            objectives_to_plot = objectives

        plot_config = {'height': 15, 'format': '{:8.2e}'}

        try:
            chart = asciichart.plot(objectives_to_plot, plot_config)
        except Exception as e:
            self.console.print(f"\n[red]Error creating plot: {e}[/red]\n")
            return

        chart_width = len(objectives_to_plot)
        num_ticks = 5
        x_axis_line = " " * 10
        x_axis_chars = [' '] * chart_width

        for i in range(num_ticks):
            tick_pos = int(i * (chart_width - 1) / (num_ticks - 1))
            tick_label = str(int(i * (original_length - 1) / (num_ticks - 1)))
            label_start = max(0, tick_pos - len(tick_label) // 2)
            if label_start + len(tick_label) > chart_width:
                label_start = chart_width - len(tick_label)
            for j, char in enumerate(tick_label):
                char_pos = label_start + j
                if 0 <= char_pos < chart_width:
                    x_axis_chars[char_pos] = char

        x_axis_line += ''.join(x_axis_chars)
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
        # Use two-tier storage (GraphRecords)
        records = self.foundry.load_all_graph_records()

        if not records:
            self.console.print("\n[dim]No optimization graphs yet[/dim]\n")
            return

        # Find best graph (minimum objective value)
        valid_records = [r for r in records if r.final_objective is not None]
        if not valid_records:
            self.console.print("\n[dim]No completed optimization graphs[/dim]\n")
            return

        best_record = min(valid_records, key=lambda r: r.final_objective)

        # Get optimizers used
        optimizers = ", ".join(set(n.optimizer for n in best_record.nodes.values()))

        info = f"""[bold green]Best Solution Across All Graphs[/bold green]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[cyan]Graph ID:[/cyan]    #{best_record.graph_id}
[cyan]Problem:[/cyan]     {best_record.problem_id}
[cyan]Pattern:[/cyan]     {best_record.pattern}
[cyan]Optimizers:[/cyan]  {optimizers}
[cyan]Objective:[/cyan]   [bold green]{best_record.final_objective:.6e}[/bold green] ✓
[cyan]Evaluations:[/cyan] {best_record.total_evaluations}
[cyan]Time:[/cyan]        {best_record.total_wall_time:.2f}s"""

        self.console.print()
        self.console.print(Panel(info, border_style="green", padding=(1, 2)))
        self.console.print()

    def handle_graph_compare(self, graph_ids: List[int]):
        """Compare multiple graphs side-by-side."""
        if len(graph_ids) < 2:
            self.console.print("\n[red]Need at least 2 graphs to compare[/red]\n")
            return

        # Load all graph records (Tier 1)
        records = []
        for graph_id in graph_ids:
            record = self.foundry.load_graph_record(graph_id)
            if record is None:
                self.console.print(f"\n[red]Graph #{graph_id} not found[/red]\n")
                return
            records.append(record)

        # Create comparison table
        table = Table(title=f"Comparison: {' vs '.join(f'Graph #{r.graph_id}' for r in records)}")
        table.add_column("Metric", style="cyan")

        for record in records:
            optimizers = ", ".join(set(n.optimizer.split(":")[0] for n in list(record.nodes.values())[:2]))
            if len(record.nodes) > 2:
                optimizers += "..."
            table.add_column(f"#{record.graph_id} ({optimizers})", justify="right")

        # Add rows for each metric
        metrics = [
            ("Problem", [str(r.problem_id) for r in records]),
            ("Pattern", [r.pattern for r in records]),
            ("Objective", [f"{r.final_objective:.6e}" if r.final_objective else "N/A" for r in records]),
            ("Nodes", [str(len(r.nodes)) for r in records]),
            ("Evaluations", [str(r.total_evaluations) for r in records]),
            ("Time (s)", [f"{r.total_wall_time:.2f}" for r in records]),
            ("Success", ["✓" if r.success else "✗" for r in records]),
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

    def handle_graph_query(
        self,
        problem_pattern: str = None,
        n_dimensions: int = None,
        success: bool = None,
        limit: int = 5,
    ):
        """
        Query past optimization graphs for cross-graph learning.

        Args:
            problem_pattern: Problem ID pattern (e.g., "ackley*", "rosenbrock*")
            n_dimensions: Filter by problem dimensions
            success: Filter by success status (True for successful only)
            limit: Maximum results to return
        """
        # Query graphs
        records = self.foundry.query_graphs(
            problem_id=problem_pattern,
            n_dimensions=n_dimensions,
            success=success,
            limit=limit,
        )

        if not records:
            filters = []
            if problem_pattern:
                filters.append(f"problem='{problem_pattern}'")
            if n_dimensions:
                filters.append(f"dims={n_dimensions}")
            if success is not None:
                filters.append(f"success={success}")
            filter_str = ", ".join(filters) if filters else "none"
            self.console.print(f"\n[dim]No graphs found matching filters: {filter_str}[/dim]\n")
            return

        # Create results table
        table = Table(title=f"Query Results ({len(records)} graphs)")
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Problem")
        table.add_column("Dims", justify="right")
        table.add_column("Pattern")
        table.add_column("Strategy")
        table.add_column("Objective", justify="right")
        table.add_column("Status")

        for record in records:
            # Get dimensions from problem signature
            dims = "?"
            if record.problem_signature:
                dims = str(record.problem_signature.n_dimensions)

            # Build strategy string (optimizers used in sequence)
            strategy_parts = []
            for node_id in sorted(record.nodes.keys()):
                node = record.nodes[node_id]
                opt_short = node.optimizer.split(":")[-1]  # Get method name
                strategy_parts.append(opt_short)
            strategy = " → ".join(strategy_parts[:3])
            if len(strategy_parts) > 3:
                strategy += " ..."

            # Status
            status = "✓" if record.success else "✗"
            status_style = "green" if record.success else "red"

            table.add_row(
                str(record.graph_id),
                str(record.problem_id),
                dims,
                record.pattern,
                strategy,
                f"{record.final_objective:.4e}" if record.final_objective else "N/A",
                f"[{status_style}]{status}[/{status_style}]",
            )

        self.console.print()
        self.console.print(table)

        # Show summary insights
        successful = [r for r in records if r.success]
        if successful:
            best = min(successful, key=lambda r: r.final_objective or float('inf'))
            self.console.print()
            self.console.print(f"[dim]Best successful: Graph #{best.graph_id} → {best.final_objective:.4e}[/dim]")

            # Show common patterns among successful graphs
            patterns = {}
            for r in successful:
                p = r.pattern
                patterns[p] = patterns.get(p, 0) + 1
            if patterns:
                most_common = max(patterns.items(), key=lambda x: x[1])
                self.console.print(f"[dim]Most common pattern: {most_common[0]} ({most_common[1]}/{len(successful)} successful)[/dim]")

        self.console.print()

    def handle_graph_analyze(self, graph_id: int, focus: str = 'overall'):
        """AI-powered analysis of optimization graph (costs money)."""
        record = self.foundry.load_graph_record(graph_id)
        detail = self.foundry.load_graph_detail(graph_id)

        if not record:
            self.console.print(f"\n[red]Graph #{graph_id} not found[/red]\n")
            return

        # Show basic graph metrics
        self.console.print("\n[cyan]Computing metrics...[/cyan]")

        # Compute basic metrics from graph
        objectives = []
        if detail:
            for node_id in sorted(detail.nodes.keys()):
                node_detail = detail.nodes[node_id]
                for point in node_detail.convergence_history:
                    objectives.append(point.objective)

        if not objectives:
            self.console.print(f"\n[yellow]No iteration data available for graph #{graph_id}[/yellow]\n")
            return

        # Basic convergence metrics
        n_iters = len(objectives)
        improvement = objectives[0] - objectives[-1] if n_iters >= 2 else 0.0
        is_stalled = False
        if n_iters >= 5:
            last_5 = objectives[-5:]
            is_stalled = abs(max(last_5) - min(last_5)) < 1e-6

        obj_str = f"{record.final_objective:.6e}" if record.final_objective is not None else "N/A"
        self.console.print(f"[dim]Iterations: {n_iters}, Improvement: {improvement:.6e}, Stalled: {is_stalled}[/dim]")
        self.console.print(f"[dim]Final objective: {obj_str}[/dim]")

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
            "graph": {
                "n_nodes": len(record.nodes),
                "pattern": record.pattern,
                "total_evaluations": record.total_evaluations,
                "total_wall_time": record.total_wall_time,
                "success": record.success,
            }
        }

        try:
            # Note: Full AI analysis to be implemented
            self.console.print(f"\n[bold cyan]Graph #{graph_id} Summary:[/bold cyan]")
            self.console.print(f"  Problem: {record.problem_id}")
            self.console.print(f"  Pattern: {record.pattern}")
            self.console.print(f"  Nodes: {len(record.nodes)}")
            self.console.print(f"  Best objective: {obj_str}")
            self.console.print(f"  Evaluations: {record.total_evaluations}")
            self.console.print(f"  Status: {'✓ Success' if record.success else '✗ Failed'}")
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

    # =========================================================================
    # Problem Commands (v0.4.1)
    # =========================================================================

    def handle_problems(self):
        """Display all registered optimization problems in table format."""
        problems = self.foundry.storage.list_problems()

        if not problems:
            self.console.print("\n[dim]No optimization problems registered yet[/dim]\n")
            return

        table = Table(title="Optimization Problems")
        table.add_column("ID", style="cyan", justify="right")
        table.add_column("Name")
        table.add_column("Type")
        table.add_column("Vars", justify="right")
        table.add_column("Constraints", justify="right")
        table.add_column("Parent", justify="right")
        table.add_column("Derivation")
        table.add_column("Graphs", justify="right")

        for p in problems:
            parent_id = p.get("parent_problem_id")
            parent = f"#{parent_id}" if parent_id else "-"
            deriv = p.get("derivation_type") or "-"
            n_graphs = len(p.get("graphs_using", []))

            # Highlight derived problems
            style = "dim" if parent != "-" else ""

            table.add_row(
                str(p.get("problem_id", "?")),
                p.get("name", p.get("problem_id", "?")),
                p.get("problem_type", "?"),
                str(p.get("n_variables", "?")),
                str(p.get("n_constraints", "?")),
                parent,
                deriv,
                str(n_graphs),
                style=style
            )

        self.console.print()
        self.console.print(table)
        self.console.print()

    def handle_problem_show(self, problem_id: int):
        """Show detailed problem information."""
        # Load full problem from storage
        problem = self.foundry.storage.load_problem(problem_id)

        if problem is None:
            self.console.print(f"\n[red]Problem #{problem_id} not found[/red]\n")
            return

        # Get index entry for graph usage info
        problems = self.foundry.storage.list_problems()
        index_entry = next((p for p in problems if p.get("problem_id") == problem_id), {})
        graphs_using = index_entry.get("graphs_using", [])
        children = index_entry.get("children", [])

        # Format bounds summary
        if hasattr(problem, 'bounds') and problem.bounds:
            if len(problem.bounds) <= 3:
                bounds_str = str(problem.bounds)
            else:
                bounds_str = f"[{problem.bounds[0]}, {problem.bounds[1]}, ..., {problem.bounds[-1]}]"
        else:
            bounds_str = "N/A"

        # Build info panel
        lineage_str = "Root problem (not derived)"
        if problem.parent_problem_id:
            lineage_str = f"Derived from: #{problem.parent_problem_id} ({problem.derivation_type})"

        children_str = ', '.join(f'#{c}' for c in children) if children else 'None'

        info = f"""[bold cyan]Problem #{problem.problem_id}: {problem.name}[/bold cyan]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[cyan]Type:[/cyan]         {problem.problem_type}
[cyan]Variables:[/cyan]    {problem.n_variables}
[cyan]Constraints:[/cyan]  {problem.n_constraints}
[cyan]Domain:[/cyan]       {problem.domain_hint or 'general'}

[bold]Bounds Summary:[/bold]
  {bounds_str}

[bold]Lineage:[/bold]
  {lineage_str}
  Version: {problem.version}

[bold]Children:[/bold]     {children_str}
[bold]Used by:[/bold]      {', '.join(f'Graph #{g}' for g in graphs_using) if graphs_using else 'Not used yet'}"""

        # Add NLP-specific info
        if hasattr(problem, 'objective_evaluator_id'):
            info += f"""

[bold]NLP Details:[/bold]
  Objective: {problem.objective_sense} {problem.objective_evaluator_id}
  Inequality constraints: {len(getattr(problem, 'inequality_constraints', []))}
  Equality constraints: {len(getattr(problem, 'equality_constraints', []))}"""

        self.console.print()
        self.console.print(Panel(info, border_style="cyan", padding=(1, 2)))
        self.console.print()

    def handle_problem_lineage(self, problem_id: int):
        """Display problem derivation lineage as tree."""
        lineage = self.foundry.storage.get_problem_lineage(problem_id)

        if not lineage:
            self.console.print(f"\n[red]Problem #{problem_id} not found[/red]\n")
            return

        children = self.foundry.storage.get_problem_children(problem_id)

        # Build ASCII tree
        tree_lines = []
        for i, entry in enumerate(lineage):
            prefix = "  " * i
            if i == 0:
                marker = "[bold green]●[/bold green]"  # Root
            elif entry.get("problem_id") == problem_id:
                marker = "[bold cyan]★[/bold cyan]"  # Current
            else:
                marker = "[dim]○[/dim]"  # Ancestor

            deriv_type = entry.get("derivation_type") or "root"
            pid = entry.get("problem_id", "?")
            name = entry.get("name", "")
            name_str = f" ({name})" if name else ""
            tree_lines.append(f"{prefix}{marker} #{pid}{name_str} - {deriv_type}")

        # Add children
        if children:
            current_depth = len(lineage)
            prefix = "  " * current_depth
            tree_lines.append(f"{prefix}[dim]Children:[/dim]")
            for child in children:
                tree_lines.append(f"{prefix}  [dim]└─ #{child}[/dim]")

        tree_str = "\n".join(tree_lines)

        info = f"""[bold cyan]Problem Lineage: #{problem_id}[/bold cyan]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{tree_str}

[dim]Legend: ● root  ★ current  ○ ancestor[/dim]"""

        self.console.print()
        self.console.print(Panel(info, border_style="cyan", padding=(1, 2)))
        self.console.print()
