"""Main REPL for PAOLA CLI."""

from functools import partial
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from langchain_core.messages import HumanMessage

from ..agent.react_agent import build_optimization_agent
from ..agent.conversational_agent import build_conversational_agent
from ..tools.evaluator_tools import (
    create_nlp_problem,
    derive_problem,
    list_problems,
    get_problem_lineage,
)
from ..tools.graph_tools import start_graph, get_graph_state, finalize_graph, query_past_graphs, get_past_graph, set_foundry
from ..tools.analysis import analyze_convergence as analyze_convergence_new, analyze_efficiency, get_all_metrics, analyze_run_with_ai
from ..callbacks import CallbackManager
from ..foundry import FileStorage, StorageBackend, OptimizationFoundry
from ..llm import TokenTracker, LangChainTokenCallback, format_session_stats
from .callback import CLICallback
from .commands import CommandHandler


class AgenticOptREPL:
    """
    Main REPL for interactive optimization.

    Provides a Claude Code-style conversational interface for optimization.
    """

    def __init__(
        self,
        llm_model: str = "qwen-flash",
        storage: StorageBackend = None,
        agent_type: str = "conversational"
    ):
        """
        Initialize REPL.

        Args:
            llm_model: LLM model to use (default: qwen-flash for cost)
            storage: Storage backend (default: FileStorage)
            agent_type: Agent type - "conversational" (default, like Claude Code) or "react" (autonomous)
        """
        self.console = Console()
        self.session = PromptSession(
            history=FileHistory('.paola_history'),
            auto_suggest=AutoSuggestFromHistory(),
            mouse_support=False,  # Allow terminal native scrollback
        )

        # Storage layer (persists independently)
        storage_backend = storage or FileStorage()

        # Initialize OptimizationFoundry (data foundation)
        self.foundry = OptimizationFoundry(storage=storage_backend)

        # Set global foundry for tools
        set_foundry(self.foundry)

        # Command handler (reads from foundry)
        self.command_handler = CommandHandler(self.foundry, self.console)

        # Agent state
        self.llm_model = llm_model
        self.agent_type = agent_type
        self.agent = None
        self.conversation_history = []

        # Token tracking
        self.token_tracker = TokenTracker()
        self.token_callback = LangChainTokenCallback(self.token_tracker, verbose=True)

        # Developer mode flag (default: on for verbose debugging)
        self.developer_mode = True

        # Callback manager with display callback
        self.callback_manager = CallbackManager()
        self.cli_callback = CLICallback(developer_mode=self.developer_mode)
        self.callback_manager.register(self.cli_callback)  # Display events

        # Import registration tools
        from ..tools.registration_tools import (
            read_file,
            write_file,
            execute_python,
            foundry_store_evaluator,
            foundry_list_evaluators,
            foundry_get_evaluator
        )

        # Import skill tools (Paola Skills infrastructure)
        from ..skills import get_skill_tools

        # Import LLM-driven optimization tools (Paola Principle)
        from ..tools.optimization_tools import (
            run_optimization,
            get_problem_info,
            list_available_optimizers,
        )

        # Import expert configuration tools (DISABLED for testing - Skills should cover this)
        # from ..tools.config_tools import (
        #     config_scipy,
        #     config_ipopt,
        #     config_nlopt,
        #     config_optuna,
        #     explain_config_option,
        # )

        # Tools - agent explicitly manages graphs (v0.3.x)
        self.tools = [
            # Problem formulation
            create_nlp_problem,
            derive_problem,
            list_problems,
            get_problem_lineage,

            # Graph management (v0.3.x)
            start_graph,
            get_graph_state,
            finalize_graph,
            query_past_graphs,
            get_past_graph,

            # Optimization execution
            run_optimization,
            get_problem_info,
            list_available_optimizers,

            # Expert configuration (escape hatch) - DISABLED for testing
            # config_scipy,
            # config_ipopt,
            # config_nlopt,
            # config_optuna,
            # explain_config_option,

            # Analysis (deterministic)
            analyze_convergence_new,
            analyze_efficiency,
            get_all_metrics,

            # Analysis (AI-powered)
            analyze_run_with_ai,

            # Evaluator management
            read_file,
            write_file,
            execute_python,
            foundry_store_evaluator,
            foundry_list_evaluators,
            foundry_get_evaluator,

            # Skill tools (optimization expertise)
            *get_skill_tools(),
        ]

        # Running state
        self.running = True

    def run(self):
        """Main REPL loop."""
        self._show_welcome()
        self._initialize_agent()

        while self.running:
            try:
                # Get user input
                user_input = self.session.prompt('> ').strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    if not self._handle_command(user_input):
                        break  # /exit was called
                else:
                    # Send to agent
                    self._process_with_agent(user_input)

            except KeyboardInterrupt:
                self.console.print("\nUse '/exit' or Ctrl+D to quit.", style="dim")
                continue
            except EOFError:
                break
            except Exception as e:
                self.console.print(f"\n[bold red]Error: {e}[/bold red]")
                import traceback
                traceback.print_exc()

        self._show_goodbye()

    def _show_welcome(self):
        """Display welcome message."""
        welcome = Panel(
            Text.from_markup(
                "[bold cyan]Paola[/bold cyan] [dim]v0.4.2[/dim]\n"
                "[dim]Package for agentic optimization with learning and analysis[/dim]\n\n"
                "Commands: /help | /graphs | /problems | /evals | /skills | /exit\n"
                "Or just tell me what you want to optimize"
            ),
            border_style="cyan",
            padding=(1, 2),
            title="[bold cyan]Welcome[/bold cyan]",
            title_align="left"
        )
        self.console.print()
        self.console.print(welcome)
        self.console.print()

    def _show_goodbye(self):
        """Display goodbye message."""
        self.console.print()
        self.console.print("[cyan]Goodbye! Happy optimizing! 🚀[/cyan]")
        self.console.print()

    def _initialize_agent(self):
        """Initialize the optimization agent."""
        self.console.print("[dim]Initializing agent...[/dim]")

        if self.agent_type == "conversational":
            self.agent = build_conversational_agent(
                tools=self.tools,
                llm_model=self.llm_model,
                callback_manager=self.callback_manager,
                temperature=0.0
            )
        else:  # "react"
            self.agent = build_optimization_agent(
                tools=self.tools,
                llm_model=self.llm_model,
                callback_manager=self.callback_manager,
                temperature=0.0
            )

        self.console.print(f"[dim]✓ Agent ready! (type: {self.agent_type})[/dim]\n")

    def _process_with_agent(self, user_input: str):
        """
        Process user input through agent.

        Args:
            user_input: User's natural language command
        """
        # Add to conversation
        self.conversation_history.append(
            HumanMessage(content=user_input)
        )

        # Build state
        state = {
            "messages": self.conversation_history,
            "context": {
                "goal": user_input,
                "iteration": len(self.conversation_history)
            },
            "done": False,
            "iteration": 0,
            "callback_manager": self.callback_manager
        }

        # Invoke agent (streaming via callback)
        try:
            # Pass token callback to LangGraph
            config = {"callbacks": [self.token_callback]}
            final_state = self.agent.invoke(state, config=config)

            # Update conversation history
            self.conversation_history = final_state["messages"]

            # Extract and display agent's final response
            if final_state["messages"]:
                last_message = final_state["messages"][-1]
                if hasattr(last_message, 'content') and last_message.content:
                    response = last_message.content.strip()
                    if response and not response.upper().startswith("DONE"):
                        self.console.print()
                        self.console.print("[bold cyan]Paola:[/bold cyan]", style="white")
                        self.console.print(f"{response}\n", style="white")

        except Exception as e:
            self.console.print(f"\n[bold red]Agent error: {e}[/bold red]\n")
            # Remove failed message from history
            if self.conversation_history:
                self.conversation_history.pop()

    def _handle_command(self, command: str) -> bool:
        """
        Handle slash commands.

        Args:
            command: Command string starting with /

        Returns:
            True to continue REPL, False to exit
        """
        cmd_parts = command.split()
        cmd = cmd_parts[0].lower()

        if cmd == '/help':
            self._show_help()
        elif cmd == '/exit':
            return False  # Signal to exit
        elif cmd == '/clear':
            self._clear_conversation()
        elif cmd == '/model':
            self._show_model_info()
        elif cmd == '/models':
            self._select_model()
        elif cmd == '/graphs':
            self.command_handler.handle_graphs()
        elif cmd == '/graph':
            # Alias for /graphs
            if len(cmd_parts) == 1:
                self.command_handler.handle_graphs()
            elif cmd_parts[1].lower() == 'show' and len(cmd_parts) > 2:
                try:
                    graph_id = int(cmd_parts[2])
                    self.command_handler.handle_graph_show(graph_id)
                except ValueError:
                    self.console.print("[red]Graph ID must be a number[/red]")
            elif cmd_parts[1].lower() == 'best':
                self.command_handler.handle_graph_best()
            elif cmd_parts[1].lower() == 'compare' and len(cmd_parts) > 3:
                try:
                    graph_ids = [int(id) for id in cmd_parts[2:]]
                    self.command_handler.handle_graph_compare(graph_ids)
                except ValueError:
                    self.console.print("[red]Graph IDs must be numbers[/red]")
            elif cmd_parts[1].lower() == 'plot' and len(cmd_parts) > 2:
                try:
                    graph_id = int(cmd_parts[2])
                    self.command_handler.handle_graph_plot(graph_id)
                except ValueError:
                    self.console.print("[red]Graph ID must be a number[/red]")
            elif cmd_parts[1].lower() == 'query':
                # Parse query options: /graph query [problem=pattern] [dims=N] [success=true/false] [limit=N]
                problem_pattern = None
                n_dimensions = None
                success = None
                limit = 5

                for part in cmd_parts[2:]:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        key = key.lower()
                        if key == 'problem':
                            problem_pattern = value
                        elif key == 'dims':
                            try:
                                n_dimensions = int(value)
                            except ValueError:
                                self.console.print(f"[red]dims must be a number, got: {value}[/red]")
                                return True
                        elif key == 'success':
                            success = value.lower() in ('true', '1', 'yes')
                        elif key == 'limit':
                            try:
                                limit = int(value)
                            except ValueError:
                                self.console.print(f"[red]limit must be a number, got: {value}[/red]")
                                return True

                self.command_handler.handle_graph_query(
                    problem_pattern=problem_pattern,
                    n_dimensions=n_dimensions,
                    success=success,
                    limit=limit,
                )
            else:
                self.console.print("[red]Usage: /graph [show|plot|compare|best|query] <id|options>[/red]")
        elif cmd == '/show':
            if len(cmd_parts) < 2:
                self.console.print("[red]Usage: /show <graph_id>[/red]")
            else:
                try:
                    graph_id = int(cmd_parts[1])
                    self.command_handler.handle_graph_show(graph_id)
                except ValueError:
                    self.console.print("[red]Graph ID must be a number[/red]")
        elif cmd == '/plot':
            if len(cmd_parts) < 2:
                self.console.print("[red]Usage: /plot <graph_id>[/red]")
            else:
                try:
                    graph_id = int(cmd_parts[1])
                    self.command_handler.handle_graph_plot(graph_id)
                except ValueError:
                    self.console.print("[red]Graph ID must be a number[/red]")
        elif cmd == '/compare':
            if len(cmd_parts) < 3:
                self.console.print("[red]Usage: /compare <graph_id1> <graph_id2> [graph_id3...][/red]")
            else:
                try:
                    graph_ids = [int(id) for id in cmd_parts[1:]]
                    self.command_handler.handle_graph_compare(graph_ids)
                except ValueError:
                    self.console.print("[red]Graph IDs must be numbers[/red]")
        elif cmd == '/best':
            self.command_handler.handle_graph_best()
        elif cmd == '/analyze':
            if len(cmd_parts) < 2:
                self.console.print("[red]Usage: /analyze <graph_id> [focus][/red]")
                self.console.print("[dim]Focus options: convergence, efficiency, algorithm, overall (default)[/dim]")
            else:
                try:
                    graph_id = int(cmd_parts[1])
                    focus = cmd_parts[2] if len(cmd_parts) > 2 else "overall"
                    self.command_handler.handle_graph_analyze(graph_id, focus)
                except ValueError:
                    self.console.print("[red]Graph ID must be a number[/red]")
        elif cmd == '/knowledge':
            if len(cmd_parts) == 1:
                self.command_handler.handle_knowledge_list()
            elif cmd_parts[1] == 'show' and len(cmd_parts) > 2:
                self.command_handler.handle_knowledge_show(cmd_parts[2])
            else:
                self.console.print("[red]Usage: /knowledge OR /knowledge show <id>[/red]")
        elif cmd == '/tokens':
            self._show_token_stats()
        elif cmd == '/register':
            if len(cmd_parts) < 2:
                self.console.print("[red]Usage: /register <file.py>[/red]")
            else:
                self.command_handler.handle_register(cmd_parts[1])
        elif cmd == '/register_eval':
            if len(cmd_parts) < 2:
                self.console.print("[red]Usage: /register_eval <file.py>[/red]")
            else:
                # Agent-driven registration - send as natural language task
                task = f"Register all evaluator functions from {cmd_parts[1]} as standalone evaluators in .paola_foundry/evaluators/. Each evaluator file must work independently (include all dependencies), provide a standard evaluate(x) interface, and be testable by running the file directly."
                self._process_with_agent(task)
        elif cmd == '/evals':
            self.command_handler.handle_evaluators()
        elif cmd == '/eval':
            if len(cmd_parts) < 2:
                self.console.print("[red]Usage: /eval <evaluator_id>[/red]")
            else:
                self.command_handler.handle_evaluator_show(cmd_parts[1])
        elif cmd == '/mode':
            self._toggle_developer_mode()
        elif cmd == '/skills':
            self._show_skills()
        elif cmd == '/skill':
            if len(cmd_parts) < 2:
                self._show_skills()
            else:
                self._show_skill_detail(cmd_parts[1])
        elif cmd == '/problems':
            self.command_handler.handle_problems()
        elif cmd == '/problem':
            if len(cmd_parts) < 2:
                self.command_handler.handle_problems()
            elif cmd_parts[1].lower() == 'show' and len(cmd_parts) > 2:
                try:
                    problem_id = int(cmd_parts[2])
                    self.command_handler.handle_problem_show(problem_id)
                except ValueError:
                    self.console.print(f"[red]Invalid problem ID: {cmd_parts[2]}. Use numeric ID (e.g., 1, 2, 3).[/red]")
            elif cmd_parts[1].lower() == 'lineage' and len(cmd_parts) > 2:
                try:
                    problem_id = int(cmd_parts[2])
                    self.command_handler.handle_problem_lineage(problem_id)
                except ValueError:
                    self.console.print(f"[red]Invalid problem ID: {cmd_parts[2]}. Use numeric ID (e.g., 1, 2, 3).[/red]")
            else:
                # Treat as problem ID for show
                try:
                    problem_id = int(cmd_parts[1])
                    self.command_handler.handle_problem_show(problem_id)
                except ValueError:
                    self.console.print(f"[red]Invalid problem ID: {cmd_parts[1]}. Use numeric ID (e.g., 1, 2, 3).[/red]")
        else:
            self.console.print(f"Unknown command: {cmd}. Type /help for available commands.", style="yellow")

        return True  # Continue REPL

    def _show_help(self):
        """Display help information."""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

[bold]Natural Language:[/bold]
  Just type your optimization goal, e.g.:
  - "optimize a 10D Rosenbrock problem"
  - "compare SLSQP and BFGS on this problem"
  - "analyze the convergence behavior"

[bold]Evaluator Registration:[/bold]
  /register <file.py>        - Register an evaluator function (manual/interactive)
  /register_eval <file.py>   - Register evaluators using AI agent (recommended)
  /evals                     - List all registered evaluators
  /eval <id>                 - Show detailed evaluator configuration

[bold]Skills (Optimization Expertise):[/bold]
  /skills                    - List all available Paola skills
  /skill <name>              - Show detailed skill information (e.g., /skill ipopt)

[bold]Graph Commands:[/bold]
  /graphs                    - List all optimization graphs
  /graph show <id>           - Show detailed graph information
  /graph plot <id>           - Plot convergence for graph
  /graph compare <id1> <id2> - Side-by-side comparison of graphs
  /graph best                - Show best solution across all graphs
  /graph query [options]     - Query past graphs for cross-graph learning
                               Options: problem=pattern dims=N success=true/false limit=N
                               Example: /graph query problem=ackley* dims=30 success=true

[bold]Shortcuts:[/bold]
  /show <id>                 - Show detailed graph results
  /plot <id>                 - Plot convergence
  /compare <id1> <id2>       - Side-by-side comparison
  /best                      - Show best solution across all graphs
  /analyze <id> [focus]      - AI-powered strategic analysis (costs ~$0.02-0.05)
                               Focus: convergence, efficiency, algorithm, overall (default)

[bold]Problems:[/bold]
  /problems                  - List all registered optimization problems
  /problem show <id>         - Show detailed problem information
  /problem lineage <id>      - Show problem derivation lineage

[bold]Knowledge:[/bold]
  /knowledge                 - List knowledge base (skeleton - not yet implemented)
  /knowledge show <id>       - Show detailed insight (skeleton)

[bold]CLI Commands:[/bold]
  /help           - Show this help message
  /exit           - Exit the CLI
  /clear          - Clear conversation history
  /model          - Show current LLM model
  /models         - Select a different LLM model
  /tokens         - Show token usage and cost statistics
  /mode           - Toggle developer mode (on by default, shows tool args/results)

[bold]Exit:[/bold]
  /exit or Ctrl+D
        """
        self.console.print(Panel(help_text, border_style="cyan", padding=(1, 2)))

    def _clear_conversation(self):
        """Clear conversation history (like Claude Code /clear)."""
        self.conversation_history = []
        self.console.print("[dim]✓ Conversation cleared[/dim]\n")

    def _show_model_info(self):
        """Show current model information."""
        self.console.print(f"\n[cyan]Current model:[/cyan] {self.llm_model}\n")

    def _select_model(self):
        """Interactive model selection."""
        available_models = [
            ("qwen-flash", "Qwen Flash - cheapest ($0.05/$0.40)"),
            ("qwen-turbo", "Qwen Turbo - fast ($0.30/$0.60)"),
            ("qwen-plus", "Qwen Plus - balanced ($0.40/$1.20)"),
            ("qwen-max", "Qwen Max - most capable ($1.2/$6)"),
            ("claude-3-5-haiku-latest", "Claude 3.5 Haiku - fast ($0.80/$4)"),
            ("claude-sonnet-4-20250514", "Claude Sonnet 4 - balanced ($3/$15)"),
        ]

        self.console.print("\n[bold cyan]Available LLM Models:[/bold cyan]\n")

        for i, (model, description) in enumerate(available_models, 1):
            current = " [green]← current[/green]" if model == self.llm_model else ""
            self.console.print(f"  {i}. [yellow]{model}[/yellow] - {description}{current}")

        self.console.print("\n[dim]Type model number or name to switch, or press Enter to cancel[/dim]")

        try:
            choice = self.session.prompt("Select model> ").strip()

            if not choice:
                self.console.print("[dim]Cancelled[/dim]\n")
                return

            # Try to parse as number
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(available_models):
                    new_model = available_models[idx][0]
                else:
                    self.console.print("[red]Invalid selection[/red]\n")
                    return
            else:
                # Try as model name
                new_model = choice
                if not any(model == new_model for model, _ in available_models):
                    self.console.print(f"[red]Unknown model: {new_model}[/red]\n")
                    return

            if new_model == self.llm_model:
                self.console.print(f"[dim]Already using {new_model}[/dim]\n")
                return

            # Switch model
            self.console.print(f"\n[cyan]Switching to {new_model}...[/cyan]")
            self.llm_model = new_model
            self._initialize_agent()
            self.console.print(f"[green]✓ Now using {new_model}[/green]\n")

            # Reset conversation when switching models
            self.conversation_history = []

        except KeyboardInterrupt:
            self.console.print("\n[dim]Cancelled[/dim]\n")
        except EOFError:
            self.console.print("\n[dim]Cancelled[/dim]\n")

    def _show_token_stats(self):
        """Display token usage statistics for current session."""
        stats = self.token_tracker.get_session_stats()

        if stats.call_count == 0:
            self.console.print("\n[dim]No LLM calls yet in this session[/dim]\n")
            return

        # Use the formatted output from token_tracker module
        formatted_stats = format_session_stats(stats)
        self.console.print(formatted_stats)

    def _toggle_developer_mode(self):
        """Toggle developer mode for verbose debugging output."""
        self.developer_mode = not self.developer_mode
        self.cli_callback.developer_mode = self.developer_mode

        if self.developer_mode:
            self.console.print("\n[bold cyan]Developer mode: ON[/bold cyan]")
            self.console.print("[dim]  - Tool arguments will be displayed")
            self.console.print("[dim]  - Tool results will be displayed[/dim]\n")
        else:
            self.console.print("\n[bold cyan]Developer mode: OFF[/bold cyan]\n")

    def _show_skills(self):
        """Display available Paola skills."""
        from ..skills import SkillIndex

        try:
            index = SkillIndex()
            skills = index.list_skills()

            if not skills:
                self.console.print("\n[dim]No skills found[/dim]\n")
                return

            self.console.print("\n[bold cyan]Available Paola Skills:[/bold cyan]\n")

            # Group by category
            by_category = {}
            for skill in skills:
                category = skill.get('category', 'unknown')
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(skill)

            for category, cat_skills in by_category.items():
                self.console.print(f"[bold]{category}:[/bold]")
                for skill in cat_skills:
                    self.console.print(f"  • [yellow]{skill['name']}[/yellow] - {skill.get('description', 'N/A')[:60]}")
                self.console.print()

            self.console.print("[dim]Use /skill <name> for details, or let the agent use list_skills/load_skill tools[/dim]\n")

        except Exception as e:
            self.console.print(f"[red]Error listing skills: {e}[/red]\n")

    def _show_skill_detail(self, skill_name: str):
        """Display detailed information about a skill."""
        from ..skills import SkillLoader

        try:
            loader = SkillLoader()
            metadata = loader.load_metadata(skill_name)
            overview = loader.load_overview(skill_name)

            self.console.print(f"\n[bold cyan]{metadata['name'].upper()} Skill[/bold cyan]\n")
            self.console.print(f"[bold]Category:[/bold] {metadata.get('category', 'unknown')}")
            self.console.print(f"[bold]Description:[/bold] {metadata.get('description', 'N/A')}\n")

            when_to_use = metadata.get('when_to_use', [])
            if when_to_use:
                self.console.print("[bold]When to use:[/bold]")
                if isinstance(when_to_use, list):
                    for item in when_to_use:
                        self.console.print(f"  • {item}")
                else:
                    self.console.print(f"  {when_to_use}")
                self.console.print()

            when_not = metadata.get('when_not_to_use', [])
            if when_not:
                self.console.print("[bold]When NOT to use:[/bold]")
                if isinstance(when_not, list):
                    for item in when_not:
                        self.console.print(f"  • {item}")
                else:
                    self.console.print(f"  {when_not}")
                self.console.print()

            # Show overview (truncated for display)
            self.console.print("[bold]Overview:[/bold]")
            overview_lines = overview.split('\n')[:15]  # First 15 lines
            for line in overview_lines:
                self.console.print(f"  {line}")
            if len(overview.split('\n')) > 15:
                self.console.print("  [dim]... (use load_skill tool for full content)[/dim]")
            self.console.print()

            # Show available sections
            self.console.print("[bold]Available sections:[/bold]")
            self.console.print("  • overview (loaded above)")
            self.console.print("  • options - Full option reference")
            self.console.print("  • options.<category> - Specific option category (e.g., options.warm_start)")
            self.console.print("  • paola - Paola integration info")
            self.console.print()
            self.console.print(f"[dim]Use load_skill(\"{skill_name}\", \"<section>\") for detailed content[/dim]\n")

        except Exception as e:
            self.console.print(f"[red]Error loading skill '{skill_name}': {e}[/red]\n")
