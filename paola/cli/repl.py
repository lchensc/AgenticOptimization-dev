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
from ..tools.optimizer_tools import run_scipy_optimization
from ..tools.evaluator_tools import create_nlp_problem
from ..tools.observation_tools import analyze_convergence
from ..tools.session_tools import start_session, finalize_session, get_active_sessions, get_session_info, set_foundry
from ..tools.graph_tools import start_graph, get_graph_state, finalize_graph, set_foundry as set_graph_foundry
from ..tools.analysis import analyze_convergence as analyze_convergence_new, analyze_efficiency, get_all_metrics, analyze_run_with_ai
from ..tools.knowledge_tools import store_optimization_insight, retrieve_optimization_knowledge, list_all_knowledge
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
        set_graph_foundry(self.foundry)

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

        # Import LLM-driven optimization tools (Paola Principle)
        from ..tools.optimization_tools import (
            run_optimization,
            get_problem_info,
            list_available_optimizers,
            get_optimizer_options,
        )

        # Import expert configuration tools
        from ..tools.config_tools import (
            config_scipy,
            config_ipopt,
            config_nlopt,
            config_optuna,
            explain_config_option,
        )

        # Tools - agent explicitly manages graphs (v0.3.0)
        self.tools = [
            # Problem formulation
            create_nlp_problem,  # NLP problems from registered evaluators

            # Graph management (v0.3.0 - recommended)
            start_graph,  # Start new optimization graph
            get_graph_state,  # Get graph state for decision making
            finalize_graph,  # Finalize and persist graph

            # Session management (v0.2.0 - legacy)
            start_session,
            finalize_session,
            get_active_sessions,
            get_session_info,

            # LLM-driven optimization (Paola Principle - recommended)
            run_optimization,  # Execute optimization with LLM-specified config
            get_problem_info,  # Get problem characteristics for LLM reasoning
            list_available_optimizers,  # List available backends
            get_optimizer_options,  # Get optimizer configuration options

            # Expert configuration (escape hatch)
            config_scipy,
            config_ipopt,
            config_nlopt,
            config_optuna,
            explain_config_option,

            # Legacy optimization (deprecated - use run_optimization)
            run_scipy_optimization,

            # Analysis (deterministic - fast & free)
            analyze_convergence_new,
            analyze_efficiency,
            get_all_metrics,

            # Analysis (AI-powered - strategic, costs money)
            analyze_run_with_ai,

            # Knowledge (skeleton - not yet implemented)
            store_optimization_insight,
            retrieve_optimization_knowledge,
            list_all_knowledge,

            # Evaluator registration
            read_file,
            write_file,
            execute_python,
            foundry_store_evaluator,
            foundry_list_evaluators,
            foundry_get_evaluator,
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
                user_input = self.session.prompt('paola> ').strip()

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
                "[bold cyan]PAOLA[/bold cyan] v0.3.0 - Agentic Optimization Platform\n\n"
                "[dim]AI-powered optimization with conversational interface[/dim]\n\n"
                "Commands: /help | /graphs | /evals | /exit\n"
                "Or just type your goal in natural language"
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
                        self.console.print(f"\n{response}\n", style="white")

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
            else:
                self.console.print("[red]Usage: /graph [show|plot|compare|best] <id>[/red]")
        elif cmd == '/sessions':
            self.command_handler.handle_sessions()
        elif cmd == '/show':
            # Try graph first, then session
            if len(cmd_parts) < 2:
                self.console.print("[red]Usage: /show <id> (shows graph by default, use /session show <id> for sessions)[/red]")
            else:
                try:
                    item_id = int(cmd_parts[1])
                    # Try graph first
                    graph = self.foundry.load_graph(item_id)
                    if graph:
                        self.command_handler.handle_graph_show(item_id)
                    else:
                        # Fall back to session
                        self.command_handler.handle_show(item_id)
                except ValueError:
                    self.console.print("[red]ID must be a number[/red]")
        elif cmd == '/plot':
            if len(cmd_parts) < 2:
                self.console.print("[red]Usage: /plot <id> OR /plot compare <id1> <id2> ...[/red]")
            elif cmd_parts[1].lower() == 'compare':
                if len(cmd_parts) < 4:
                    self.console.print("[red]Usage: /plot compare <id1> <id2> [id3...][/red]")
                else:
                    try:
                        ids = [int(id) for id in cmd_parts[2:]]
                        # Try graph first
                        if self.foundry.load_graph(ids[0]):
                            # TODO: Add handle_graph_plot_compare
                            self.console.print("[yellow]Graph comparison plot not yet implemented, using session compare[/yellow]")
                            self.command_handler.handle_plot_compare(ids)
                        else:
                            self.command_handler.handle_plot_compare(ids)
                    except ValueError:
                        self.console.print("[red]IDs must be numbers[/red]")
            else:
                try:
                    item_id = int(cmd_parts[1])
                    # Try graph first
                    graph = self.foundry.load_graph(item_id)
                    if graph:
                        self.command_handler.handle_graph_plot(item_id)
                    else:
                        self.command_handler.handle_plot(item_id)
                except ValueError:
                    self.console.print("[red]ID must be a number[/red]")
        elif cmd == '/compare':
            if len(cmd_parts) < 3:
                self.console.print("[red]Usage: /compare <id1> <id2> [id3...][/red]")
            else:
                try:
                    ids = [int(id) for id in cmd_parts[1:]]
                    # Try graph first
                    if self.foundry.load_graph(ids[0]):
                        self.command_handler.handle_graph_compare(ids)
                    else:
                        self.command_handler.handle_compare(ids)
                except ValueError:
                    self.console.print("[red]IDs must be numbers[/red]")
        elif cmd == '/best':
            # Try graph first
            graphs = self.foundry.load_all_graphs()
            if graphs:
                self.command_handler.handle_graph_best()
            else:
                self.command_handler.handle_best()
        elif cmd == '/analyze':
            if len(cmd_parts) < 2:
                self.console.print("[red]Usage: /analyze <session_id> [focus][/red]")
                self.console.print("[dim]Focus options: convergence, efficiency, algorithm, overall (default)[/dim]")
            else:
                try:
                    session_id = int(cmd_parts[1])
                    focus = cmd_parts[2] if len(cmd_parts) > 2 else "overall"
                    self.command_handler.handle_analyze(session_id, focus)
                except ValueError:
                    self.console.print("[red]Session ID must be a number[/red]")
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
                task = f"Register all evaluator functions from {cmd_parts[1]} as standalone evaluators in .paola_data/evaluators/. Each evaluator file must work independently (include all dependencies), provide a standard evaluate(x) interface, and be testable by running the file directly."
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

[bold]Graph Commands (v0.3.0):[/bold]
  /graphs                    - List all optimization graphs
  /graph show <id>           - Show detailed graph information
  /graph plot <id>           - Plot convergence for graph
  /graph compare <id1> <id2> - Side-by-side comparison of graphs
  /graph best                - Show best solution across all graphs

[bold]Inspection Commands:[/bold]
  /show <id>                 - Show detailed results (graph or session)
  /plot <id>                 - Plot convergence
  /plot compare <id1> <id2>  - Overlay convergence curves
  /compare <id1> <id2>       - Side-by-side comparison
  /best                      - Show best solution
  /analyze <id> [focus]      - AI-powered strategic analysis (costs ~$0.02-0.05)
                               Focus: convergence, efficiency, algorithm, overall (default)

[bold]Legacy Session Commands (v0.2.0):[/bold]
  /sessions                  - List all optimization sessions
  /knowledge                 - List knowledge base (skeleton - not yet implemented)
  /knowledge show <id>       - Show detailed insight (skeleton)

[bold]Session Commands:[/bold]
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
            ("qwen-max", "Qwen Max - most capable ($20/$60)"),
            ("claude-3-haiku-20240307", "Claude 3 Haiku - cheapest ($0.25/$1.25)"),
            ("claude-3-5-haiku-latest", "Claude 3.5 Haiku - fast ($0.80/$4.00)"),
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
