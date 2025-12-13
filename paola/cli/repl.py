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
from ..tools.optimizer_tools import run_scipy_optimization
from ..tools.evaluator_tools import create_benchmark_problem
from ..tools.observation_tools import analyze_convergence
from ..tools.run_tools import start_optimization_run, finalize_optimization_run, get_active_runs, set_foundry
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

    def __init__(self, llm_model: str = "qwen-flash", storage: StorageBackend = None):
        """
        Initialize REPL.

        Args:
            llm_model: LLM model to use (default: qwen-flash for cost)
            storage: Storage backend (default: FileStorage)
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
        self.agent = None
        self.conversation_history = []

        # Token tracking
        self.token_tracker = TokenTracker()
        self.token_callback = LangChainTokenCallback(self.token_tracker, verbose=True)

        # Callback manager with display callback
        self.callback_manager = CallbackManager()
        self.callback_manager.register(CLICallback())  # Display events

        # Tools - agent explicitly manages runs
        self.tools = [
            # Problem formulation
            create_benchmark_problem,

            # Run management
            start_optimization_run,
            finalize_optimization_run,
            get_active_runs,

            # Optimization
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
                "[bold cyan]PAOLA[/bold cyan] - Platform for Agentic Optimization with Learning and Analysis\n"
                "[dim]The optimization platform that learns from every run[/dim]\n\n"
                "Version 0.1.0\n\n"
                "Type your optimization goals in natural language.\n"
                "Type '/help' for commands, '/exit' to quit."
            ),
            border_style="cyan",
            padding=(1, 2)
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

        self.agent = build_optimization_agent(
            tools=self.tools,
            llm_model=self.llm_model,
            callback_manager=self.callback_manager,
            temperature=0.0
        )

        self.console.print("[dim]✓ Agent ready![/dim]\n")

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
        elif cmd == '/runs':
            self.command_handler.handle_runs()
        elif cmd == '/show':
            if len(cmd_parts) < 2:
                self.console.print("[red]Usage: /show <run_id>[/red]")
            else:
                try:
                    run_id = int(cmd_parts[1])
                    self.command_handler.handle_show(run_id)
                except ValueError:
                    self.console.print("[red]Run ID must be a number[/red]")
        elif cmd == '/plot':
            if len(cmd_parts) < 2:
                self.console.print("[red]Usage: /plot <run_id> OR /plot compare <run1> <run2> ...[/red]")
            elif cmd_parts[1].lower() == 'compare':
                # /plot compare <run1> <run2> ...
                if len(cmd_parts) < 4:
                    self.console.print("[red]Usage: /plot compare <run1> <run2> [run3...][/red]")
                else:
                    try:
                        run_ids = [int(id) for id in cmd_parts[2:]]
                        self.command_handler.handle_plot_compare(run_ids)
                    except ValueError:
                        self.console.print("[red]Run IDs must be numbers[/red]")
            else:
                # /plot <run_id>
                try:
                    run_id = int(cmd_parts[1])
                    self.command_handler.handle_plot(run_id)
                except ValueError:
                    self.console.print("[red]Run ID must be a number[/red]")
        elif cmd == '/compare':
            if len(cmd_parts) < 3:  # Need at least 2 run IDs
                self.console.print("[red]Usage: /compare <run1> <run2> [run3...][/red]")
            else:
                try:
                    run_ids = [int(id) for id in cmd_parts[1:]]
                    self.command_handler.handle_compare(run_ids)
                except ValueError:
                    self.console.print("[red]Run IDs must be numbers[/red]")
        elif cmd == '/best':
            self.command_handler.handle_best()
        elif cmd == '/analyze':
            if len(cmd_parts) < 2:
                self.console.print("[red]Usage: /analyze <run_id> [focus][/red]")
                self.console.print("[dim]Focus options: convergence, efficiency, algorithm, overall (default)[/dim]")
            else:
                try:
                    run_id = int(cmd_parts[1])
                    focus = cmd_parts[2] if len(cmd_parts) > 2 else "overall"
                    self.command_handler.handle_analyze(run_id, focus)
                except ValueError:
                    self.console.print("[red]Run ID must be a number[/red]")
        elif cmd == '/knowledge':
            if len(cmd_parts) == 1:
                self.command_handler.handle_knowledge_list()
            elif cmd_parts[1] == 'show' and len(cmd_parts) > 2:
                self.command_handler.handle_knowledge_show(cmd_parts[2])
            else:
                self.console.print("[red]Usage: /knowledge OR /knowledge show <id>[/red]")
        elif cmd == '/tokens':
            self._show_token_stats()
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

[bold]Inspection Commands:[/bold]
  /runs                      - List all optimization runs
  /show <id>                 - Show detailed results for run (with metrics)
  /analyze <id> [focus]      - AI-powered strategic analysis (costs ~$0.02-0.05)
                               Focus: convergence, efficiency, algorithm, overall (default)
  /plot <id>                 - Plot convergence for run
  /plot compare <id1> <id2>  - Overlay convergence curves for multiple runs
  /compare <id1> <id2>       - Side-by-side comparison of runs
  /best                      - Show best solution across all runs
  /knowledge                 - List knowledge base (skeleton - not yet implemented)
  /knowledge show <id>       - Show detailed insight (skeleton)

[bold]Session Commands:[/bold]
  /help           - Show this help message
  /exit           - Exit the CLI
  /clear          - Clear conversation history
  /model          - Show current LLM model
  /models         - Select a different LLM model
  /tokens         - Show token usage and cost statistics

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
            ("qwen-flash", "Fast, cheap, good for testing"),
            ("qwen-plus", "Balanced performance and cost"),
            ("qwen-max", "Most capable, higher cost"),
            ("qwen-turbo", "Fast with good quality"),
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
