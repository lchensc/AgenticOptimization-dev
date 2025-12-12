"""Main REPL for AgenticOpt CLI."""

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from langchain_core.messages import HumanMessage

from ..agent.react_agent import build_aopt_agent
from ..tools.optimizer_tools import run_scipy_optimization
from ..tools.evaluator_tools import create_benchmark_problem
from ..tools.observation_tools import analyze_convergence
from ..callbacks import CallbackManager
from .callback import CLICallback


class AgenticOptREPL:
    """
    Main REPL for interactive optimization.

    Provides a Claude Code-style conversational interface for optimization.
    """

    def __init__(self, llm_model: str = "qwen-flash"):
        """
        Initialize REPL.

        Args:
            llm_model: LLM model to use (default: qwen-flash for cost)
        """
        self.console = Console()
        self.session = PromptSession(
            history=FileHistory('.aopt_history'),
            auto_suggest=AutoSuggestFromHistory(),
        )

        # Agent state
        self.llm_model = llm_model
        self.agent = None
        self.conversation_history = []
        self.callback_manager = CallbackManager()
        self.callback_manager.register(CLICallback())

        # Tools
        self.tools = [
            create_benchmark_problem,
            run_scipy_optimization,
            analyze_convergence
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
                user_input = self.session.prompt('aopt> ').strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                elif user_input.startswith('/'):
                    self._handle_command(user_input)
                else:
                    # Send to agent
                    self._process_with_agent(user_input)

            except KeyboardInterrupt:
                self.console.print("\nUse 'exit' or Ctrl+D to quit.", style="dim")
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
                "[bold cyan]AgenticOpt[/bold cyan] - AI Optimization Assistant\n"
                "Version 0.1.0\n\n"
                "Type your optimization goals in natural language.\n"
                "Type '/help' for commands, 'exit' to quit."
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

        self.agent = build_aopt_agent(
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
            final_state = self.agent.invoke(state)

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

    def _handle_command(self, command: str):
        """
        Handle slash commands.

        Args:
            command: Command string starting with /
        """
        cmd = command.lower().split()[0]

        if cmd == '/help':
            self._show_help()
        elif cmd == '/clear':
            self.console.clear()
        elif cmd == '/reset':
            self._reset_conversation()
        elif cmd == '/model':
            self._show_model_info()
        else:
            self.console.print(f"Unknown command: {cmd}. Type /help for available commands.", style="yellow")

    def _show_help(self):
        """Display help information."""
        help_text = """
[bold cyan]Available Commands:[/bold cyan]

[bold]Natural Language:[/bold]
  Just type your optimization goal, e.g.:
  - "optimize a 10D Rosenbrock problem"
  - "compare SLSQP and BFGS on this problem"
  - "analyze the convergence behavior"

[bold]Slash Commands:[/bold]
  /help    - Show this help message
  /clear   - Clear the screen
  /reset   - Reset conversation history
  /model   - Show current LLM model info

[bold]Exit:[/bold]
  exit, quit, or Ctrl+D
        """
        self.console.print(Panel(help_text, border_style="cyan", padding=(1, 2)))

    def _reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
        self.console.print("[dim]✓ Conversation reset[/dim]\n")

    def _show_model_info(self):
        """Show current model information."""
        self.console.print(f"\n[cyan]Current model:[/cyan] {self.llm_model}\n")
