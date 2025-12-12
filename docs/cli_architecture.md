# AgenticOpt CLI Architecture

## Vision

A Claude Code-style interactive CLI where users collaborate with an AI agent to solve optimization problems through natural conversation.

## Core Concept

```bash
$ aopt
╭──────────────────────────────────────────╮
│  AgenticOpt - AI Optimization Assistant │
╰──────────────────────────────────────────╯

aopt> optimize a 10D Rosenbrock problem

💭 I'll create a 10D Rosenbrock benchmark problem...
🔧 create_benchmark_problem(name="rosenbrock", dimension=10)
✓ Created problem 'rosenbrock_10d' [0.05s]

💭 Now I'll run SLSQP optimization with default settings...
🔧 run_scipy_optimization(problem_id="rosenbrock_10d", algorithm="SLSQP")
✓ Converged to objective 1.2e-8 in 23 iterations [2.3s]

The optimization succeeded! SLSQP found the global minimum at x=(1,1,...,1)
with objective value 1.2e-8. Would you like to try a different algorithm?

aopt> yes, try BFGS
...
```

## Key Principles

1. **Conversational** - Natural language commands, not rigid syntax
2. **Streaming** - Agent thinking displayed in real-time
3. **Stateful** - Maintains context across commands
4. **Extensible** - Easy to add new commands and capabilities
5. **Familiar** - Feels like Claude Code, not a traditional CLI tool

## Architecture Components

### 1. REPL Core (`aopt/cli/repl.py`)

The main interactive loop using `prompt_toolkit`:

```python
class AgenticOptREPL:
    """Main REPL for interactive optimization."""

    def __init__(self):
        self.session = PromptSession(
            history=FileHistory('.aopt_history'),
            auto_suggest=AutoSuggestFromHistory(),
        )
        self.project = None
        self.agent = None
        self.conversation_history = []
        self.running = True

    def run(self):
        """Main REPL loop."""
        self._show_welcome()

        while self.running:
            try:
                # Build dynamic prompt
                prompt_str = self._build_prompt()

                # Get user input
                user_input = self.session.prompt(prompt_str).strip()

                if not user_input:
                    continue

                # Handle input
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                elif user_input in ['exit', 'quit']:
                    break
                else:
                    self._process_with_agent(user_input)

            except KeyboardInterrupt:
                continue
            except EOFError:
                break

        self._show_goodbye()
```

### 2. Project/Workspace (`aopt/cli/project.py`)

Manages optimization projects with persistent state:

```python
class Project:
    """Optimization project/workspace."""

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path
        self.metadata = {}
        self.problems = {}  # problem_id -> problem info
        self.results = {}   # optimization results
        self.history = []   # command history

    def create(self):
        """Initialize project directory structure."""
        self.path.mkdir(exist_ok=True)
        (self.path / "results").mkdir(exist_ok=True)
        (self.path / "problems").mkdir(exist_ok=True)
        (self.path / ".aopt").mkdir(exist_ok=True)
        self._save_metadata()

    def load(self):
        """Load project from disk."""
        self._load_metadata()
        self._load_results()

    def save_result(self, result: dict):
        """Save optimization result."""
        result_id = len(self.results) + 1
        self.results[result_id] = result
        self._save_results()
```

### 3. CLI Callback (`aopt/cli/callback.py`)

Displays agent events in real-time with proper formatting:

```python
class CLICallback(BaseCallback):
    """Real-time streaming display for CLI."""

    def __init__(self):
        self.console = Console()

    def __call__(self, event: AgentEvent):
        """Handle and display event."""

        if event.event_type == EventType.REASONING:
            # Agent thinking - dim style
            reasoning = event.data.get('reasoning', '')
            self.console.print(f"💭 {reasoning}", style="dim")

        elif event.event_type == EventType.TOOL_CALL:
            # Tool invocation - yellow
            tool_name = event.data.get('tool_name')
            args = event.data.get('arguments', {})
            args_str = self._format_args(args)
            self.console.print(f"🔧 {tool_name}({args_str})", style="yellow")

        elif event.event_type == EventType.TOOL_RESULT:
            # Tool completion - green
            tool_name = event.data.get('tool_name')
            duration = event.data.get('duration', 0)
            self.console.print(f"✓ {tool_name} completed ({duration:.2f}s)", style="green")

        elif event.event_type == EventType.TOOL_ERROR:
            # Tool error - red
            tool_name = event.data.get('tool_name')
            error = event.data.get('error')
            self.console.print(f"✗ {tool_name} failed: {error}", style="red")
```

### 4. Slash Commands (`aopt/cli/commands.py`)

Special commands for workspace operations:

```python
class CommandRegistry:
    """Registry of slash commands."""

    def __init__(self, repl):
        self.repl = repl
        self.commands = {
            '/help': self.help_command,
            '/status': self.status_command,
            '/history': self.history_command,
            '/save': self.save_command,
            '/load': self.load_command,
            '/clear': self.clear_command,
            '/project': self.project_command,
        }

    def execute(self, command_str: str):
        """Execute a slash command."""
        parts = command_str.split()
        cmd = parts[0]
        args = parts[1:]

        if cmd not in self.commands:
            print(f"Unknown command: {cmd}. Type /help for available commands.")
            return

        self.commands[cmd](args)

    def history_command(self, args):
        """Show optimization history."""
        if not self.repl.project:
            print("No active project. Create one with: create project <name>")
            return

        table = Table(title="Optimization History")
        table.add_column("#", style="cyan")
        table.add_column("Problem", style="white")
        table.add_column("Algorithm", style="yellow")
        table.add_column("Objective", style="green")
        table.add_column("Iterations", style="blue")

        for i, result in enumerate(self.repl.project.results.values(), 1):
            table.add_row(
                str(i),
                result['problem_id'],
                result['algorithm'],
                f"{result['objective']:.2e}",
                str(result['iterations'])
            )

        console.print(table)
```

### 5. Agent Integration (`aopt/cli/agent_manager.py`)

Manages agent lifecycle and conversation:

```python
class AgentManager:
    """Manages agent for CLI interactions."""

    def __init__(self, project: Optional[Project] = None):
        self.project = project
        self.agent = None
        self.conversation_history = []
        self.callback_manager = CallbackManager()
        self.callback_manager.register(CLICallback())

    def initialize_agent(self, llm_model: str = "qwen-plus"):
        """Initialize or reinitialize agent."""
        tools = get_all_optimizer_tools()

        self.agent = build_aopt_agent(
            tools=tools,
            llm_model=llm_model,
            callback_manager=self.callback_manager,
            temperature=0.0
        )

    def process_message(self, user_input: str) -> str:
        """Process user message through agent."""
        # Add user message to history
        self.conversation_history.append(
            HumanMessage(content=user_input)
        )

        # Build state
        state = {
            "messages": self.conversation_history,
            "context": self._build_context(),
            "done": False,
            "iteration": len(self.conversation_history),
            "callback_manager": self.callback_manager
        }

        # Invoke agent
        final_state = self.agent.invoke(state)

        # Update conversation history with agent's response
        self.conversation_history = final_state["messages"]

        # Extract agent's final response
        last_message = final_state["messages"][-1]
        response = last_message.content if hasattr(last_message, 'content') else ""

        return response

    def _build_context(self) -> dict:
        """Build context from project state."""
        context = {
            "iteration": len(self.conversation_history)
        }

        if self.project:
            context.update({
                "project_name": self.project.name,
                "problems": list(self.project.problems.keys()),
                "results_count": len(self.project.results)
            })

        return context
```

## File Structure

```
aopt/
├── cli/
│   ├── __init__.py           # Public API
│   ├── repl.py               # Main REPL loop
│   ├── project.py            # Project/workspace management
│   ├── callback.py           # CLI streaming callback
│   ├── commands.py           # Slash command registry
│   ├── agent_manager.py      # Agent lifecycle management
│   └── utils.py              # Helper functions (prompt formatting, etc.)
└── ... (existing modules)

bin/
└── aopt                      # Entry point script
```

## Entry Point (`bin/aopt`)

```python
#!/usr/bin/env python
"""AgenticOpt CLI entry point."""

import sys
from aopt.cli import AgenticOptREPL

def main():
    repl = AgenticOptREPL()
    try:
        repl.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

## User Experience Flow

### 1. Starting the CLI

```bash
$ aopt
╭──────────────────────────────────────────╮
│  AgenticOpt - AI Optimization Assistant │
│  Version 0.1.0                           │
╰──────────────────────────────────────────╯

Type 'help' or '/help' for available commands.
Type 'exit' or press Ctrl+D to quit.

aopt>
```

### 2. Creating a Project

```bash
aopt> create project wing_optimization

✓ Created project 'wing_optimization' at ./wing_optimization/
✓ Initialized workspace structure

aopt [wing_optimization]>
```

### 3. Natural Conversation with Agent

```bash
aopt [wing_optimization]> I need to optimize a transonic wing for minimum drag

💭 I'll help you set up a transonic wing optimization problem. Let me start
   by understanding your requirements better. Could you specify:
   - The flight condition (Mach number, altitude)
   - Design variables (shape parameters, twist distribution, etc.)
   - Constraints (structural, geometric, performance)

aopt [wing_optimization]> Mach 0.85, CL = 0.5, minimize drag

💭 I'll formulate a transonic wing optimization problem at Mach 0.85 with
   a lift coefficient constraint CL = 0.5...
```

### 4. Using Slash Commands

```bash
aopt [wing_optimization]> /history

╭─────────────── Optimization History ────────────────╮
│ #  Problem    Algorithm  Objective    Iterations   │
├────┼───────────┼──────────┼───────────┼────────────┤
│ 1  wing_M085  SLSQP      0.0234       45          │
│ 2  wing_M085  BFGS       0.0231       38          │
╰─────────────────────────────────────────────────────╯

aopt [wing_optimization]> /save results/best.json
✓ Saved optimization results to wing_optimization/results/best.json
```

### 5. Follow-up Questions

```bash
aopt [wing_optimization]> why did BFGS perform better?

💭 Let me analyze the convergence behavior of both optimizations...
🔧 analyze_convergence(problem_id="wing_M085", algorithms=["SLSQP", "BFGS"])
✓ Analysis complete [0.12s]

BFGS performed better for several reasons:
1. Better gradient approximation: The quasi-Newton Hessian approximation
   in BFGS was more accurate for this problem's curvature.
2. Fewer function evaluations: BFGS used 38 iterations vs SLSQP's 45.
3. Smoother convergence: BFGS showed monotonic improvement, while SLSQP
   had some oscillations in the constraint handling.

Would you like me to visualize the convergence comparison?
```

## Implementation Phases

### Phase 1: Core REPL (Day 1, 4-6 hours)
- [x] Design architecture
- [ ] Basic REPL with prompt_toolkit
- [ ] Simple message loop (no agent yet)
- [ ] Exit handling
- [ ] History support

### Phase 2: Agent Integration (Day 1-2, 4-6 hours)
- [ ] CLICallback for streaming
- [ ] AgentManager integration
- [ ] Conversation history management
- [ ] Real-time agent display

### Phase 3: Project Management (Day 2, 3-4 hours)
- [ ] Project class
- [ ] Directory structure
- [ ] State persistence
- [ ] Load/save functionality

### Phase 4: Slash Commands (Day 2-3, 3-4 hours)
- [ ] Command registry
- [ ] /help, /status, /history
- [ ] /save, /load
- [ ] /clear, /project

### Phase 5: Polish (Day 3, 2-3 hours)
- [ ] Welcome/goodbye screens
- [ ] Error handling
- [ ] Tab completion
- [ ] Documentation

## Key Dependencies

```python
# requirements.txt additions
prompt-toolkit>=3.0.0  # For REPL interface
```

## Advantages Over Traditional Optimization Tools

1. **Conversational** - "optimize a wing" not "run_optimization(...)"
2. **Intelligent** - Agent understands context and intent
3. **Interactive** - Iterative refinement through dialogue
4. **Transparent** - See agent thinking in real-time
5. **Flexible** - No rigid command syntax
6. **Collaborative** - User and agent work together

## Example Session

```bash
$ aopt
╭──────────────────────────────────────────╮
│  AgenticOpt - AI Optimization Assistant │
╰──────────────────────────────────────────╯

aopt> create project turbine_design
✓ Created project 'turbine_design'

aopt [turbine_design]> I want to maximize turbine efficiency

💭 I'll help design a turbine optimization problem. What type of turbine?
   - Gas turbine
   - Steam turbine
   - Wind turbine
   - Hydraulic turbine

aopt [turbine_design]> gas turbine

💭 Great! For gas turbine efficiency optimization, I'll need to know...

[Interactive dialogue continues]

aopt [turbine_design]> /save best_design
✓ Saved to turbine_design/results/best_design.json

aopt [turbine_design]> exit
Goodbye! Your work is saved in ./turbine_design/
```

This architecture makes optimization **accessible, intelligent, and collaborative** - exactly what the agentic vision is about!
