# Callback & Streaming Architecture

**Date**: December 10, 2025
**Status**: Design specification for Milestone 1

---

## 1. Overview

The agent emits **real-time events** during optimization, enabling:
- Rich console output (progress bars, colored logs, tables)
- Jupyter notebook integration (inline plots, interactive controls)
- Web dashboards (for remote HPC monitoring)
- File logging (for batch jobs, debugging)
- Testing/debugging (capture events for assertions)

**Key principle**: Callbacks are **optional** - agent works without them, but becomes observable when enabled.

---

## 2. Event Types

The agent emits structured events at key points in the optimization:

```python
from enum import Enum
from pydantic import BaseModel
from typing import Any, Optional

class EventType(str, Enum):
    """All event types emitted by agent."""

    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_STEP = "agent_step"           # Start of ReAct cycle
    AGENT_DONE = "agent_done"

    # Formulation
    FORMULATION_START = "formulation_start"
    FORMULATION_COMPLETE = "formulation_complete"
    FORMULATION_QUESTION = "formulation_question"  # Agent needs user input

    # Agent reasoning
    REASONING = "reasoning"             # Agent's thought process

    # Tool execution
    TOOL_CALL = "tool_call"             # About to call tool
    TOOL_RESULT = "tool_result"         # Tool returned result
    TOOL_ERROR = "tool_error"           # Tool failed

    # Optimization progress
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"
    EVALUATION = "evaluation"           # Function evaluation
    CACHE_HIT = "cache_hit"             # Evaluation from cache

    # Convergence & observation
    CONVERGENCE_CHECK = "convergence_check"
    PATTERN_DETECTED = "pattern_detected"  # Agent detected issue

    # Adaptation
    ADAPTATION_START = "adaptation_start"
    ADAPTATION_COMPLETE = "adaptation_complete"
    RESTART = "restart"                 # Optimizer restart

    # Resource tracking
    BUDGET_UPDATE = "budget_update"


class AgentEvent(BaseModel):
    """
    Structured event emitted by agent.

    All callbacks receive AgentEvent instances.
    """

    # Event metadata
    event_type: EventType
    timestamp: float  # Unix timestamp
    iteration: int    # Current iteration number

    # Event-specific data
    data: dict[str, Any]

    # Context snapshot (optional, for rich events)
    context: Optional[dict] = None


# Example events:

# Tool call event
AgentEvent(
    event_type=EventType.TOOL_CALL,
    timestamp=1702234567.89,
    iteration=12,
    data={
        "tool_name": "evaluate_function",
        "arguments": {
            "design": [0.5, 0.3, 0.2],
            "objectives": ["drag"],
            "gradient": True
        }
    }
)

# Tool result event
AgentEvent(
    event_type=EventType.TOOL_RESULT,
    timestamp=1702234568.12,
    iteration=12,
    data={
        "tool_name": "evaluate_function",
        "result": {
            "objectives": [0.0245],
            "gradient": [...],
            "cost": 0.5,
            "cache_hit": False
        },
        "duration": 0.23  # seconds
    }
)

# Cache hit event (important for efficiency tracking)
AgentEvent(
    event_type=EventType.CACHE_HIT,
    timestamp=1702234569.45,
    iteration=13,
    data={
        "design": [0.5, 0.3, 0.2],
        "saved_cost": 0.5  # CPU hours saved
    }
)

# Reasoning event (agent's thoughts)
AgentEvent(
    event_type=EventType.REASONING,
    timestamp=1702234570.12,
    iteration=13,
    data={
        "reasoning": "Convergence is slow. Gradient variance is high (0.08). I should check for numerical noise and consider switching gradient method."
    }
)

# Pattern detection event
AgentEvent(
    event_type=EventType.PATTERN_DETECTED,
    timestamp=1702234571.34,
    iteration=15,
    data={
        "pattern_type": "repeated_constraint_violation",
        "details": {
            "constraint": "CL >= 0.8",
            "violations": 8,
            "recent_values": [0.78, 0.79, 0.77, 0.79]
        },
        "recommendation": "Tighten constraint to CL >= 0.82"
    }
)

# Adaptation event
AgentEvent(
    event_type=EventType.ADAPTATION_START,
    timestamp=1702234572.56,
    iteration=15,
    data={
        "adaptation_type": "constraint_modification",
        "reasoning": "Constraint CL >= 0.8 violated 8 times. Tightening to CL >= 0.82.",
        "old_constraint": {"CL": 0.8},
        "new_constraint": {"CL": 0.82}
    }
)
```

---

## 3. Callback Interface

### 3.1 Callback Function Signature

```python
from typing import Callable

CallbackFunction = Callable[[AgentEvent], None]

# Simple callback example
def my_callback(event: AgentEvent) -> None:
    """
    User-defined callback receives events.

    Can do anything: print, log, update UI, collect metrics, etc.
    """
    if event.event_type == EventType.ITERATION_COMPLETE:
        print(f"Iteration {event.iteration} done: {event.data}")
```

### 3.2 Callback Manager

```python
class CallbackManager:
    """
    Manages multiple callbacks, handles errors.

    Allows registering multiple callbacks that run in order.
    If one callback fails, others still execute.
    """

    def __init__(self):
        self.callbacks: list[CallbackFunction] = []

    def register(self, callback: CallbackFunction) -> None:
        """Add callback to list."""
        self.callbacks.append(callback)

    def emit(self, event: AgentEvent) -> None:
        """
        Send event to all registered callbacks.

        Catches exceptions to prevent callback failures from
        breaking the optimization.
        """
        for callback in self.callbacks:
            try:
                callback(event)
            except Exception as e:
                # Log error but continue
                print(f"Callback error: {e}")
                # Don't let callback failures break optimization
```

---

## 4. Integration with ReAct Loop

### 4.1 Modified Agent Structure

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Optional
import operator
import time

class AgentState(TypedDict):
    """Agent working memory."""
    messages: Annotated[list, operator.add]
    context: dict
    done: bool
    iteration: int  # Track iteration number
    callback_manager: Optional[CallbackManager]  # For event emission


def build_aopt_agent(tools: list, callback_manager: Optional[CallbackManager] = None):
    """
    Build ReAct agent with optional callbacks.

    Args:
        tools: List of available tools
        callback_manager: Optional callback manager for events
    """

    workflow = StateGraph(AgentState)

    # Single node: ReAct step (now with callbacks)
    workflow.add_node("react", create_react_node(tools))

    workflow.set_entry_point("react")

    workflow.add_conditional_edges(
        "react",
        lambda state: "end" if state["done"] else "continue",
        {
            "continue": "react",
            "end": END
        }
    )

    return workflow.compile()


def create_react_node(tools):
    """
    ReAct node with event emission at key points.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, ToolMessage

    llm = ChatAnthropic(model="claude-sonnet-4-5-20250929")
    llm_with_tools = llm.bind_tools(tools)

    def react_step(state: AgentState) -> dict:
        """
        Execute one ReAct cycle with event streaming.
        """
        context = state["context"]
        iteration = state.get("iteration", 0) + 1
        callback_manager = state.get("callback_manager")

        # EMIT: Agent step start
        if callback_manager:
            callback_manager.emit(AgentEvent(
                event_type=EventType.AGENT_STEP,
                timestamp=time.time(),
                iteration=iteration,
                data={"step": iteration}
            ))

        # Build prompt
        prompt = build_optimization_prompt(context)
        messages = state["messages"] + [HumanMessage(content=prompt)]

        # Get LLM decision
        response = llm_with_tools.invoke(messages)

        # EMIT: Reasoning (if agent provided text)
        if callback_manager and response.content:
            callback_manager.emit(AgentEvent(
                event_type=EventType.REASONING,
                timestamp=time.time(),
                iteration=iteration,
                data={"reasoning": response.content}
            ))

        new_messages = [response]

        # Execute tool calls
        if response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                # EMIT: Tool call start
                if callback_manager:
                    callback_manager.emit(AgentEvent(
                        event_type=EventType.TOOL_CALL,
                        timestamp=time.time(),
                        iteration=iteration,
                        data={
                            "tool_name": tool_call["name"],
                            "arguments": tool_call["args"]
                        }
                    ))

                # Execute tool
                start_time = time.time()
                try:
                    result = execute_tool(tool_call, tools)
                    duration = time.time() - start_time

                    # EMIT: Tool result
                    if callback_manager:
                        callback_manager.emit(AgentEvent(
                            event_type=EventType.TOOL_RESULT,
                            timestamp=time.time(),
                            iteration=iteration,
                            data={
                                "tool_name": tool_call["name"],
                                "result": result,
                                "duration": duration
                            }
                        ))

                    # EMIT: Special events for specific tools
                    if callback_manager:
                        emit_tool_specific_events(
                            tool_call["name"],
                            result,
                            iteration,
                            callback_manager
                        )

                    tool_results.append(result)
                    new_messages.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"]
                        )
                    )

                except Exception as e:
                    # EMIT: Tool error
                    if callback_manager:
                        callback_manager.emit(AgentEvent(
                            event_type=EventType.TOOL_ERROR,
                            timestamp=time.time(),
                            iteration=iteration,
                            data={
                                "tool_name": tool_call["name"],
                                "error": str(e)
                            }
                        ))
                    raise

            new_context = update_context(context, tool_results)

            return {
                "messages": new_messages,
                "context": new_context,
                "done": False,
                "iteration": iteration
            }

        # Check if done
        if "DONE" in response.content.upper() or "CONVERGED" in response.content.upper():
            # EMIT: Agent done
            if callback_manager:
                callback_manager.emit(AgentEvent(
                    event_type=EventType.AGENT_DONE,
                    timestamp=time.time(),
                    iteration=iteration,
                    data={
                        "reason": "converged",
                        "final_context": context
                    }
                ))

            return {
                "messages": new_messages,
                "context": context,
                "done": True,
                "iteration": iteration
            }

        return {
            "messages": new_messages,
            "context": context,
            "done": False,
            "iteration": iteration
        }

    return react_step


def emit_tool_specific_events(
    tool_name: str,
    result: dict,
    iteration: int,
    callback_manager: CallbackManager
) -> None:
    """
    Emit specialized events based on tool results.

    Examples:
    - evaluate_function with cache_hit=True → CACHE_HIT event
    - analyze_convergence → CONVERGENCE_CHECK event
    - detect_pattern with pattern found → PATTERN_DETECTED event
    """

    # Cache hit detection
    if tool_name == "evaluate_function" and result.get("cache_hit"):
        callback_manager.emit(AgentEvent(
            event_type=EventType.CACHE_HIT,
            timestamp=time.time(),
            iteration=iteration,
            data={
                "design": result.get("design"),
                "saved_cost": result.get("cost", 0)
            }
        ))

    # Convergence check
    if tool_name == "analyze_convergence":
        callback_manager.emit(AgentEvent(
            event_type=EventType.CONVERGENCE_CHECK,
            timestamp=time.time(),
            iteration=iteration,
            data=result
        ))

    # Pattern detection
    if tool_name == "detect_pattern" and result:
        callback_manager.emit(AgentEvent(
            event_type=EventType.PATTERN_DETECTED,
            timestamp=time.time(),
            iteration=iteration,
            data=result
        ))

    # Optimizer restart
    if tool_name == "optimizer_restart":
        callback_manager.emit(AgentEvent(
            event_type=EventType.RESTART,
            timestamp=time.time(),
            iteration=iteration,
            data=result
        ))
```

---

## 5. User-Facing API

### 5.1 Agent Class with Callback Support

```python
from aopt import Agent
from aopt.callbacks import RichConsoleCallback, FileLogger

# Option 1: Use built-in rich console (default)
agent = Agent(llm_model="claude-sonnet-4-5", verbose=True)
# verbose=True automatically registers RichConsoleCallback

# Option 2: Custom callback
def my_callback(event: AgentEvent):
    if event.event_type == EventType.ITERATION_COMPLETE:
        print(f"Iteration {event.iteration}: {event.data}")

agent = Agent(llm_model="claude-sonnet-4-5")
agent.register_callback(my_callback)

# Option 3: Multiple callbacks
agent = Agent(llm_model="claude-sonnet-4-5")
agent.register_callback(RichConsoleCallback())  # Pretty terminal output
agent.register_callback(FileLogger("optimization.log"))  # File logging
agent.register_callback(my_custom_tracker)  # Custom metrics

# Option 4: No callbacks (headless, for testing)
agent = Agent(llm_model="claude-sonnet-4-5", verbose=False)
# Runs silently, only returns final result
```

### 5.2 Agent Implementation

```python
class Agent:
    """
    Main agent class with callback support.
    """

    def __init__(
        self,
        llm_model: str = "claude-sonnet-4-5",
        verbose: bool = True,
        log_file: Optional[str] = None
    ):
        self.llm_model = llm_model
        self.callback_manager = CallbackManager()

        # Auto-register default callbacks
        if verbose:
            from aopt.callbacks import RichConsoleCallback
            self.callback_manager.register(RichConsoleCallback())

        if log_file:
            from aopt.callbacks import FileLogger
            self.callback_manager.register(FileLogger(log_file))

        # Build agent with tools
        from aopt.tools import get_all_tools
        self.tools = get_all_tools()
        self.graph = build_aopt_agent(self.tools)

    def register_callback(self, callback: CallbackFunction) -> None:
        """Register additional callback."""
        self.callback_manager.register(callback)

    def run(self, goal: str, budget: Optional[float] = None) -> dict:
        """
        Run optimization with goal.

        Emits events to all registered callbacks during execution.
        """
        # Emit start event
        self.callback_manager.emit(AgentEvent(
            event_type=EventType.AGENT_START,
            timestamp=time.time(),
            iteration=0,
            data={"goal": goal, "budget": budget}
        ))

        # Initialize state
        initial_state = {
            "messages": [],
            "context": {"goal": goal, "budget_total": budget},
            "done": False,
            "iteration": 0,
            "callback_manager": self.callback_manager
        }

        # Run agent graph
        final_state = self.graph.invoke(initial_state)

        # Extract result
        result = {
            "converged": final_state["done"],
            "iterations": final_state["iteration"],
            "final_context": final_state["context"],
            "reasoning_log": extract_reasoning(final_state["messages"])
        }

        return result
```

---

## 6. Built-in Callback Implementations

### 6.1 Rich Console Callback

```python
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

class RichConsoleCallback:
    """
    Beautiful terminal output using rich library.

    Features:
    - Colored output based on event type
    - Progress bars for iterations
    - Tables for convergence metrics
    - Panels for important events (adaptations, convergence)
    """

    def __init__(self):
        self.console = Console()
        self.progress = None
        self.current_task = None

    def __call__(self, event: AgentEvent) -> None:
        """Handle event and render to console."""

        if event.event_type == EventType.AGENT_START:
            self.console.print(Panel(
                f"[bold cyan]Optimization Started[/bold cyan]\n"
                f"Goal: {event.data['goal']}\n"
                f"Budget: {event.data.get('budget', 'Unlimited')} CPU hours",
                title="AOpt Agent"
            ))

        elif event.event_type == EventType.REASONING:
            self.console.print(
                f"[dim]💭 Agent: {event.data['reasoning']}[/dim]"
            )

        elif event.event_type == EventType.TOOL_CALL:
            self.console.print(
                f"[yellow]🔧 Calling {event.data['tool_name']}...[/yellow]"
            )

        elif event.event_type == EventType.TOOL_RESULT:
            duration = event.data.get('duration', 0)
            self.console.print(
                f"[green]✓ {event.data['tool_name']} completed "
                f"({duration:.2f}s)[/green]"
            )

        elif event.event_type == EventType.CACHE_HIT:
            saved = event.data.get('saved_cost', 0)
            self.console.print(
                f"[bright_green]⚡ Cache hit! Saved {saved:.2f} CPU hours[/bright_green]"
            )

        elif event.event_type == EventType.ITERATION_COMPLETE:
            iter_num = event.iteration
            obj = event.data.get('objective', 'N/A')
            self.console.print(
                f"[blue]Iteration {iter_num}: objective = {obj}[/blue]"
            )

        elif event.event_type == EventType.CONVERGENCE_CHECK:
            # Render convergence table
            table = Table(title=f"Convergence Analysis (Iter {event.iteration})")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            for key, value in event.data.items():
                table.add_row(key, str(value))

            self.console.print(table)

        elif event.event_type == EventType.PATTERN_DETECTED:
            self.console.print(Panel(
                f"[bold yellow]⚠️  Pattern Detected[/bold yellow]\n"
                f"Type: {event.data['pattern_type']}\n"
                f"Recommendation: {event.data.get('recommendation', 'N/A')}",
                border_style="yellow"
            ))

        elif event.event_type == EventType.ADAPTATION_START:
            self.console.print(Panel(
                f"[bold magenta]🔄 Adaptation[/bold magenta]\n"
                f"Type: {event.data['adaptation_type']}\n"
                f"Reasoning: {event.data['reasoning']}",
                border_style="magenta"
            ))

        elif event.event_type == EventType.AGENT_DONE:
            self.console.print(Panel(
                f"[bold green]✅ Optimization Complete[/bold green]\n"
                f"Iterations: {event.iteration}\n"
                f"Reason: {event.data.get('reason', 'Unknown')}",
                title="Success",
                border_style="green"
            ))


### 6.2 File Logger Callback

```python
import json
from pathlib import Path

class FileLogger:
    """
    Log all events to JSON file for replay/debugging.
    """

    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.events = []

    def __call__(self, event: AgentEvent) -> None:
        """Append event to log."""
        self.events.append(event.model_dump())

        # Write to file (append mode)
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(event.model_dump()) + "\n")

    def get_events(self) -> list[AgentEvent]:
        """Get all logged events."""
        return [AgentEvent(**e) for e in self.events]


### 6.3 Simple Progress Callback

```python
class SimpleProgressCallback:
    """
    Minimal callback that just prints iteration progress.
    """

    def __call__(self, event: AgentEvent) -> None:
        if event.event_type == EventType.ITERATION_COMPLETE:
            data = event.data
            print(f"Iter {event.iteration}: "
                  f"obj={data.get('objective'):.6f}, "
                  f"converged={data.get('converged', False)}")
```

---

## 7. Testing with Callbacks

### 7.1 Event Capture for Testing

```python
class EventCapture:
    """
    Callback that captures events for testing assertions.
    """

    def __init__(self):
        self.events: list[AgentEvent] = []

    def __call__(self, event: AgentEvent) -> None:
        self.events.append(event)

    def get_events_by_type(self, event_type: EventType) -> list[AgentEvent]:
        """Filter events by type."""
        return [e for e in self.events if e.event_type == event_type]

    def count(self, event_type: EventType) -> int:
        """Count events of specific type."""
        return len(self.get_events_by_type(event_type))


# Test example
def test_agent_emits_events():
    """Verify agent emits expected events."""

    # Setup event capture
    capture = EventCapture()

    agent = Agent(llm_model="claude-sonnet-4-5", verbose=False)
    agent.register_callback(capture)

    # Run optimization
    result = agent.run("Minimize 2D Rosenbrock")

    # Assertions
    assert capture.count(EventType.AGENT_START) == 1
    assert capture.count(EventType.AGENT_DONE) == 1
    assert capture.count(EventType.TOOL_CALL) > 0
    assert capture.count(EventType.ITERATION_COMPLETE) > 0

    # Check cache hits occurred
    cache_hits = capture.get_events_by_type(EventType.CACHE_HIT)
    assert len(cache_hits) > 0, "Expected some cache hits during optimization"
```

---

## 8. Future Extensions

### 8.1 Jupyter Notebook Integration

```python
class JupyterCallback:
    """
    Display optimization progress in Jupyter with inline plots.
    """

    def __init__(self):
        from IPython.display import display, clear_output
        import matplotlib.pyplot as plt

        self.display = display
        self.clear_output = clear_output
        self.objectives = []

    def __call__(self, event: AgentEvent) -> None:
        if event.event_type == EventType.ITERATION_COMPLETE:
            self.objectives.append(event.data['objective'])

            # Update plot
            self.clear_output(wait=True)
            plt.figure(figsize=(10, 4))
            plt.plot(self.objectives, 'b-o')
            plt.xlabel('Iteration')
            plt.ylabel('Objective')
            plt.title('Optimization Progress')
            plt.grid(True)
            plt.show()
```

### 8.2 Web Dashboard Callback

```python
class WebDashboardCallback:
    """
    Send events to web dashboard via WebSocket.
    """

    def __init__(self, websocket_url: str):
        import websocket
        self.ws = websocket.WebSocket()
        self.ws.connect(websocket_url)

    def __call__(self, event: AgentEvent) -> None:
        # Send event as JSON to web dashboard
        self.ws.send(event.model_dump_json())
```

---

## 9. Summary

**Architecture decisions:**

1. ✅ **Event-driven**: Agent emits structured `AgentEvent` instances
2. ✅ **Optional callbacks**: Agent works without callbacks, becomes observable when enabled
3. ✅ **Multiple callbacks**: Can have console + file + custom simultaneously
4. ✅ **Error isolation**: Callback failures don't break optimization
5. ✅ **Rich by default**: `verbose=True` gives beautiful console output
6. ✅ **Testing friendly**: Can capture events for assertions

**Milestone 1 deliverables:**

- [x] `AgentEvent` + `EventType` definitions
- [x] `CallbackManager` implementation
- [x] Modified `react_step` with event emission
- [x] `RichConsoleCallback` (beautiful terminal)
- [x] `FileLogger` (JSON event log)
- [x] `EventCapture` (for testing)
- [x] User API: `Agent.register_callback()`

**Future (Milestone 2+):**
- Jupyter notebook integration
- Web dashboard
- Real-time plot updates
- Interactive controls (pause/modify mid-run)

---

**Status**: Ready for implementation in Milestone 1! 🚀
