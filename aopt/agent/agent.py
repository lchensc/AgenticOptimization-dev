"""
Main Agent class for user-facing API.

Provides simple interface for running optimizations with full agent autonomy.
"""

from typing import Optional
import time
import logging

from .react_agent import build_aopt_agent
from ..callbacks import (
    CallbackManager,
    CallbackFunction,
    EventType,
    create_event,
    RichConsoleCallback,
    FileLogger
)

logger = logging.getLogger(__name__)


class Agent:
    """
    Main agent class for agentic optimization.

    Provides clean API for running optimizations with autonomous agent control.

    Example:
        >>> agent = Agent(llm_model="claude-sonnet-4-5", verbose=True)
        >>> result = agent.run("Minimize drag, maintain CL >= 0.8")
        >>> print(f"Converged: {result['converged']}")
        >>> print(f"Iterations: {result['iterations']}")

    The agent has full autonomy:
    - Formulates problems from natural language
    - Chooses optimizers
    - Executes iterations
    - Observes and adapts
    - Decides when to stop
    """

    def __init__(
        self,
        llm_model: str = "qwen-flash",
        temperature: float = 0.0,
        verbose: bool = True,
        log_file: Optional[str] = None,
        max_iterations: Optional[int] = None
    ):
        """
        Initialize agent.

        Args:
            llm_model: LLM model identifier
                      - Qwen: "qwen-flash" (default for development), "qwen-turbo", "qwen-plus"
                      - Claude: "claude-sonnet-4", "claude-3-5-sonnet-20241022"
                      - OpenAI: "gpt-4", "gpt-3.5-turbo"
            temperature: LLM temperature (0.0 = deterministic, 1.0 = creative)
            verbose: If True, register RichConsoleCallback
            log_file: If specified, register FileLogger
            max_iterations: Optional safety limit on iterations
                           (agent can still stop earlier)
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.callback_manager = CallbackManager()

        # Auto-register default callbacks
        if verbose:
            self.callback_manager.register(RichConsoleCallback(verbose=True))
            logger.info("Registered RichConsoleCallback")

        if log_file:
            self.callback_manager.register(FileLogger(log_file))
            logger.info(f"Registered FileLogger: {log_file}")

        # Tools will be initialized when needed
        self.tools = []

        # Agent graph will be built when needed
        self.graph = None

    def register_callback(self, callback: CallbackFunction) -> None:
        """
        Register additional callback.

        Args:
            callback: Callback function or callable class

        Example:
            >>> def my_callback(event):
            ...     if event.event_type == EventType.CACHE_HIT:
            ...         print(f"Cache hit! Saved {event.data['saved_cost']:.2f} hours")
            >>> agent.register_callback(my_callback)
        """
        self.callback_manager.register(callback)
        logger.info(f"Registered callback: {callback}")

    def run(
        self,
        goal: str,
        budget: Optional[float] = None,
        initial_problem: Optional[dict] = None
    ) -> dict:
        """
        Run optimization with given goal.

        Agent has full autonomy - decides everything from formulation to stopping.

        Args:
            goal: Natural language optimization goal
            budget: Optional computational budget (CPU hours)
            initial_problem: Optional pre-formulated problem
                            (if None, agent formulates from goal)

        Returns:
            {
                "converged": bool,
                "iterations": int,
                "final_context": dict,
                "reasoning_log": list[str],
                "total_time": float
            }

        Example:
            >>> result = agent.run('''
            ...     Minimize the 10-dimensional Rosenbrock function.
            ...     Bounds: [-5, 5] for all variables.
            ... ''')
            >>> print(f"Converged in {result['iterations']} iterations")
        """
        start_time = time.time()

        # Emit start event
        self.callback_manager.emit(create_event(
            event_type=EventType.AGENT_START,
            iteration=0,
            data={
                "goal": goal,
                "budget": budget,
                "llm_model": self.llm_model
            }
        ))

        # Initialize tools if not already done
        if not self.tools:
            self._initialize_tools()

        # Build agent graph if not already done
        if self.graph is None:
            self.graph = build_aopt_agent(
                tools=self.tools,
                llm_model=self.llm_model,
                callback_manager=self.callback_manager,
                temperature=self.temperature
            )
            logger.info(f"Built agent graph with model: {self.llm_model}")

        # Initialize state
        initial_state = {
            "messages": [],
            "context": {
                "goal": goal,
                "budget_total": budget,
                "budget_used": 0.0,
                "problem": initial_problem,
                "optimizer_type": None,
                "iteration": 0,
                "current_objectives": None,
                "best_objectives": None,
                "history": [],
                "observations": {},
                "cache_stats": {"hit_rate": 0.0},
                "budget_status": {
                    "total": budget,
                    "used": 0.0,
                    "remaining_pct": 100.0
                }
            },
            "done": False,
            "iteration": 0,
            "callback_manager": self.callback_manager
        }

        # Run agent graph
        try:
            final_state = self.graph.invoke(initial_state)
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            self.callback_manager.emit(create_event(
                event_type=EventType.TOOL_ERROR,
                iteration=initial_state.get("iteration", 0),
                data={"error": str(e), "tool_name": "agent"}
            ))
            raise

        # Extract result
        total_time = time.time() - start_time

        result = {
            "converged": final_state["done"],
            "iterations": final_state["iteration"],
            "final_context": final_state["context"],
            "reasoning_log": self._extract_reasoning(final_state["messages"]),
            "total_time": total_time
        }

        logger.info(
            f"Agent completed: converged={result['converged']}, "
            f"iterations={result['iterations']}, "
            f"time={total_time:.2f}s"
        )

        return result

    def _initialize_tools(self) -> None:
        """
        Initialize tools for agent.

        Registers all available tools:
        - Evaluator tools (2): evaluate_function, compute_gradient
        - Optimizer tools (4): optimizer_create, optimizer_propose, optimizer_update, optimizer_restart
        - Gate control tools (5): gate_continue, gate_stop, gate_restart_from, gate_get_history, gate_get_statistics
        - Observation tools (5): analyze_convergence, detect_pattern, check_feasibility, get_gradient_quality, compute_improvement_statistics
        - Cache tools (3): cache_stats, cache_clear, run_db_query
        """
        from ..tools import (
            # Evaluator tools
            evaluate_function,
            compute_gradient,
            # Optimizer tools
            optimizer_create,
            optimizer_propose,
            optimizer_update,
            optimizer_restart,
            # Gate control tools
            gate_continue,
            gate_stop,
            gate_restart_from,
            gate_get_history,
            gate_get_statistics,
            # Observation tools
            analyze_convergence,
            detect_pattern,
            check_feasibility,
            get_gradient_quality,
            compute_improvement_statistics,
            # Cache tools
            cache_stats,
            cache_clear,
            run_db_query,
        )

        self.tools = [
            # Evaluator tools (agent uses these most)
            evaluate_function,
            compute_gradient,
            # Optimizer tools
            optimizer_create,
            optimizer_propose,
            optimizer_update,
            optimizer_restart,
            # Gate control tools
            gate_continue,
            gate_stop,
            gate_restart_from,
            gate_get_history,
            gate_get_statistics,
            # Observation tools
            analyze_convergence,
            detect_pattern,
            check_feasibility,
            get_gradient_quality,
            compute_improvement_statistics,
            # Cache tools
            cache_stats,
            cache_clear,
            run_db_query,
        ]

        logger.info(f"Initialized {len(self.tools)} tools")

    def _extract_reasoning(self, messages: list) -> list[str]:
        """
        Extract agent reasoning from message history.

        Args:
            messages: Message history

        Returns:
            List of reasoning strings
        """
        reasoning = []

        for msg in messages:
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                # Skip if it's just a tool call or result
                if not msg.content.startswith("{"):
                    reasoning.append(msg.content)

        return reasoning

    def reset(self) -> None:
        """
        Reset agent state.

        Clears graph and prepares for new run.
        """
        self.graph = None
        logger.info("Agent reset")

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"Agent(llm_model='{self.llm_model}', "
            f"callbacks={len(self.callback_manager)})"
        )
