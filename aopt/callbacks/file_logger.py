"""
File logger callback for saving events to JSON log.

Useful for debugging, replay, and post-analysis.
"""

from .base import AgentEvent, EventType
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class FileLogger:
    """
    Log all events to JSON file for replay/debugging.

    Each event is written as a JSON line (JSONL format).
    Can be used to replay optimization runs or analyze agent behavior.

    Example:
        >>> logger = FileLogger("optimization_run.log")
        >>> agent.register_callback(logger)
        >>> # Events automatically logged to file
    """

    def __init__(self, log_file: str, mode: str = "w"):
        """
        Initialize file logger.

        Args:
            log_file: Path to log file
            mode: File mode ('w' for overwrite, 'a' for append)
        """
        self.log_file = Path(log_file)
        self.mode = mode
        self.events: list[AgentEvent] = []

        # Create parent directory if needed
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize file (clear if mode='w')
        if mode == "w":
            self.log_file.write_text("")

        logger.info(f"FileLogger initialized: {self.log_file}")

    def __call__(self, event: AgentEvent) -> None:
        """
        Log event to file.

        Args:
            event: Event to log
        """
        # Store in memory
        self.events.append(event)

        # Write to file (append mode)
        try:
            with open(self.log_file, 'a') as f:
                # Write as JSON line
                json_str = event.model_dump_json()
                f.write(json_str + "\n")
        except Exception as e:
            logger.error(f"Failed to write event to {self.log_file}: {e}")

    def get_events(self) -> list[AgentEvent]:
        """Get all logged events from memory."""
        return self.events.copy()

    def get_events_by_type(self, event_type: EventType) -> list[AgentEvent]:
        """
        Get events filtered by type.

        Args:
            event_type: Event type to filter

        Returns:
            List of matching events
        """
        return [e for e in self.events if e.event_type == event_type]

    def count(self, event_type: EventType) -> int:
        """
        Count events of specific type.

        Args:
            event_type: Event type to count

        Returns:
            Number of events
        """
        return len(self.get_events_by_type(event_type))

    @classmethod
    def load_from_file(cls, log_file: str) -> list[AgentEvent]:
        """
        Load events from JSON log file.

        Args:
            log_file: Path to log file

        Returns:
            List of events
        """
        events = []
        log_path = Path(log_file)

        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")

        with open(log_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        event_dict = json.loads(line)
                        events.append(AgentEvent(**event_dict))
                    except Exception as e:
                        logger.warning(f"Failed to parse event line: {e}")

        return events
