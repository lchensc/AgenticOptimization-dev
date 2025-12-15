"""File-based storage implementation using JSON files."""

import json
from pathlib import Path
from typing import List, Optional

from .backend import StorageBackend
from ..schema import SessionRecord
from ..problem import Problem


class FileStorage(StorageBackend):
    """
    File-based storage using JSON files.

    v0.2.0 Directory structure:
        .paola_runs/
        ├── sessions/
        │   ├── session_0001.json
        │   ├── session_0002.json
        │   └── ...
        ├── problems/
        │   └── problem_id.json
        └── metadata.json (tracks next_session_id)
    """

    def __init__(self, base_dir: str = ".paola_runs"):
        """
        Initialize file storage.

        Args:
            base_dir: Base directory for storage (default: .paola_runs)
        """
        self.base_dir = Path(base_dir)
        self.sessions_dir = self.base_dir / "sessions"
        self.problems_dir = self.base_dir / "problems"
        self.metadata_file = self.base_dir / "metadata.json"

        # Create directories if needed
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.problems_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        if not self.metadata_file.exists():
            self._save_metadata({"next_session_id": 1})

    def save_session(self, session: SessionRecord) -> None:
        """Save session to JSON file."""
        file_path = self.sessions_dir / f"session_{session.session_id:04d}.json"
        with open(file_path, 'w') as f:
            f.write(session.to_json())

    def load_session(self, session_id: int) -> Optional[SessionRecord]:
        """Load session from JSON file."""
        file_path = self.sessions_dir / f"session_{session_id:04d}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            return SessionRecord.from_json(f.read())

    def load_all_sessions(self) -> List[SessionRecord]:
        """Load all sessions, sorted by session_id."""
        sessions = []
        for file_path in sorted(self.sessions_dir.glob("session_*.json")):
            with open(file_path, 'r') as f:
                sessions.append(SessionRecord.from_json(f.read()))
        return sessions

    def save_problem(self, problem: Problem) -> None:
        """Save problem metadata."""
        file_path = self.problems_dir / f"{problem.problem_id}.json"
        with open(file_path, 'w') as f:
            json.dump(problem.to_dict(), f, indent=2)

    def load_problem(self, problem_id: str) -> Optional[Problem]:
        """Load problem metadata."""
        file_path = self.problems_dir / f"{problem_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            return Problem.from_dict(json.load(f))

    def get_next_session_id(self) -> int:
        """Get and increment next session ID."""
        metadata = self._load_metadata()
        session_id = metadata.get("next_session_id", 1)
        metadata["next_session_id"] = session_id + 1
        self._save_metadata(metadata)
        return session_id

    def _load_metadata(self) -> dict:
        """Load metadata from file."""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def _save_metadata(self, metadata: dict) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
