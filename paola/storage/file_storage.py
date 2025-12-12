"""File-based storage implementation using JSON files."""

import json
from pathlib import Path
from typing import List, Optional

from .base import StorageBackend
from .models import OptimizationRun, Problem


class FileStorage(StorageBackend):
    """File-based storage using JSON files."""

    def __init__(self, base_dir: str = ".paola_runs"):
        self.base_dir = Path(base_dir)
        self.runs_dir = self.base_dir / "runs"
        self.problems_dir = self.base_dir / "problems"
        self.metadata_file = self.base_dir / "metadata.json"

        # Create directories if needed
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.problems_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        if not self.metadata_file.exists():
            self._save_metadata({"next_run_id": 1})

    def save_run(self, run: OptimizationRun) -> None:
        """Save run to JSON file."""
        file_path = self.runs_dir / f"run_{run.run_id:03d}.json"
        with open(file_path, 'w') as f:
            f.write(run.to_json())

    def load_run(self, run_id: int) -> Optional[OptimizationRun]:
        """Load run from JSON file."""
        file_path = self.runs_dir / f"run_{run_id:03d}.json"
        if not file_path.exists():
            return None

        with open(file_path, 'r') as f:
            return OptimizationRun.from_json(f.read())

    def load_all_runs(self) -> List[OptimizationRun]:
        """Load all runs, sorted by run_id."""
        runs = []
        for file_path in sorted(self.runs_dir.glob("run_*.json")):
            with open(file_path, 'r') as f:
                runs.append(OptimizationRun.from_json(f.read()))
        return runs

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

    def get_next_run_id(self) -> int:
        """Get and increment next run ID."""
        metadata = self._load_metadata()
        run_id = metadata["next_run_id"]
        metadata["next_run_id"] = run_id + 1
        self._save_metadata(metadata)
        return run_id

    def _load_metadata(self) -> dict:
        """Load metadata from file."""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def _save_metadata(self, metadata: dict) -> None:
        """Save metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
