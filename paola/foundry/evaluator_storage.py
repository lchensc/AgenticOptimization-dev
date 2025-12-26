"""
Evaluator storage layer for Foundry.

Manages storage and retrieval of evaluator configurations.
Integrates with Foundry's storage backend.
"""

import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .evaluator_schema import EvaluatorConfig
from .storage import StorageBackend


class EvaluatorStorage:
    """
    Storage and retrieval of evaluator configurations.

    Stores evaluator configurations (JSON) in Foundry storage backend.
    NOT code - just metadata that FoundryEvaluator uses.
    """

    def __init__(self, storage_backend: StorageBackend):
        """
        Initialize evaluator storage.

        Args:
            storage_backend: Foundry storage backend
        """
        self.storage = storage_backend
        self._evaluator_cache: Dict[str, EvaluatorConfig] = {}

    def store_evaluator(self, config: EvaluatorConfig) -> str:
        """
        Store evaluator configuration.

        Args:
            config: EvaluatorConfig to store

        Returns:
            evaluator_id
        """
        # Set registration time if not set
        if config.lineage.registered_at is None:
            config.lineage.registered_at = datetime.now()

        # Convert to dict for storage
        config_dict = config.dict()

        # Store in backend
        self._store_config_dict(config.evaluator_id, config_dict)

        # Update cache
        self._evaluator_cache[config.evaluator_id] = config

        return config.evaluator_id

    def retrieve_evaluator(self, evaluator_id: str) -> EvaluatorConfig:
        """
        Retrieve evaluator configuration.

        Args:
            evaluator_id: Evaluator ID

        Returns:
            EvaluatorConfig

        Raises:
            KeyError: If evaluator not found
        """
        # Check cache first
        if evaluator_id in self._evaluator_cache:
            return self._evaluator_cache[evaluator_id]

        # Load from storage
        config_dict = self._load_config_dict(evaluator_id)

        if config_dict is None:
            raise KeyError(f"Evaluator not found: {evaluator_id}")

        # Parse and cache
        config = EvaluatorConfig.from_dict(config_dict)
        self._evaluator_cache[evaluator_id] = config

        return config

    def list_evaluators(
        self,
        evaluator_type: Optional[str] = None,
        status: Optional[str] = None,
        domain: Optional[str] = None
    ) -> List[EvaluatorConfig]:
        """
        List evaluators with optional filters.

        Args:
            evaluator_type: Filter by source type (python_function, cli_executable)
            status: Filter by status (registered, validated, active)
            domain: Filter by problem domain

        Returns:
            List of EvaluatorConfig
        """
        all_configs = self._load_all_configs()

        # Apply filters
        filtered = []
        for config in all_configs:
            # Type filter
            if evaluator_type and config.source.type != evaluator_type:
                continue

            # Status filter
            if status and config.status != status:
                continue

            # Domain filter
            if domain and config.metadata.domain != domain:
                continue

            filtered.append(config)

        return filtered

    def update_performance(
        self,
        evaluator_id: str,
        execution_time: float,
        success: bool
    ):
        """
        Update performance metrics after evaluation.

        Args:
            evaluator_id: Evaluator ID
            execution_time: Time taken (seconds)
            success: Whether evaluation succeeded
        """
        config = self.retrieve_evaluator(evaluator_id)

        # Update metrics
        config.performance.total_calls += 1

        if success:
            # Update success rate
            total = config.performance.total_calls
            old_success_rate = config.performance.success_rate
            new_success_rate = (old_success_rate * (total - 1) + 1.0) / total
            config.performance.success_rate = new_success_rate

            # Update median and std (simplified - just track recent values)
            # In production, would use rolling statistics
            if config.performance.median_time is None:
                config.performance.median_time = execution_time
                config.performance.std_time = 0.0
            else:
                # Simple exponential moving average
                alpha = 0.1
                config.performance.median_time = (
                    alpha * execution_time +
                    (1 - alpha) * config.performance.median_time
                )

        else:
            # Update success rate for failure
            total = config.performance.total_calls
            old_success_rate = config.performance.success_rate
            new_success_rate = (old_success_rate * (total - 1)) / total
            config.performance.success_rate = new_success_rate

        # Store updated config
        self.store_evaluator(config)

    def add_run_reference(self, evaluator_id: str, run_id: str):
        """
        Add reference to run that uses this evaluator.

        Args:
            evaluator_id: Evaluator ID
            run_id: Run ID
        """
        config = self.retrieve_evaluator(evaluator_id)

        if run_id not in config.lineage.used_in_runs:
            config.lineage.used_in_runs.append(run_id)
            self.store_evaluator(config)

    def add_problem_reference(self, evaluator_id: str, problem_id: str):
        """
        Add reference to problem that uses this evaluator.

        Args:
            evaluator_id: Evaluator ID
            problem_id: Problem ID
        """
        config = self.retrieve_evaluator(evaluator_id)

        if problem_id not in config.lineage.used_in_problems:
            config.lineage.used_in_problems.append(problem_id)
            self.store_evaluator(config)

    def update_status(self, evaluator_id: str, status: str):
        """
        Update evaluator status.

        Args:
            evaluator_id: Evaluator ID
            status: New status (registered, validated, active, failed, deprecated)
        """
        config = self.retrieve_evaluator(evaluator_id)
        config.status = status
        self.store_evaluator(config)

    def delete_evaluator(self, evaluator_id: str):
        """
        Delete evaluator configuration.

        Args:
            evaluator_id: Evaluator ID
        """
        # Remove from cache
        self._evaluator_cache.pop(evaluator_id, None)

        # Remove from storage
        self._delete_config(evaluator_id)

    # ===== Storage Backend Interface =====

    def _store_config_dict(self, evaluator_id: str, config_dict: Dict[str, Any]):
        """Store configuration dict to backend."""
        # Use storage backend's base directory
        if hasattr(self.storage, 'base_dir'):
            base_dir = Path(self.storage.base_dir)
            evaluators_dir = base_dir / "evaluators"
            evaluators_dir.mkdir(exist_ok=True)

            config_file = evaluators_dir / f"{evaluator_id}.json"
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            # Fallback: store in memory only
            # (For storage backends that don't use filesystem)
            pass

    def _load_config_dict(self, evaluator_id: str) -> Optional[Dict[str, Any]]:
        """Load configuration dict from backend."""
        if hasattr(self.storage, 'base_dir'):
            base_dir = Path(self.storage.base_dir)
            config_file = base_dir / "evaluators" / f"{evaluator_id}.json"

            if config_file.exists():
                try:
                    with open(config_file) as f:
                        return json.load(f)
                except json.JSONDecodeError as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Corrupted evaluator file {evaluator_id}.json: {e}"
                    )
                    return None

        return None

    def _load_all_configs(self) -> List[EvaluatorConfig]:
        """Load all evaluator configurations."""
        configs = []

        if hasattr(self.storage, 'base_dir'):
            base_dir = Path(self.storage.base_dir)
            evaluators_dir = base_dir / "evaluators"

            if evaluators_dir.exists():
                for config_file in evaluators_dir.glob("*.json"):
                    try:
                        with open(config_file) as f:
                            config_dict = json.load(f)
                            config = EvaluatorConfig.from_dict(config_dict)
                            configs.append(config)
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        # Skip corrupted files but log warning
                        import logging
                        logging.getLogger(__name__).warning(
                            f"Skipping corrupted evaluator file {config_file.name}: {e}"
                        )
                        continue

        return configs

    def _delete_config(self, evaluator_id: str):
        """Delete configuration from backend."""
        if hasattr(self.storage, 'base_dir'):
            base_dir = Path(self.storage.base_dir)
            config_file = base_dir / "evaluators" / f"{evaluator_id}.json"

            if config_file.exists():
                config_file.unlink()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dict with statistics
        """
        all_configs = self._load_all_configs()

        stats = {
            'total_evaluators': len(all_configs),
            'by_type': {},
            'by_status': {},
            'by_domain': {},
            'total_calls': 0,
            'avg_success_rate': 0.0
        }

        for config in all_configs:
            # Count by type
            source_type = config.source.type
            stats['by_type'][source_type] = stats['by_type'].get(source_type, 0) + 1

            # Count by status
            status = config.status
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1

            # Count by domain
            domain = config.metadata.domain or 'unknown'
            stats['by_domain'][domain] = stats['by_domain'].get(domain, 0) + 1

            # Accumulate calls and success rate
            stats['total_calls'] += config.performance.total_calls

        # Calculate average success rate
        if all_configs:
            total_success_rate = sum(c.performance.success_rate for c in all_configs)
            stats['avg_success_rate'] = total_success_rate / len(all_configs)

        return stats

    def clear_cache(self):
        """Clear in-memory cache."""
        self._evaluator_cache.clear()

    def __str__(self) -> str:
        """String representation."""
        stats = self.get_statistics()
        return (
            f"EvaluatorStorage(\n"
            f"  total_evaluators={stats['total_evaluators']},\n"
            f"  by_type={stats['by_type']},\n"
            f"  total_calls={stats['total_calls']}\n"
            f")"
        )
