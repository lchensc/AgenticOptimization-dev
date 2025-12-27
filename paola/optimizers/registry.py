"""
Backend registry for optimizer management.

Provides centralized registration and lookup of optimizer backends,
replacing the scattered registration in the old backends.py.
"""

from typing import Dict, List, Optional, Any, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .base import OptimizerBackend

logger = logging.getLogger(__name__)


class BackendRegistry:
    """
    Registry for optimizer backends.

    Provides:
    - Backend registration
    - Lookup by name (with optional method extraction)
    - Listing of available backends
    - Lazy initialization

    Usage:
        registry = BackendRegistry()
        registry.register(SciPyBackend())
        backend = registry.get("scipy")
        backend = registry.get("scipy:SLSQP")  # Extracts method
    """

    def __init__(self):
        """Initialize empty registry."""
        self._backends: Dict[str, "OptimizerBackend"] = {}
        self._initialized = False

    def register(self, backend: "OptimizerBackend") -> None:
        """
        Register an optimizer backend.

        Args:
            backend: Backend instance to register
        """
        self._backends[backend.name] = backend
        logger.debug(f"Registered backend: {backend.name}")

    def get(self, name: str) -> Optional["OptimizerBackend"]:
        """
        Get backend by name.

        Handles "backend:method" format by extracting backend name.

        Args:
            name: Backend name or "backend:method" specification

        Returns:
            Backend instance or None if not found
        """
        self._ensure_initialized()

        # Handle "backend:method" format
        backend_name = name.split(":")[0].lower()
        return self._backends.get(backend_name)

    def get_available(self) -> List[str]:
        """
        Get list of available (installed) backend names.

        Returns:
            List of backend names that are installed and ready to use
        """
        self._ensure_initialized()
        return [
            name for name, backend in self._backends.items()
            if backend.is_available()
        ]

    def list_all(self) -> Dict[str, Dict[str, Any]]:
        """
        List all backends with their capabilities.

        Returns:
            Dict mapping backend name to info dict
        """
        self._ensure_initialized()

        result = {}
        for name, backend in self._backends.items():
            info = backend.get_info()
            info["available"] = backend.is_available()
            result[name] = info
        return result

    def _ensure_initialized(self) -> None:
        """Lazy initialization of backends."""
        if not self._initialized:
            self._initialize_backends()
            self._initialized = True

    def _initialize_backends(self) -> None:
        """
        Initialize and register all built-in backends.

        Override or extend this method to add custom backends.
        """
        # Import backends here to avoid circular imports
        from .backends import SciPyBackend, IPOPTBackend, OptunaBackend, PymooBackend

        self.register(SciPyBackend())
        self.register(IPOPTBackend())
        self.register(OptunaBackend())
        self.register(PymooBackend())

        logger.info(f"Initialized {len(self._backends)} optimizer backends")


# Global registry instance
_REGISTRY: Optional[BackendRegistry] = None


def get_registry() -> BackendRegistry:
    """Get the global backend registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = BackendRegistry()
    return _REGISTRY


def get_backend(name: str) -> Optional["OptimizerBackend"]:
    """
    Get backend by name (convenience function).

    Args:
        name: Backend name or "backend:method" specification

    Returns:
        Backend instance or None if not found
    """
    return get_registry().get(name)


def list_backends() -> Dict[str, Dict[str, Any]]:
    """
    List all backends with capabilities (convenience function).

    Returns:
        Dict mapping backend name to info dict
    """
    return get_registry().list_all()


def get_available_backends() -> List[str]:
    """
    Get list of available backend names (convenience function).

    Returns:
        List of installed backend names
    """
    return get_registry().get_available()


def register_backend(backend: "OptimizerBackend") -> None:
    """
    Register a custom backend (convenience function).

    Args:
        backend: Backend instance to register
    """
    get_registry().register(backend)
