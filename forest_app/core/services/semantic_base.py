"""
Base interface for SemanticMemoryManager to avoid circular imports.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime

class SemanticMemoryManagerBase:
    """Interface for semantic memory management."""
    async def store_memory(self, event_type: str, content: str, metadata: Optional[Dict[str, Any]] = None, importance: float = 0.5) -> Dict[str, Any]:
        raise NotImplementedError

    async def query_memories(self, query: str, k: int = 5, filter_event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def get_recent_memories(self, n: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError
