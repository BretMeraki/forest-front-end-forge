from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
import json
import os

from ..protocols import SemanticMemoryProtocol

class MemoryEntry:
    def __init__(
        self,
        memory_type: str,
        content: str,
        timestamp: datetime,
        metadata: Optional[Dict[str, Any]] = None
    ):
        if not memory_type or not content:
            raise ValueError("memory_type and content must not be empty")
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime object")
            
        self.memory_type = memory_type.strip()
        self.content = content.strip()
        self.timestamp = timestamp
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        try:
            return {
                'memory_type': self.memory_type,
                'content': self.content,
                'timestamp': self.timestamp.isoformat(),
                'metadata': self.metadata
            }
        except Exception as e:
            raise ValueError(f"Error converting memory to dict: {e}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")
            
        required_fields = {'memory_type', 'content', 'timestamp'}
        missing_fields = required_fields - set(data.keys())
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
            
        try:
            return cls(
                memory_type=str(data['memory_type']),
                content=str(data['content']),
                timestamp=datetime.fromisoformat(str(data['timestamp'])),
                metadata=data.get('metadata', {})
            )
        except Exception as e:
            raise ValueError(f"Error creating memory from dict: {e}")

class SemanticMemoryManager:
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or 'memory_store.json'
        self.memories: List[MemoryEntry] = []
        self.current_context: Dict[str, Any] = {}
        self._load_memories()

    def store_milestone(self, node_id: UUID, description: str, impact: float) -> None:
        """Store a milestone memory with its impact and context."""
        if not isinstance(node_id, UUID):
            raise TypeError("node_id must be a UUID")
        if not description:
            raise ValueError("description must not be empty")
        if not isinstance(impact, (int, float)):
            raise TypeError("impact must be a number")
        if impact < 0.0 or impact > 1.0:
            raise ValueError("impact must be between 0.0 and 1.0")
            
        try:
            memory = MemoryEntry(
                memory_type='milestone',
                content=description,
                timestamp=datetime.utcnow(),
                metadata={
                    'node_id': str(node_id),
                    'impact': float(impact),
                    'context': self.current_context.copy()
                }
            )
            self.memories.append(memory)
            self._save_memories()
        except Exception as e:
            raise ValueError(f"Error storing milestone: {e}")

    def store_reflection(
        self,
        reflection_type: str,
        content: str,
        emotion: Optional[str] = None
    ) -> None:
        """Store a reflection with optional emotional context."""
        if not reflection_type or not content:
            raise ValueError("reflection_type and content must not be empty")
        if emotion is not None and not isinstance(emotion, str):
            raise TypeError("emotion must be a string or None")
            
        try:
            memory = MemoryEntry(
                memory_type='reflection',
                content=content,
                timestamp=datetime.utcnow(),
                metadata={
                    'reflection_type': reflection_type,
                    'emotion': emotion,
                    'context': self.current_context.copy()
                }
            )
            self.memories.append(memory)
            self._save_memories()
        except Exception as e:
            raise ValueError(f"Error storing reflection: {e}")

    def get_relevant_memories(
        self,
        context: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve memories relevant to the given context."""
        if not isinstance(context, str):
            raise TypeError("context must be a string")
        if not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
            
        try:
            scored_memories = []
            context_lower = context.lower()
            
            for memory in self.memories:
                if not isinstance(memory, MemoryEntry):
                    continue
                    
                score = 0.0
                
                # Check content match
                if context_lower in memory.content.lower():
                    score += 1.0
                
                # Check metadata context match
                memory_context = memory.metadata.get('context', {})
                if isinstance(memory_context, dict):
                    for key, value in memory_context.items():
                        if str(value).lower() in context_lower:
                            score += 0.5
                
                # Boost recent memories
                time_diff = (datetime.utcnow() - memory.timestamp).total_seconds()
                recency_boost = 1.0 / (1.0 + time_diff / 86400.0)  # Decay over days
                score += recency_boost
                
                # Boost high-impact memories
                if memory.memory_type == 'milestone':
                    impact = memory.metadata.get('impact', 0.0)
                    if isinstance(impact, (int, float)):
                        score *= (1.0 + float(impact))
                
                if score > 0:
                    scored_memories.append((score, memory))
            
            # Sort by relevance score and return top memories
            scored_memories.sort(reverse=True, key=lambda x: x[0])
            return [memory.to_dict() for _, memory in scored_memories[:limit]]
        except Exception as e:
            raise ValueError(f"Error getting relevant memories: {e}")

    def update_context(self, new_context: Dict[str, Any]) -> None:
        """Update the current context for new memories."""
        if not isinstance(new_context, dict):
            raise TypeError("new_context must be a dictionary")
        self.current_context.update(new_context)

    def _load_memories(self) -> None:
        """Load memories from storage."""
        if not os.path.exists(self.storage_path):
            self.memories = []
            return
            
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise TypeError("Memory storage must contain a list")
                self.memories = [
                    MemoryEntry.from_dict(memory_data)
                    for memory_data in data
                    if isinstance(memory_data, dict)
                ]
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Error loading memories: {e}")
        except Exception as e:
            self.memories = []
            raise ValueError(f"Unexpected error loading memories: {e}")

    def _save_memories(self) -> None:
        """Save memories to storage."""
        try:
            data = [memory.to_dict() for memory in self.memories]
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise ValueError(f"Error saving memories: {e}") 