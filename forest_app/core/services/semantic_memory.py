"""Semantic Memory Service for Forest App."""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
import numpy as np
from forest_app.integrations.llm import LLMClient

logger = logging.getLogger(__name__)

from forest_app.core.services.semantic_base import SemanticMemoryManagerBase

class SemanticMemoryManager(SemanticMemoryManagerBase):
    """Manages semantic episodic memory for the Forest application."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.memories: List[Dict[str, Any]] = []
        
    async def store_memory(self, 
                          event_type: str,
                          content: str,
                          metadata: Optional[Dict[str, Any]] = None,
                          importance: float = 0.5) -> Dict[str, Any]:
        """
        Store a new memory with semantic embedding.
        
        Args:
            event_type: Type of event (e.g., 'task_completion', 'reflection', 'milestone')
            content: The actual content/description of the memory
            metadata: Additional structured data about the memory
            importance: Float between 0-1 indicating memory importance
        """
        # Generate embedding for the content using LLM
        embedding = await self.llm_client.get_embedding(content)
        
        memory = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "content": content,
            "metadata": metadata or {},
            "importance": importance,
            "embedding": embedding,
            "access_count": 0,
            "last_accessed": None
        }
        
        self.memories.append(memory)
        logger.info(f"Stored new memory of type {event_type}")
        return memory
    
    async def query_memories(self, 
                           query: str, 
                           k: int = 5,
                           event_types: Optional[List[str]] = None,
                           time_window_days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Query memories semantically similar to the input query.
        
        Args:
            query: The search query
            k: Number of memories to return
            event_types: Optional filter for specific event types
            time_window_days: Optional time window to search within
        """
        if not self.memories:
            return []
            
        # Get query embedding
        query_embedding = await self.llm_client.get_embedding(query)
        
        # Filter memories by event type and time window if specified
        filtered_memories = self.memories
        if event_types:
            filtered_memories = [m for m in filtered_memories if m["event_type"] in event_types]
            
        if time_window_days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=time_window_days)
            filtered_memories = [
                m for m in filtered_memories 
                if datetime.fromisoformat(m["timestamp"]) >= cutoff
            ]
            
        # Calculate cosine similarities
        similarities = []
        for memory in filtered_memories:
            similarity = self._cosine_similarity(query_embedding, memory["embedding"])
            similarities.append((similarity, memory))
            
        # Sort by similarity and return top k
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_memories = [memory for _, memory in similarities[:k]]
        
        # Update access stats
        for memory in top_memories:
            await self.update_memory_stats(memory["id"], 1)
            
        return top_memories

    async def get_recent_memories(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent memories."""
        sorted_memories = sorted(
            self.memories,
            key=lambda x: datetime.fromisoformat(x["timestamp"]),
            reverse=True
        )
        return sorted_memories[:limit]

    async def extract_themes(self, memories: List[Dict[str, Any]]) -> List[str]:
        """Extract common themes from a list of memories."""
        if not memories:
            return []

        # Combine all memory content for theme extraction
        combined_content = " ".join([m["content"] for m in memories])
        
        # Use LLM to extract themes
        themes = await self.llm_client.extract_themes(combined_content)
        return themes
    
    async def update_memory_stats(self, memory_id: str, access_count: int = 1) -> bool:
        """Update access statistics for a memory."""
        for memory in self.memories:
            if memory.get("id") == memory_id:
                memory["access_count"] += access_count
                memory["last_accessed"] = datetime.now(timezone.utc).isoformat()
                return True
        return False
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        if not self.memories:
            return {
                "total_memories": 0,
                "memory_types": {},
                "avg_importance": 0,
                "avg_access_count": 0
            }
            
        memory_types = {}
        total_importance = 0
        total_access_count = 0
        
        for memory in self.memories:
            event_type = memory["event_type"]
            memory_types[event_type] = memory_types.get(event_type, 0) + 1
            total_importance += memory["importance"]
            total_access_count += memory["access_count"]
            
        return {
            "total_memories": len(self.memories),
            "memory_types": memory_types,
            "avg_importance": total_importance / len(self.memories),
            "avg_access_count": total_access_count / len(self.memories)
        }
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory store to serializable dictionary."""
        return {
            "memories": self.memories,
            "stats": self.get_memory_stats()
        }
        
    async def from_dict(self, data: Dict[str, Any]) -> None:
        """Load memories from dictionary."""
        if "memories" in data and isinstance(data["memories"], list):
            self.memories = data["memories"]
            logger.info(f"Loaded {len(self.memories)} memories from dictionary") 