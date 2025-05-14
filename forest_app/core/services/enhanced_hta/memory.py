"""Semantic and episodic memory logic for Enhanced HTA Service.

This module provides functionality for:
- Memory retrieval and storage operations
- Semantic memory context generation
- Episodic memory integration with HTA nodes
- Memory snapshots for state preservation

These components help create a personalized experience that leverages
the user's journey history and contextual preferences.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from datetime import datetime, timezone, timedelta
from sqlalchemy import select, desc, func

from forest_app.core.cache_service import CacheService, cacheable
from forest_app.core.session_manager import SessionManager
from forest_app.core.transaction_decorator import transaction_protected
from forest_app.core.snapshot import MemorySnapshot
from forest_app.persistence.models import MemorySnapshotModel, HTANodeModel

logger = logging.getLogger(__name__)


class HTAMemoryManager:
    """Manages semantic and episodic memory operations for the Enhanced HTA service.
    
    This component handles memory storage, retrieval, and context integration,
    providing a way to personalize HTA nodes based on the user's history and preferences.
    """
    
    def __init__(self, session_manager: Optional[SessionManager] = None):
        """Initialize the memory manager with session management.
        
        Args:
            session_manager: Optional session manager for database operations
        """
        self.session_manager = session_manager or SessionManager.get_instance()
        self.cache = CacheService.get_instance()

    @transaction_protected()
    async def store_memory(self, user_id: UUID, memory_data: Dict[str, Any]) -> bool:
        """Store a new memory item for a user.
        
        This method creates a persistent memory snapshot that preserves the user's
        current context, preferences, and journey state. This enables a more
        personalized and continuous experience across sessions.
        
        Args:
            user_id: The UUID of the user
            memory_data: Dictionary containing memory content
            
        Returns:
            Boolean indicating success
        """
        try:
            # Create a new memory snapshot model
            memory_snapshot = MemorySnapshotModel(
                id=UUID(memory_data.get('snapshot_id', str(uuid.uuid4()))),
                user_id=user_id,
                snapshot_type=memory_data.get('snapshot_type', 'user_state'),
                timestamp=memory_data.get('timestamp', datetime.now(timezone.utc)),
                content=memory_data,
                tags=memory_data.get('tags', []),
                metadata=memory_data.get('metadata', {})
            )
            
            # Persist to database using session manager
            async with self.session_manager.session() as session:
                session.add(memory_snapshot)
                await session.commit()
            
            # Invalidate any cached snapshots
            cache_key = f"memory:user:{user_id}:latest"
            await self.cache.delete(cache_key)
            
            logger.info(f"Stored memory snapshot {memory_snapshot.id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store memory for user {user_id}: {e}")
            return False
    
    @cacheable(key_pattern="memory:user:{0}:latest", ttl=300)
    async def get_latest_snapshot(self, user_id: UUID) -> Optional[MemorySnapshot]:
        """Retrieve the latest memory snapshot for a user with caching.
        
        This method retrieves the most recent memory snapshot for a user, enabling
        personalized context-aware operations. Results are cached for performance.
        
        Args:
            user_id: The UUID of the user
            
        Returns:
            Optional MemorySnapshot containing user's latest memory context
        """
        try:
            async with self.session_manager.session() as session:
                # Query for the latest snapshot
                query = (
                    select(MemorySnapshotModel)
                    .where(MemorySnapshotModel.user_id == user_id)
                    .order_by(desc(MemorySnapshotModel.timestamp))
                    .limit(1)
                )
                
                result = await session.execute(query)
                model = result.scalars().first()
                
                if not model:
                    logger.info(f"No memory snapshots found for user {user_id}")
                    return None
                    
                # Convert to domain model
                snapshot = MemorySnapshot(
                    id=model.id,
                    user_id=model.user_id,
                    timestamp=model.timestamp,
                    content=model.content,
                    snapshot_type=model.snapshot_type,
                    tags=model.tags,
                    metadata=model.metadata
                )
                
                logger.debug(f"Retrieved memory snapshot {snapshot.id} for user {user_id}")
                return snapshot
                
        except Exception as e:
            logger.error(f"Error retrieving memory snapshot for user {user_id}: {e}")
            return None
            
    async def update_memory_with_completion(self, user_id: UUID, node: HTANodeModel) -> bool:
        """Update user memory with details of a completed HTA node.
        
        This method captures task completion data into the user's memory context,
        enhancing future personalization through their journey history.
        
        Args:
            user_id: UUID of the user
            node: HTANodeModel that was just completed
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get existing memory snapshot or create a new one
            current_snapshot = await self.get_latest_snapshot(user_id)
            
            # Build completion memory data
            completion_data = {
                "snapshot_type": "task_completion",
                "timestamp": datetime.now(timezone.utc),
                "content": {
                    "completed_node": {
                        "id": str(node.id),
                        "title": node.title,
                        "tree_id": str(node.tree_id),
                        "completed_at": datetime.now(timezone.utc).isoformat(),
                        "is_major_phase": getattr(node, 'is_major_phase', False)
                    },
                    "previous_state": current_snapshot.content if current_snapshot else {}
                },
                "tags": ["task_completion", node.title.lower().replace(" ", "_")[:20]],
                "metadata": {
                    "tree_id": str(node.tree_id),
                    "node_type": "major_phase" if getattr(node, 'is_major_phase', False) else "task"
                }
            }
            
            # Store the updated memory
            success = await self.store_memory(user_id, completion_data)
            
            if success:
                logger.info(f"Updated memory for user {user_id} with completion of node {node.id}")
            else:
                logger.warning(f"Failed to update memory for user {user_id} with node {node.id} completion")
                
            return success
            
        except Exception as e:
            logger.error(f"Error updating memory with completion for user {user_id}: {e}")
            return False
            
    @cacheable(key_pattern="memory:user:{0}:completion_count", ttl=600)
    async def get_user_completion_stats(self, user_id: UUID) -> Dict[str, Any]:
        """Get statistics on user's task completion history.
        
        Retrieves analytics about the user's journey to provide insights for
        personalization and user experience enhancement.
        
        Args:
            user_id: UUID of the user
            
        Returns:
            Dictionary with completion statistics
        """
        try:
            async with self.session_manager.session() as session:
                # Get total completed nodes
                completed_query = (
                    select(func.count())
                    .where(
                        (HTANodeModel.user_id == user_id) &
                        (HTANodeModel.status == "completed")
                    )
                )
                result = await session.execute(completed_query)
                completed_count = result.scalar_one() or 0
                
                # Get major milestones completed
                milestone_query = (
                    select(func.count())
                    .where(
                        (HTANodeModel.user_id == user_id) &
                        (HTANodeModel.status == "completed") &
                        (HTANodeModel.hta_metadata['is_major_phase'].as_boolean() == True)
                    )
                )
                result = await session.execute(milestone_query)
                milestone_count = result.scalar_one() or 0
                
                # Get count of recent completions (last 7 days)
                recent_query = (
                    select(func.count())
                    .where(
                        (HTANodeModel.user_id == user_id) &
                        (HTANodeModel.status == "completed") &
                        (HTANodeModel.updated_at >= datetime.now(timezone.utc) - timedelta(days=7))
                    )
                )
                result = await session.execute(recent_query)
                recent_count = result.scalar_one() or 0
                
                return {
                    "total_completed": completed_count,
                    "major_milestones": milestone_count,
                    "recent_completions": recent_count,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error retrieving completion stats for user {user_id}: {e}")
            return {
                "total_completed": 0,
                "major_milestones": 0,
                "recent_completions": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }
