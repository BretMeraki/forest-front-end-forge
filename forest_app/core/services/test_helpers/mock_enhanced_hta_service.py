"""
Mock implementation of EnhancedHTAService for testing

This module provides a simplified version of the EnhancedHTAService
that can be used in unit tests without requiring all dependencies.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable, Awaitable
from uuid import UUID

from forest_app.core.roadmap_models import RoadmapManifest
from forest_app.persistence.models import HTATreeModel, HTANodeModel, TaskFootprintModel
from forest_app.core.schema_contract import HTASchemaContract

logger = logging.getLogger(__name__)

# Mock class for memory manager component
class MockHTAMemoryManager:
    """Mock implementation of the HTAMemoryManager for testing."""
    
    def __init__(self, semantic_memory_manager=None):
        self.semantic_memory_manager = semantic_memory_manager
        self.memory_snapshots = {}
        
    async def store_memory(self, user_id: UUID, memory_data: Dict[str, Any]) -> bool:
        """Store memory data for a user."""
        self.memory_snapshots[str(user_id)] = memory_data
        return True
        
    async def get_snapshot(self, user_id: UUID) -> Dict[str, Any]:
        """Get memory snapshot for a user."""
        return self.memory_snapshots.get(str(user_id), {})
        
    async def update_memory_with_task_completion(self, user_id: UUID, node_id: UUID, 
                                             completion_data: Dict[str, Any]) -> bool:
        """Update memory with task completion data."""
        return True


# Mock class for reinforcement manager component
class MockReinforcementManager:
    """Mock implementation of the ReinforcementManager for testing."""
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        
    async def generate_reinforcement(self, node: HTANodeModel, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate positive reinforcement for completing a task."""
        return f"Great job completing the task: {node.title}!"


# Mock class for event manager component
class MockEventManager:
    """Mock implementation of the EventManager for testing."""
    
    def __init__(self):
        self.event_handlers = {}
        self.published_events = []
        
    async def publish_event(self, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Publish an event to the event bus."""
        self.published_events.append((event_type, event_data))
        return True
        
    async def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> bool:
        """Subscribe to an event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        return True


# Mock class for background task manager component
class MockBackgroundTaskManager:
    """Mock implementation of the BackgroundTaskManager for testing."""
    
    def __init__(self):
        self.tasks = []
        
    async def enqueue_task(self, task_func: Callable[..., Awaitable[Any]], *args, **kwargs) -> bool:
        """Enqueue a task for background processing."""
        self.tasks.append((task_func, args, kwargs))
        return True


class MockEnhancedHTAService:
    """
    Mock implementation of the EnhancedHTAService for testing.
    
    This implementation provides a simplified interface that satisfies
    the contract expected by tests but doesn't require all the external
    dependencies that the real service needs.
    """
    
    def __init__(self, llm_client=None, semantic_memory_manager=None, session_manager=None):
        """Initialize the mock service with optional dependencies."""
        self.llm_client = llm_client
        self.semantic_memory_manager = semantic_memory_manager
        self.session_manager = session_manager
        self.trees = {}  # In-memory store for trees
        
        # Initialize modular components
        self.memory_manager = MockHTAMemoryManager(semantic_memory_manager)
        self.reinforcement_manager = MockReinforcementManager(llm_client)
        self.event_manager = MockEventManager()
        self.background_manager = MockBackgroundTaskManager()
    
    async def generate_initial_hta_from_manifest(self, manifest: RoadmapManifest) -> HTATreeModel:
        """
        Generate a simple HTA tree from a manifest for testing.
        
        This mock implementation doesn't need all the parameters that the real
        implementation requires, making tests simpler to write.
        
        Args:
            manifest: The RoadmapManifest to generate a tree from
            
        Returns:
            HTATreeModel: A simple tree model for testing
        """
        logger.info(f"Mock: Generating initial HTA from manifest for testing")
        
        # Create a simple tree model
        tree_id = manifest.tree_id if hasattr(manifest, 'tree_id') else uuid.uuid4()
        user_id = manifest.user_id if hasattr(manifest, 'user_id') else uuid.uuid4()
        
        # Create top node
        top_node = HTANodeModel(
            id=uuid.uuid4(),
            tree_id=tree_id,
            user_id=user_id,
            title=f"Goal: {manifest.user_goal}",
            description="Top level goal node",
            status="pending",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            is_major_phase=True,
            internal_task_details={"from_manifest": True}
        )
        
        # Create tree model
        tree_model = HTATreeModel(
            id=tree_id,
            user_id=user_id,
            manifest={} if not hasattr(manifest, 'dict') else manifest.dict(),
            top_node_id=top_node.id,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        # Store in our mock database
        self.trees[str(tree_id)] = {
            "tree": tree_model,
            "nodes": {str(top_node.id): top_node}
        }
        
        return tree_model

    async def complete_node(self, node_id: UUID, user_id: UUID, 
                     completion_time: Optional[datetime] = None) -> TaskFootprintModel:
        """
        Mark a node as completed and trigger memory updates.
        
        Args:
            node_id: The ID of the node to complete
            user_id: The ID of the user completing the node
            completion_time: Optional completion timestamp
            
        Returns:
            TaskFootprintModel: A record of the task completion
        """
        logger.info(f"Mock: Completing node {node_id} for user {user_id}")
        
        # Find the node in our mock database
        node = None
        for tree_data in self.trees.values():
            if str(node_id) in tree_data["nodes"]:
                node = tree_data["nodes"][str(node_id)]
                break
                
        if not node:
            # Create a dummy node if not found
            node = HTANodeModel(
                id=node_id,
                user_id=user_id,
                tree_id=uuid.uuid4(),
                title="Mock node",
                description="Mock node description",
                status="pending"
            )
            
        # Mark as completed
        node.status = "completed"
        node.completed_at = completion_time or datetime.now(timezone.utc)
        
        # Generate reinforcement message
        reinforcement = await self.reinforcement_manager.generate_reinforcement(node)
        
        # Create footprint
        footprint = TaskFootprintModel(
            id=uuid.uuid4(),
            user_id=user_id,
            node_id=node_id,
            completed_at=node.completed_at,
            reinforcement_message=reinforcement
        )
        
        # Publish event
        await self.event_manager.publish_event(
            "task_completed",
            {"node_id": node_id, "user_id": user_id, "footprint": footprint}
        )
        
        # Update memory
        await self.memory_manager.update_memory_with_task_completion(
            user_id, 
            node_id, 
            {"node_title": node.title, "completed_at": node.completed_at}
        )
        
        return footprint


# Helper function to get a mock service instance for tests
def get_mock_enhanced_hta_service(llm_client=None, semantic_memory_manager=None, session_manager=None):
    """
    Get a configured mock enhanced HTA service for testing.
    
    Args:
        llm_client: Optional mock LLM client
        semantic_memory_manager: Optional mock semantic memory manager
        session_manager: Optional mock session manager
        
    Returns:
        MockEnhancedHTAService: A configured mock service
    """
    return MockEnhancedHTAService(
        llm_client=llm_client,
        semantic_memory_manager=semantic_memory_manager,
        session_manager=session_manager
    )
