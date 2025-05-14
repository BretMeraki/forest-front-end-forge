"""
Core logic for Enhanced HTA operations.
"""

"""Core logic for Enhanced HTA operations.

This module provides the main Enhanced HTA Service class which serves as the
central orchestrator for the modular components of the service.

It integrates the various specialized modules including:
- Memory management
- Reinforcement generation
- Event handling
- Background task processing
- Utility functions

This orchestrator approach maintains a clean separation of concerns while
providing a unified interface for the application to interact with.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Awaitable
from datetime import datetime, timezone
from uuid import UUID

# Import base service for extension
from forest_app.core.services.hta_service import HTAService
from forest_app.core.circuit_breaker import circuit_protected, CircuitBreaker, CircuitBreakerConfig
from forest_app.core.event_bus import EventType, EventData
from forest_app.core.transaction_decorator import transaction_protected

# Import our modular components
from forest_app.core.services.enhanced_hta.memory import HTAMemoryManager
from forest_app.core.services.enhanced_hta.reinforcement import ReinforcementManager
from forest_app.core.services.enhanced_hta.events import EventManager
from forest_app.core.services.enhanced_hta.background import BackgroundTaskManager
from forest_app.core.services.enhanced_hta.utils import Result, format_uuid

# Import framework components
from forest_app.core.context_infused_generator import ContextInfusedNodeGenerator
from forest_app.persistence.hta_tree_repository import HTATreeRepository

# Import core domain models
from forest_app.core.roadmap_models import RoadmapManifest
from forest_app.core.snapshot import MemorySnapshot
from forest_app.modules.hta_tree import HTATree
from forest_app.persistence.models import HTANodeModel, HTATreeModel
from forest_app.integrations.llm import (
    LLMClient,
    LLMError,
    LLMValidationError
)

logger = logging.getLogger(__name__)

class EnhancedHTAService(HTAService):
    """
    Enhanced HTAService with a modular, clean architecture optimized for maintainability.
    
    This service extends the base HTAService with specialized components:
    - Memory Manager: Handles semantic and episodic memory operations
    - Reinforcement Manager: Generates personalized positive feedback
    - Event Manager: Centralizes event handling and cache invalidation
    - Background Task Manager: Processes intensive operations asynchronously
    
    These modular components maintain the intimate, personal experience that makes
    The Forest special, while providing a clean, maintainable architecture.
    """
    
    def __init__(self, llm_client, semantic_memory_manager, session_manager=None):
        # ... existing constructor code ...
        super().__init__(llm_client, semantic_memory_manager)
        self.llm_client = llm_client
        self.semantic_memory_manager = semantic_memory_manager
        self.memory_manager = HTAMemoryManager(session_manager)
        self.reinforcement_manager = ReinforcementManager(llm_client)
        self.event_manager = EventManager()
        self.background_manager = BackgroundTaskManager()
        self.node_generator = ContextInfusedNodeGenerator(
            llm_client=llm_client,
            memory_service=semantic_memory_manager,
            session_manager=session_manager
        )
        self.tree_repository = HTATreeRepository(session_manager)
        self.llm_circuit = CircuitBreaker(
            CircuitBreakerConfig(
                name="llm_service",
                failure_threshold=3,
                recovery_timeout=60,
                expected_exceptions=[LLMError, LLMValidationError, asyncio.TimeoutError],
                fallback_function=self._llm_fallback
            )
        )
        logger.info("EnhancedHTAService initialized with modular components")

    async def generate_initial_hta_from_manifest(self, manifest, user_id, request_context):
        """
        Generate and persist an initial HTA tree from a RoadmapManifest.
        Args:
            manifest: RoadmapManifest object
            user_id: UUID of the user
            request_context: dict for request metadata
        Returns:
            HTATreeModel instance or compatible object with 'id' and 'user_id'
        """
        # Create the HTATreeModel instance
        tree_model = HTATreeModel(
            id=manifest.tree_id,
            user_id=user_id,
            goal_name=getattr(manifest, 'user_goal', "HTA Tree"),  # Using goal_name instead of title
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
            # Removed status field as it's not in the model
        )
        # Persist the tree to the database using the repository
        await self.tree_repository.save_tree_model(tree_model)
        # Optionally, create an audit log or similar if required by your domain
        # Publish an event to the event bus if present
        if hasattr(self, 'event_bus') and self.event_bus:
            # Convert UUID to string and ensure event_type is provided correctly
            try:
                await self.event_bus.publish(
                    EventType.TREE_EVOLVED,  # Use TREE_EVOLVED since we don't have TREE_CREATED
                    EventData(
                        event_type=EventType.TREE_EVOLVED,  # Must match a valid EventType enum value
                        user_id=str(user_id),  # Convert UUID to string
                        payload={
                            "tree_id": str(manifest.tree_id),
                            "action": "created",  # To indicate this is a creation event
                            "context": request_context
                        }
                    )
                )
            except Exception as e:
                # Log but don't let event publishing failure block the tree creation
                print(f"WARNING: Failed to publish event: {e}")
                # Continue execution even if event publishing fails
        return tree_model

        """
        Initialize the enhanced HTA service with modular components.
        
        Args:
            llm_client: LLM client for generating content
            semantic_memory_manager: Manager for semantic memory operations
            session_manager: Optional SessionManager for database operations
        """
        # Initialize base service
        super().__init__(llm_client, semantic_memory_manager)
        
        # Store core dependencies
        self.llm_client = llm_client
        self.semantic_memory_manager = semantic_memory_manager
        
        # Initialize our modular components
        self.memory_manager = HTAMemoryManager(session_manager)
        self.reinforcement_manager = ReinforcementManager(llm_client)
        self.event_manager = EventManager()
        self.background_manager = BackgroundTaskManager()
        
        # Initialize our framework components
        self.node_generator = ContextInfusedNodeGenerator(
            llm_client=llm_client,
            memory_service=semantic_memory_manager,
            session_manager=session_manager
        )
        
        self.tree_repository = HTATreeRepository(session_manager)
        
        # Set up circuit breakers for external services
        self.llm_circuit = CircuitBreaker(
            CircuitBreakerConfig(
                name="llm_service",
                failure_threshold=3,
                recovery_timeout=60,
                expected_exceptions=[LLMError, LLMValidationError, asyncio.TimeoutError],
                fallback_function=self._llm_fallback
            )
        )
        
        logger.info("EnhancedHTAService initialized with modular components")
    
    @transaction_protected()
    async def complete_node(self, node_id: UUID, user_id: UUID):
        """Mark a node as complete, update memory, and trigger positive reinforcement.
        
        This method orchestrates the task completion process by delegating to the
        appropriate specialized components:
        1. Update the node status in the repository
        2. Update the memory with completion details
        3. Generate positive reinforcement message
        4. Publish the completion event
        5. Schedule background expansion if needed
        
        Args:
            node_id: UUID of the node to complete
            user_id: UUID of the user completing the node
            
        Returns:
            Dictionary with completion results, including positive reinforcement message
        """
        logger.info(f"Processing node completion for node {node_id} by user {user_id}")
        
        # Get the node and validate ownership
        node = await self.tree_repository.get_node_by_id(node_id)
        if not node:
            logger.error(f"Node {node_id} not found")
            raise ValueError(f"Node {node_id} not found")
            
        if node.user_id != user_id:
            logger.error(f"User {user_id} does not own node {node_id}")
            raise ValueError(f"User {user_id} does not own node {node_id}")
            
        if node.status == "completed":
            logger.info(f"Node {node_id} already completed")
            return {
                "status": "already_completed",
                "message": "This task is already completed."
            }
            
        # Get tree for manifest update
        tree = await self.tree_repository.get_tree_by_id(node.tree_id)
        if not tree:
            logger.error(f"Tree {node.tree_id} not found")
            raise ValueError(f"Tree {node.tree_id} not found")
            
        # Update node status to completed using the repository
        success = await self.tree_repository.update_node_status(
            node_id=node_id,
            new_status="completed",
            update_internal_details={
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "completion_context": {
                    "completed_by": str(user_id),
                    "completion_timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
        if not success:
            logger.error(f"Failed to update node {node_id} status")
            raise RuntimeError(f"Failed to update node {node_id} status")
            
        # Update the manifest to keep it synchronized
        manifest = RoadmapManifest(**tree.manifest)
        if node.roadmap_step_id:
            manifest = manifest.update_step_status(node.roadmap_step_id, "completed")
            
            # Save updated manifest to tree
            tree.manifest = manifest.dict()
            await self.tree_repository.update_tree(tree)
        
        # Update memory with completion using the memory manager
        await self.memory_manager.update_memory_with_completion(user_id, node)
        
        # Check if parent node should be updated with completion count
        if node.parent_id:
            increment_success, new_count = await self.tree_repository.increment_branch_completion_count(node.parent_id)
            if increment_success:
                logger.info(f"Incremented completion count for parent {node.parent_id} to {new_count}")
        
        # Generate positive reinforcement message using the reinforcement manager
        reinforcement = await self.reinforcement_manager.generate_reinforcement(node)
        
        # Publish completion event using the event manager
        await self.event_manager.publish_event(
            event_type=EventType.TASK_COMPLETED,
            user_id=user_id,
            payload={
                "node_id": str(node_id),
                "tree_id": str(node.tree_id),
                "is_major_phase": getattr(node, 'is_major_phase', False)
            }
        )
        
        # Check if we need to expand any nodes based on completion triggers
        expand_nodes = await self.tree_repository.get_nodes_ready_for_expansion(node.tree_id)
        
        if expand_nodes:
            # Schedule background expansion using the background manager
            await self.background_manager.enqueue_task(
                self.background_manager.expand_nodes_in_background,
                expand_nodes, user_id,
                priority=3,  # Medium priority
                metadata={"type": "node_expansion", "user_id": str(user_id)}
            )

    def _llm_fallback(self, *args, **kwargs):
        """
        Fallback function when LLM service is unavailable.
        
        This provides a graceful degradation path to maintain the user experience
        even when external AI services are temporarily unavailable.
        
        Returns:
            Dictionary with status and fallback message
        """
        logger.warning("LLM service unavailable, using fallback.")
        return {"status": "unavailable", "message": "LLM service is temporarily unavailable."}
        
    @transaction_protected()
    async def save_tree(self, snapshot: MemorySnapshot, tree: HTATree) -> bool:
        """Save the HTA tree with transaction safety and event publication.
        
        This orchestrator method saves the tree data and coordinates events,
        delegating specialized operations to the appropriate components.
        
        Args:
            snapshot: The MemorySnapshot to update
            tree: The HTATree to save
            
        Returns:
            Boolean indicating success
        """
        try:
            # Save the tree using the base implementation
            success = await super().save_tree(snapshot, tree)
            
            if success and hasattr(tree, 'user_id'):
                # Publish event using the event manager
                await self.event_manager.publish_event(
                    event_type=EventType.TREE_UPDATED,
                    user_id=tree.user_id,
                    payload={
                        "tree_id": format_uuid(tree.id),
                        "node_count": len(tree.nodes) if hasattr(tree, 'nodes') else 0
                    }
                )
                
                # Schedule background check for meaningful moments
                if hasattr(tree, '_meaningful_transitions') and tree._meaningful_transitions:
                    await self.background_manager.enqueue_task(
                        self.background_manager.process_meaningful_moments,
                        tree, snapshot,
                        priority=3,  # Medium priority
                        metadata={"type": "meaningful_moments", "user_id": format_uuid(tree.user_id)}
                    )
            
            return success
            
        except Exception as e:
            logger.error(f"Error saving tree: {e}")
            return False
