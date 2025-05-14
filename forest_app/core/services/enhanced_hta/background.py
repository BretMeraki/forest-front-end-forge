"""Background and asynchronous task processing for Enhanced HTA Service.

This module provides functionality for:
- Processing intensive operations in the background
- Handling meaningful moment detection
- Scheduling recurring analysis and reflection tasks
- Managing task priorities and execution

These components ensure the application stays responsive while still
providing rich, computationally intensive features.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from uuid import UUID
from datetime import datetime, timezone
import asyncio

from forest_app.core.task_queue import TaskQueue
from forest_app.core.transaction_decorator import transaction_protected
from forest_app.modules.hta_tree import HTATree
from forest_app.core.snapshot import MemorySnapshot

logger = logging.getLogger(__name__)


class BackgroundTaskManager:
    """Manages background task processing for the Enhanced HTA service.
    
    This component handles the execution of computationally intensive or non-blocking
    operations, ensuring the main application flow remains responsive while still
    providing rich features that might require significant processing.
    """
    
    def __init__(self):
        """Initialize the background task manager."""
        self.task_queue = TaskQueue.get_instance()
    
    async def enqueue_task(self, 
                        task_func: Callable[..., Awaitable[Any]], 
                        *args, 
                        priority: int = 5,
                        metadata: Optional[Dict[str, Any]] = None,
                        **kwargs) -> bool:
        """Enqueue a task for background processing.
        
        Args:
            task_func: The async function to execute
            *args: Positional arguments for the task function
            priority: Task priority (1-10, lower is higher priority)
            metadata: Optional metadata for tracking and logging
            **kwargs: Keyword arguments for the task function
            
        Returns:
            Boolean indicating if the task was successfully queued
        """
        try:
            await self.task_queue.enqueue(
                task_func,
                *args,
                priority=priority,
                metadata=metadata or {},
                **kwargs
            )
            return True
        except Exception as e:
            logger.error(f"Error enqueueing background task: {e}")
            return False
    
    @transaction_protected()
    async def process_meaningful_moments(self, tree: HTATree, snapshot: MemorySnapshot) -> bool:
        """Process meaningful moments detected in the HTA tree.
        
        This background operation analyzes transitions and milestones in the user's
        journey, generating insights and potential adaptations for future tasks.
        
        Args:
            tree: The HTATree containing meaningful transitions
            snapshot: The user's memory snapshot for context
            
        Returns:
            Boolean indicating processing success
        """
        try:
            if not hasattr(tree, '_meaningful_transitions') or not tree._meaningful_transitions:
                logger.debug("No meaningful transitions to process")
                return True
                
            user_id = getattr(tree.root, 'user_id', None) if tree.root else None
            if not user_id:
                logger.warning("Cannot process meaningful moments: missing user_id")
                return False
                
            logger.info(f"Processing {len(tree._meaningful_transitions)} meaningful moments for user {user_id}")
            
            # Group transitions by type for more efficient processing
            grouped_transitions = {}
            for transition in tree._meaningful_transitions:
                transition_type = transition.get('type', 'unknown')
                if transition_type not in grouped_transitions:
                    grouped_transitions[transition_type] = []
                grouped_transitions[transition_type].append(transition)
                
            # Process each type of transition
            for transition_type, transitions in grouped_transitions.items():
                if transition_type == 'completion_streak':
                    await self._process_completion_streak(user_id, transitions, snapshot)
                elif transition_type == 'milestone':
                    await self._process_milestone_reached(user_id, transitions, snapshot)
                elif transition_type == 'pattern':
                    await self._process_pattern_detected(user_id, transitions, snapshot)
                else:
                    logger.warning(f"Unknown transition type: {transition_type}")
                    
            return True
            
        except Exception as e:
            logger.error(f"Error processing meaningful moments: {e}")
            return False
            
    async def _process_completion_streak(self, user_id: UUID, transitions: List[Dict[str, Any]], snapshot: MemorySnapshot) -> None:
        """Process completion streak transitions.
        
        Args:
            user_id: The user's UUID
            transitions: List of completion streak transitions
            snapshot: The user's memory snapshot for context
        """
        try:
            for transition in transitions:
                streak_count = transition.get('streak_count', 0)
                if streak_count >= 3:
                    logger.info(f"User {user_id} has completed {streak_count} tasks in a row")
                    
                    # Here we would trigger reinforcement, notifications, or other responses
                    # based on the detected streak
                    
        except Exception as e:
            logger.error(f"Error processing completion streak for user {user_id}: {e}")
            
    async def _process_milestone_reached(self, user_id: UUID, transitions: List[Dict[str, Any]], snapshot: MemorySnapshot) -> None:
        """Process milestone completion transitions.
        
        Args:
            user_id: The user's UUID
            transitions: List of milestone transitions
            snapshot: The user's memory snapshot for context
        """
        try:
            for transition in transitions:
                milestone_id = transition.get('milestone_id')
                milestone_title = transition.get('title', 'Unknown milestone')
                
                logger.info(f"User {user_id} has completed milestone: {milestone_title}")
                
                # Here we would update recommendation models, trigger celebratory
                # notifications, or update long-term user journey metrics
                
        except Exception as e:
            logger.error(f"Error processing milestone for user {user_id}: {e}")
            
    async def _process_pattern_detected(self, user_id: UUID, transitions: List[Dict[str, Any]], snapshot: MemorySnapshot) -> None:
        """Process pattern detection transitions.
        
        Args:
            user_id: The user's UUID
            transitions: List of pattern transitions
            snapshot: The user's memory snapshot for context
        """
        try:
            for transition in transitions:
                pattern_type = transition.get('pattern_type', 'unknown')
                confidence = transition.get('confidence', 0.0)
                
                if confidence >= 0.75:  # Only process high-confidence patterns
                    logger.info(f"Detected {pattern_type} pattern for user {user_id} with {confidence:.2f} confidence")
                    
                    # Here we would adapt the user's journey based on the detected pattern
                    
        except Exception as e:
            logger.error(f"Error processing pattern for user {user_id}: {e}")
            
    async def expand_nodes_in_background(self, nodes: List, user_id: UUID) -> bool:
        """Expand nodes in the background based on completion triggers.
        
        Args:
            nodes: List of nodes to expand
            user_id: UUID of the user
            
        Returns:
            Boolean indicating expansion success
        """
        try:
            from forest_app.core.services.enhanced_hta.memory import HTAMemoryManager
            
            memory_manager = HTAMemoryManager()
            node_generator = self._get_node_generator()
            tree_repository = self._get_tree_repository()
            
            if not nodes:
                logger.debug(f"No nodes to expand for user {user_id}")
                return True
                
            logger.info(f"Expanding {len(nodes)} nodes in background for user {user_id}")
            
            expanded_count = 0
            for node in nodes:
                # Get latest memory snapshot for context
                memory_snapshot = await memory_manager.get_latest_snapshot(user_id)
                
                # Generate branch nodes
                branch_nodes = await node_generator.generate_branch_from_parent(
                    parent_node=node,
                    memory_snapshot=memory_snapshot
                )
                
                if branch_nodes:
                    # Add new nodes to the tree
                    branch_node_ids = await tree_repository.add_nodes_bulk(branch_nodes)
                    expanded_count += len(branch_nodes)
                    
                    # Update the parent node to mark expansion complete
                    await tree_repository.update_branch_triggers(
                        node_id=node.id,
                        new_triggers={
                            "expand_now": False,
                            "current_completion_count": 0,
                            "last_expanded_at": datetime.now(timezone.utc).isoformat()
                        }
                    )
            
            logger.info(f"Successfully expanded {expanded_count} nodes in {len(nodes)} branches")
            return True
            
        except Exception as e:
            logger.error(f"Error expanding nodes in background: {e}")
            return False
            
    def _get_node_generator(self):
        """Get the node generator for creating new nodes.
        
        This is abstracted to allow easier testing and mocking.
        
        Returns:
            NodeGenerator instance
        """
        # This would typically be injected or imported, but we're creating it here
        # to avoid circular imports
        from forest_app.modules.node_generator import NodeGenerator
        return NodeGenerator()
        
    def _get_tree_repository(self):
        """Get the tree repository for database operations.
        
        This is abstracted to allow easier testing and mocking.
        
        Returns:
            TreeRepository instance
        """
        # This would typically be injected or imported, but we're creating it here
        # to avoid circular imports
        from forest_app.persistence.repositories import HTATreeRepository
        return HTATreeRepository()
