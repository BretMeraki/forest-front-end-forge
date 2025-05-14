"""
Mock implementation of ContextInfusedNodeGenerator for testing

This module provides a simplified version of the ContextInfusedNodeGenerator
that can be used in unit tests without requiring LLM or database dependencies.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from uuid import UUID

from forest_app.persistence.models import HTANodeModel

logger = logging.getLogger(__name__)

class MockContextInfusedNodeGenerator:
    """
    Mock implementation of the ContextInfusedNodeGenerator for testing.
    
    This implementation provides a simplified interface that satisfies
    the contract expected by tests but doesn't require LLM or database access.
    """
    
    def __init__(self, llm_client=None, memory_service=None, session_manager=None):
        """Initialize the mock generator with optional dependencies."""
        self.llm_client = llm_client
        self.memory_service = memory_service
        self.session_manager = session_manager
    
    async def generate_trunk_node(self, tree_id: UUID, user_id: UUID, 
                               user_goal: str, memory_snapshot: Optional[Dict] = None) -> HTANodeModel:
        """
        Generate a mock trunk node for testing.
        
        Args:
            tree_id: UUID of the tree
            user_id: UUID of the user
            user_goal: User's goal statement
            memory_snapshot: Optional memory snapshot for context
            
        Returns:
            HTANodeModel: A simple trunk node for testing
        """
        logger.info(f"Mock: Generating trunk node for testing")
        
        # Create a personalized title based on user goal and preferences
        title = f"Personalized plan for: {user_goal}"
        
        # Include user preferences from memory snapshot if available
        preferences = ""
        if memory_snapshot and isinstance(memory_snapshot, dict):
            user_prefs = memory_snapshot.get("user_preferences", {})
            if user_prefs and isinstance(user_prefs, dict):
                pref_time = user_prefs.get("preferred_time")
                if pref_time:
                    preferences = f" (optimized for {pref_time})"
        
        # Create a mock trunk node
        return HTANodeModel(
            id=uuid.uuid4(),
            tree_id=tree_id,
            user_id=user_id,
            title=title + preferences,
            description=f"Achieve {user_goal} with a structured approach",
            status="pending",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            is_major_phase=True,
            internal_task_details={
                "priority_level": "high",
                "estimated_duration": "medium",
                "joy_factor": 0.8
            }
        )
    
    async def generate_branch_from_parent(self, parent_node: HTANodeModel, 
                                      memory_snapshot: Optional[Dict] = None) -> List[HTANodeModel]:
        """
        Generate mock branch nodes from a parent node for testing.
        
        Args:
            parent_node: The parent node to generate branches from
            memory_snapshot: Optional memory snapshot for context
            
        Returns:
            List[HTANodeModel]: A list of simple branch nodes for testing
        """
        logger.info(f"Mock: Generating branch nodes for testing")
        
        # Create 2-3 mock branch nodes with personalized content
        branches = []
        branch_titles = [
            "Get started with the basics",
            "Build a regular practice routine",
            "Track your progress"
        ]
        
        for i, title in enumerate(branch_titles):
            branch = HTANodeModel(
                id=uuid.uuid4(),
                tree_id=parent_node.tree_id,
                user_id=parent_node.user_id,
                parent_id=parent_node.id,
                title=title,
                description=f"Step {i+1}: {title}",
                status="pending",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                is_major_phase=False,
                internal_task_details={
                    "priority_level": "medium",
                    "estimated_duration": "short",
                    "joy_factor": 0.7
                }
            )
            branches.append(branch)
        
        return branches
    
    async def generate_micro_actions(self, branch_node: HTANodeModel, count: int = 3) -> List[HTANodeModel]:
        """
        Generate mock micro-actions for testing.
        
        Args:
            branch_node: The parent branch node
            count: Number of micro-actions to generate
            
        Returns:
            List[HTANodeModel]: A list of simple micro-action nodes for testing
        """
        logger.info(f"Mock: Generating {count} micro-actions for testing")
        
        # Create mock micro-actions
        micro_actions = []
        
        for i in range(count):
            action = HTANodeModel(
                id=uuid.uuid4(),
                tree_id=branch_node.tree_id,
                user_id=branch_node.user_id,
                parent_id=branch_node.id,
                title=f"Action {i+1}: Quick win to build momentum",
                description=f"A small, achievable action to make progress",
                status="pending",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                is_leaf=True,
                internal_task_details={
                    "priority_level": "high",
                    "estimated_duration": "very_short",
                    "joy_factor": 0.9,
                    "positive_reinforcement": "You'll feel great after completing this!"
                }
            )
            micro_actions.append(action)
        
        return micro_actions

# Helper function to get a mock generator instance for tests
def get_mock_node_generator(llm_client=None, memory_service=None, session_manager=None):
    """
    Get a configured mock node generator for testing.
    
    Args:
        llm_client: Optional mock LLM client
        memory_service: Optional mock memory service
        session_manager: Optional mock session manager
        
    Returns:
        MockContextInfusedNodeGenerator: A configured mock generator
    """
    return MockContextInfusedNodeGenerator(
        llm_client=llm_client,
        memory_service=memory_service,
        session_manager=session_manager
    )
