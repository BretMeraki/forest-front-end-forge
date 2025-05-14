"""
Mock implementation of HTATreeRepository for testing

This module provides a simplified version of the HTATreeRepository
that can be used in unit tests without requiring database access.
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from uuid import UUID

from forest_app.persistence.models import HTATreeModel, HTANodeModel

logger = logging.getLogger(__name__)

class MockHTATreeRepository:
    """
    Mock implementation of the HTATreeRepository for testing.
    
    This implementation provides an in-memory storage solution that
    satisfies the contract expected by tests but doesn't require database access.
    """
    
    def __init__(self, session_manager=None):
        """Initialize the mock repository with optional dependencies."""
        self.session_manager = session_manager
        self.trees = {}  # Dict[tree_id, tree_model]
        self.nodes = {}  # Dict[node_id, node_model]
    
    async def create_tree(self, user_id: UUID, manifest: Dict[str, Any],
                       goal_name: str, initial_context: Optional[str] = None) -> HTATreeModel:
        """
        Create a mock tree in memory for testing.
        
        Args:
            user_id: UUID of the user
            manifest: Dictionary representation of the manifest
            goal_name: Name of the goal
            initial_context: Optional initial context
            
        Returns:
            HTATreeModel: The created tree model
        """
        logger.info(f"Mock: Creating tree for user {user_id}")
        
        # Create a mock tree
        tree_id = uuid.uuid4()
        tree_model = HTATreeModel(
            id=tree_id,
            user_id=user_id,
            manifest=manifest,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        # Store in our mock database
        self.trees[str(tree_id)] = tree_model
        
        return tree_model
    
    async def add_node(self, node: HTANodeModel) -> HTANodeModel:
        """
        Add a node to the mock repository.
        
        Args:
            node: The node to add
            
        Returns:
            HTANodeModel: The added node
        """
        logger.info(f"Mock: Adding node {node.id} to tree {node.tree_id}")
        
        # Store the node
        self.nodes[str(node.id)] = node
        
        return node
    
    async def add_nodes(self, nodes: List[HTANodeModel]) -> List[HTANodeModel]:
        """
        Add multiple nodes to the mock repository.
        
        Args:
            nodes: The list of nodes to add
            
        Returns:
            List[HTANodeModel]: The list of added nodes
        """
        logger.info(f"Mock: Adding {len(nodes)} nodes")
        
        # Store all nodes
        for node in nodes:
            self.nodes[str(node.id)] = node
        
        return nodes
    
    async def add_nodes_bulk(self, nodes: List[HTANodeModel]) -> List[str]:
        """
        Add multiple nodes in bulk to the mock repository.
        
        Args:
            nodes: The list of nodes to add
            
        Returns:
            List[str]: The list of added node IDs
        """
        logger.info(f"Mock: Bulk adding {len(nodes)} nodes")
        
        # Store all nodes and collect IDs
        node_ids = []
        
        for node in nodes:
            self.nodes[str(node.id)] = node
            node_ids.append(str(node.id))
        
        return node_ids
    
    async def update_tree(self, tree: HTATreeModel) -> HTATreeModel:
        """
        Update a tree in the mock repository.
        
        Args:
            tree: The updated tree model
            
        Returns:
            HTATreeModel: The updated tree model
        """
        logger.info(f"Mock: Updating tree {tree.id}")
        
        # Update the tree
        tree.updated_at = datetime.now(timezone.utc)
        self.trees[str(tree.id)] = tree
        
        return tree
    
    async def get_tree(self, tree_id: UUID) -> Optional[HTATreeModel]:
        """
        Get a tree from the mock repository.
        
        Args:
            tree_id: UUID of the tree to get
            
        Returns:
            HTATreeModel: The tree model, or None if not found
        """
        logger.info(f"Mock: Getting tree {tree_id}")
        
        # Get the tree if it exists
        return self.trees.get(str(tree_id))
    
    async def get_node(self, node_id: UUID) -> Optional[HTANodeModel]:
        """
        Get a node from the mock repository.
        
        Args:
            node_id: UUID of the node to get
            
        Returns:
            HTANodeModel: The node model, or None if not found
        """
        logger.info(f"Mock: Getting node {node_id}")
        
        # Get the node if it exists
        return self.nodes.get(str(node_id))
    
    async def get_nodes_for_tree(self, tree_id: UUID) -> List[HTANodeModel]:
        """
        Get all nodes for a tree from the mock repository.
        
        Args:
            tree_id: UUID of the tree
            
        Returns:
            List[HTANodeModel]: The list of nodes for the tree
        """
        logger.info(f"Mock: Getting all nodes for tree {tree_id}")
        
        # Filter nodes by tree_id
        tree_nodes = [
            node for node in self.nodes.values()
            if str(node.tree_id) == str(tree_id)
        ]
        
        return tree_nodes

# Helper function to get a mock repository instance for tests
def get_mock_tree_repository(session_manager=None):
    """
    Get a configured mock tree repository for testing.
    
    Args:
        session_manager: Optional mock session manager
        
    Returns:
        MockHTATreeRepository: A configured mock repository
    """
    return MockHTATreeRepository(session_manager=session_manager)
