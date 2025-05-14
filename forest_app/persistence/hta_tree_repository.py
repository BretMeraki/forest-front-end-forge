"""
HTATreeRepository

This module implements an optimized repository for HTA trees with performance
enhancements like bulk operations, efficient querying patterns, and denormalized
fields. It ensures that tree operations are fast and reliable even as the tree grows.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
from sqlalchemy import select, update, delete, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from forest_app.persistence.models import HTATreeModel, HTANodeModel, UserModel
from forest_app.core.session_manager import SessionManager

logger = logging.getLogger(__name__)

class HTATreeRepository:
    """
    Repository for HTA trees with optimizations for performance.
    """
    
    def __init__(self, session_manager: SessionManager):
        """
        Initialize the repository with a session manager.
        
        Args:
            session_manager: SessionManager for database operations
        """
        self.session_manager = session_manager
        
    async def create_tree(self, user_id: uuid.UUID, manifest: Dict[str, Any], 
                        goal_name: str, initial_context: Optional[str] = None) -> HTATreeModel:
        """
        Create a new tree with optimized structure.
        
        Args:
            user_id: UUID of the user
            manifest: Tree manifest data
            goal_name: Name of the tree's goal
            initial_context: Optional initial context for the tree
            
        Returns:
            HTATreeModel instance
        """
        async with self.session_manager.session() as session:
            tree = HTATreeModel(
                id=uuid.uuid4(),
                user_id=user_id,
                goal_name=goal_name,
                initial_context=initial_context,
                manifest=manifest,
                # Add additional fields for optimization
                initial_roadmap_depth=0,  # Will update after nodes are added
                initial_task_count=0,     # Will update after nodes are added
            )
            session.add(tree)
            await session.commit()
            await session.refresh(tree)
            
            logger.info(f"Created new HTA tree: {tree.id} for user: {user_id}")
            return tree
    
    async def update_tree(self, tree: HTATreeModel) -> HTATreeModel:
        """
        Update an existing tree.
        
        Args:
            tree: HTATreeModel to update
            
        Returns:
            Updated HTATreeModel instance
        """
        async with self.session_manager.session() as session:
            session.add(tree)
            await session.commit()
            await session.refresh(tree)
            
            logger.debug(f"Updated HTA tree: {tree.id}")
            return tree
    
    async def add_node(self, node: HTANodeModel) -> HTANodeModel:
        """
        Add a single node to a tree.
        
        Args:
            node: HTANodeModel to add
            
        Returns:
            Added HTANodeModel with ID
        """
        async with self.session_manager.session() as session:
            session.add(node)
            await session.commit()
            await session.refresh(node)
            
            # Update leaf status of parent if needed
            if node.parent_id:
                await self._update_parent_leaf_status(session, node.parent_id, is_leaf=False)
                
            logger.debug(f"Added HTA node: {node.id} to tree: {node.tree_id}")
            return node
    
    async def add_nodes_bulk(self, nodes: List[HTANodeModel]) -> List[uuid.UUID]:
        """
        Add multiple nodes in a single transaction for performance.
        
        Args:
            nodes: List of HTANodeModel instances to add
            
        Returns:
            List of added node IDs
        """
        if not nodes:
            return []
            
        async with self.session_manager.session() as session:
            # Add all nodes in a batch
            session.add_all(nodes)
            await session.commit()
            
            # Refresh to get IDs
            for node in nodes:
                await session.refresh(node)
                
            # Update leaf status of parents in one operation
            parent_ids = {node.parent_id for node in nodes if node.parent_id}
            if parent_ids:
                parent_update = update(HTANodeModel).where(
                    HTANodeModel.id.in_(parent_ids)
                ).values(is_leaf=False)
                await session.execute(parent_update)
                await session.commit()
                
            logger.info(f"Added {len(nodes)} nodes in bulk to tree: {nodes[0].tree_id if nodes else None}")
            return [node.id for node in nodes]
    
    async def get_tree_by_id(self, tree_id: uuid.UUID) -> Optional[HTATreeModel]:
        """
        Get a tree by its ID.
        
        Args:
            tree_id: UUID of the tree
            
        Returns:
            HTATreeModel instance or None if not found
        """
        async with self.session_manager.session() as session:
            stmt = select(HTATreeModel).where(HTATreeModel.id == tree_id)
            result = await session.execute(stmt)
            tree = result.scalars().first()
            
            if tree:
                logger.debug(f"Retrieved HTA tree: {tree_id}")
            else:
                logger.warning(f"HTA tree not found: {tree_id}")
                
            return tree
    
    async def get_tree_with_nodes(self, tree_id: uuid.UUID) -> Tuple[Optional[HTATreeModel], List[HTANodeModel]]:
        """
        Get a tree with all its nodes efficiently.
        Uses denormalized fields for fast filtering and sorting.
        
        Args:
            tree_id: UUID of the tree
            
        Returns:
            Tuple of (HTATreeModel, List[HTANodeModel]) or (None, []) if not found
        """
        async with self.session_manager.session() as session:
            # Get tree
            tree_stmt = select(HTATreeModel).where(HTATreeModel.id == tree_id)
            tree_result = await session.execute(tree_stmt)
            tree = tree_result.scalars().first()
            
            if not tree:
                logger.warning(f"HTA tree not found: {tree_id}")
                return None, []
            
            # Get nodes with a single query
            nodes_stmt = select(HTANodeModel).where(HTANodeModel.tree_id == tree_id)
            nodes_result = await session.execute(nodes_stmt)
            nodes = nodes_result.scalars().all()
            
            logger.debug(f"Retrieved HTA tree: {tree_id} with {len(nodes)} nodes")
            return tree, nodes
    
    async def get_node_by_id(self, node_id: uuid.UUID) -> Optional[HTANodeModel]:
        """
        Get a node by its ID.
        
        Args:
            node_id: UUID of the node
            
        Returns:
            HTANodeModel instance or None if not found
        """
        async with self.session_manager.session() as session:
            stmt = select(HTANodeModel).where(HTANodeModel.id == node_id)
            result = await session.execute(stmt)
            node = result.scalars().first()
            
            if node:
                logger.debug(f"Retrieved HTA node: {node_id}")
            else:
                logger.warning(f"HTA node not found: {node_id}")
                
            return node
    
    async def get_nodes_by_parent(self, parent_id: uuid.UUID) -> List[HTANodeModel]:
        """
        Get all child nodes for a parent.
        
        Args:
            parent_id: UUID of the parent node
            
        Returns:
            List of child HTANodeModel instances
        """
        async with self.session_manager.session() as session:
            stmt = select(HTANodeModel).where(HTANodeModel.parent_id == parent_id)
            result = await session.execute(stmt)
            nodes = result.scalars().all()
            
            logger.debug(f"Retrieved {len(nodes)} child nodes for parent: {parent_id}")
            return nodes
    
    async def get_nodes_by_tree(self, tree_id: uuid.UUID, 
                             status: Optional[str] = None,
                             is_major_phase: Optional[bool] = None) -> List[HTANodeModel]:
        """
        Get nodes for a tree, optionally filtered by status and/or major phase flag.
        Uses indexes for efficient querying.
        
        Args:
            tree_id: UUID of the tree
            status: Optional status filter
            is_major_phase: Optional major phase filter
            
        Returns:
            List of matching HTANodeModel instances
        """
        async with self.session_manager.session() as session:
            # Build query with conditionals
            conditions = [HTANodeModel.tree_id == tree_id]
            
            if status is not None:
                conditions.append(HTANodeModel.status == status)
                
            if is_major_phase is not None:
                conditions.append(HTANodeModel.is_major_phase == is_major_phase)
            
            stmt = select(HTANodeModel).where(and_(*conditions))
            result = await session.execute(stmt)
            nodes = result.scalars().all()
            
            filter_desc = f"status={status}, is_major_phase={is_major_phase}" if status or is_major_phase is not None else "all"
            logger.debug(f"Retrieved {len(nodes)} nodes for tree {tree_id} with filters: {filter_desc}")
            return nodes
    
    async def update_node(self, node: HTANodeModel) -> HTANodeModel:
        """
        Update a node.
        
        Args:
            node: HTANodeModel to update
            
        Returns:
            Updated HTANodeModel instance
        """
        async with self.session_manager.session() as session:
            session.add(node)
            await session.commit()
            await session.refresh(node)
            
            logger.debug(f"Updated HTA node: {node.id}")
            return node
    
    async def update_node_status(self, node_id: uuid.UUID, 
                             new_status: str, 
                             update_internal_details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update a node's status and optionally its internal details.
        Uses specific update statement for efficiency.
        
        Args:
            node_id: UUID of the node
            new_status: New status value
            update_internal_details: Optional dict of internal details to update
            
        Returns:
            True if update was successful
        """
        async with self.session_manager.session() as session:
            # Start with basic update
            update_values = {
                "status": new_status,
                "updated_at": datetime.utcnow()
            }
            
            # Add internal details if provided
            if update_internal_details:
                # Get current internal details first
                stmt = select(HTANodeModel).where(HTANodeModel.id == node_id)
                result = await session.execute(stmt)
                node = result.scalars().first()
                
                if node:
                    current_details = node.internal_task_details or {}
                    # Merge details
                    merged_details = {**current_details, **update_internal_details}
                    update_values["internal_task_details"] = merged_details
            
            # Execute update
            update_stmt = update(HTANodeModel).where(HTANodeModel.id == node_id).values(**update_values)
            result = await session.execute(update_stmt)
            await session.commit()
            
            success = result.rowcount > 0
            if success:
                logger.info(f"Updated HTA node {node_id} status to {new_status}")
            else:
                logger.warning(f"Failed to update HTA node {node_id} status")
                
            return success
    
    async def update_branch_triggers(self, node_id: uuid.UUID, 
                                  new_triggers: Dict[str, Any]) -> bool:
        """
        Update a node's branch triggers.
        
        Args:
            node_id: UUID of the node
            new_triggers: New triggers dict
            
        Returns:
            True if update was successful
        """
        async with self.session_manager.session() as session:
            # Get current triggers first
            stmt = select(HTANodeModel).where(HTANodeModel.id == node_id)
            result = await session.execute(stmt)
            node = result.scalars().first()
            
            if not node:
                logger.warning(f"HTA node not found for trigger update: {node_id}")
                return False
                
            current_triggers = node.branch_triggers or {}
            # Merge triggers
            merged_triggers = {**current_triggers, **new_triggers}
            
            # Execute update
            update_stmt = update(HTANodeModel).where(
                HTANodeModel.id == node_id
            ).values(
                branch_triggers=merged_triggers,
                updated_at=datetime.utcnow()
            )
            result = await session.execute(update_stmt)
            await session.commit()
            
            success = result.rowcount > 0
            if success:
                logger.debug(f"Updated HTA node {node_id} branch triggers")
            else:
                logger.warning(f"Failed to update HTA node {node_id} branch triggers")
                
            return success
    
    async def increment_branch_completion_count(self, node_id: uuid.UUID) -> Tuple[bool, int]:
        """
        Increment the branch completion count trigger and return the new count.
        
        Args:
            node_id: UUID of the node
            
        Returns:
            Tuple of (success, new_count)
        """
        async with self.session_manager.session() as session:
            # Get current count first
            stmt = select(HTANodeModel).where(HTANodeModel.id == node_id)
            result = await session.execute(stmt)
            node = result.scalars().first()
            
            if not node or not node.branch_triggers:
                logger.warning(f"HTA node or branch_triggers not found: {node_id}")
                return False, 0
                
            current_count = node.branch_triggers.get("current_completion_count", 0)
            new_count = current_count + 1
            
            # Update branch triggers
            node.branch_triggers["current_completion_count"] = new_count
            
            # Check if we've hit the threshold for expansion
            threshold = node.branch_triggers.get("completion_count_for_expansion_trigger", 3)
            if new_count >= threshold:
                node.branch_triggers["expand_now"] = True
                logger.info(f"HTA node {node_id} hit completion threshold, flagging for expansion")
                
            session.add(node)
            await session.commit()
            
            return True, new_count
    
    async def get_nodes_ready_for_expansion(self, tree_id: uuid.UUID) -> List[HTANodeModel]:
        """
        Get all nodes flagged for expansion in a tree.
        
        Args:
            tree_id: UUID of the tree
            
        Returns:
            List of nodes ready for expansion
        """
        async with self.session_manager.session() as session:
            # Need to use JSONB operator to query inside branch_triggers
            stmt = select(HTANodeModel).where(
                and_(
                    HTANodeModel.tree_id == tree_id,
                    # This is PostgreSQL-specific JSONB query
                    HTANodeModel.branch_triggers["expand_now"].astext == "true"
                )
            )
            
            result = await session.execute(stmt)
            nodes = result.scalars().all()
            
            logger.debug(f"Found {len(nodes)} nodes ready for expansion in tree {tree_id}")
            return nodes
    
    async def _update_parent_leaf_status(self, session: AsyncSession, 
                                     parent_id: uuid.UUID, 
                                     is_leaf: bool) -> bool:
        """
        Update a parent node's leaf status.
        
        Args:
            session: Active database session
            parent_id: UUID of the parent node
            is_leaf: New leaf status
            
        Returns:
            True if update was successful
        """
        update_stmt = update(HTANodeModel).where(
            HTANodeModel.id == parent_id
        ).values(
            is_leaf=is_leaf
        )
        result = await session.execute(update_stmt)
        return result.rowcount > 0
    
    async def build_tree_statistics(self, tree_id: uuid.UUID) -> Dict[str, Any]:
        """
        Build statistics for a tree to help with optimization.
        
        Args:
            tree_id: UUID of the tree
            
        Returns:
            Dictionary of tree statistics
        """
        async with self.session_manager.session() as session:
            # Get node counts by status
            status_counts = {}
            for status in ["pending", "in_progress", "completed", "deferred", "cancelled"]:
                stmt = select(func.count()).where(
                    and_(
                        HTANodeModel.tree_id == tree_id,
                        HTANodeModel.status == status
                    )
                )
                result = await session.execute(stmt)
                status_counts[status] = result.scalar() or 0
            
            # Get depth statistics
            stmt = select(HTANodeModel).where(HTANodeModel.tree_id == tree_id)
            result = await session.execute(stmt)
            nodes = result.scalars().all()
            
            # Calculate depth for each node
            depths = {}
            for node in nodes:
                depth = await self._calculate_node_depth(session, node)
                depths[str(node.id)] = depth
            
            max_depth = max(depths.values()) if depths else 0
            avg_depth = sum(depths.values()) / len(depths) if depths else 0
            
            # Get branching statistics
            branches = {}
            for node in nodes:
                if node.parent_id:
                    parent_id = str(node.parent_id)
                    branches[parent_id] = branches.get(parent_id, 0) + 1
            
            max_branch = max(branches.values()) if branches else 0
            avg_branch = sum(branches.values()) / len(branches) if branches else 0
            
            return {
                "total_nodes": len(nodes),
                "status_counts": status_counts,
                "max_depth": max_depth,
                "avg_depth": avg_depth,
                "max_branch": max_branch,
                "avg_branch": avg_branch,
                "leaf_count": sum(1 for node in nodes if node.is_leaf),
                "major_phase_count": sum(1 for node in nodes if node.is_major_phase)
            }
    
    async def _calculate_node_depth(self, session: AsyncSession, node: HTANodeModel) -> int:
        """
        Calculate the depth of a node in the tree.
        
        Args:
            session: Active database session
            node: HTANodeModel to calculate depth for
            
        Returns:
            Depth of the node (0 for root)
        """
        depth = 0
        current_id = node.parent_id
        visited = set()
        
        while current_id and current_id not in visited:
            visited.add(current_id)
            depth += 1
            
            stmt = select(HTANodeModel.parent_id).where(HTANodeModel.id == current_id)
            result = await session.execute(stmt)
            parent_id = result.scalar()
            
            current_id = parent_id
            
        return depth

logger.debug("HTATreeRepository defined.")
