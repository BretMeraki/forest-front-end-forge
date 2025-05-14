"""
Context-Infused Node Generator

This module implements a generator for HTA nodes that infuses each node with the user's
unique context, preferences, and memories. It ensures that each tree generation is
truly personalized while maintaining performance and structural integrity.
"""

import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime

from forest_app.core.schema_contract import HTASchemaContract
from forest_app.core.roadmap_models import RoadmapManifest, RoadmapStep
from forest_app.persistence.models import HTANodeModel, MemorySnapshotModel
from forest_app.core.transaction_decorator import transaction_protected as transaction_protected
from forest_app.integrations.llm import LLMClient
from forest_app.core.session_manager import SessionManager

logger = logging.getLogger(__name__)


class ContextInfusedNodeGenerator:
    """
    Generates unique, context-aware nodes following the schema contract.
    Never uses templates - always creates fresh content based on user context.
    """
    
    def __init__(self, llm_client: LLMClient, memory_service, session_manager: SessionManager):
        """
        Initialize the context-infused node generator.
        
        Args:
            llm_client: LLM client for generating context-rich content
            memory_service: Service for accessing user memory snapshots
            session_manager: Manager for database sessions
        """
        self.llm = llm_client
        self.memory = memory_service
        self.session_manager = session_manager
        
    async def generate_trunk_node(self, tree_id: uuid.UUID, user_id: uuid.UUID, 
                                 user_goal: str, memory_snapshot: Optional[Dict] = None) -> HTANodeModel:
        """
        Generate a trunk node infused with user context.
        
        Args:
            tree_id: UUID of the tree
            user_id: UUID of the user
            user_goal: User's goal statement
            memory_snapshot: Optional memory snapshot data
            
        Returns:
            HTANodeModel instance for the trunk node
        """
        # Get fresh context data if memory_snapshot not provided
        if memory_snapshot is None:
            memory_snapshot = await self._get_memory_context(user_id)
            
        recent_reflections = await self._extract_recent_reflections(user_id)
        user_preferences = await self._extract_user_preferences(user_id)
        
        # Generate a unique trunk node based on user context
        node_content = await self._generate_node_content(
            prompt_type="trunk_node",
            context={
                "user_goal": user_goal,
                "memory_snapshot": memory_snapshot,
                "recent_reflections": recent_reflections,
                "user_preferences": user_preferences,
                "node_type": "trunk",
                "node_purpose": "Major phase in achieving the goal",
            }
        )
        
        # Create node model with optimizations for performance
        return HTANodeModel(
            id=uuid.uuid4(),
            tree_id=tree_id,
            user_id=user_id,
            title=node_content.get("title"),
            description=node_content.get("description"),
            status="pending",
            parent_id=None,
            is_leaf=False,
            is_major_phase=True,
            internal_task_details={
                "phase_type": node_content.get("phase_type", "journey_beginning"),
                "expected_duration": node_content.get("expected_duration", "medium"),
                "joy_factor": node_content.get("joy_factor", 0.7),
                "context_relevance_score": node_content.get("relevance_score", 0.8)
            },
            branch_triggers={
                "expand_now": True,
                "completion_count_for_expansion_trigger": 0,
                "current_completion_count": 0
            }
        )
    
    async def generate_branch_from_parent(self, parent_node: HTANodeModel, 
                                        memory_snapshot: Optional[Dict] = None) -> List[HTANodeModel]:
        """
        Generate branches specifically for this parent based on context.
        
        Args:
            parent_node: Parent node to generate branches for
            memory_snapshot: Optional memory snapshot data
            
        Returns:
            List of HTANodeModel instances representing branches
        """
        # Get fresh context data if memory_snapshot not provided
        if memory_snapshot is None:
            memory_snapshot = await self._get_memory_context(parent_node.user_id)
            
        # Get contextual information
        tree_metadata = await self._get_tree_metadata(parent_node.tree_id)
        user_preferences = await self._extract_user_preferences(parent_node.user_id)
        recent_activities = await self._extract_recent_activities(parent_node.user_id)
        
        # Generate branch content based on parent context
        branches_content = await self._generate_node_content(
            prompt_type="branch_nodes",
            context={
                "parent_node": {
                    "title": parent_node.title,
                    "description": parent_node.description,
                    "is_major_phase": parent_node.is_major_phase,
                    "metadata": parent_node.internal_task_details
                },
                "tree_metadata": tree_metadata,
                "user_preferences": user_preferences,
                "recent_activities": recent_activities,
                "memory_snapshot": memory_snapshot,
                "count": 3 if parent_node.is_major_phase else 2
            }
        )
        
        # Create branch node models
        branch_nodes = []
        for branch in branches_content.get("branches", []):
            node = HTANodeModel(
                id=uuid.uuid4(),
                tree_id=parent_node.tree_id,
                user_id=parent_node.user_id,
                title=branch.get("title"),
                description=branch.get("description"),
                status="pending",
                parent_id=parent_node.id,
                is_leaf=False,
                is_major_phase=False,
                internal_task_details={
                    "estimated_energy": branch.get("estimated_energy", "medium"),
                    "estimated_time": branch.get("estimated_time", "medium"),
                    "joy_factor": branch.get("joy_factor", 0.5),
                    "context_relevance_score": branch.get("relevance_score", 0.7),
                    "branch_type": branch.get("branch_type", "task")
                },
                branch_triggers={
                    "expand_now": False,
                    "completion_count_for_expansion_trigger": 2,
                    "current_completion_count": 0
                }
            )
            branch_nodes.append(node)
            
        return branch_nodes
    
    async def generate_micro_actions(self, branch_node: HTANodeModel, 
                                  count: int = 3) -> List[HTANodeModel]:
        """
        Generate unique, actionable micro-tasks based on branch context.
        
        Args:
            branch_node: Branch node to generate micro-actions for
            count: Number of micro-actions to generate
            
        Returns:
            List of HTANodeModel instances representing micro-actions
        """
        memory_snapshot = await self._get_memory_context(branch_node.user_id)
        parent_context = await self._get_parent_context(branch_node)
        
        # Generate micro-action content
        micro_actions_content = await self._generate_node_content(
            prompt_type="micro_actions",
            context={
                "branch_node": {
                    "title": branch_node.title,
                    "description": branch_node.description,
                    "metadata": branch_node.internal_task_details
                },
                "parent_context": parent_context,
                "memory_snapshot": memory_snapshot,
                "count": count
            }
        )
        
        # Create micro-action node models with optimization for bulk insertion
        micro_action_nodes = []
        for action in micro_actions_content.get("micro_actions", []):
            node = HTANodeModel(
                id=uuid.uuid4(),
                tree_id=branch_node.tree_id,
                user_id=branch_node.user_id,
                title=action.get("title"),
                description=action.get("description"),
                status="pending",
                parent_id=branch_node.id,
                is_leaf=True,
                is_major_phase=False,
                internal_task_details={
                    "actionability_score": action.get("actionability_score", 0.8),
                    "joy_factor": action.get("joy_factor", 0.6),
                    "estimated_time": action.get("estimated_time", "low"),
                    "framing": action.get("framing", "action"),
                    "positive_reinforcement": action.get("positive_reinforcement", 
                                                       "Great job completing this step!")
                }
            )
            micro_action_nodes.append(node)
            
        return micro_action_nodes
    
    async def _generate_node_content(self, prompt_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate node content using the LLM with the given context.
        
        Args:
            prompt_type: Type of prompt to use
            context: Context data for the prompt
            
        Returns:
            Dictionary with generated content
        """
        try:
            # Generate content using LLM
            # The LLM client should have different prompt templates for different node types
            response = await self.llm.generate(
                prompt_type=prompt_type,
                context=context
            )
            
            # Validate response against schema contract
            if prompt_type == "trunk_node":
                errors = HTASchemaContract.validate_model("node", response)
                if errors:
                    logger.warning(f"LLM response validation errors for {prompt_type}: {errors}")
                    # Attempt repair of common issues
                    response = self._repair_response(response, errors)
                
            # Check for context infusion
            if not HTASchemaContract.check_context_infusion("node", "title", response.get("title", "")):
                logger.warning("Generated title lacks proper context infusion")
                
            if not HTASchemaContract.check_context_infusion("node", "description", response.get("description", "")):
                logger.warning("Generated description lacks proper context infusion")
                
            return response
            
        except Exception as e:
            logger.error(f"Error generating node content: {str(e)}")
            # Return fallback content
            return self._get_fallback_content(prompt_type)
    
    def _repair_response(self, response: Dict[str, Any], errors: List[str]) -> Dict[str, Any]:
        """
        Attempt to repair common validation errors in LLM responses.
        
        Args:
            response: The LLM response to repair
            errors: List of validation errors
            
        Returns:
            Repaired response dictionary
        """
        repaired = response.copy()
        
        for error in errors:
            if "Missing required field" in error:
                field = error.split(": ")[1]
                if field == "title":
                    repaired["title"] = "Untitled Task"
                elif field == "description":
                    repaired["description"] = "Task details will be provided soon."
                    
            elif "Invalid value for" in error:
                field = error.split(" for ")[1]
                if field == "title":
                    # Truncate if too long, pad if too short
                    title = repaired.get("title", "Untitled")
                    repaired["title"] = title[:100] if len(title) > 100 else title
                    
        return repaired
    
    def _get_fallback_content(self, prompt_type: str) -> Dict[str, Any]:
        """
        Get fallback content when LLM generation fails.
        
        Args:
            prompt_type: Type of prompt that failed
            
        Returns:
            Dictionary with fallback content
        """
        if prompt_type == "trunk_node":
            return {
                "title": "Journey Phase",
                "description": "This phase of your journey will be detailed soon.",
                "phase_type": "general",
                "expected_duration": "medium",
                "joy_factor": 0.5,
                "relevance_score": 0.5
            }
        elif prompt_type == "branch_nodes":
            return {
                "branches": [
                    {
                        "title": "Next Step",
                        "description": "A step towards completing this phase.",
                        "estimated_energy": "medium",
                        "estimated_time": "medium",
                        "joy_factor": 0.5,
                        "relevance_score": 0.5,
                        "branch_type": "task"
                    }
                ]
            }
        elif prompt_type == "micro_actions":
            return {
                "micro_actions": [
                    {
                        "title": "Take Action",
                        "description": "A specific action to move forward.",
                        "actionability_score": 0.8,
                        "joy_factor": 0.6,
                        "estimated_time": "low",
                        "framing": "action",
                        "positive_reinforcement": "Great job completing this step!"
                    }
                ]
            }
        return {}
    
    async def _get_memory_context(self, user_id: uuid.UUID) -> Dict[str, Any]:
        """
        Get memory context for the user.
        
        Args:
            user_id: UUID of the user
            
        Returns:
            Dictionary with memory context
        """
        try:
            return await self.memory.get_latest_snapshot(user_id)
        except Exception as e:
            logger.error(f"Error getting memory context: {str(e)}")
            return {}
    
    async def _extract_recent_reflections(self, user_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        Extract recent reflections for the user.
        
        Args:
            user_id: UUID of the user
            
        Returns:
            List of recent reflections
        """
        try:
            return await self.memory.get_recent_reflections(user_id)
        except Exception as e:
            logger.error(f"Error extracting recent reflections: {str(e)}")
            return []
    
    async def _extract_user_preferences(self, user_id: uuid.UUID) -> Dict[str, Any]:
        """
        Extract user preferences from memory.
        
        Args:
            user_id: UUID of the user
            
        Returns:
            Dictionary with user preferences
        """
        try:
            snapshot = await self.memory.get_latest_snapshot(user_id)
            return snapshot.get("user_preferences", {})
        except Exception as e:
            logger.error(f"Error extracting user preferences: {str(e)}")
            return {}
    
    async def _extract_recent_activities(self, user_id: uuid.UUID) -> List[Dict[str, Any]]:
        """
        Extract recent activities for the user.
        
        Args:
            user_id: UUID of the user
            
        Returns:
            List of recent activities
        """
        try:
            snapshot = await self.memory.get_latest_snapshot(user_id)
            return snapshot.get("recent_activities", [])
        except Exception as e:
            logger.error(f"Error extracting recent activities: {str(e)}")
            return []
    
    async def _get_tree_metadata(self, tree_id: uuid.UUID) -> Dict[str, Any]:
        """
        Get metadata for the tree.
        
        Args:
            tree_id: UUID of the tree
            
        Returns:
            Dictionary with tree metadata
        """
        async with self.session_manager.session() as session:
            from forest_app.persistence.models import HTATreeModel
            tree = await session.query(HTATreeModel).filter(HTATreeModel.id == tree_id).first()
            if tree:
                return {
                    "goal_name": tree.goal_name,
                    "initial_context": tree.initial_context,
                    "created_at": tree.created_at.isoformat() if tree.created_at else None
                }
            return {}
    
    async def _get_parent_context(self, node: HTANodeModel) -> Dict[str, Any]:
        """
        Get context from parent nodes.
        
        Args:
            node: Node to get parent context for
            
        Returns:
            Dictionary with parent context
        """
        async with self.session_manager.session() as session:
            # Get ancestors in one query with path-based optimization
            # For now, we do basic ancestor lookups
            ancestors = []
            current_id = node.parent_id
            visited = set()
            
            while current_id and current_id not in visited:
                visited.add(current_id)
                parent = await session.query(HTANodeModel).filter(HTANodeModel.id == current_id).first()
                if parent:
                    ancestors.append({
                        "title": parent.title,
                        "description": parent.description,
                        "is_major_phase": parent.is_major_phase
                    })
                    current_id = parent.parent_id
                else:
                    break
                    
            return {"ancestors": ancestors}

logger.debug("Context-Infused Node Generator defined.")
