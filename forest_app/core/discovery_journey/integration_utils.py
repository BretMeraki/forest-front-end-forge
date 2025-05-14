"""
Discovery Journey Integration Utilities

Contains utility functions to seamlessly integrate the Discovery Journey into the
core user experience without creating a separate visible feature.
"""

import logging
from typing import Any, Dict, Optional, List, Union, cast
from uuid import UUID
import asyncio
from datetime import datetime, timezone

from forest_app.core.discovery_journey import DiscoveryJourneyService
from forest_app.core.services.enhanced_hta_service import EnhancedHTAService
from forest_app.core.snapshot import MemorySnapshot
from forest_app.modules.hta_tree import HTATree

logger = logging.getLogger(__name__)

async def enrich_hta_with_discovery_insights(
    discovery_service: DiscoveryJourneyService,
    hta_service: EnhancedHTAService,
    user_id: Union[str, UUID],
    hta_tree: Dict[str, Any],
    raw_goal: str,
    raw_context: str
) -> Dict[str, Any]:
    """
    Invisibly enhances the HTA with insights from the Discovery Journey.
    
    This function imperceptibly integrates Discovery Journey capabilities into
    the HTA generation process, making the HTA more adaptive to the user's
    abstract-to-concrete journey without creating a separate visible feature.
    
    Args:
        discovery_service: The Discovery Journey service
        hta_service: The HTA service
        user_id: User identifier
        hta_tree: The generated HTA tree
        raw_goal: The user's initial goal
        raw_context: The user's context reflection
        
    Returns:
        Enhanced HTA tree with discovery insights
    """
    if not discovery_service or not hta_tree:
        return hta_tree
        
    try:
        # Assess the abstraction level of the user's goal
        abstraction_level = await discovery_service.assess_abstraction_level(
            user_id=user_id,
            goal_description=raw_goal,
            context_reflection=raw_context
        )
        
        # If the goal is abstract, subtly modify the HTA to include exploratory elements
        if abstraction_level and abstraction_level.get('level', 0) > 6:  # On a scale of 1-10
            # Get the HTA tree object
            tree_obj = HTATree.from_dict(hta_tree)
            
            # Find appropriate parent nodes for exploratory tasks
            root_id = tree_obj.root_id
            if root_id:
                # Get the trunk nodes (direct children of root)
                trunk_nodes = [node for node in tree_obj.nodes.values() 
                               if node.parent_id == root_id]
                
                # Select a trunk node that's appropriate for exploration
                if trunk_nodes:
                    target_node = _select_exploration_target(trunk_nodes, abstraction_level)
                    
                    # Generate exploratory tasks
                    exploratory_tasks = await discovery_service.generate_exploratory_tasks(
                        user_id=str(user_id),  # Ensure user_id is a string
                        tree=tree_obj,
                        parent_node_id=target_node.node_id,
                        count=2  # Keep it minimal to blend in with regular tasks
                    )
                    
                    # Imperceptibly blend exploratory tasks with regular ones
                    for task in exploratory_tasks:
                        # Add special metadata but keep it invisible to the user
                        task['metadata'] = task.get('metadata', {})
                        task['metadata']['discovery_type'] = 'exploratory'
                        task['metadata']['abstraction_level'] = abstraction_level.get('level')
                        
                        # Create a node that looks like a normal HTA node
                        tree_obj.add_node(
                            node_id=task.get('id') or f"discovery_{task.get('title', '')[:10]}_{id(task)}",
                            title=task.get('title', 'Explore deeper'),
                            description=task.get('description', ''),
                            parent_id=target_node.node_id,
                            metadata=task.get('metadata', {})
                        )
            
            # Return the enhanced tree
            return tree_obj.to_dict()
                    
        return hta_tree
    except Exception as e:
        logger.warning(f"Non-critical: Could not enrich HTA with discovery insights: {e}")
        return hta_tree

def _select_exploration_target(nodes: List[Any], abstraction_data: Dict[str, Any]) -> Any:
    """
    Select the most appropriate node for adding exploratory tasks.
    
    Args:
        nodes: List of potential target nodes
        abstraction_data: Data about the abstraction level
        
    Returns:
        Selected node
    """
    # Use abstraction insights to select the most appropriate node
    abstract_areas = abstraction_data.get('abstract_areas', [])
    
    # If we have identified abstract areas, try to match with node titles/descriptions
    if abstract_areas:
        for area in abstract_areas:
            for node in nodes:
                # Simple keyword matching (could be made more sophisticated)
                if (area.lower() in node.title.lower() or 
                    area.lower() in node.description.lower()):
                    return node
    
    # Default to the first node if no match found
    return nodes[0]

async def track_task_completion_for_discovery(
    discovery_service: DiscoveryJourneyService,
    user_id: Union[str, UUID],
    task_id: str,
    feedback: Optional[Dict[str, Any]] = None
) -> None:
    """
    Invisibly track task completion for discovery journey insights.
    
    Args:
        discovery_service: The Discovery Journey service
        user_id: User identifier
        task_id: Completed task identifier
        feedback: Optional user feedback
    """
    if not discovery_service:
        return
        
    try:
        # Process the task completion in the background
        await discovery_service.process_task_completion(
            user_id=user_id,
            task_id=task_id,
            feedback=feedback or {}
        )
    except Exception as e:
        logger.warning(f"Non-critical: Could not track task completion for discovery: {e}")

async def infuse_recommendations_into_snapshot(
    discovery_service: DiscoveryJourneyService,
    snapshot: MemorySnapshot,
    user_id: Union[str, UUID]
) -> MemorySnapshot:
    """
    Invisibly infuse discovery recommendations into the memory snapshot.
    
    This allows the system to subtly guide the user's journey without
    creating a separate visible feature.
    
    Args:
        discovery_service: The Discovery Journey service
        snapshot: Current memory snapshot
        user_id: User identifier
        
    Returns:
        Enhanced memory snapshot
    """
    if not discovery_service or not snapshot:
        return snapshot
        
    try:
        # Get tree from snapshot
        tree_dict = snapshot.core_state.get('hta_tree')
        if not tree_dict:
            return snapshot
            
        tree = HTATree.from_dict(tree_dict)
        
        # Get evolution recommendations
        evolution = await discovery_service.evolve_focus_based_on_patterns(
            user_id=user_id,
            tree=tree
        )
        
        if evolution:
            # Store evolution data invisibly in the snapshot
            if not "discovery_journey" in snapshot.component_state:
                snapshot.component_state["discovery_journey"] = {}
                
            snapshot.component_state["discovery_journey"]["evolution_insights"] = evolution
            
            # If we have recommended tasks, blend them invisibly into the HTA
            recommended_tasks = evolution.get('recommended_tasks', [])
            if recommended_tasks and tree.root_id:
                # Find appropriate nodes to attach recommendations
                for task in recommended_tasks[:2]:  # Limit to 2 to avoid overwhelming
                    target_node_id = _find_appropriate_target_node(tree, task)
                    if target_node_id:
                        # Add the task as a normal-looking HTA node
                        tree.add_node(
                            node_id=task.get('id') or f"discovery_rec_{id(task)}",
                            title=task.get('title', 'Try this approach'),
                            description=task.get('description', ''),
                            parent_id=target_node_id,
                            metadata={
                                "discovery_type": "recommendation",
                                "confidence": task.get('confidence', 0.7),
                                "visible_to_user": True  # This looks like a normal task
                            }
                        )
                
                # Update the snapshot with modified tree
                snapshot.core_state['hta_tree'] = tree.to_dict()
        
        return snapshot
    except Exception as e:
        logger.warning(f"Non-critical: Could not infuse recommendations into snapshot: {e}")
        return snapshot

def _find_appropriate_target_node(tree: HTATree, task: Dict[str, Any]) -> Optional[str]:
    """
    Find an appropriate node to attach a recommended task.
    
    Args:
        tree: HTA tree
        task: Task to attach
        
    Returns:
        Node ID or None if no appropriate node found
    """
    # Start with the root node
    root_id = tree.root_id
    if not root_id:
        return None
        
    # Look for nodes that match the task's theme
    task_theme = task.get('theme', '').lower()
    for node in tree.nodes.values():
        # Skip the root node
        if node.node_id == root_id:
            continue
            
        # Look for thematic matches
        if (task_theme in node.title.lower() or 
            task_theme in node.description.lower()):
            return node.node_id
            
    # If no good match, return a trunk node
    trunk_nodes = [node.node_id for node in tree.nodes.values() 
                  if node.parent_id == root_id]
    return trunk_nodes[0] if trunk_nodes else root_id
