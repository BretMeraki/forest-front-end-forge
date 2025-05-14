# forest_app/core/services/hta_service.py

import logging
import json
import random
from typing import Optional, Dict, Any, List, Set, TYPE_CHECKING, Union, Tuple
from datetime import datetime, timezone, timedelta
from uuid import UUID

# Core imports with error handling
try:
    from forest_app.core.snapshot import MemorySnapshot
    from forest_app.modules.hta_tree import HTATree, HTANode
    from forest_app.modules.seed import SeedManager, Seed
    from forest_app.core.services.semantic_base import SemanticMemoryManagerBase
    from forest_app.integrations.llm import (
        LLMClient,
        HTAEvolveResponse,
        DistilledReflectionResponse,
        LLMError,
        LLMValidationError
    )
    from forest_app.modules.hta_models import HTANodeModel
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import required modules: {e}")
    # Define dummy classes if imports fail
    class MemorySnapshot: pass
    class HTATree: pass
    class HTANode: pass
    class SeedManager: pass
    class Seed: pass
    # No fallback needed; use SemanticMemoryManagerBase for type hints
    class LLMClient: pass
    class HTAEvolveResponse: pass
    class DistilledReflectionResponse: pass
    class LLMError(Exception): pass
    class LLMValidationError(Exception): pass
    class HTANodeModel: pass

# Feature flags with error handling
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    def is_enabled(feature): return True

logger = logging.getLogger(__name__)

class HTAService:
    """Service for managing HTA (Hierarchical Task Analysis) with semantic memory integration."""

    def __init__(self, llm_client, semantic_memory_manager):
        self.llm_client = llm_client
        self.semantic_memory_manager = semantic_memory_manager
        self.logger = logging.getLogger(__name__)
        self.task_hierarchies: Dict[str, Dict[str, Any]] = {}
        
        # Basic dependency checks
        if not isinstance(self.llm_client, LLMClient):
            logger.critical("HTAService initialized with invalid LLMClient!")
        if not isinstance(self.semantic_memory_manager, SemanticMemoryManagerBase):  # Fixed type check
            logger.critical("HTAService initialized with invalid SemanticMemoryManager!")
            
        logger.info("HTAService initialized with dependencies.")
        
    async def initialize_task_hierarchy(self, task_id: str, hierarchy_data: Dict[str, Any]) -> None:
        """
        Initialize a task hierarchy for a given task ID.
        
        Args:
            task_id: Unique identifier for the task
            hierarchy_data: Dictionary containing the task hierarchy structure
        """
        self.task_hierarchies[task_id] = hierarchy_data
        logger.info(f"Initialized HTA hierarchy for task {task_id}")
        
    async def update_task_state(self, task_id: str, state_data: Dict[str, Any]) -> None:
        """
        Update the state of a task in the hierarchy.
        
        Args:
            task_id: Unique identifier for the task
            state_data: Dictionary containing the updated state information
        """
        if task_id not in self.task_hierarchies:
            logger.warning(f"Task {task_id} not found in hierarchies")
            return
            
        self.task_hierarchies[task_id].update(state_data)
        logger.debug(f"Updated state for task {task_id}")
        
    async def get_task_hierarchy(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the task hierarchy for a given task ID.
        
        Args:
            task_id: Unique identifier for the task
            
        Returns:
            Dictionary containing the task hierarchy, or None if not found
        """
        return self.task_hierarchies.get(task_id)

    async def _get_active_seed(self, snapshot: MemorySnapshot) -> Optional[Seed]:
        """Helper to get the primary active seed from the snapshot or SeedManager."""
        # This logic might vary based on how active seed is tracked
        # Option 1: Check snapshot first
        active_seed_id = snapshot.component_state.get("seed_manager", {}).get("active_seed_id")
        if active_seed_id and hasattr(self.semantic_memory_manager, 'get_seed_by_id'):
            try:
                # Assume get_seed_by_id might be async if it involves DB lookups
                seed = await self.semantic_memory_manager.get_seed_by_id(active_seed_id)
                if seed: return seed
            except Exception as e:
                logger.warning(f"Could not get seed by ID {active_seed_id} from snapshot: {e}")

        # Option 2: Fallback to getting the first active seed from SeedManager
        if hasattr(self.semantic_memory_manager, 'get_primary_active_seed'):
             # Assume get_primary_active_seed might be async
            return await self.semantic_memory_manager.get_primary_active_seed()

        logger.warning("Could not determine active seed.")
        return None


    async def load_tree(self, snapshot: MemorySnapshot) -> Optional[HTATree]:
        """
        Loads the HTA tree, prioritizing the version stored in the active Seed,
        falling back to the snapshot's core_state if necessary.
        Returns an HTATree object or None if not found/invalid.
        """
        logger.debug("Attempting to load HTA tree...")
        current_hta_dict: Optional[Dict] = None
        active_seed = await self._get_active_seed(snapshot)

        # Priority 1: Load from active seed
        if active_seed and hasattr(active_seed, 'hta_tree') and isinstance(active_seed.hta_tree, dict) and active_seed.hta_tree.get('root'):
            logger.debug(f"Loading HTA tree from active seed ID: {getattr(active_seed, 'seed_id', 'N/A')}")
            current_hta_dict = active_seed.hta_tree
        # Priority 2: Load from snapshot core_state
        elif isinstance(snapshot.core_state, dict) and snapshot.core_state.get('hta_tree'):
            logger.warning("Loading HTA tree from snapshot core_state (fallback).") # Log as warning - Seed should ideally be source of truth
            current_hta_dict = snapshot.core_state.get('hta_tree')
        else:
            logger.warning("Could not find HTA tree dictionary in active seed or snapshot core_state.")
            return None

        # Parse the dictionary into an HTATree object
        if current_hta_dict and isinstance(current_hta_dict, dict):
            try:
                tree = HTATree.from_dict(current_hta_dict)
                if tree and tree.root:
                    logger.info(f"Successfully loaded HTA tree with root: {tree.root.id} - '{tree.root.title}'")
                    return tree
                else:
                    logger.error("Failed to parse valid HTATree object from loaded dictionary (root missing?).")
                    return None
            except ValueError as ve:
                logger.error(f"ValueError parsing HTA tree dictionary: {ve}")
                return None
            except Exception as e:
                logger.exception(f"Unexpected error parsing HTA tree dictionary: {e}")
                return None
        else:
            logger.error("Loaded HTA data is not a valid dictionary.")
            return None


    async def save_tree(self, snapshot: MemorySnapshot, tree: HTATree) -> bool:
        """
        Saves the current state of the HTATree object back to the active Seed
        and the snapshot's core_state.

        Args:
            snapshot: The MemorySnapshot object (its core_state will be updated).
            tree: The HTATree object to save.

        Returns:
            True if saving was successful (at least to the snapshot), False otherwise.
        """
        if not tree or not tree.root:
            logger.error("Cannot save HTA tree: Tree object or root node is missing.")
            return False

        try:
            final_hta_dict_to_save = tree.to_dict()
        except Exception as e:
            logger.exception(f"Failed to serialize HTATree object to dictionary: {e}")
            return False

        if not final_hta_dict_to_save or not final_hta_dict_to_save.get('root'):
             logger.error("Failed to serialize HTA tree or root node is missing in dict.")
             return False

        # 1. Update Snapshot Core State (Primary target)
        try:
            if not hasattr(snapshot, 'core_state') or not isinstance(snapshot.core_state, dict):
                snapshot.core_state = {}
            snapshot.core_state['hta_tree'] = final_hta_dict_to_save
            logger.info("Updated HTA tree in snapshot core_state.")
            snapshot_save_ok = True
        except Exception as e:
            logger.exception(f"Failed to update HTA tree in snapshot core_state: {e}")
            snapshot_save_ok = False # Still try to save to seed

        # 2. Update Active Seed (Secondary, Source of Truth)
        seed_save_ok = False
        active_seed = await self._get_active_seed(snapshot)
        if active_seed and hasattr(self.semantic_memory_manager, 'update_seed'):
            try:
                # Assume update_seed is async if it interacts with DB
                success = await self.semantic_memory_manager.update_seed(
                    active_seed.seed_id,
                    hta_tree=final_hta_dict_to_save
                )
                if success:
                    logger.info(f"Successfully updated HTA tree in active seed ID: {active_seed.seed_id}")
                    seed_save_ok = True
                else:
                    logger.error(f"SeedManager failed to update HTA tree for seed ID: {active_seed.seed_id}")
            except Exception as seed_update_err:
                logger.exception(f"Failed to update seed {active_seed.seed_id} with final HTA: {seed_update_err}")
        elif not active_seed:
            logger.error("Cannot save HTA to seed: Active seed not found.")
        else: # SeedManager missing method
             logger.error("Cannot save HTA to seed: Injected SeedManager lacks update_seed method.")

        # Check for progress triggers that might warrant celebration
        if snapshot_save_ok and active_seed and hasattr(active_seed, 'hta_tree'):
            try:
                await self._check_for_meaningful_moments(tree, snapshot)
            except Exception as moment_err:
                # Don't let this block the update, just log
                logger.exception(f"Error checking for meaningful moments: {moment_err}")
        
        # Return overall success (prioritize snapshot save)
        return snapshot_save_ok


    def update_node_status(self, tree: HTATree, node_id: str, new_status: str) -> bool:
        """
        Updates the status of a specific node within the tree object and triggers propagation.
        Note: This modifies the tree object in place. Saving must be done separately.

        Args:
            tree: The HTATree object to modify.
            node_id: The ID of the node to update.
            new_status: The new status string (e.g., "completed", "pending").

        Returns:
            True if the node was found and status potentially updated, False otherwise.
        """
        if not tree or not tree.root:
            logger.error("Cannot update node status: HTATree object is invalid.")
            return False  # Fail gracefully, protecting the user's experience

        # Find the node and its context in the tree
        node = tree.find_node_by_id(node_id)
        if node:
            # Check if this is a meaningful state transition
            is_completion = (node.status != "completed" and new_status == "completed")
            is_meaningful_node = self._is_meaningful_node(node)
            
            logger.info(f"Updating status for node '{node.title}' ({node_id}) to '{new_status}'.")
            
            # Perform the update, which handles propagation
            tree.update_node_status(node_id, new_status)
            
            # Track this as a potentially meaningful moment in the user's journey
            if is_completion and is_meaningful_node:
                self._record_meaningful_transition(node, tree)  # This doesn't modify the tree
                
            return True
        else:
            logger.warning(f"Cannot update status: Node with id '{node_id}' not found in tree.")
            return False  # Failure handled gracefully to protect the user experience


    async def evolve_tree(self, tree: HTATree, reflections: List[str], user_mood: Optional[str] = None) -> Optional[HTATree]:
        """
        Handles the HTA evolution process using the LLM client.

        Args:
            tree: The current HTATree object.
            reflections: List of user reflections from the completed batch.

        Returns:
            A new HTATree object with the evolved structure if successful and valid,
            otherwise None.
        """
        if not tree or not tree.root:
            logger.error("Cannot evolve HTA: Initial tree is invalid.")
            return None
        if not isinstance(self.llm_client, LLMClient) or type(self.llm_client).__name__ == 'DummyService':
            logger.error("Cannot evolve HTA: LLMClient is invalid or dummy.")
            return None

        evolution_goal = "Previous task batch complete. Re-evaluate the plan and suggest next steps." # Default goal

        # 1. (Optional) Distill reflections
        if reflections and hasattr(self.llm_client, 'distill_reflections'):
            logger.info(f"Distilling {len(reflections)} reflections for evolution goal...")
            try:
                distilled_response: Optional[DistilledReflectionResponse] = await self.llm_client.distill_reflections(
                    reflections=reflections
                )
                if distilled_response and distilled_response.distilled_text:
                    evolution_goal = distilled_response.distilled_text
                    logger.info(f"Using distilled reflection as evolution goal: '{evolution_goal[:100]}...'")
                else:
                    logger.warning("Reflection distillation failed or returned empty text. Using default goal.")
            except Exception as distill_err:
                logger.exception(f"Error during reflection distillation: {distill_err}. Using default goal.")
        elif not reflections:
            logger.info("No reflections provided for evolution. Using default goal.")

        # 2. Call LLM for evolution
        try:
            current_hta_json = json.dumps(tree.to_dict()) # Serialize current tree
            logger.debug(f"Calling request_hta_evolution. Goal: '{evolution_goal[:100]}...'")

            evolved_hta_response: Optional[HTAEvolveResponse] = await self.llm_client.request_hta_evolution(
                current_hta_json=current_hta_json,
                evolution_goal=evolution_goal,
                # use_advanced_model=False # TODO: Consider making this configurable
            )

            # 3. Validate and Process LLM Response
            if not isinstance(evolved_hta_response, HTAEvolveResponse) or not evolved_hta_response.hta_root:
                 log_response = evolved_hta_response if len(str(evolved_hta_response)) < 500 else str(type(evolved_hta_response))
                 logger.error(f"Failed to get valid evolved HTA from LLM. Type: {type(evolved_hta_response)}. Response: {log_response}")
                 return None # Evolution failed

            # Check root ID match (critical!)
            llm_root_id = getattr(evolved_hta_response.hta_root, 'id', 'LLM_MISSING_ID')
            original_root_id = getattr(tree.root, 'id', 'ORIGINAL_MISSING_ID')
            if llm_root_id != original_root_id:
                 logger.error(f"LLM HTA evolution root ID mismatch ('{llm_root_id}' vs '{original_root_id}'). Discarding evolved tree.")
                 return None # Evolution result is invalid

            # Convert Pydantic model back to dictionary for HTATree parsing
            # Add extra validation here if needed before dumping
            evolved_hta_root_dict = evolved_hta_response.hta_root.model_dump(mode='json')
            evolved_hta_dict = {'root': evolved_hta_root_dict}

            # Attempt to parse the evolved dictionary back into an HTATree object
            try:
                new_tree = HTATree.from_dict(evolved_hta_dict)
                if new_tree and new_tree.root:
                    logger.info("Successfully received and parsed evolved HTA tree.")
                    return new_tree # Return the new, evolved tree object
                else:
                    logger.error("Failed to re-parse evolved HTA dictionary into valid HTATree object.")
                    return None # Parsing failed
            except ValueError as ve:
                 logger.error(f"ValueError parsing evolved HTA dictionary: {ve}")
                 return None
            except Exception as parse_err:
                 logger.exception(f"Unexpected error parsing evolved HTA dictionary: {parse_err}")
                 return None

        except (LLMError, LLMValidationError) as llm_evolve_err:
            logger.error(f"LLM/Validation Error during HTA evolution request: {llm_evolve_err}")
            return None # Evolution failed
        except Exception as evolve_err:
            logger.exception(f"Unexpected error during HTA evolution process: {evolve_err}")
            return None # Evolution failed

    async def update_task_completion(self, task_id: str, completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update task completion status with semantic memory context.
        
        Args:
            task_id: The ID of the completed task
            completion_data: Data about the completion including memory context
        """
        try:
            # Extract relevant data from completion
            memory_context = completion_data.get("memory_context", "")
            user_mood = completion_data.get("user_mood", "") 
            timestamp = completion_data.get("timestamp", datetime.now(timezone.utc).isoformat())
            
            # Track emotional journey alongside task completion
            await self._track_emotional_journey(task_id, user_mood, timestamp)
            
            # Update task status with enriched context
            update_result = await self._update_task_status(
                task_id=task_id,
                status="completed",
                context=memory_context
            )
            
            # Store completion as semantic memory with emotional context
            importance = 0.5  # Default importance
            
            # Adjust importance based on user engagement signals
            if user_mood:
                # Emotionally significant moments deserve higher importance
                importance = 0.7
            if memory_context and len(memory_context) > 100:
                # Detailed reflections suggest this was meaningful to the user
                importance = 0.8
            if update_result.get("progress_metrics", {}).get("milestone_reached", False):
                # Milestones are highly important moments in the journey
                importance = 0.9
            
            # Create enriched memory entry with emotional awareness
            content = f"Completed '{update_result.get('title', 'a task')}'"
            if user_mood:
                content += f" feeling {user_mood}"
            
            # Store as semantic memory with enriched metadata
            await self.semantic_memory_manager.store_memory(
                event_type="task_completion",  # More specific event type
                content=content,
                metadata={
                    "task_id": task_id,
                    "title": update_result.get("title"),
                    "user_mood": user_mood,
                    "has_reflection": bool(memory_context),
                    "supportive_message": update_result.get("supportive_message"),
                    "progress_metrics": update_result.get("progress_metrics"),
                    "timestamp": timestamp
                },
                importance=importance
            )
            
            return update_result
            
        except Exception as e:
            self.logger.error(f"Error updating task completion: {e}")
            raise
            
    async def get_task_history(self, task_id: str) -> List[Dict[str, Any]]:
        """Get task history from semantic memory."""
        try:
            memories = await self.semantic_memory_manager.query_memories(
                query=f"Get history for task {task_id}",
                k=10,
                event_types=["hta_update", "task_completion"]
            )
            return memories
        except Exception as e:
            self.logger.error(f"Error getting task history: {e}")
            raise
            
    async def _update_task_status(self, 
                                 task_id: str, 
                                 status: str,
                                 context: str = "") -> Dict[str, Any]:
        """
        Update task status with context and return enriched result with emotional awareness.
        
        This method updates the status of a task in the hierarchy and enriches the response
        with supportive messaging and progress recognition to create a more emotionally
        resonant experience for users completing their journey.
        
        Args:
            task_id: Unique identifier for the task
            status: New status (e.g., "completed", "pending")
            context: Optional user-provided context or reflection
            
        Returns:
            Dict containing update results, supportive messaging, and progress metrics
        """
        if task_id not in self.task_hierarchies:
            self.logger.warning(f"Cannot update task {task_id}: Not found in hierarchies")
            return {
                "success": False,
                "message": "We couldn't find this part of your journey. Let's make sure we're on the same path.",
                "task_id": task_id
            }
        
        # Get the current state of the task
        task_data = self.task_hierarchies.get(task_id, {})
        old_status = task_data.get("status", "unknown")
        title = task_data.get("title", "this step")
        
        # Update the task status
        task_data["status"] = status
        task_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        # Store user context if provided
        if context:
            if "reflections" not in task_data:
                task_data["reflections"] = []
            
            task_data["reflections"].append({
                "content": context,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        # Update the hierarchy
        self.task_hierarchies[task_id] = task_data
        
        # Calculate progress metrics
        progress_metrics = self._calculate_progress_metrics(task_id)
        
        # Generate supportive messaging based on progress
        supportive_message = self._generate_supportive_message(
            task_title=title,
            old_status=old_status,
            new_status=status,
            progress_metrics=progress_metrics,
            has_reflection=bool(context)
        )
        
        # Create enriched result
        result = {
            "success": True,
            "task_id": task_id,
            "old_status": old_status,
            "new_status": status,
            "title": title,
            "supportive_message": supportive_message,
            "progress_metrics": progress_metrics,
            "reflection_added": bool(context)
        }
        
        self.logger.info(f"Updated task '{title}' ({task_id}) status from '{old_status}' to '{status}'")
        return result
    
    async def _track_emotional_journey(self, task_id: str, user_mood: Optional[str], timestamp: str) -> None:
        """
        Track the emotional journey of the user through task completion.
        
        Args:
            task_id: The task identifier
            user_mood: The user's reported mood (if provided)
            timestamp: When the mood was recorded
        """
        if not user_mood:
            return  # No mood to track
        
        # Ensure the task exists in our hierarchy
        if task_id not in self.task_hierarchies:
            self.logger.warning(f"Cannot track emotional journey: Task {task_id} not found")
            return
        
        task_data = self.task_hierarchies[task_id]
        
        # Initialize emotional journey tracking if not present
        if "emotional_journey" not in task_data:
            task_data["emotional_journey"] = []
        
        # Add the current mood to the journey
        task_data["emotional_journey"].append({
            "mood": user_mood,
            "timestamp": timestamp
        })
        
        # Update the hierarchy
        self.task_hierarchies[task_id] = task_data
        self.logger.debug(f"Tracked emotional journey for task {task_id}: {user_mood}")
    
    def _calculate_progress_metrics(self, task_id: str) -> Dict[str, Any]:
        """
        Calculate progress metrics for a task and its context in the hierarchy.
        
        Args:
            task_id: The task identifier
            
        Returns:
            Dictionary of progress metrics
        """
        # Default metrics
        metrics = {
            "total_steps": 0,
            "completed_steps": 0,
            "completion_percentage": 0,
            "milestone_reached": False,
            "streak": 0,
            "recent_velocity": 0
        }
        
        # Ensure the task exists
        if task_id not in self.task_hierarchies:
            return metrics
        
        task_data = self.task_hierarchies[task_id]
        
        # If task has children, calculate child completion metrics
        if "children" in task_data and task_data["children"]:
            children = task_data["children"]
            metrics["total_steps"] = len(children)
            metrics["completed_steps"] = sum(1 for child_id in children 
                                        if child_id in self.task_hierarchies 
                                        and self.task_hierarchies[child_id].get("status") == "completed")
            
            if metrics["total_steps"] > 0:
                metrics["completion_percentage"] = round((metrics["completed_steps"] / metrics["total_steps"]) * 100)
            
            # Check if this completion is a milestone (25%, 50%, 75%, 100%)
            milestone_thresholds = [25, 50, 75, 100]
            for threshold in milestone_thresholds:
                exact_count_for_threshold = (threshold / 100) * metrics["total_steps"]
                if metrics["completed_steps"] == round(exact_count_for_threshold):
                    metrics["milestone_reached"] = True
                    metrics["milestone_percentage"] = threshold
                    break
        
        # Calculate completion streak and velocity
        if "reflections" in task_data:
            # Sort reflections by timestamp
            reflections = sorted(task_data["reflections"], 
                                key=lambda r: r.get("timestamp", "2000-01-01T00:00:00Z"))
            
            # Calculate streak (consecutive days with completions)
            if reflections:
                # Get dates of activities
                dates = set()
                for reflection in reflections:
                    timestamp = reflection.get("timestamp")
                    if timestamp:
                        try:
                            # Convert timestamp to date string
                            date_str = timestamp.split("T")[0]  # Extract YYYY-MM-DD
                            dates.add(date_str)
                        except (IndexError, AttributeError):
                            continue
                
                # Sort dates
                sorted_dates = sorted(list(dates))
                
                # Calculate streak
                if sorted_dates:
                    current_streak = 1
                    max_streak = 1
                    
                    for i in range(1, len(sorted_dates)):
                        # Convert to datetime for date comparison
                        prev_date = datetime.strptime(sorted_dates[i-1], "%Y-%m-%d").date()
                        curr_date = datetime.strptime(sorted_dates[i], "%Y-%m-%d").date()
                        
                        # Check if dates are consecutive
                        if (curr_date - prev_date).days == 1:
                            current_streak += 1
                            max_streak = max(max_streak, current_streak)
                        else:
                            current_streak = 1
                    
                    metrics["streak"] = max_streak
                
                # Calculate velocity (completions per week)
                if len(reflections) >= 2:
                    try:
                        first_timestamp = reflections[0].get("timestamp")
                        last_timestamp = reflections[-1].get("timestamp")
                        
                        if first_timestamp and last_timestamp:
                            first_dt = datetime.fromisoformat(first_timestamp.replace('Z', '+00:00'))
                            last_dt = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
                            
                            # Calculate duration in weeks
                            duration_days = (last_dt - first_dt).total_seconds() / (24 * 3600)
                            duration_weeks = max(duration_days / 7, 1)  # At least 1 week to avoid division by zero
                            
                            # Calculate velocity
                            metrics["recent_velocity"] = round(len(reflections) / duration_weeks, 1)
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Error calculating velocity: {e}")
        
        return metrics
    
    def _generate_supportive_message(self, task_title: str, old_status: str, new_status: str, 
                                     progress_metrics: Dict[str, Any], has_reflection: bool) -> str:
        """
        Generate a supportive, emotionally resonant message based on the task completion context.
        
        Args:
            task_title: Title of the completed task
            old_status: Previous status
            new_status: New status
            progress_metrics: Dictionary of progress metrics
            has_reflection: Whether the user added a reflection
            
        Returns:
            A supportive message string
        """
        # Check if this is a completion event
        is_completion = (old_status != "completed" and new_status == "completed")
        if not is_completion:
            return f"Status updated to {new_status}."  # Simple message for non-completion updates
        
        # Create a pool of base messages
        base_messages = [
            f"You've completed '{task_title}'! Well done on taking this step forward.",
            f"Wonderful progress! '{task_title}' is now complete.",
            f"'{task_title}' is finished! Each step brings you closer to your goal.",
            f"You've finished '{task_title}'. This moment is worth celebrating.",
            f"'{task_title}' complete! Your journey continues to unfold beautifully."
        ]
        
        # Select a random base message
        message = random.choice(base_messages)
        
        # Enhance the message based on context
        enhancements = []
        
        # 1. Milestone reached
        if progress_metrics.get("milestone_reached"):
            milestone_percentage = progress_metrics.get("milestone_percentage", 0)
            if milestone_percentage == 100:
                enhancements.append(f"You've completed all the steps in this branch! What an accomplishment.")
            elif milestone_percentage == 75:
                enhancements.append(f"You're now 75% of the way through this branch. The end is in sight!")
            elif milestone_percentage == 50:
                enhancements.append(f"You've reached the halfway point of this branch. Momentum is building!")
            elif milestone_percentage == 25:
                enhancements.append(f"You've completed 25% of this branch. A strong start!")
        
        # 2. Streaks
        streak = progress_metrics.get("streak", 0)
        if streak >= 3:
            enhancements.append(f"You're on a {streak}-day streak! Consistency is transformative.")
        
        # 3. Velocity
        velocity = progress_metrics.get("recent_velocity", 0)
        if velocity >= 3:
            enhancements.append(f"You're moving at an impressive pace of {velocity} completions per week.")
        
        # 4. Reflection added
        if has_reflection:
            reflection_notes = [
                "Your reflection adds meaning to this accomplishment.",
                "Taking time to reflect deepens your journey.",
                "Your thoughts enrich this moment of progress."
            ]
            enhancements.append(random.choice(reflection_notes))
        
        # Add up to two enhancements to avoid overwhelming
        if enhancements:
            # Shuffle and select up to 2 enhancements
            random.shuffle(enhancements)
            selected_enhancements = enhancements[:min(2, len(enhancements))]
            message += f" {' '.join(selected_enhancements)}"
        
        return message
    
    def _is_meaningful_node(self, node: HTANode) -> bool:
        """
        Determine if a node represents a meaningful moment in the user's journey.
        
        Args:
            node: The HTANode to evaluate
            
        Returns:
            True if the node is considered meaningful
        """
        if not node:
            return False
        
        # Criteria for meaningful nodes
        
        # 1. Major phase nodes are always meaningful
        if hasattr(node, 'is_major_phase') and node.is_major_phase:
            return True
        
        # 2. Nodes with high priority or significance
        if hasattr(node, 'priority') and node.priority and node.priority >= 0.7:
            return True
        
        # 3. Nodes with certain keywords in title suggesting importance
        meaningful_keywords = ['milestone', 'breakthrough', 'key', 'critical', 
                              'essential', 'foundation', 'complete', 'finish']
        if node.title and any(keyword in node.title.lower() for keyword in meaningful_keywords):
            return True
        
        # 4. Leaf nodes (no children) are often meaningful completions
        if not node.children or len(node.children) == 0:
            return True
        
        # Default to false for other nodes
        return False
    
    def _record_meaningful_transition(self, node: HTANode, tree: HTATree) -> None:
        """
        Record a meaningful state transition for potential celebration or reflection prompts.
        This method doesn't modify the tree itself.
        
        Args:
            node: The node that had a meaningful status change
            tree: The HTATree containing the node
        """
        if not hasattr(tree, '_meaningful_transitions'):
            tree._meaningful_transitions = []
        
        # Record this transition for potential later use
        transition = {
            "node_id": node.id,
            "title": node.title,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "is_major_phase": getattr(node, 'is_major_phase', False),
            "parent_id": node.parent.id if node.parent else None,
            "parent_title": node.parent.title if node.parent else None
        }
        
        tree._meaningful_transitions.append(transition)
        self.logger.debug(f"Recorded meaningful transition for node: {node.title}")
    
    async def _check_for_meaningful_moments(self, tree: HTATree, snapshot: MemorySnapshot) -> None:
        """
        Check for meaningful moments in the user's journey that might warrant celebration,
        reflection prompts, or special recognition.
        
        Args:
            tree: The current HTATree
            snapshot: The current MemorySnapshot
        """
        # Skip if tree hasn't recorded any meaningful transitions
        if not hasattr(tree, '_meaningful_transitions') or not tree._meaningful_transitions:
            return
        
        # Process each recorded transition
        for transition in tree._meaningful_transitions:
            is_major_phase = transition.get('is_major_phase', False)
            node_title = transition.get('title', 'unknown')
            node_id = transition.get('node_id')
            
            # Major phase completions are especially significant
            if is_major_phase:
                message = f"Major milestone achieved: {node_title}"
                importance = 0.9  # High importance
                
                # Store this celebration moment in semantic memory
                try:
                    await self.semantic_memory_manager.store_memory(
                        event_type="milestone",  # Special event type
                        content=message,
                        metadata={
                            "transition": transition,
                            "milestone_type": "major_phase",
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        },
                        importance=importance
                    )
                    self.logger.info(f"Recorded major phase completion: {node_title}")
                except Exception as e:
                    self.logger.error(f"Failed to record major phase completion: {e}")
            
            # Other meaningful transitions (could expand with more specialized handling)
            elif node_id:
                # Store in semantic memory with moderate importance
                try:
                    await self.semantic_memory_manager.store_memory(
                        event_type="progress",
                        content=f"Completed meaningful step: {node_title}",
                        metadata={
                            "transition": transition,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        },
                        importance=0.7  # Moderate importance
                    )
                except Exception as e:
                    self.logger.error(f"Failed to record meaningful step completion: {e}")
        
        # Clear processed transitions
        tree._meaningful_transitions = []

    async def analyze_task_patterns(self, task_id: str) -> Dict[str, Any]:
        """
        Analyze patterns in task history using semantic memory to create meaningful moments of
        discovery and insight for the user throughout their journey.
        
        This enhanced analysis goes beyond technical patterns to identify emotional trends,
        growth moments, and personalized insights that can surprise and delight users with
        meaningful revelations about their journey.
        """
        try:
            # Get rich task history with emotional context
            history = await self.get_task_history(task_id)
            
            if not history:
                return {"patterns": [], "insights": [], "emotional_journey": [], "growth_narrative": ""}
            
            # Extract emotional journey data if available
            emotional_journey = []
            for entry in history:
                if isinstance(entry, dict) and entry.get("metadata") and entry["metadata"].get("user_mood"):
                    emotional_journey.append({
                        "timestamp": entry.get("timestamp"),
                        "mood": entry["metadata"]["user_mood"],
                        "task": entry["metadata"].get("title", "")
                    })
            
            # Build enriched analysis prompt with emotional awareness
            history_text = "\n".join([
                f"- [{m.get('timestamp', 'unknown')}] {m.get('content', '')} {' (Mood: ' + m.get('metadata', {}).get('user_mood', '') + ')' if m.get('metadata', {}).get('user_mood') else ''}"
                for m in history
            ])
            
            # Create a more nuanced prompt that looks for meaningful patterns
            prompt = f"""
            Task History for User's Journey:
            {history_text}
            
            Analyze this journey history and identify:
            1. Meaningful patterns in how the user approaches their goals
            2. Emotional trends or shifts throughout their journey
            3. Growth moments or breakthroughs
            4. Personal strengths the user has demonstrated
            5. Potential helpful insights for their continued journey
            
            Most importantly, craft a brief, supportive narrative about their growth journey so far.
            Use a warm, encouraging tone that celebrates their progress while acknowledging any challenges.
            """
            
            # Generate enhanced analysis with emotional intelligence
            analysis = await self.llm_client.generate(prompt)
            
            # Parse analysis into a structured format with enhanced categories
            lines = [l.strip() for l in analysis.split("\n") if l.strip()]
            patterns = []
            insights = []
            strengths = []
            emotional_trends = []
            growth_narrative = ""
            
            current_section = None
            narrative_lines = []
            
            for line in lines:
                line_lower = line.lower()
                
                # Detect sections
                if "pattern" in line_lower:
                    current_section = "patterns"
                elif "emotional trend" in line_lower:
                    current_section = "emotional_trends"
                elif "growth moment" in line_lower or "breakthrough" in line_lower:
                    current_section = "growth_moments"  
                elif "strength" in line_lower:
                    current_section = "strengths"
                elif "insight" in line_lower:
                    current_section = "insights"
                elif any(marker in line_lower for marker in ["narrative", "journey", "story", "progress"]):
                    current_section = "narrative"
                    continue  # Skip the header itself
                
                # Capture content in appropriate categories
                if line.startswith("-") or line.startswith("*"):
                    clean_line = line[1:].strip()
                    if current_section == "patterns":
                        patterns.append(clean_line)
                    elif current_section == "emotional_trends":
                        emotional_trends.append(clean_line)
                    elif current_section == "strengths":
                        strengths.append(clean_line)
                    elif current_section == "insights":
                        insights.append(clean_line)
                # Collect narrative lines
                elif current_section == "narrative":
                    narrative_lines.append(line)
            
            # Combine narrative lines into a cohesive paragraph
            if narrative_lines:
                growth_narrative = " ".join(narrative_lines)
            
            # Return enriched analysis with emotional intelligence
            return {
                "patterns": patterns,
                "emotional_trends": emotional_trends,
                "strengths": strengths,
                "insights": insights,
                "emotional_journey": emotional_journey,
                "growth_narrative": growth_narrative
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing task patterns: {e}")
            raise
