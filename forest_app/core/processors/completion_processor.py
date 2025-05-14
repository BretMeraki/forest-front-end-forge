# forest_app/core/processors/completion_processor.py

import logging
import inspect
import uuid
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
from sqlalchemy.orm import Session
import json
from datetime import datetime, timezone
from uuid import UUID

# Core imports with error handling
try:
    from forest_app.core.snapshot import MemorySnapshot
    from forest_app.core.utils import clamp01
    from forest_app.core.services import HTAService, SemanticMemoryManager
    from forest_app.core.services.enhanced_hta.memory import HTAMemoryManager
    from forest_app.modules.xp_mastery import XPMastery
    from forest_app.modules.task_engine import TaskEngine
    from forest_app.modules.logging_tracking import TaskFootprintLogger, ReflectionLogLogger
    from forest_app.integrations.llm import LLMClient
    from forest_app.persistence.models import HTANodeModel, MemorySnapshotModel, TaskFootprintModel
    from forest_app.persistence.repository import HTATreeRepository
    from forest_app.core.transaction_decorator import transaction_protected
    from forest_app.core.roadmap_models import RoadmapManifest
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import required modules: {e}")
    # Define dummy classes if imports fail
    class MemorySnapshot: pass
    class HTAService: pass
    class SemanticMemoryManager: pass
    class HTAMemoryManager: pass
    class XPMastery: pass
    class TaskEngine: pass
    class TaskFootprintLogger: pass
    class ReflectionLogLogger: pass
    class LLMClient: pass
    class HTANodeModel: pass
    class HTATreeRepository: pass
    class RoadmapManifest: pass
    def clamp01(x): return x
    def transaction_protected(func=None): 
        def decorator(f): return f
        return decorator if func is None else decorator(func)

# Feature flags with error handling
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    def is_enabled(feature): return False
    class Feature:
        CORE_HTA = "FEATURE_ENABLE_CORE_HTA"
        XP_MASTERY = "FEATURE_ENABLE_XP_MASTERY"
        # Add others if needed for checks within this processor

# Constants with error handling
try:
    from forest_app.config.constants import WITHERING_COMPLETION_RELIEF
except ImportError:
    WITHERING_COMPLETION_RELIEF = 0.1  # Default value if import fails

logger = logging.getLogger(__name__)

# --- Completion Processor Class ---

class CompletionProcessor:
    """Processes task completions with semantic memory integration and transactional integrity.
    
    This processor ensures that task completion updates are atomic, properly logged,
    and include positive reinforcement messages for users. It maintains transactional 
    integrity across node status updates, roadmap manifest updates, and memory snapshots.
    """

    def __init__(self, 
                 llm_client: LLMClient,
                 hta_service: HTAService, 
                 tree_repository: HTATreeRepository,
                 memory_manager: HTAMemoryManager,
                 semantic_memory_manager: Optional[SemanticMemoryManager] = None,
                 task_logger: Optional[TaskFootprintLogger] = None,
                 reflection_logger: Optional[ReflectionLogLogger] = None):
        """Initialize the CompletionProcessor with required services.
        
        Args:
            llm_client: Client for LLM operations (insights, reinforcement)
            hta_service: Service for HTA operations
            tree_repository: Repository for tree/node operations
            memory_manager: Manager for memory snapshot operations
            semantic_memory_manager: Optional manager for semantic memory operations
            task_logger: Optional logger for task footprints
            reflection_logger: Optional logger for reflections
        """
        self.llm_client = llm_client
        self.hta_service = hta_service
        self.tree_repository = tree_repository
        self.memory_manager = memory_manager
        self.semantic_memory_manager = semantic_memory_manager
        self.task_logger = task_logger
        self.reflection_logger = reflection_logger
        self.logger = logging.getLogger(__name__)

    @transaction_protected()
    async def process_node_completion(self, 
                               node_id: Union[str, UUID], 
                               user_id: Union[str, UUID],
                               success: bool = True,
                               reflection: Optional[str] = None,
                               db_session: Optional[Session] = None) -> Dict[str, Any]:
        """Process a task/node completion with full transactional integrity.
        
        This method implements the requirements from Task 1.5, ensuring atomic updates 
        to both the HTA node and roadmap manifest, memory snapshot updates, audit logging,
        and positive reinforcement.
        
        Args:
            node_id: The UUID of the HTA node to mark as completed
            user_id: The UUID of the user completing the node
            success: Whether the task was completed successfully
            reflection: Optional reflection from the user about the completion
            db_session: Optional DB session for transaction context
            
        Returns:
            Dictionary with completion results, including supportive message
            
        Raises:
            ValueError: If the node doesn't exist or doesn't belong to the user
            RuntimeError: If the update operation fails
        """
        # Convert string IDs to UUID if necessary
        if isinstance(node_id, str):
            node_id = UUID(node_id)
        if isinstance(user_id, str):
            user_id = UUID(user_id)
            
        try:
            # 1. Retrieve the node
            node = await self.tree_repository.get_node_by_id(node_id)
            if not node:
                self.logger.error(f"Node {node_id} not found")
                raise ValueError(f"Node {node_id} not found")
                
            if node.user_id != user_id:
                self.logger.error(f"User {user_id} does not own node {node_id}")
                raise ValueError(f"User {user_id} does not own node {node_id}")
                
            # 2. Check if already completed (idempotency)
            if node.status == "completed":
                self.logger.info(f"Node {node_id} already completed")
                # Find existing reinforcement message if available
                existing_footprint = await self.tree_repository.get_task_footprint(node_id)
                message = existing_footprint.reinforcement_message if existing_footprint else "This task is already completed."
                return {
                    "status": "already_completed",
                    "message": message,
                    "reinforcement_message": message,
                    "node_id": str(node_id),
                    "title": node.title
                }
                
            # 3. Get the tree for manifest update
            tree = await self.tree_repository.get_tree_by_id(node.tree_id)
            if not tree:
                self.logger.error(f"Tree {node.tree_id} not found")
                raise ValueError(f"Tree {node.tree_id} not found")
                
            # 4. Update node status to completed
            old_status = node.status
            completion_time = datetime.now(timezone.utc)
            success = await self.tree_repository.update_node_status(
                node_id=node_id,
                new_status="completed",
                update_internal_details={
                    "completed_at": completion_time.isoformat(),
                    "completion_status": 1.0 if success else 0.0,
                    "reflection": reflection or ""
                }
            )
            
            if not success:
                self.logger.error(f"Failed to update node {node_id} status")
                raise RuntimeError(f"Failed to update node {node_id} status")
                
            # 5. Update the manifest to keep it synchronized
            manifest_updated = False
            if node.roadmap_step_id:
                # Ensure required fields for RoadmapManifest are present
                manifest_data = dict(tree.manifest)
                if 'tree_id' not in manifest_data:
                    manifest_data['tree_id'] = str(getattr(tree, 'id', uuid.uuid4()))
                if 'user_goal' not in manifest_data:
                    manifest_data['user_goal'] = "dummy goal"
                manifest = RoadmapManifest(**manifest_data)
                manifest = manifest.update_step_status(node.roadmap_step_id, "completed")
                
                # Save updated manifest to tree
                tree.manifest = manifest.dict()
                await self.tree_repository.update_tree(tree)
                manifest_updated = True
                
            # 6. Update parent node completion count if applicable
            if node.parent_id:
                increment_success, new_count = await self.tree_repository.increment_branch_completion_count(node.parent_id)
                if increment_success:
                    self.logger.info(f"Incremented completion count for parent {node.parent_id} to {new_count}")
                
            # 7. Generate positive reinforcement message
            reinforcement = await self._generate_positive_reinforcement(node)
            
            # 8. Create task footprint for audit logging
            footprint = None
            if self.task_logger:
                footprint = await self.task_logger.log_task_completion(
                    user_id=user_id,
                    node_id=node_id,
                    completed_at=completion_time,
                    success=success,
                    reinforcement_message=reinforcement
                )
                
            # 9. Add reflection log if provided
            if reflection and self.reflection_logger:
                await self.reflection_logger.log_reflection(
                    user_id=user_id,
                    node_id=node_id,
                    content=reflection,
                    timestamp=completion_time
                )
                
            # 10. Update memory with completion (MemorySnapshot)
            await self.memory_manager.update_memory_with_completion(user_id=user_id, node=node)
            
            # 11. Create result with supportive message
            result = {
                "status": "completed",
                "node_id": str(node_id),
                "title": node.title,
                "old_status": old_status,
                "new_status": "completed",
                "reinforcement_message": reinforcement,
                "is_major_phase": node.is_major_phase,
                "completion_time": completion_time.isoformat(),
                "success": success,
                "manifest_updated": manifest_updated,
                "roadmap_step_id": str(node.roadmap_step_id) if node.roadmap_step_id else None
            }
            
            self.logger.info(f"Successfully completed node {node_id} for user {user_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing node completion: {e}")
            # The transaction_protected decorator will handle rollback
            raise
            
    async def _generate_positive_reinforcement(self, node: HTANodeModel) -> str:
        """Generate a positive, supportive message for task completion.
        
        The message is more enthusiastic and supportive for major phases.
        
        Args:
            node: The completed node
            
        Returns:
            Supportive message string
        """
        try:
            if self.llm_client and is_enabled(Feature.CORE_HTA):
                # We have an LLM client available, generate personalized message
                prompt = f"""
                Generate a brief, positive reinforcement message for a user who just completed 
                a task titled: "{node.title}".
                
                Task description: {node.description}
                
                {"This is a MAJOR milestone in their journey!" if node.is_major_phase else "This is a regular task in their journey."}
                
                The message should be:
                1. Supportive and encouraging
                2. Brief (1-2 sentences)
                3. Genuine and not overly cheerful
                4. {'More celebratory for this significant achievement' if node.is_major_phase else 'Appropriate for a routine task'}
                
                Output the message directly with no additional text or formatting.
                """
                
                response = await self.llm_client.generate(prompt, temperature=0.7)
                if response and len(response.strip()) > 0:
                    return response.strip()
                    
            # Fall back to template messages if LLM fails or isn't available
            if node.is_major_phase:
                templates = [
                    f"Excellent work completing '{node.title}'! This is a significant milestone in your journey.",
                    f"Major achievement unlocked! '{node.title}' complete - you've made important progress!",
                    f"Amazing job on completing '{node.title}'! This is a meaningful step forward.",
                    f"Congratulations on this important milestone! '{node.title}' is now behind you.",
                ]
            else:
                templates = [
                    f"Well done completing '{node.title}'!",
                    f"Nice work! '{node.title}' is now complete.",
                    f"Task complete! You're making steady progress.",
                    f"Great job finishing '{node.title}'.",
                ]
                
            # Select a template based on a hash of the node ID for consistent messages
            index = int(str(node.id)[-1], 16) % len(templates)
            return templates[index]
            
        except Exception as e:
            self.logger.warning(f"Error generating reinforcement message: {e}")
            return f"Task completed: {node.title}"
    
    @transaction_protected()
    async def process_completion(self, 
                               task_id: str, 
                               completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy method for compatibility with existing code.
        
        Args:
            task_id: The ID of the completed task
            completion_data: Data about the completion
            
        Returns:
            Dictionary with completion results
        """
        try:
            # Query relevant task memories if semantic memory manager available
            relevant_memories = []
            memory_context = ""
            
            if self.semantic_memory_manager:
                try:
                    relevant_memories = await self.semantic_memory_manager.query_memories(
                        query=completion_data.get("summary", ""),
                        k=3,
                        event_types=["task_completion"]
                    )
                    
                    # Build memory context
                    memory_context = self._build_memory_context(relevant_memories)
                except Exception as mem_err:
                    self.logger.warning(f"Non-critical error querying semantic memories: {mem_err}")
            
            # Update HTA with completion and memory context
            hta_result = await self.hta_service.update_task_completion(
                task_id=task_id,
                completion_data={
                    **completion_data,
                    "memory_context": memory_context
                }
            )
            
            # Generate completion insights
            insights = []
            if self.llm_client:
                try:
                    insights = await self._generate_completion_insights(
                        task_id=task_id,
                        completion_data=completion_data,
                        hta_result=hta_result,
                        memory_context=memory_context
                    )
                except Exception as insight_err:
                    self.logger.warning(f"Non-critical error generating insights: {insight_err}")
            
            return {
                "task_id": task_id,
                "hta_result": hta_result,
                "insights": insights,
                "relevant_memories": relevant_memories
            }
            
        except Exception as e:
            self.logger.error(f"Error processing task completion: {e}")
            raise

    def _build_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """Build a context string from relevant task memories."""
        if not memories:
            return ""
            
        context_parts = ["Previous related task completions:"]
        for memory in memories:
            timestamp = datetime.fromisoformat(memory.get("timestamp", "")).strftime("%Y-%m-%d")
            content = memory.get("content", "")
            metadata = memory.get("metadata", {})
            context_parts.append(f"- [{timestamp}] {content}")
            if metadata.get("learnings"):
                context_parts.append(f"  Learnings: {metadata['learnings']}")
            
        return "\n".join(context_parts)

    async def _generate_completion_insights(self,
                                         task_id: str,
                                         completion_data: Dict[str, Any],
                                         hta_result: Dict[str, Any],
                                         memory_context: str) -> List[str]:
        """Generate insights about the task completion using LLM."""
        prompt = f"""
        Task Completion:
        - ID: {task_id}
        - Summary: {completion_data.get('summary', '')}
        - Status: {completion_data.get('status', '')}
        
        HTA Update Result:
        {json.dumps(hta_result)}
        
        Memory Context:
        {memory_context}
        
        Based on this task completion and historical context, generate insights about:
        1. Task completion patterns
        2. Progress towards goals
        3. Learning opportunities
        
        Provide 2-3 brief, thoughtful insights.
        """
        
        response = await self.llm_client.generate(prompt, temperature=0.7)
        
        # Parse insights from response
        insights = [line.strip() for line in response.split("\n") if line.strip()]
        return insights
