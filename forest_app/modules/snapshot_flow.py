# Rewritten snapshot module (e.g., forest_app/modules/snapshot_utils.py or similar)
import json
import logging
import os
import sys # For stderr
from datetime import datetime, timezone
from collections import deque
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from forest_app.core.snapshot import MemorySnapshot

# Import shared types
from forest_app.modules.types import (
    SemanticMemoryProtocol,
    MemoryDict,
    SnapshotDict
)

# --- Import Feature Flags ---
# Need to check flags to determine if data *should* be present
try:
    from forest_app.core.feature_flags import Feature, is_enabled
    # Use the real is_enabled if available
except ImportError:
    # Fallback if feature flags module isn't available
    print("ERROR: Snapshot module could not import feature flags. Cannot reliably check features.", file=sys.stderr)
    # Define minimal dummy Feature enum and is_enabled
    class Feature:
        XP_MASTERY = "FEATURE_ENABLE_XP_MASTERY"
        SHADOW_ANALYSIS = "FEATURE_ENABLE_SHADOW_ANALYSIS"
        METRICS_SPECIFIC = "FEATURE_ENABLE_METRICS_SPECIFIC"
        DEVELOPMENT_INDEX = "FEATURE_ENABLE_DEVELOPMENT_INDEX"
        NARRATIVE_MODES = "FEATURE_ENABLE_NARRATIVE_MODES"
        SEED_MANAGER = "FEATURE_ENABLE_SEED_MANAGER"

    def is_enabled(feature) -> bool: return False # Assume all off if flags missing

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from forest_app.core.snapshot import MemorySnapshot

logger = logging.getLogger(__name__)

class SnapshotFlowController:
    """
    Manages the flow of snapshots, including semantic memory integration.
    Combines CallbackTrigger and SnapshotRotatingSaver functionality.
    """

    def __init__(self, 
                 frequency: int = 5, 
                 max_snapshots: int = 10,
                 semantic_memory_manager: Optional[SemanticMemoryProtocol] = None):
        if frequency <= 0:
            logger.warning("Snapshot frequency must be positive, defaulting to 5.")
            frequency = 5
        if max_snapshots <= 0:
            logger.warning("Max snapshots must be positive, defaulting to 10.")
            max_snapshots = 10

        self.counter = 0
        self.frequency = frequency
        self.snapshots = deque(maxlen=max_snapshots)
        self.builder = CompressedSnapshotBuilder()
        self.gpt_memory_sync = GPTMemorySync()
        self.semantic_memory_manager = semantic_memory_manager
        self.logger = logging.getLogger(__name__)
        logger.info("SnapshotFlowController initialized with frequency %d, max_snapshots %d", frequency, max_snapshots)

    async def register_user_submission(self, full_snapshot: Any) -> Dict[str, Any]:
        """
        Processes a user submission, potentially triggering a new snapshot and memory storage.
        Now includes semantic memory processing.
        """
        self.counter += 1
        logger.debug("User submission registered, counter at %d/%d", self.counter, self.frequency)

        # Extract memory-worthy content from the snapshot
        memory_content = self._extract_memory_content(full_snapshot)
        
        # Store semantic memories if we have content and a memory manager
        if memory_content and self.semantic_memory_manager:
            try:
                for content_item in memory_content:
                    await self.semantic_memory_manager.store_memory(
                        event_type=content_item["type"],
                        content=content_item["content"],
                        metadata=content_item.get("metadata", {}),
                        importance=content_item.get("importance", 0.5)
                    )
                logger.debug("Stored %d new memories from snapshot", len(memory_content))
            except Exception as e:
                logger.error("Failed to store semantic memories: %s", e)

        # Regular snapshot processing
        if self.counter >= self.frequency:
            logger.info("Snapshot frequency reached. Building new snapshot.")
            self.counter = 0
            snapshot = self.builder.build(full_snapshot)
            
            # Update semantic memory context if available
            if self.semantic_memory_manager:
                try:
                    # Query relevant memories based on current context
                    relevant_memories = await self.semantic_memory_manager.query_memories(
                        query=self._get_context_query(full_snapshot),
                        k=5
                    )
                    
                    # Update memory context in snapshot
                    if "memory_context" not in snapshot:
                        snapshot["memory_context"] = {}
                    snapshot["memory_context"]["relevant_memories"] = relevant_memories
                    snapshot["memory_context"]["memory_themes"] = self._extract_themes(relevant_memories)
                except Exception as e:
                    logger.error("Failed to update memory context: %s", e)

            # Store the snapshot
            self.snapshots.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "snapshot": snapshot
            })
            logger.info("New snapshot stored with timestamp %s", snapshot.get("timestamp"))
            return snapshot
        return None

    def _extract_memory_content(self, snapshot: Any) -> List[Dict[str, Any]]:
        """
        Extracts memory-worthy content from a snapshot.
        Returns a list of dictionaries containing content to be stored as memories.
        """
        memory_content = []
        
        try:
            # Extract from reflection log
            if hasattr(snapshot, 'reflection_log'):
                for reflection in snapshot.reflection_log:
                    if isinstance(reflection, dict) and 'content' in reflection:
                        memory_content.append({
                            "type": "reflection",
                            "content": reflection['content'],
                            "metadata": {"timestamp": reflection.get('timestamp')},
                            "importance": 0.7  # Reflections are generally important
                        })

            # Extract from task completions
            if hasattr(snapshot, 'task_footprints'):
                for task in snapshot.task_footprints:
                    if isinstance(task, dict) and task.get('status') == 'completed':
                        memory_content.append({
                            "type": "task_completion",
                            "content": f"Completed task: {task.get('title', 'Unknown')}",
                            "metadata": {
                                "task_id": task.get('id'),
                                "timestamp": task.get('completion_time')
                            },
                            "importance": 0.6
                        })

            # Extract from story beats
            if hasattr(snapshot, 'story_beats'):
                for beat in snapshot.story_beats:
                    if isinstance(beat, dict) and 'content' in beat:
                        memory_content.append({
                            "type": "story_beat",
                            "content": beat['content'],
                            "metadata": {"timestamp": beat.get('timestamp')},
                            "importance": 0.8  # Story beats are highly important
                        })

        except Exception as e:
            logger.error("Error extracting memory content: %s", e)

        return memory_content

    def _get_context_query(self, snapshot: Any) -> str:
        """
        Generates a context query for memory retrieval based on current snapshot state.
        """
        context_elements = []
        
        try:
            # Add current priority if available
            if hasattr(snapshot, 'reflection_context'):
                current_priority = snapshot.reflection_context.get('current_priority')
                if current_priority:
                    context_elements.append(f"Current priority: {current_priority}")

            # Add recent insight if available
            recent_insight = snapshot.reflection_context.get('recent_insight')
            if recent_insight:
                context_elements.append(f"Recent insight: {recent_insight}")

            # Add current task context
            if hasattr(snapshot, 'task_backlog') and snapshot.task_backlog:
                current_tasks = [task.get('title') for task in snapshot.task_backlog[:3] if isinstance(task, dict)]
                if current_tasks:
                    context_elements.append(f"Current tasks: {', '.join(current_tasks)}")

        except Exception as e:
            logger.error("Error generating context query: %s", e)
            return "Find relevant memories for the current context"

        # Combine elements or use default
        if context_elements:
            return " ".join(context_elements)
        return "Find relevant memories for the current context"

    def _extract_themes(self, memories: List[Dict[str, Any]]) -> List[str]:
        """
        Extracts common themes from a list of memories.
        """
        themes = set()
        
        try:
            # Extract themes from memory content
            for memory in memories:
                content = memory.get('content', '')
                # Add basic theme extraction logic here
                # For now, just use memory types as themes
                if memory.get('event_type'):
                    themes.add(memory['event_type'])
                
                # You could add more sophisticated theme extraction here
                # For example, using NLP to identify key topics
                
        except Exception as e:
            logger.error("Error extracting themes: %s", e)
            
        return list(themes)

    def get_latest_context(self) -> str:
        """
        Gets the LLM context string based on the most recently stored snapshot.
        """
        latest_record = self.snapshots[-1] if self.snapshots else None
        if latest_record and 'snapshot' in latest_record:
            logger.info("Providing LLM context from latest stored snapshot.")
            # Use the snapshot part of the record
            return self.gpt_memory_sync.inject_into_context(latest_record["snapshot"])
        else:
            logger.warning("No recent snapshot available in saver to generate context.")
            # Provide default message via memory_sync
            return self.gpt_memory_sync.inject_into_context(None)

    def force_snapshot(self, full_snapshot: Any) -> Dict[str, Any]:
         """Forces snapshot creation, storage, and context generation."""
         logger.info("SnapshotFlowController forcing snapshot.")
         # Force trigger to build
         forced_snapshot = self.builder.build(full_snapshot)
         # Store it
         self.snapshots.append({
             "timestamp": datetime.now(timezone.utc).isoformat(),
             "snapshot": forced_snapshot
         })
         # Prepare context
         context_string = self.gpt_memory_sync.inject_into_context(forced_snapshot)
         return {
             "synced": True, # Considered synced as it was just forced
             "context_injection": context_string,
             "compressed_snapshot": forced_snapshot,
         }

    # --- Optional: Add methods for saving/loading state ---
    def save_state_to_json(self, filepath: str):
         """Exports the rotating snapshots to JSON."""
         self.snapshots.export_to_json(filepath)

    def load_state_from_json(self, filepath: str):
         """Loads rotating snapshots from JSON."""
         self.snapshots.load_from_json(filepath)

    async def save_snapshot(self, snapshot: MemorySnapshot) -> bool:
        """Save a snapshot with semantic memory integration."""
        try:
            # Update semantic memory context before saving
            if self.semantic_memory_manager:
                # Get recent memories
                recent_memories = await self.semantic_memory_manager.get_recent_memories(limit=5)
                
                # Get relevant memories based on current context
                context_query = self._build_context_query(snapshot)
                relevant_memories = await self.semantic_memory_manager.query_memories(
                    query=context_query,
                    k=5
                )
                
                # Extract themes from memories
                memory_themes = await self.semantic_memory_manager.extract_themes(
                    memories=relevant_memories
                )
                
                # Update memory context in snapshot
                snapshot.update_memory_context(
                    recent_memories=recent_memories,
                    relevant_memories=relevant_memories,
                    memory_themes=memory_themes,
                    query_info={
                        "query": context_query,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "relevance_score": sum(m.get("relevance", 0) for m in relevant_memories) / len(relevant_memories) if relevant_memories else 0.0,
                        "themes": memory_themes
                    }
                )
                
                # Update semantic memory stats
                stats = await self.semantic_memory_manager.get_memory_stats()
                snapshot.update_semantic_memories(stats_update=stats)

            # Add snapshot to rotation
            self.snapshots.append(snapshot)
            self.counter += 1
            
            # Store snapshot state as memory if significant
            if self.semantic_memory_manager and self._is_significant_snapshot(snapshot):
                await self.semantic_memory_manager.store_memory(
                    event_type="snapshot_state",
                    content=self._generate_snapshot_summary(snapshot),
                    metadata={
                        "timestamp": snapshot.timestamp,
                        "state_metrics": {
                            "shadow_score": snapshot.shadow_score,
                            "capacity": snapshot.capacity,
                            "withering_level": snapshot.withering_level
                        }
                    },
                    importance=self._calculate_snapshot_importance(snapshot)
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving snapshot: {e}")
            return False

    def _build_context_query(self, snapshot: MemorySnapshot) -> str:
        """Build a context query for memory retrieval based on snapshot state."""
        query_parts = []
        
        # Add recent reflections
        if hasattr(snapshot, "current_batch_reflections"):
            recent_reflections = " ".join(snapshot.current_batch_reflections[-3:])  # Last 3 reflections
            query_parts.append(f"Recent reflections: {recent_reflections}")
        
        # Add active tasks
        if hasattr(snapshot, "task_backlog"):
            active_tasks = [t.get("title", "") for t in snapshot.task_backlog[:3] if isinstance(t, dict)]
            if active_tasks:
                query_parts.append(f"Active tasks: {', '.join(active_tasks)}")
        
        # Add emotional state
        if hasattr(snapshot, "shadow_score") and hasattr(snapshot, "capacity"):
            query_parts.append(f"Emotional state - Shadow: {snapshot.shadow_score:.2f}, Capacity: {snapshot.capacity:.2f}")
        
        return " | ".join(query_parts)

    def _is_significant_snapshot(self, snapshot: MemorySnapshot) -> bool:
        """Determine if a snapshot represents a significant state change."""
        try:
            # Check for significant metric changes
            if len(self.snapshots) > 1:
                previous = self.snapshots[-2]
                shadow_change = abs(snapshot.shadow_score - previous.shadow_score) > 0.2
                capacity_change = abs(snapshot.capacity - previous.capacity) > 0.2
                if shadow_change or capacity_change:
                    return True
            
            # Check for task completions
            if hasattr(snapshot, "task_backlog"):
                completed_tasks = [t for t in snapshot.task_backlog if isinstance(t, dict) and t.get("status") == "completed"]
                if completed_tasks:
                    return True
            
            # Check for new reflections
            if hasattr(snapshot, "current_batch_reflections") and snapshot.current_batch_reflections:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking snapshot significance: {e}")
            return False

    def _calculate_snapshot_importance(self, snapshot: MemorySnapshot) -> float:
        """Calculate importance score for snapshot memory."""
        try:
            importance = 0.5  # Base importance
            
            # Adjust based on metric changes
            if len(self.snapshots) > 1:
                previous = self.snapshots[-2]
                shadow_change = abs(snapshot.shadow_score - previous.shadow_score)
                capacity_change = abs(snapshot.capacity - previous.capacity)
                importance += min(0.3, max(shadow_change, capacity_change))
            
            # Adjust based on task activity
            if hasattr(snapshot, "task_backlog"):
                completed_tasks = [t for t in snapshot.task_backlog if isinstance(t, dict) and t.get("status") == "completed"]
                importance += min(0.2, len(completed_tasks) * 0.05)
            
            # Adjust based on reflection activity
            if hasattr(snapshot, "current_batch_reflections"):
                importance += min(0.2, len(snapshot.current_batch_reflections) * 0.05)
            
            return min(1.0, importance)
            
        except Exception as e:
            self.logger.error(f"Error calculating snapshot importance: {e}")
            return 0.5

    def _generate_snapshot_summary(self, snapshot: MemorySnapshot) -> str:
        """Generate a human-readable summary of the snapshot state."""
        try:
            summary_parts = []
            
            # Add metric state
            summary_parts.append(
                f"State metrics - Shadow: {snapshot.shadow_score:.2f}, "
                f"Capacity: {snapshot.capacity:.2f}, "
                f"Withering: {snapshot.withering_level:.2f}"
            )
            
            # Add task summary
            if hasattr(snapshot, "task_backlog"):
                active_tasks = [t.get("title", "") for t in snapshot.task_backlog[:3] if isinstance(t, dict)]
                if active_tasks:
                    summary_parts.append(f"Active tasks: {', '.join(active_tasks)}")
            
            # Add reflection summary
            if hasattr(snapshot, "current_batch_reflections") and snapshot.current_batch_reflections:
                latest_reflection = snapshot.current_batch_reflections[-1]
                summary_parts.append(f"Latest reflection: {latest_reflection[:100]}...")
            
            return " | ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating snapshot summary: {e}")
            return "Error generating snapshot summary"
