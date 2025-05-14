# forest_app/core/orchestrator.py (REFACTORED)

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from sqlalchemy.orm import Session

# Core imports with error handling
try:
    from forest_app.core.snapshot import MemorySnapshot
    from forest_app.core.utils import clamp01
    from forest_app.core.processors import ReflectionProcessor, CompletionProcessor
    from forest_app.core.services import (
        HTAService,
        ComponentStateManager,
        SemanticMemoryManager
    )
    from forest_app.core.services.enhanced_hta.memory import HTAMemoryManager
    from forest_app.modules.seed import SeedManager, Seed
    from forest_app.modules.logging_tracking import TaskFootprintLogger, ReflectionLogLogger
    from forest_app.persistence.repository import HTATreeRepository
    from forest_app.modules.soft_deadline_manager import hours_until_deadline
    from forest_app.persistence.models import HTANodeModel
    from uuid import UUID
except ImportError as e:
    logging.getLogger(__name__).error(f"Failed to import required modules: {e}")
    # Define dummy classes if imports fail
    class MemorySnapshot: pass
    class ReflectionProcessor: pass
    class CompletionProcessor: pass
    class HTAService: pass
    class ComponentStateManager: pass
    class SemanticMemoryManager: pass
    class HTAMemoryManager: pass
    class SeedManager: pass
    class Seed: pass
    class TaskFootprintLogger: pass
    class ReflectionLogLogger: pass
    class HTATreeRepository: pass
    class HTANodeModel: pass
    class UUID: pass
    def clamp01(x): return x
    def hours_until_deadline(x): return 0

# Feature flags with error handling
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    def is_enabled(feature): return False
    class Feature:
        SOFT_DEADLINES = "FEATURE_ENABLE_SOFT_DEADLINES"

# Constants with error handling
try:
    from forest_app.config.constants import (
        MAGNITUDE_THRESHOLDS,
        WITHERING_COMPLETION_RELIEF,
        WITHERING_IDLE_COEFF,
        WITHERING_OVERDUE_COEFF,
        WITHERING_DECAY_FACTOR
    )
except ImportError:
    MAGNITUDE_THRESHOLDS = {"HIGH": 8.0, "MEDIUM": 5.0, "LOW": 2.0}
    WITHERING_COMPLETION_RELIEF = 0.1
    WITHERING_IDLE_COEFF = {"structured": 0.1}
    WITHERING_OVERDUE_COEFF = 0.1
    WITHERING_DECAY_FACTOR = 0.9

# Import shared types
from forest_app.modules.types import SemanticMemoryProtocol

logger = logging.getLogger(__name__)

# ═════════════════════════════ ForestOrchestrator (Refactored) ══════════════

class ForestOrchestrator:
    """
    Coordinates the main Forest application workflows by delegating to
    specialized processors and services. Manages top-level state transitions.
    """

    # ───────────────────────── 1. INITIALISATION (DI Based) ─────────────────
    def __init__(
        self,
        reflection_processor: ReflectionProcessor,
        completion_processor: CompletionProcessor,
        state_manager: ComponentStateManager,
        hta_service: HTAService,
        seed_manager: SeedManager,
        semantic_memory_manager: SemanticMemoryProtocol,
        memory_manager: Optional[HTAMemoryManager] = None,
        tree_repository: Optional[HTATreeRepository] = None,
        task_logger: Optional[TaskFootprintLogger] = None,
        reflection_logger: Optional[ReflectionLogLogger] = None,
        llm_client = None,
    ):
        """Initializes the orchestrator with injected processors and services."""
        self.reflection_processor = reflection_processor
        self.completion_processor = completion_processor
        self.state_manager = state_manager
        self.hta_service = hta_service
        self.seed_manager = seed_manager
        self.semantic_memory_manager = semantic_memory_manager
        self.memory_manager = memory_manager
        self.tree_repository = tree_repository
        self.task_logger = task_logger
        self.reflection_logger = reflection_logger
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)

        # Check critical dependencies
        if not isinstance(self.reflection_processor, ReflectionProcessor):
             raise TypeError("Invalid ReflectionProcessor provided.")
        if not isinstance(self.completion_processor, CompletionProcessor):
             raise TypeError("Invalid CompletionProcessor provided.")
        if not isinstance(self.state_manager, ComponentStateManager):
             raise TypeError("Invalid ComponentStateManager provided.")
        if not isinstance(self.seed_manager, SeedManager):
             raise TypeError("Invalid SeedManager provided.")
        # Add checks for other injected components like hta_service if kept

        logger.info("ForestOrchestrator (Refactored) initialized.")


    # ───────────────────────── 2. CORE WORKFLOWS ────────────────────────────

    async def process_reflection(self, reflection_text: str, snapshot: Any = None) -> Dict[str, Any]:
        """Process a reflection with semantic memory integration."""
        try:
            # Store reflection as semantic memory
            await self.semantic_memory_manager.store_memory(
                event_type="reflection",
                content=reflection_text,
                metadata={"timestamp": datetime.now(timezone.utc).isoformat()},
                importance=0.7  # Reflections are generally important
            )

            # Query relevant memories for context
            relevant_memories = await self.semantic_memory_manager.query_memories(
                query=reflection_text,
                k=3,
                event_types=["reflection"]
            )

            # Process reflection with context from semantic memory
            result = await self.reflection_processor.process_reflection(
                reflection_text=reflection_text,
                context={"relevant_memories": relevant_memories},
                snapshot=snapshot
            )

            return {
                "processed_reflection": result,
                "relevant_memories": relevant_memories
            }

        except Exception as e:
            self.logger.error(f"Error in process_reflection: {e}")
            raise

    async def process_task_completion(self,
                                    task_id: str,
                                    success: bool,
                                    snap: Optional[MemorySnapshot] = None,
                                    db: Optional[Session] = None,
                                    task_logger: Optional[TaskFootprintLogger] = None,
                                    reflection: Optional[str] = None) -> Dict[str, Any]:
        """Process a task completion with full transactional integrity.
        
        This method implements Task 1.5 requirements, ensuring atomic updates,
        memory snapshot updates, audit logging, and positive reinforcement.
        It handles both task node updates and roadmap manifest synchronization.
        
        Args:
            task_id: UUID or string ID of the task/node to complete
            success: Whether the task was completed successfully
            snap: Optional MemorySnapshot for context and update
            db: Optional database session for transaction context
            task_logger: Optional logger for task footprints (overrides instance logger)
            reflection: Optional user reflection on the completion
            
        Returns:
            Dictionary with completion results, including supportive message
        """
        try:
            self.logger.info(f"Processing task completion for task {task_id}, success={success}")
            
            # Convert string UUID to UUID if needed
            node_id = UUID(task_id) if isinstance(task_id, str) else task_id
            
            # Extract user_id from snapshot if available
            user_id = None
            if snap and hasattr(snap, 'user_id'):
                user_id = snap.user_id
            elif snap and hasattr(snap, 'user') and hasattr(snap.user, 'id'):
                user_id = snap.user.id
                
            if not user_id:
                self.logger.warning("No user ID available in snapshot for task completion")
                # Attempt to retrieve user_id from the node if possible
                if self.tree_repository:
                    try:
                        node = await self.tree_repository.get_node_by_id(node_id)
                        if node:
                            user_id = node.user_id
                    except Exception as node_err:
                        self.logger.warning(f"Error getting node for user ID: {node_err}")
                                
            if not user_id:
                self.logger.error("Cannot complete task: No user ID available")
                raise ValueError("No user ID available for task completion")
            
            # Use enhanced implementation if all required components are available
            if all([self.completion_processor, self.tree_repository, self.memory_manager]):
                # Use the enhanced CompletionProcessor implementation for Task 1.5
                result = await self.completion_processor.process_node_completion(
                    node_id=node_id,
                    user_id=user_id,
                    success=success,
                    reflection=reflection,
                    db_session=db
                )
                
                # Update snapshot with reinforcement message if available
                if snap and result.get("reinforcement_message"):
                    # Add the message to the snapshot if it has a messages array
                    if hasattr(snap, 'messages') and isinstance(snap.messages, list):
                        snap.messages.append({
                            "type": "system",
                            "content": result.get("reinforcement_message"),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "is_reinforcement": True
                        })
                    # Update last activity timestamp
                    if hasattr(snap, 'component_state') and isinstance(snap.component_state, dict):
                        snap.component_state["last_activity_ts"] = datetime.now(timezone.utc).isoformat()
                    
                    # Reduce withering level on task completion
                    if hasattr(snap, 'withering_level'):
                        current_level = getattr(snap, 'withering_level', 0.0)
                        if isinstance(current_level, (int, float)):
                            snap.withering_level = max(0.0, current_level - WITHERING_COMPLETION_RELIEF)
                return result
            
            # Fall back to legacy implementation if components are missing
            self.logger.warning("Using legacy completion flow - some Task 1.5 features unavailable")
            completion_data = {
                "success": success,
                "reflection": reflection or "",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            return await self.complete_task(task_id, completion_data)
            
        except Exception as e:
            self.logger.error(f"Error processing task completion: {e}")
            raise

    async def complete_task(self, task_id: str, completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete a task with semantic memory integration (legacy method).
        
        Note: This is maintained for backward compatibility. For new code,
        prefer using process_task_completion which implements Task 1.5 requirements.
        """
        try:
            # Store task completion as semantic memory
            await self.semantic_memory_manager.store_memory(
                event_type="task_completion",
                content=f"Completed task {task_id}: {completion_data.get('summary', '')}",
                metadata={
                    "task_id": task_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **completion_data
                },
                importance=0.6
            )

            # Process task completion
            result = await self.completion_processor.process_completion(task_id, completion_data)

            return result

        except Exception as e:
            self.logger.error(f"Error in complete_task: {e}")
            raise

    async def get_memory_context(self, query: str = None) -> Dict[str, Any]:
        """Get relevant memory context for the current state."""
        try:
            memories = await self.semantic_memory_manager.query_memories(
                query=query or "Get recent important memories",
                k=5
            )

            stats = self.semantic_memory_manager.get_memory_stats()

            return {
                "relevant_memories": memories,
                "memory_stats": stats
            }

        except Exception as e:
            self.logger.error(f"Error in get_memory_context: {e}")
            raise

    # ───────────────────────── 3. UTILITY & DELEGATION ──────────────────────

    # Keeping _update_withering here for now, could be moved to its own class
    def _update_withering(self, snap: MemorySnapshot):
        """Adjusts withering level based on inactivity and deadlines."""
        # (Implementation is the same as the original orchestrator version)
        if not hasattr(snap, 'withering_level'): snap.withering_level = 0.0
        if not hasattr(snap, 'component_state') or not isinstance(snap.component_state, dict): snap.component_state = {}
        if not hasattr(snap, 'task_backlog') or not isinstance(snap.task_backlog, list): snap.task_backlog = []

        current_path = getattr(snap, "current_path", "structured")
        now_utc = datetime.now(timezone.utc)
        last_iso = snap.component_state.get("last_activity_ts")
        idle_hours = 0.0
        if last_iso and isinstance(last_iso, str):
            try:
                # Ensure TZ info for comparison
                last_dt_aware = datetime.fromisoformat(last_iso.replace("Z", "+00:00"))
                if last_dt_aware.tzinfo is None: last_dt_aware = last_dt_aware.replace(tzinfo=timezone.utc)
                idle_delta = now_utc - last_dt_aware
                idle_hours = max(0.0, idle_delta.total_seconds() / 3600.0)
            except ValueError: logger.warning("Could not parse last_activity_ts: %s", last_iso)
            except Exception as ts_err: logger.exception("Error processing last_activity_ts: %s", ts_err)
        elif last_iso is not None: logger.warning("last_activity_ts is not a string: %s", type(last_iso))

        idle_coeff = WITHERING_IDLE_COEFF.get(current_path, WITHERING_IDLE_COEFF["structured"])
        idle_penalty = idle_coeff * idle_hours

        overdue_hours = 0.0
        if is_enabled(Feature.SOFT_DEADLINES) and current_path != "open" and isinstance(snap.task_backlog, list):
            try:
                overdue_list = []
                for task in snap.task_backlog:
                    if isinstance(task, dict) and task.get("soft_deadline"):
                        overdue = hours_until_deadline(task) # Use imported helper
                        if isinstance(overdue, (int, float)) and overdue < 0:
                            overdue_list.append(abs(overdue))
                if overdue_list: overdue_hours = max(overdue_list)
            except Exception as e: logger.error("Error calculating overdue hours: %s", e) # Simplified error handling
        elif not is_enabled(Feature.SOFT_DEADLINES):
            logger.debug("Skipping overdue hours calculation: SOFT_DEADLINES feature disabled.")

        soft_coeff = WITHERING_OVERDUE_COEFF.get(current_path, 0.0) if is_enabled(Feature.SOFT_DEADLINES) else 0.0
        soft_penalty = soft_coeff * overdue_hours

        current_withering = getattr(snap, 'withering_level', 0.0)
        if not isinstance(current_withering, (int, float)): current_withering = 0.0
        new_level = float(current_withering) + idle_penalty + soft_penalty
        snap.withering_level = clamp01(new_level * WITHERING_DECAY_FACTOR)
        logger.debug(f"Withering updated: Level={snap.withering_level:.4f} (IdleHrs={idle_hours:.2f}, OverdueHrs={overdue_hours:.2f})")


    # Example: Keeping get_primary_active_seed here, but could be moved to SeedManager
    async def get_primary_active_seed(self) -> Optional[Seed]:
        """Retrieves the first active seed using the injected SeedManager."""
        if not self.seed_manager or not hasattr(self.seed_manager, 'get_primary_active_seed'):
             logger.error("Injected SeedManager missing or invalid for get_primary_active_seed.")
             return None
        try:
            # Assuming get_primary_active_seed is now async in SeedManager
            return await self.seed_manager.get_primary_active_seed()
        except Exception as e:
            logger.exception("Error getting primary active seed via orchestrator: %s", e)
            return None


    # Convenience APIs delegating to SeedManager
    async def plant_seed( self, intention: str, domain: str, addl_ctx: Optional[Dict[str, Any]] = None) -> Optional[Seed]:
        logger.info(f"Orchestrator: Delegating plant_seed to SeedManager...")
        if not self.seed_manager or not hasattr(self.seed_manager, 'plant_seed'):
            logger.error("Injected SeedManager missing or invalid for plant_seed.")
            return None
        try:
            # Assuming plant_seed is now async in SeedManager
            return await self.seed_manager.plant_seed(intention, domain, addl_ctx)
        except Exception as exc:
            logger.exception("Orchestrator plant_seed delegation error: %s", exc)
            return None


    async def trigger_seed_evolution( self, seed_id: str, evolution: str, new_intention: Optional[str] = None ) -> bool:
        logger.info(f"Orchestrator: Delegating trigger_seed_evolution to SeedManager...")
        if not self.seed_manager or not hasattr(self.seed_manager, 'evolve_seed'):
            logger.error("Injected SeedManager missing or invalid for evolve_seed.")
            return False
        try:
             # Assuming evolve_seed is now async in SeedManager
            return await self.seed_manager.evolve_seed(seed_id, evolution, new_intention)
        except Exception as exc:
            logger.exception("Orchestrator trigger_seed_evolution delegation error: %s", exc)
            return False


    # Static utility method can remain
    @staticmethod
    def describe_magnitude(value: float) -> str:
        # (Implementation is the same as the original orchestrator version)
        try:
            float_value = float(value)
            valid_thresholds = {k: float(v) for k, v in MAGNITUDE_THRESHOLDS.items() if isinstance(v, (int, float))}
            if not valid_thresholds: return "Unknown"
            sorted_thresholds = sorted(valid_thresholds.items(), key=lambda item: item[1], reverse=True)
            for label, thresh in sorted_thresholds:
                if float_value >= thresh: return str(label)
            return str(sorted_thresholds[-1][0]) if sorted_thresholds else "Dormant"
        except (ValueError, TypeError) as e:
            logger.error("Error converting value/threshold for magnitude: %s (Value: %s)", e, value)
            return "Unknown"
        except Exception as e:
            logger.exception("Error describing magnitude for value %s: %s", value, e)
            return "Unknown"
