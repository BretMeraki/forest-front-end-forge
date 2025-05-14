"""
Phase Notification Service for The Forest application - Lean MVP Edition.

This module implements simplified phase completion detection and basic notifications
as specified in PRD v3.15 under the PhaseLogic-HTA Flow section, which states:
"[LeanMVP - Simplify]: Basic UI notification on completion of all tasks under a major phase node. 
Complex suggestion logic deferred."

The service detects phase completions in the HTA tree and provides basic next phase information.
"""
from typing import List, Dict, Any, Optional
from uuid import UUID
import logging

# Set up logger
logger = logging.getLogger(__name__)

from forest_app.core.roadmap_models import RoadmapManifest, RoadmapStep
from forest_app.persistence.models import HTANodeModel
from forest_app.integrations.llm_service import BaseLLMService


class PhaseCompletionEvent:
    """Represents a simplified phase completion event for the Lean MVP.
    
    [LeanMVP - Simplify]: Focuses on basic phase completion information with minimal
    next phase details. Complex suggestions are deferred to post-MVP.
    """
    
    def __init__(
        self,
        phase_id: UUID,
        phase_title: str,
        completed_at: str,
        next_phase_id: Optional[UUID] = None,
        next_phase_title: Optional[str] = None,
        custom_message: Optional[str] = None
    ):
        """
        Initialize a new phase completion event.
        
        Args:
            phase_id: ID of the completed phase
            phase_title: Title of the completed phase
            completed_at: ISO-formatted timestamp when phase was completed
            next_phase_id: ID of the recommended next phase (if any)
            next_phase_title: Title of the recommended next phase (if any)
            custom_message: Simple message to display to the user
        """
        self.phase_id = phase_id
        self.phase_title = phase_title
        self.completed_at = completed_at
        self.next_phase_id = next_phase_id
        self.next_phase_title = next_phase_title
        self.custom_message = custom_message


class PhaseNotificationService:
    """
    Service for managing basic phase completion notifications.
    
    [LeanMVP - Simplify]: This service detects when a major phase has been completed
    and provides basic information about the next phase, without LLM-driven suggestions.
    Complex suggestion logic is deferred to post-MVP.
    """
    
    def __init__(self):
        """
        Initialize the phase notification service.
        
        [LeanMVP - Simplify]: LLM integration deferred for the initial MVP.
        """
        pass
    
    def is_phase_complete(self, phase_node: HTANodeModel) -> bool:
        """
        Check if a phase node is complete (all child tasks completed).
        
        Args:
            phase_node: The phase node to check
            
        Returns:
            True if the phase is complete, False otherwise
        """
        # A phase is complete when:
        # 1. It has children (otherwise it's a leaf task, not a phase)
        # 2. All its children have status="completed"
        if not phase_node.children:
            return False
            
        return all(child.status == "completed" for child in phase_node.children)
    
    def find_next_phase(self, manifest: RoadmapManifest, current_phase_id: UUID) -> Optional[RoadmapStep]:
        """
        Find the next logical phase to suggest after the current one.
        
        Args:
            manifest: The roadmap manifest
            current_phase_id: ID of the current (completed) phase
            
        Returns:
            The next phase step, or None if no suitable next phase exists
        """
        current_phase = manifest.get_step_by_id(current_phase_id)
        if not current_phase:
            logger.warning(f"Cannot find next phase: current phase {current_phase_id} not found in manifest")
            return None
            
        # Find all phases (steps with is_major_phase=True in metadata)
        phases = [
            step for step in manifest.steps 
            if step.hta_metadata.get("is_major_phase", False) and step.status != "completed"
        ]
        
        # Log warning if no major phases found - this indicates RoadmapParser isn't setting the flag
        if not phases:
            logger.warning(
                "No major phases found in manifest. Ensure RoadmapParser is correctly populating "
                "the 'is_major_phase' flag in step.hta_metadata as noted in PRD F5.2.1"
            )
        
        # Filter phases that have no pending dependencies
        ready_phases = []
        for phase in phases:
            # Skip the current phase
            if phase.id == current_phase_id:
                continue
                
            # Check if all dependencies are met
            deps_met = True
            for dep_id in phase.dependencies:
                dep_step = manifest.get_step_by_id(dep_id)
                if not dep_step or dep_step.status != "completed":
                    deps_met = False
                    break
                    
            if deps_met:
                ready_phases.append(phase)
                
        # If no ready phases, return None
        if not ready_phases:
            return None
            
        # Choose the phase with highest priority
        priority_map = {"high": 3, "medium": 2, "low": 1}
        ready_phases.sort(key=lambda p: priority_map.get(p.priority, 0), reverse=True)
        
        return ready_phases[0] if ready_phases else None
    
    def generate_phase_completion_event(
        self, 
        manifest: RoadmapManifest, 
        completed_phase_id: UUID
    ) -> PhaseCompletionEvent:
        """
        Generate a phase completion event with next steps and suggestions.
        
        Args:
            manifest: The roadmap manifest
            completed_phase_id: ID of the completed phase
            memory_context: Optional semantic memory context to inform suggestions
            
        Returns:
            A PhaseCompletionEvent object with completion details and suggestions
        """
        completed_phase = manifest.get_step_by_id(completed_phase_id)
        if not completed_phase:
            error_msg = f"Phase with ID {completed_phase_id} not found in manifest"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        # Verify this is actually a phase
        if not completed_phase.hta_metadata.get("is_major_phase", False):
            logger.warning(
                f"Step '{completed_phase.title}' (ID: {completed_phase_id}) is not marked as a major phase. "
                f"This may indicate inconsistent phase flagging in the manifest."
            )
            
        # Find the next phase
        next_phase = self.find_next_phase(manifest, completed_phase_id)
        
        # Generate a custom message
        custom_message = f"Congratulations on completing the '{completed_phase.title}' phase!"
        if next_phase:
            custom_message += f" Next up: {next_phase.title}"
        else:
            custom_message += " You're making great progress on your journey!"
            
        # [LeanMVP - Simplify]: No LLM-driven suggestions for initial MVP
        # Just provide a basic completion message
        
        # Create and return the simplified event
        return PhaseCompletionEvent(
            phase_id=completed_phase.id,
            phase_title=completed_phase.title,
            completed_at=completed_phase.updated_at.isoformat(),
            next_phase_id=next_phase.id if next_phase else None,
            next_phase_title=next_phase.title if next_phase else None,
            custom_message=custom_message
        )
"""
"""
