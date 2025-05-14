"""
Pydantic models for the Roadmap Manifest, which serves as the single source of truth
for the HTA (Hierarchical Task Analysis) tree and its evolution.

The Roadmap Manifest consists of:
- A collection of RoadmapSteps with dependencies
- Metadata about the overall tree structure
- Status tracking for steps that are synchronized with HTANodes

This module implements the [Manifest-HTA - Core] PRD requirement where the 
RoadmapManifest is the single source of truth for the HTA tree.
"""

from pydantic import BaseModel, Field, validator, ConfigDict, PrivateAttr
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal, FrozenSet
from uuid import UUID, uuid4


class RoadmapStep(BaseModel):
    """
    A single step in the roadmap manifest.
    
    Each step represents a task or action to be taken, and can have dependencies
    on other steps in the manifest. The status field is synchronized with the 
    corresponding HTANode's status.
    
    [LeanMVP - Simplify]: Focusing on essential fields for MVP, deferring estimated_duration and semantic_context.
    """
    # Core Identification & Content
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the step")
    title: str = Field(description="Title of the roadmap step")
    description: str = Field(description="Detailed description of the step")
    
    # Status Management
    status: Literal["pending", "in_progress", "completed", "deferred", "cancelled"] = "pending"
    
    # Dependency Management
    dependencies: FrozenSet[UUID] = Field(
        default_factory=frozenset, 
        description="Set of step IDs this step depends on"
    )

    @validator('dependencies', pre=True, always=True)
    def convert_dependencies(cls, v):
        if v is None:
            return frozenset()
        if isinstance(v, (list, set, tuple)):
            return frozenset(v)
        return v
    
    # Priority & Metadata
    priority: Literal["high", "medium", "low"] = "medium"
    hta_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Metadata about this step in the HTA structure (e.g., {\"is_major_phase\": true/false})"
    )
    
    # Audit Trail
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of step creation")
    # updated_at is set only on logical mutation (e.g., status change or adding steps), not on every validation.
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last step update")

    model_config = ConfigDict(
        frozen=True,
        extra='forbid',
        validate_assignment=False,
        populate_by_name=True,
        validate_default=False,
        arbitrary_types_allowed=True
    )


class RoadmapManifest(BaseModel):
    """
    The manifest that serves as the single source of truth for the HTA tree.
    Implements internal indexes and helper methods per PRD v4.0.
    """
    model_config = ConfigDict(
        extra='forbid',
        validate_assignment=False,
        populate_by_name=True,
        arbitrary_types_allowed=True
    )
    # Internal indexes and caches as PrivateAttr (not serialized)
    _step_index: Dict[UUID, RoadmapStep] = PrivateAttr(default_factory=dict)
    _dependency_graph: Dict[UUID, FrozenSet[UUID]] = PrivateAttr(default_factory=dict)
    _reverse_dependency_graph: Dict[UUID, list] = PrivateAttr(default_factory=dict)
    _topological_sort_cache: list = PrivateAttr(default=None)

    """
    The manifest that serves as the single source of truth for the HTA tree.
    
    This model contains all the steps in the roadmap, with their dependencies and status.
    When the HTA tree is updated, the manifest is updated to reflect those changes,
    ensuring consistency between the two.
    
    [LeanMVP - Simplify]: Focusing on essential fields for MVP, deferring some context capture fields.
    """
    # Core Identification
    id: UUID = Field(default_factory=uuid4, description="Unique identifier for the manifest")
    tree_id: UUID = Field(description="Corresponds to HTATreeModel.id")
    
    # Versioning - Simplified for MVP
    manifest_version: str = Field(default="1.0", description="Version of the manifest schema")
    
    # Goal Capture - Essential for context
    user_goal: str = Field(description="Primary user goal or objective")
    
    # Q&A Traceability - Keeping basic structure for future expansion
    q_and_a_responses: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Recorded clarifying questions and user responses during onboarding"
    )
    
    # Steps Content
    steps: List[RoadmapStep] = Field(default_factory=list)
    
    # Audit Trail
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of manifest creation")
    # updated_at is set only on logical mutation (e.g., status change or adding steps), not on every validation.
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last manifest update")

    def __init__(self, **data):
        super().__init__(**data)
        self._build_indexes()

    def _build_indexes(self):
        self._step_index = {step.id: step for step in self.steps}
        self._dependency_graph = {step.id: step.dependencies for step in self.steps}
        self._reverse_dependency_graph = {}
        for step in self.steps:
            for dep in step.dependencies:
                self._reverse_dependency_graph.setdefault(dep, []).append(step.id)
        self._topological_sort_cache = None

    def get_step_by_id(self, step_id: UUID) -> Optional[RoadmapStep]:
        """
        Find and return a step by its ID.
        
        Args:
            step_id: The UUID of the step to find
            
        Returns:
            The step with the given ID, or None if not found
        """
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def update_step_status(self, step_id: UUID, new_status: Literal["pending", "in_progress", "completed", "deferred", "cancelled"]) -> 'RoadmapManifest':
        """
        Return a new manifest with the step status updated (immutability pattern).
        """
        steps = [
            step.copy(update={"status": new_status, "updated_at": datetime.utcnow()}) if step.id == step_id else step
            for step in self.steps
        ]
        return self.copy(update=self._ensure_updated_timestamp({"steps": steps}))

    
    def _ensure_updated_timestamp(self, update_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Helper to ensure that updated_at is always set on mutation operations.
        
        Args:
            update_dict: Dictionary of updates to apply
            
        Returns:
            Dictionary with updated_at added if not present
        """
        if "updated_at" not in update_dict:
            update_dict["updated_at"] = datetime.utcnow()
        return update_dict
        
    def add_step(self, step: RoadmapStep) -> 'RoadmapManifest':
        """
        Return a new manifest with the additional step (immutability pattern).
        """
        return self.copy(update=self._ensure_updated_timestamp({"steps": self.steps + [step]}))

    def get_pending_actionable_steps(self) -> List[RoadmapStep]:
        """
        Returns all steps that are pending and have all dependencies completed.
        """
        actionable = []
        for step in self.steps:
            if step.status == "pending" and all(
                self.get_step_by_id(dep_id).status == "completed"
                for dep_id in step.dependencies
            ):
                actionable.append(step)
        return actionable

    def get_major_phases(self) -> List[RoadmapStep]:
        """
        Returns all steps marked as major phases in hta_metadata.
        """
        return [step for step in self.steps if step.hta_metadata.get("is_major_phase", False)]

    def check_circular_dependencies(self) -> List[str]:
        """
        Check for circular dependencies in the manifest steps.
        
        Returns:
            A list of error messages describing any circular dependencies found
        """
        errors = []
        visited = {}  # Maps step_id to visit status: 0=unvisited, 1=in progress, 2=visited
        
        def dfs(step_id):
            # Mark as in-progress
            visited[step_id] = 1
            
            step = self.get_step_by_id(step_id)
            if not step:
                return
            
            for dep_id in step.dependencies:
                if dep_id not in visited:
                    # Not visited yet
                    visited[dep_id] = 0
                    
                if visited.get(dep_id) == 0:
                    # Visit unvisited node
                    if dfs(dep_id):
                        # Propagate cycle detection
                        return True
                elif visited.get(dep_id) == 1:
                    # Found a cycle
                    cycle_step = self.get_step_by_id(dep_id)
                    current_step = self.get_step_by_id(step_id)
                    if cycle_step and current_step:
                        errors.append(
                            f"Circular dependency detected: '{current_step.title}' ({step_id}) "
                            f"depends on '{cycle_step.title}' ({dep_id}) which creates a cycle."
                        )
                    return True
            
            # Mark as visited
            visited[step_id] = 2
            return False
        
        # Check each step that hasn't been visited
        for step in self.steps:
            if step.id not in visited:
                visited[step.id] = 0
                dfs(step.id)
        
        return errors
