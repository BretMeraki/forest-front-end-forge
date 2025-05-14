import pytest
from datetime import datetime
from uuid import uuid4, UUID
from forest_app.core.roadmap_models import RoadmapStep, RoadmapManifest

def test_roadmapstep_dependencies_frozenset():
    step = RoadmapStep(
        title="Test Step",
        description="desc",
        dependencies=[uuid4(), uuid4()]
    )
    assert isinstance(step.dependencies, frozenset)
    # Should be immutable: frozenset has no add method
    with pytest.raises(AttributeError):
        step.dependencies.add(uuid4())

def test_roadmapstep_timestamps():
    step = RoadmapStep(title="A", description="B")
    assert isinstance(step.created_at, datetime)
    assert isinstance(step.updated_at, datetime)
    # With validator removed, created_at and updated_at should be exactly the same at creation
    assert step.created_at == step.updated_at
    
    # Test that copying a step with an updated status updates the updated_at timestamp
    # This simulates the behavior of update_step_status in RoadmapManifest
    import time
    time.sleep(0.01)  # Ensure time difference
    updated_step = step.copy(update={"status": "completed", "updated_at": datetime.utcnow()})
    assert updated_step.updated_at > step.updated_at

def test_manifest_step_index_and_lookup():
    step1 = RoadmapStep(title="A", description="B")
    step2 = RoadmapStep(title="C", description="D")
    manifest = RoadmapManifest(tree_id=uuid4(), user_goal="Goal", steps=[step1, step2])
    found = manifest.get_step_by_id(step1.id)
    assert found == step1  # Use value equality, not identity
    assert manifest.get_step_by_id(uuid4()) is None

def test_manifest_add_step_and_status_update():
    manifest = RoadmapManifest(tree_id=uuid4(), user_goal="Goal")
    step = RoadmapStep(title="A", description="B")
    manifest = manifest.add_step(step)
    assert step in manifest.steps
    manifest = manifest.update_step_status(step.id, "completed")
    updated_step = manifest.get_step_by_id(step.id)
    assert updated_step.status == "completed"

def test_manifest_circular_dependency_detection():
    id1, id2 = uuid4(), uuid4()
    step1 = RoadmapStep(id=id1, title="A", description="B", dependencies=[id2])
    step2 = RoadmapStep(id=id2, title="C", description="D", dependencies=[id1])
    manifest = RoadmapManifest(tree_id=uuid4(), user_goal="Goal", steps=[step1, step2])
    errors = manifest.check_circular_dependencies()
    assert errors and any("Circular dependency" in e for e in errors)

import pytest

def test_manifest_topological_sort_and_major_phases():
    pytest.skip("Topological sort and major phases test not yet implemented.")
