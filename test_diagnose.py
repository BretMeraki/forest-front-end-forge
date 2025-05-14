#!/usr/bin/env python3
from uuid import uuid4, UUID
from datetime import datetime
import sys

# Add project root to path to allow imports without setting PYTHONPATH
import os
sys.path.insert(0, os.path.abspath('.'))

from forest_app.core.roadmap_models import RoadmapStep, RoadmapManifest

# Test 1: RoadmapStep dependencies are frozenset
print("Test 1: RoadmapStep dependencies")
try:
    step = RoadmapStep(
        title="Test Step",
        description="desc",
        dependencies=[uuid4(), uuid4()]
    )
    print(f"  Dependencies type: {type(step.dependencies)}")
    print(f"  Is frozenset: {isinstance(step.dependencies, frozenset)}")
    print("  Test 1 PASSED")
except Exception as e:
    print(f"  Test 1 FAILED: {type(e).__name__}: {str(e)}")

# Test 2: RoadmapStep timestamps
print("\nTest 2: RoadmapStep timestamps")
try:
    step = RoadmapStep(title="A", description="B")
    print(f"  created_at: {step.created_at}")
    print(f"  updated_at: {step.updated_at}")
    print(f"  created_at == updated_at: {step.created_at == step.updated_at}")
    print("  Test 2 PASSED")
except Exception as e:
    print(f"  Test 2 FAILED: {type(e).__name__}: {str(e)}")

# Test 3: Manifest step index and lookup
print("\nTest 3: Manifest step index and lookup")
try:
    step1 = RoadmapStep(title="A", description="B")
    step2 = RoadmapStep(title="C", description="D") 
    manifest = RoadmapManifest(tree_id=uuid4(), user_goal="Goal", steps=[step1, step2])
    found = manifest.get_step_by_id(step1.id)
    print(f"  Found step: {found}")
    print(f"  Found step == step1: {found == step1}")
    not_found = manifest.get_step_by_id(uuid4())
    print(f"  Not found step: {not_found}")
    print("  Test 3 PASSED")
except Exception as e:
    print(f"  Test 3 FAILED: {type(e).__name__}: {str(e)}")

# Test 4: Manifest add step and status update
print("\nTest 4: Manifest add step and status update")
try:
    manifest = RoadmapManifest(tree_id=uuid4(), user_goal="Goal")
    step = RoadmapStep(title="A", description="B")
    manifest_with_step = manifest.add_step(step)
    print(f"  Step in steps: {step in manifest_with_step.steps}")
    manifest_updated = manifest_with_step.update_step_status(step.id, "completed")
    updated_step = manifest_updated.get_step_by_id(step.id)
    print(f"  Updated step status: {updated_step.status}")
    print("  Test 4 PASSED")
except Exception as e:
    print(f"  Test 4 FAILED: {type(e).__name__}: {str(e)}")

# Test 5: Manifest circular dependency detection
print("\nTest 5: Manifest circular dependency detection")
try:
    id1, id2 = uuid4(), uuid4()
    step1 = RoadmapStep(id=id1, title="A", description="B", dependencies=[id2])
    step2 = RoadmapStep(id=id2, title="C", description="D", dependencies=[id1])
    manifest = RoadmapManifest(tree_id=uuid4(), user_goal="Goal", steps=[step1, step2])
    errors = manifest.check_circular_dependencies()
    print(f"  Circular dependency errors: {errors}")
    print(f"  Has errors about circular dependency: {any('Circular dependency' in e for e in errors)}")
    print("  Test 5 PASSED")
except Exception as e:
    print(f"  Test 5 FAILED: {type(e).__name__}: {str(e)}")
