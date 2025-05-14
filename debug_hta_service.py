#!/usr/bin/env python3

"""Debug script for enhanced_hta_service issues."""

import asyncio
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock

async def main():
    print("\n=== ENHANCED HTA SERVICE DEBUG TESTING ===\n")
    
    # Import components with error handling
    try:
        from forest_app.core.services.enhanced_hta.core import EnhancedHTAService
        print("✓ Successfully imported EnhancedHTAService")
    except Exception as e:
        print(f"✗ Error importing EnhancedHTAService: {e}")
        return
        
    try:
        from forest_app.core.roadmap_models import RoadmapManifest, RoadmapStep
        print("✓ Successfully imported RoadmapManifest and RoadmapStep")
    except Exception as e:
        print(f"✗ Error importing roadmap models: {e}")
        return
        
    try:
        from forest_app.persistence.models import HTATreeModel
        print("✓ Successfully imported HTATreeModel")
    except Exception as e:
        print(f"✗ Error importing HTATreeModel: {e}")
        return
    
    try: 
        from forest_app.core.event_bus import EventType, EventData
        # Define necessary EventType enum values
        if not hasattr(EventType, 'TREE_CREATED'):
            EventType.TREE_CREATED = "hta_tree_created"
        if not hasattr(EventType, 'NODE_CREATED'):
            EventType.NODE_CREATED = "hta_node_created"
            
        print("✓ Successfully imported and configured EventBus components")
    except Exception as e:
        print(f"✗ Error importing EventBus: {e}")
        return
        
    # Create test data
    print("\n--- Creating Test Data ---")
    tree_id = uuid.uuid4()
    user_id = uuid.uuid4()
    step_id = uuid.uuid4()
    print(f"Tree ID: {tree_id}")
    print(f"User ID: {user_id}")
    
    # Create test manifest
    manifest = RoadmapManifest(
        tree_id=tree_id,
        user_goal="Debug Test Goal",
        steps=[
            RoadmapStep(
                id=step_id,
                title="Debug Step",
                description="Test step",
                status="pending",
                priority="high",
                hta_metadata={"is_major_phase": True},
                dependencies=[]
            )
        ],
    )
    print("Created test manifest")
    
    # Create mocks
    print("\n--- Setting Up Mocks ---")
    mock_llm = MagicMock()
    mock_memory = MagicMock()
    
    # Mock all necessary components
    try:
        # Set up SessionManager
        from forest_app.core.session_manager import SessionManager
        from forest_app.core.services.enhanced_hta.memory import HTAMemoryManager
        
        print("Mocking SessionManager and HTAMemoryManager...")
        # Create a mock SessionManager with get_instance
        mock_session_manager = MagicMock()
        mock_session_manager.add = MagicMock()
        mock_session_manager.commit = AsyncMock()
        mock_session_manager.session = MagicMock(return_value=mock_session_manager)
        
        # Patch the SessionManager.get_instance class method
        original_session_get_instance = getattr(SessionManager, 'get_instance', None)
        SessionManager.get_instance = classmethod(lambda cls: mock_session_manager)
        print("✓ Successfully mocked SessionManager.get_instance")
        
        # Create a mock HTAMemoryManager
        mock_memory_manager = MagicMock(spec=HTAMemoryManager)
        mock_memory_manager.get_tasks_for_tree = AsyncMock(return_value=[])
        print("✓ Successfully created mock HTAMemoryManager")
        
        # Set up TaskQueue
        from forest_app.core.task_queue import TaskQueue
        print("Mocking TaskQueue...")
        
        # Create a mock TaskQueue
        mock_task_queue = MagicMock()
        mock_task_queue.add_task = AsyncMock()
        
        # Patch the TaskQueue.get_instance class method
        original_task_get_instance = getattr(TaskQueue, 'get_instance', None)
        TaskQueue.get_instance = classmethod(lambda cls: mock_task_queue)
        print("✓ Successfully mocked TaskQueue.get_instance")
        
    except Exception as e:
        print(f"✗ Error setting up component mocks: {repr(e)}")
        import traceback
        traceback.print_exc()
    
    # Create and configure mock repository
    mock_repo = MagicMock()
    async def mock_save_tree(tree_model):
        print("\nMOCK REPOSITORY called with:")
        print(f"- type: {type(tree_model)}")
        
        # Print all available attributes
        for attr in dir(tree_model):
            if not attr.startswith('_') and not callable(getattr(tree_model, attr)):
                try:
                    print(f"- {attr}: {getattr(tree_model, attr)}")
                except Exception as ex:
                    print(f"- {attr}: ERROR accessing ({ex})")
                    
        # Check for required attributes
        title = getattr(tree_model, 'title', None)
        goal_name = getattr(tree_model, 'goal_name', None)
        print(f"- Has 'title': {title is not None}")
        print(f"- Has 'goal_name': {goal_name is not None}")
        
        # Return the same tree model for simplicity
        return tree_model
    mock_repo.save_tree_model = AsyncMock(side_effect=mock_save_tree)
    print("Created mock repository")
    
    # Create and configure mock event bus
    mock_event_bus = MagicMock()
    # Create a very permissive mock for event_bus.publish to accept any arguments
    async def mock_publish(*args, **kwargs):
        print(f"\nMOCK EVENT BUS publishing:")
        if args:
            print(f"- Args: {args}")
        if kwargs:
            print(f"- Kwargs: {kwargs}")
        return None
    mock_event_bus.publish = AsyncMock(side_effect=mock_publish)
    print("Created mock event bus with permissive publish method")
    
    # Create service and inject mocks
    print("\n--- Creating Service ---")
    try:
        service = EnhancedHTAService(mock_llm, mock_memory)
        service.tree_repository = mock_repo
        service.event_bus = mock_event_bus
        print("✓ Successfully created service with injected mocks")
    except Exception as e:
        print(f"✗ Error creating service: {repr(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Execute the method
    print("\n--- Testing Method ---")
    try:
        print("Calling generate_initial_hta_from_manifest...")
        result = await service.generate_initial_hta_from_manifest(
            manifest=manifest,
            user_id=user_id,
            request_context={"source": "debug_test"}
        )
        print("\n✓ Successfully executed method!")
        print(f"Result type: {type(result)}")
        print(f"Result ID: {result.id}")
        print(f"Result user_id: {result.user_id}")
        
        # Print all result attributes
        print("\nResult attributes:")
        for attr in dir(result):
            if not attr.startswith('_') and not callable(getattr(result, attr)):
                try:
                    print(f"- {attr}: {getattr(result, attr)}")
                except Exception as ex:
                    print(f"- {attr}: ERROR accessing ({ex})")
                    
    except Exception as e:
        print(f"\n✗ Error executing method: {repr(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
