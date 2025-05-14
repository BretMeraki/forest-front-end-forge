"""
Debug script for test_enhanced_hta_service
This will help identify the source of the error.
"""

import asyncio
import uuid
import sys
from datetime import datetime, timezone

# Try importing our enhanced HTA service
try:
    from forest_app.core.services.enhanced_hta.core import EnhancedHTAService
    print("Successfully imported EnhancedHTAService from modular architecture")
except ImportError as e:
    print(f"Error importing EnhancedHTAService: {e}")
    sys.exit(1)

# Import the test helpers
try:
    from forest_app.core.services.test_helpers.mock_enhanced_hta_service import get_mock_enhanced_hta_service
    print("Successfully imported mock service")
except ImportError as e:
    print(f"Error importing mock service: {e}")
    sys.exit(1)

# Import test dependencies
try:
    from forest_app.core.roadmap_models import RoadmapManifest, RoadmapStep
    print("Successfully imported roadmap models")
except ImportError as e:
    print(f"Error importing roadmap models: {e}")
    sys.exit(1)

# Mock classes from test file
class MockLLMClient:
    """Mock LLM client for testing."""
    
    async def generate(self, prompt_type, context):
        """Mock generation based on prompt type."""
        return {}

class MockSemanticMemoryManager:
    """Mock semantic memory manager for testing."""
    
    def __init__(self):
        self.snapshots = {}
        
    async def get_latest_snapshot(self, user_id):
        """Get the latest snapshot for a user."""
        return self.snapshots.get(str(user_id), {})
        
    async def update_snapshot(self, user_id, snapshot):
        """Update a user's snapshot."""
        self.snapshots[str(user_id)] = snapshot
        return True

class MockSessionManager:
    """Mock session manager for testing."""
    
    @staticmethod
    def get_instance():
        """Get the singleton instance."""
        return MockSessionManager()
    
    def session(self):
        """Get a new session."""
        return self.MockSession()
        
    class MockSession:
        """Mock database session."""
        
        def __init__(self):
            self.trees = {}
            self.nodes = {}
            
        async def __aenter__(self):
            return self
            
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
            
        def add(self, obj):
            """Add an object to the session."""
            pass
            
        async def commit(self):
            """Commit the session."""
            pass

async def run_test():
    """Run the test with direct instantiation and detailed error handling."""
    # Set up dependencies
    print("Setting up test dependencies...")
    
    # Try importing HTATreeModel first to ensure we can access it
    try:
        from forest_app.persistence.models import HTATreeModel
        print("Successfully imported HTATreeModel")
    except ImportError as e:
        print(f"Error importing HTATreeModel: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # Try importing HTATreeRepository 
    try:
        from forest_app.persistence.hta_tree_repository import HTATreeRepository
        print("Successfully imported HTATreeRepository")
    except ImportError as e:
        print(f"Error importing HTATreeRepository: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # Try importing SessionManager, HTAMemoryManager, and TaskQueue
    try:
        from forest_app.core.session_manager import SessionManager
        from forest_app.core.services.enhanced_hta.memory import HTAMemoryManager
        from forest_app.core.task_queue import TaskQueue
        print("Successfully imported SessionManager, HTAMemoryManager, and TaskQueue")
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Import the event bus
    try:
        from forest_app.core.event_bus import EventBus, EventType, EventData
        print("Successfully imported EventBus")
    except ImportError as e:
        print(f"Error importing EventBus: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Create test user ID and tree ID
    user_id = uuid.uuid4()
    tree_id = uuid.uuid4()
    print(f"Test user ID: {user_id}")
    print(f"Test tree ID: {tree_id}")
    
    # Create test roadmap
    manifest = RoadmapManifest(
        tree_id=tree_id,
        user_goal="Learn to play the guitar",
        steps=[
            RoadmapStep(
                id=uuid.uuid4(),
                title="Master the basics",
                description="Learn fundamental chords and techniques",
                status="pending",
                priority="high",
                hta_metadata={"is_major_phase": True},
                dependencies=[]
            )
        ],
    )
    print(f"Created test roadmap manifest with ID: {manifest.tree_id}")
    
    # Create mocks for dependencies
    from unittest.mock import MagicMock, AsyncMock
    
    # Create a mock for HTATreeRepository
    mock_repo = MagicMock(spec=HTATreeRepository)
    async def mock_save_tree(tree_model):
        print(f"MOCK: Saving tree model with ID: {tree_model.id}, user_id: {tree_model.user_id}")
        return tree_model
    mock_repo.save_tree_model = AsyncMock(side_effect=mock_save_tree)
    print("Created mock repository")
    
    # Create mock event bus with publish method that validates properly
    mock_event_bus = None
    try:
        mock_event_bus = MagicMock()
        async def mock_publish(event_type, event_data):
            print(f"\nMock event_bus publishing:")
            print(f" - Event type: {event_type}")
            print(f" - Event data type: {type(event_data)}")
            
            # Print all event_data attributes
            if hasattr(event_data, 'event_type'):
                print(f" - Event data event_type: {event_data.event_type}")
            if hasattr(event_data, 'user_id'):
                print(f" - Event data user_id: {event_data.user_id}")
            if hasattr(event_data, 'payload'):
                print(f" - Event data payload: {event_data.payload}")
            
            # Validate event_type is one of the enum values
            if event_type not in list(EventType):
                print(f"WARNING: event_type {event_type} is not in EventType enum: {list(EventType)}")
            
            # Validate event_data.event_type matches event_type
            if hasattr(event_data, 'event_type') and event_data.event_type != event_type:
                print(f"WARNING: event_data.event_type {event_data.event_type} doesn't match event_type {event_type}")
                
            return None
        mock_event_bus.publish = AsyncMock(side_effect=mock_publish)
        print("Created enhanced mock event bus with validation")
        
        # Print all available event types
        print("\nAvailable event types:")
        for event_type in EventType:
            print(f" - {event_type.name}: {event_type.value}")
            
        # Ensure TREE_EVOLVED is available
        if hasattr(EventType, 'TREE_EVOLVED'):
            print(f"\nTREE_EVOLVED is available: {EventType.TREE_EVOLVED}")
        else:
            print("WARNING: TREE_EVOLVED is not available in EventType enum")
    except Exception as e:
        print(f"Error creating mock event bus: {e}")
        import traceback
        traceback.print_exc()
    
    # Create your LLM and Memory mocks
    mock_llm = MagicMock()
    mock_memory = MagicMock()
    
    # Create a simple SessionManager mock without using spec
    mock_session_manager = MagicMock()
    mock_session_manager.add = MagicMock()
    mock_session_manager.commit = AsyncMock()
    # Create a session function that returns the manager itself
    mock_session_manager.session = MagicMock(return_value=mock_session_manager)
    # Setup get_instance class method mock 
    SessionManager.get_instance = classmethod(MagicMock(return_value=mock_session_manager))
    print("Created mock session manager")
    
    # Create TaskQueue mock
    mock_task_queue = MagicMock()
    mock_task_queue.add_task = AsyncMock()
    # Setup get_instance class method mock
    TaskQueue.get_instance = classmethod(MagicMock(return_value=mock_task_queue))
    print("Created mock task queue")
    
    # Create service instance directly without patching
    try:
        print("Creating service directly...")
        service = EnhancedHTAService(mock_llm, mock_memory)
        
        # Manually inject our mocks
        service.tree_repository = mock_repo
        service.event_bus = mock_event_bus
        print("Service created successfully")
        
        # Test the target method
        print("\nTesting generate_initial_hta_from_manifest...")
        try:
            # Create request context
            request_context = {"source": "test"}
            
            print(f"Calling method with:\n - manifest tree_id: {manifest.tree_id}\n - user_id: {user_id}")
            tree_model = await service.generate_initial_hta_from_manifest(
                manifest=manifest,
                user_id=user_id,
                request_context=request_context
            )
            print("\nMethod executed successfully!")
            print(f"Result: {tree_model}")
            print(f"Tree ID: {tree_model.id}")
            print(f"Tree user ID: {tree_model.user_id}")
            print("Test passed!")
        except Exception as e:
            print(f"\nERROR during method execution: {repr(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"\nERROR creating service: {repr(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_test())
