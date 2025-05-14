"""
Tests for the HTAService and its enhanced version, focusing on manifest-to-HTA conversion.
"""

import pytest
import asyncio
import json
import time
import uuid
from uuid import UUID
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

# Import models and services
from forest_app.core.roadmap_models import RoadmapManifest, RoadmapStep
from forest_app.persistence.models import HTANodeModel, HTATreeModel
from forest_app.core.services.enhanced_hta_service import EnhancedHTAService
from forest_app.core.event_bus import EventBus, EventType


class TestHTAService:
    """Tests for the HTAService implementation."""

    @pytest.mark.asyncio
    async def test_hta_minimal(self):
        """A minimal test for generate_initial_hta_from_manifest that avoids fixture complexities."""
        import uuid
        from unittest.mock import MagicMock, AsyncMock, patch
        from datetime import datetime, timezone
        from forest_app.core.services.enhanced_hta.core import EnhancedHTAService
        from forest_app.core.roadmap_models import RoadmapManifest, RoadmapStep
        from forest_app.persistence.models import HTATreeModel
        
        # Create minimal test data
        tree_id = uuid.uuid4()
        user_id = uuid.uuid4()
        step_id = uuid.uuid4()
        
        # Create a simple manifest
        manifest = RoadmapManifest(
            tree_id=tree_id,
            user_goal="Test Goal",
            q_and_a_responses=[],
            steps=[RoadmapStep(
                id=step_id,
                title="Step 1",
                description="First step",
                status="pending",
                priority="high",
                dependencies=[]
            )]
        )
        
        # Create service with minimal mocked dependencies
        mock_llm = MagicMock()
        mock_memory = MagicMock()
        
        service = EnhancedHTAService(mock_llm, mock_memory)
        
        # Inject test dependencies directly 
        mock_repo = MagicMock()
        async def save_tree_model_side_effect(tree_model):
            # Check that we're creating the model with goal_name, not title
            print(f"\nMOCK REPO: Creating HTATreeModel with:")
            print(f" - id: {tree_model.id}")
            print(f" - user_id: {tree_model.user_id}")
            print(f" - has goal_name: {hasattr(tree_model, 'goal_name')}")
            if hasattr(tree_model, 'goal_name'):
                print(f" - goal_name: {tree_model.goal_name}")
            return tree_model
            
        mock_repo.save_tree_model = AsyncMock(side_effect=save_tree_model_side_effect)
        service.tree_repository = mock_repo
        
        # Create mock event bus
        from forest_app.core.event_bus import EventType, EventData
        mock_event_bus = MagicMock()
        async def mock_publish(event_type, event_data):
            print(f"\nMock event_bus.publish called with:")
            print(f" - Event type: {event_type}")
            # This test will succeed whether event_bus.publish is called or not
            return None
        mock_event_bus.publish = AsyncMock(side_effect=mock_publish)
        service.event_bus = mock_event_bus
        
        # Create other required mocks
        mock_event_bus = MagicMock()
        mock_event_bus.publish = AsyncMock(return_value=None)
        service.event_bus = mock_event_bus
        
        # Execute the method
        try:
            print(f"\n\n=== Running minimal test with tree_id={tree_id}, user_id={user_id}")
            result = await service.generate_initial_hta_from_manifest(
                manifest=manifest,
                user_id=user_id,
                request_context={"source": "test"}
            )
            print(f"Success! Result: {result}")
            print(f"Result id={result.id}, user_id={result.user_id}")
            
            # Assertions
            assert result is not None
            assert result.id == tree_id
            assert result.user_id == user_id
            assert mock_repo.save_tree_model.called
            assert mock_event_bus.publish.called
            
        except Exception as e:
            import traceback
            print(f"Exception in minimal test: {repr(e)}")
            traceback.print_exc()
            raise

    @pytest.fixture
    def mock_llm_client(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_semantic_memory_manager(self):
        return MagicMock()
    
    @pytest.fixture
    def mock_event_bus(self):
        mock = AsyncMock()
        mock.publish = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_cache_service(self):
        mock = AsyncMock()
        mock.delete = AsyncMock()
        mock.get = AsyncMock(return_value=None)
        mock.set = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_task_queue(self):
        mock = AsyncMock()
        mock.enqueue = AsyncMock()
        return mock
    
    @pytest.fixture
    def hta_service(self, mock_llm_client, mock_semantic_memory_manager, 
                    mock_event_bus, mock_cache_service, mock_task_queue):
        """Create a test HTA service with mocked dependencies and repository."""
        service = EnhancedHTAService(mock_llm_client, mock_semantic_memory_manager)
        # Inject a MagicMock as the repository
        mock_repo = MagicMock()
        from forest_app.persistence.models import HTATreeModel
        async def save_tree_model_side_effect(tree_model):
            # Return a real HTATreeModel with the same id and user_id
            # Convert 'title' to 'goal_name' since that's what the actual model uses
            print(f"\nMOCK REPOSITORY: Saving tree model with attributes:\n - id: {tree_model.id}\n - user_id: {tree_model.user_id}")
            # If tree_model has 'title', use it for goal_name
            goal_name = getattr(tree_model, 'title', None) or "Default Goal"
            print(f" - Converting title '{goal_name}' to goal_name")
            
            return HTATreeModel(
                id=tree_model.id,
                user_id=tree_model.user_id,
                goal_name=goal_name,  # Use goal_name instead of title
                created_at=getattr(tree_model, 'created_at', None),
                updated_at=getattr(tree_model, 'updated_at', None)
                # Note: 'status' is removed as it's not in the model
            )
        mock_repo.save_tree_model = AsyncMock(side_effect=save_tree_model_side_effect)
        service.tree_repository = mock_repo
        
        # Explicitly setup event_bus with a proper publish method that won't raise errors
        async def mock_publish(event_type, event_data):
            print(f"\nMock event_bus publishing: {event_type}")
            print(f"Event data: {event_data}")
            return None
        mock_event_bus.publish = AsyncMock(side_effect=mock_publish)
        service.event_bus = mock_event_bus
        
        service.cache = mock_cache_service
        service.task_queue = mock_task_queue
        return service
    
    @pytest.fixture
    def sample_manifest(self):
        """Create a sample RoadmapManifest for testing."""
        tree_id = uuid.uuid4()
        
        # Create a few steps with dependencies
        step1_id = uuid.uuid4()
        step2_id = uuid.uuid4()
        step3_id = uuid.uuid4()
        
        step1 = RoadmapStep(
            id=step1_id,
            title="First step",
            description="This is the first step",
            status="pending",
            priority="high",
            dependencies=[]
        )
        
        step2 = RoadmapStep(
            id=step2_id,
            title="Second step",
            description="This is the second step",
            status="pending",
            priority="medium",
            dependencies=[step1_id]
        )
        
        step3 = RoadmapStep(
            id=step3_id,
            title="Third step",
            description="This is the third step",
            status="pending",
            priority="low",
            dependencies=[step1_id, step2_id]
        )
        
        return RoadmapManifest(
            tree_id=tree_id,
            user_goal="Test goal",
            q_and_a_responses=[],
            steps=[step1, step2, step3]
        )
    
    @pytest.fixture
    def mock_db_session(self):
        """Create a mock DB session for testing."""
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        mock_session.bulk_save_objects = MagicMock()
        mock_session.commit = AsyncMock()
        return mock_session
    
    @pytest.mark.asyncio
    async def test_direct_debug_hta_service(self):
        """Direct test of HTA service with controlled mocks."""
        # Import needed components
        from forest_app.core.services.enhanced_hta.core import EnhancedHTAService
        from forest_app.core.roadmap_models import RoadmapManifest, RoadmapStep
        from forest_app.persistence.models import HTATreeModel
        from unittest.mock import MagicMock, AsyncMock, patch
        from forest_app.core.event_bus import EventType, EventData
        import traceback
        import uuid
        from datetime import datetime, timezone
        
        print("\n\n=== DIRECT DEBUG TEST ===")
        
        # Create test data
        tree_id = uuid.uuid4()
        user_id = uuid.uuid4()
        step_id = uuid.uuid4()
        
        manifest = RoadmapManifest(
            tree_id=tree_id,
            user_goal="Debug Test Goal",
            q_and_a_responses=[],
            steps=[RoadmapStep(
                id=step_id,
                title="Debug Step",
                description="Test step",
                status="pending",
                priority="high",
                dependencies=[]
            )]
        )
        
        # Create minimal mocks
        mock_llm = MagicMock()
        mock_memory = MagicMock()
        
        # Create service instance
        service = EnhancedHTAService(mock_llm, mock_memory)
        
        # Create repository mock with controlled behavior
        mock_repo = MagicMock()
        async def mock_save_tree(tree_model):
            print(f"\nMock repository saving tree:")
            print(f" - id: {tree_model.id}")
            print(f" - user_id: {tree_model.user_id}")
            
            # Print all available attributes
            for attr in dir(tree_model):
                if not attr.startswith('_') and not callable(getattr(tree_model, attr)):
                    print(f" - {attr}: {getattr(tree_model, attr)}")
            
            # Return the same model as passed in
            return tree_model
        mock_repo.save_tree_model = AsyncMock(side_effect=mock_save_tree)
        
        # Create event bus mock with proper publish method
        mock_event_bus = MagicMock()
        async def mock_publish(event_type, event_data):
            print(f"Mock event_bus publishing: {event_type}, data={event_data}")
            return None
        mock_event_bus.publish = AsyncMock(side_effect=mock_publish)
        
        # Inject mocks directly
        service.tree_repository = mock_repo
        service.event_bus = mock_event_bus
        
        print("Mocks prepared and injected")
        print(f"service.tree_repository={service.tree_repository}")
        print(f"service.event_bus={service.event_bus}")
        
        # Execute test
        try:
            print("Calling generate_initial_hta_from_manifest...")
            result = await service.generate_initial_hta_from_manifest(
                manifest=manifest,
                user_id=user_id,
                request_context={"source": "debug_test"}
            )
            print("Call completed successfully!")
            if result:
                print(f"Result id: {result.id}")
                print(f"Result user_id: {result.user_id}")
            
            # Assertions
            assert result is not None, "Result is None!"
            assert hasattr(result, 'id')
            assert result.id == tree_id
            assert hasattr(result, 'user_id')
            assert result.user_id == user_id
            
            # Verify mocks were called
            mock_repo.save_tree_model.assert_called_once()
            mock_event_bus.publish.assert_called_once()
            
            print("All assertions passed!")
            
        except Exception as e:
            print(f"Exception: {repr(e)}")
            print(f"Exception type: {type(e)}")
            traceback.print_exc()
            raise
            
    @patch('forest_app.persistence.database.get_db_session')
    async def test_generate_initial_hta_from_manifest_success(
        self, mock_get_session, hta_service, sample_manifest, mock_db_session
    ):
        """Test successful generation of HTA from manifest."""
        # Arrange
        user_id = uuid.uuid4()
        request_context = {"source": "test"}
        
        print("\n\n=== Test Setup ===")
        print(f"sample_manifest.tree_id: {sample_manifest.tree_id}")
        print(f"user_id: {user_id}")
        print(f"hta_service.tree_repository: {hta_service.tree_repository}")
        print(f"hta_service.event_bus: {hta_service.event_bus}")
        print(f"has publish method: {hasattr(hta_service.event_bus, 'publish')}")
        
        # Ensure event bus publish is properly mocked
        if not hasattr(hta_service.event_bus, 'publish') or not callable(getattr(hta_service.event_bus, 'publish')):
            print("WARNING: event_bus.publish is not callable!")
            hta_service.event_bus.publish = AsyncMock(return_value=None)
        
        # Configure the mock session context manager
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_db_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        mock_get_session.return_value = mock_session_context
        
        # Act
        import traceback
        start_time = time.time()
        try:
            print("Calling generate_initial_hta_from_manifest...")
            result = await hta_service.generate_initial_hta_from_manifest(
                manifest=sample_manifest,
                user_id=user_id,
                request_context=request_context
            )
            print("Call completed successfully!")
            if result:
                print(f"Result id: {result.id}")
                print(f"Result user_id: {result.user_id}")
        except Exception as e:
            print('Exception during HTA generation:', repr(e))
            print(f"Exception type: {type(e)}")
            traceback.print_exc()
            raise
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000  # ms
        
        print("\n=== Test Assertions ===")
        print(f"result: {result if result else 'None'}")
        
        # Assert
        assert result is not None, "Result is None!"
        assert hasattr(result, 'id'), "Result has no id attribute!"
        assert hasattr(result, 'user_id'), "Result has no user_id attribute!"
        # Verify repository method was called
        hta_service.tree_repository.save_tree_model.assert_called_once()
        # Verify event was published
        hta_service.event_bus.publish.assert_called_once()
        
        # Verify performance (<1s)
        assert execution_time < 1000, f"Execution took {execution_time}ms, exceeding the 1s target"

    @patch('forest_app.persistence.database.get_db_session')
    async def test_generate_initial_hta_from_manifest_db_error(
        self, mock_get_session, hta_service, sample_manifest, mock_db_session
    ):
        """Test handling of database errors during HTA generation."""
        # Arrange
        user_id = uuid.uuid4()
        request_context = {"source": "test"}
        
        # Configure mock to simulate DB error
        mock_db_session.commit.side_effect = SQLAlchemyError("Test DB error")
        
        # Configure the mock session context manager
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_db_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        mock_get_session.return_value = mock_session_context
        
        # Simulate repository raising DB error
        hta_service.tree_repository.save_tree_model.side_effect = SQLAlchemyError("Test DB error")
        # Act & Assert
        with pytest.raises(SQLAlchemyError):
            await hta_service.generate_initial_hta_from_manifest(
                manifest=sample_manifest,
                user_id=user_id,
                request_context=request_context
            )
        # Verify repository method was called
        hta_service.tree_repository.save_tree_model.assert_called_once()
        # Verify no event was published (since operation failed)
        hta_service.event_bus.publish.assert_not_called()

    @patch('forest_app.persistence.database.get_db_session')
    async def test_generate_initial_hta_with_empty_manifest(
        self, mock_get_session, hta_service, mock_db_session
    ):
        """Test handling of empty manifest."""
        # Arrange
        user_id = uuid.uuid4()
        request_context = {"source": "test"}
        
        # Empty manifest with only tree_id
        empty_manifest = RoadmapManifest(
            tree_id=uuid.uuid4(),
            user_goal="Empty goal",
            q_and_a_responses=[],
            steps=[]
        )
        
        # Configure the mock session context manager
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_db_session)
        mock_session_context.__aexit__ = AsyncMock(return_value=None)
        mock_get_session.return_value = mock_session_context
        
        # Act
        result = await hta_service.generate_initial_hta_from_manifest(
            manifest=empty_manifest,
            user_id=user_id,
            request_context=request_context
        )
        
        # Assert
        assert result is not None
        assert hasattr(result, 'id')
        assert hasattr(result, 'user_id')
        # Verify repository method was called
        hta_service.tree_repository.save_tree_model.assert_called_once()
        # Verify event was published
        hta_service.event_bus.publish.assert_called_once()


# Integration tests - require actual DB connection
@pytest.mark.integration
class TestHTAServiceIntegration:
    """Integration tests for the HTAService implementation."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_manifest_to_tree(self):
        """
        End-to-end test of manifest to tree conversion with actual DB.
        Requires database connection to be configured in test environment.
        """
        # This would be implemented in a full test suite with proper DB fixtures
        pass
