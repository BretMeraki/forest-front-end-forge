import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone
from uuid import uuid4, UUID

import sys
sys.modules['forest_app.core.utils'] = MagicMock()

from forest_app.core.processors.completion_processor import CompletionProcessor

class DummyNode:
    def __init__(self, id=None, user_id=None, status="pending", parent_id=None, roadmap_step_id=None, title="Test Task", tree_id=None, is_major_phase=False, description="", branch_triggers=None):
        self.id = id or uuid4()
        self.user_id = user_id or uuid4()
        self.status = status
        self.parent_id = parent_id
        self.roadmap_step_id = roadmap_step_id
        self.title = title
        self.tree_id = tree_id or uuid4()
        self.is_major_phase = is_major_phase
        self.description = description
        self.branch_triggers = branch_triggers or MagicMock(current_completion_count=0)

class DummyTree:
    def __init__(self, id=None, manifest=None):
        self.id = id or uuid4()
        self.manifest = manifest or {}

@pytest.mark.asyncio
async def test_idempotent_completion_returns_reinforcement():
    # Arrange
    node_id = uuid4()
    user_id = uuid4()
    dummy_node = DummyNode(id=node_id, user_id=user_id, status="completed")
    dummy_tree = DummyTree(id=dummy_node.tree_id)
    dummy_footprint = MagicMock(reinforcement_message="Great work! Already done.")

    processor = CompletionProcessor(
        llm_client=MagicMock(),
        hta_service=MagicMock(),
        tree_repository=MagicMock(),
        memory_manager=MagicMock(),
        task_logger=MagicMock(),
        reflection_logger=MagicMock()
    )
    processor.tree_repository.get_node_by_id = AsyncMock(return_value=dummy_node)
    processor.tree_repository.get_task_footprint = AsyncMock(return_value=dummy_footprint)
    processor.tree_repository.get_tree_by_id = AsyncMock(return_value=dummy_tree)

    # Act
    result = await processor.process_node_completion(node_id, user_id)

    # Assert
    assert result["status"] == "already_completed"
    assert "reinforcement_message" in result
    assert result["reinforcement_message"] == "Great work! Already done."

@pytest.mark.asyncio
async def test_successful_completion_updates_memory_and_returns_reinforcement():
    # Arrange
    node_id = uuid4()
    user_id = uuid4()
    dummy_node = DummyNode(id=node_id, user_id=user_id, status="pending", roadmap_step_id=uuid4(), is_major_phase=True)
    dummy_tree = DummyTree(
        id=dummy_node.tree_id,
        manifest={
            "tree_id": str(dummy_node.tree_id),
            "user_goal": "dummy goal"
        }
    )

    processor = CompletionProcessor(
        llm_client=MagicMock(),
        hta_service=MagicMock(),
        tree_repository=MagicMock(),
        memory_manager=MagicMock(),
        task_logger=MagicMock(),
        reflection_logger=MagicMock()
    )
    processor.tree_repository.get_node_by_id = AsyncMock(return_value=dummy_node)
    processor.tree_repository.get_tree_by_id = AsyncMock(return_value=dummy_tree)
    processor.tree_repository.update_node_status = AsyncMock(return_value=True)
    processor.tree_repository.increment_branch_completion_count = AsyncMock(return_value=(True, 1))
    processor.tree_repository.get_task_footprint = AsyncMock(return_value=None)
    processor.tree_repository.update_tree = AsyncMock()
    processor.task_logger.log_task_completion = AsyncMock()
    processor.memory_manager.update_memory_with_completion = AsyncMock(return_value=True)
    processor._generate_positive_reinforcement = AsyncMock(return_value="You did it!")

    # Act
    result = await processor.process_node_completion(node_id, user_id)

    # Assert
    assert result["status"] == "completed"
    assert result["reinforcement_message"] == "You did it!"
    processor.memory_manager.update_memory_with_completion.assert_awaited_once()
    processor.task_logger.log_task_completion.assert_awaited()

@pytest.mark.asyncio
async def test_transaction_rollback_on_failure(monkeypatch):
    # Arrange
    node_id = uuid4()
    user_id = uuid4()
    dummy_node = DummyNode(id=node_id, user_id=user_id, status="pending")
    dummy_tree = DummyTree(id=dummy_node.tree_id)

    processor = CompletionProcessor(
        llm_client=MagicMock(),
        hta_service=MagicMock(),
        tree_repository=MagicMock(),
        memory_manager=MagicMock(),
        task_logger=MagicMock(),
        reflection_logger=MagicMock()
    )
    processor.tree_repository.get_node_by_id = AsyncMock(return_value=dummy_node)
    processor.tree_repository.get_tree_by_id = AsyncMock(return_value=dummy_tree)
    processor.tree_repository.update_node_status = AsyncMock(return_value=False)  # Simulate failure

    # Act & Assert
    with pytest.raises(RuntimeError):
        await processor.process_node_completion(node_id, user_id)

@pytest.mark.asyncio
async def test_reflection_logged_when_provided():
    # Arrange
    node_id = uuid4()
    user_id = uuid4()
    dummy_node = DummyNode(id=node_id, user_id=user_id, status="pending")
    dummy_tree = DummyTree(id=dummy_node.tree_id)

    processor = CompletionProcessor(
        llm_client=MagicMock(),
        hta_service=MagicMock(),
        tree_repository=MagicMock(),
        memory_manager=MagicMock(),
        task_logger=MagicMock(),
        reflection_logger=MagicMock()
    )
    processor.tree_repository.get_node_by_id = AsyncMock(return_value=dummy_node)
    processor.tree_repository.get_tree_by_id = AsyncMock(return_value=dummy_tree)
    processor.tree_repository.update_node_status = AsyncMock(return_value=True)
    processor.tree_repository.get_task_footprint = AsyncMock(return_value=None)
    processor.tree_repository.update_tree = AsyncMock()
    processor.memory_manager.update_memory_with_completion = AsyncMock(return_value=True)
    processor._generate_positive_reinforcement = AsyncMock(return_value="Well done!")
    processor.task_logger.log_task_completion = AsyncMock()
    processor.reflection_logger.log_reflection = AsyncMock()

    # Act
    reflection = "I learned a lot!"
    await processor.process_node_completion(node_id, user_id, reflection=reflection)

    # Assert
    processor.reflection_logger.log_reflection.assert_awaited()
