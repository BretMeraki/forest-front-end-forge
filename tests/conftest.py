"""Common test fixtures for Forest App tests."""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any

# --- Windows event loop policy fix for pytest-asyncio ---
import sys
import asyncio
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@pytest.fixture
def mock_feature_flags(mocker):
    """Mock feature flags to always return True."""
    mock_feature = mocker.patch('forest_app.core.feature_flags.Feature')
    mock_is_enabled = mocker.patch('forest_app.core.feature_flags.is_enabled')
    mock_is_enabled.return_value = True
    return mock_is_enabled

@pytest.fixture
def sample_hta_node():
    """Create a sample HTA node for testing."""
    return {
        "id": "test_node_1",
        "parent_id": None,
        "title": "Test Node",
        "description": "A test node for testing",
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "status": "pending",
        "priority": 0.7,
        "magnitude": 5.0,
        "metadata": {}
    }

@pytest.fixture
def sample_snapshot():
    """Create a sample memory snapshot for testing."""
    return {
        "core_state": {
            "hta_tree": {
                "root": {
                    "id": "root",
                    "title": "Root Node",
                    "children": []
                }
            }
        },
        "capacity": 0.8,
        "shadow_score": 0.3,
        "reflection_log": [],
        "task_footprints": [],
        "totems": [],
        "component_state": {}
    }

@pytest.fixture
def mock_llm_client(mocker):
    """Mock LLM client for testing."""
    mock_client = mocker.Mock()
    mock_client.generate.return_value = {"text": "Test response"}
    return mock_client 