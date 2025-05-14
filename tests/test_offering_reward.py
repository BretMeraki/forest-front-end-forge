"""Tests for offering reward module."""

import pytest
from datetime import datetime, timezone
from forest_app.modules.offering_reward import (
    OfferingRouter,
    OfferingSuggestion,
    OfferingResponseModel,
    DEFAULT_OFFERING_ERROR_MSG,
    DEFAULT_OFFERING_DISABLED_MSG
)

class MockDesireEngine:
    """Mock DesireEngine for testing."""
    def __init__(self, wants=None):
        self.wants = wants or ["reading", "exercise", "meditation"]
    
    def get_all_wants(self):
        return self.wants

class MockFinancialEngine:
    """Mock FinancialReadinessEngine for testing."""
    def __init__(self, readiness=0.7):
        self.readiness = readiness

@pytest.fixture
def offering_router():
    """Create an OfferingRouter instance for testing."""
    desire_engine = MockDesireEngine()
    financial_engine = MockFinancialEngine()
    return OfferingRouter(desire_engine=desire_engine, financial_engine=financial_engine)

@pytest.fixture
def sample_snapshot():
    """Create a sample snapshot for testing."""
    return {
        "wants_cache": {
            "reading": 0.8,
            "exercise": 0.7,
            "meditation": 0.6
        },
        "totems": [],
        "component_state": {
            "OfferingRouter": {
                "totems": []
            }
        }
    }

@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    return {
        "id": "task_1",
        "title": "Complete Project Report",
        "description": "Write and submit the final project report",
        "magnitude": 0.8
    }

@pytest.fixture
def mock_feature_flags(monkeypatch):
    def _set_enabled(return_value):
        monkeypatch.setattr("forest_app.modules.offering_reward.is_enabled", lambda feature: return_value)
    return _set_enabled

def test_offering_router_initialization(offering_router):
    """Test OfferingRouter initialization."""
    assert offering_router.desire_engine is not None
    assert offering_router.financial_engine is not None
    assert offering_router.logger is not None

def test_preview_offering_disabled(offering_router, mock_feature_flags):
    """Test preview_offering_for_task when feature is disabled."""
    mock_feature_flags(False)
    
    result = offering_router.preview_offering_for_task({}, None, 0.5)
    assert isinstance(result, list)
    assert len(result) == 0

def test_preview_offering_success(offering_router, sample_snapshot, sample_task, mock_feature_flags):
    """Test successful preview offering generation."""
    result = offering_router.preview_offering_for_task(sample_snapshot, sample_task, 0.7)
    
    assert isinstance(result, list)
    assert result == []  # Rewards are disabled for MVP

@pytest.mark.asyncio
async def test_generate_offering_disabled(offering_router, mock_feature_flags):
    """Test maybe_generate_offering when feature is disabled."""
    # Override mock to return False for REWARDS feature
    mock_feature_flags(False)
    
    result = await offering_router.maybe_generate_offering({})
    assert isinstance(result, list)
    assert len(result) == 0

@pytest.mark.asyncio
async def test_generate_offering_success(offering_router, sample_snapshot, mock_feature_flags):
    """Test successful offering generation."""
    result = await offering_router.maybe_generate_offering(sample_snapshot, reward_scale=0.7)
    assert isinstance(result, list)
    assert result == []  # Rewards are disabled for MVP

def test_record_acceptance_disabled(offering_router, mock_feature_flags):
    """Test record_acceptance when feature is disabled."""
    # Override mock to return False for REWARDS feature
    mock_feature_flags(False)
    
    result = offering_router.record_acceptance({}, "Test suggestion")
    assert result == {"error": "Cannot record acceptance; rewards disabled or error occurred."}

def test_record_acceptance_success(offering_router, sample_snapshot, mock_feature_flags):
    """Test successful acceptance recording."""
    suggestion = "Take a relaxing walk"
    result = offering_router.record_acceptance(sample_snapshot, suggestion)
    assert result == {"error": "Cannot record acceptance; rewards disabled or error occurred."}

def test_record_acceptance_invalid_suggestion(offering_router, sample_snapshot, mock_feature_flags):
    """Test record_acceptance with invalid suggestion."""
    result = offering_router.record_acceptance(sample_snapshot, "")
    assert result == {"error": "Cannot record acceptance; rewards disabled or error occurred."}

def test_record_acceptance_invalid_snapshot(offering_router, mock_feature_flags):
    """Test record_acceptance with invalid snapshot structure."""
    invalid_snapshot = {"totems": "not_a_list"}
    result = offering_router.record_acceptance(invalid_snapshot, "Test suggestion")
    assert "error" in result

def test_get_snapshot_data(offering_router):
    """Test _get_snapshot_data helper method."""
    # Test with dict
    dict_snap = {"test_key": "test_value"}
    assert offering_router._get_snapshot_data(dict_snap, "test_key") == "test_value"
    assert offering_router._get_snapshot_data(dict_snap, "missing_key", "default") == "default"
    
    # Test with object
    class TestSnapshot:
        test_key = "test_value"
    
    obj_snap = TestSnapshot()
    assert offering_router._get_snapshot_data(obj_snap, "test_key") == "test_value"
    assert offering_router._get_snapshot_data(obj_snap, "missing_key", "default") == "default"

def test_pydantic_models():
    """Test Pydantic model validation."""
    # Test OfferingSuggestion
    suggestion = OfferingSuggestion(suggestion="Test suggestion")
    assert suggestion.suggestion == "Test suggestion"
    
    with pytest.raises(ValueError):
        OfferingSuggestion(suggestion="")  # Empty suggestion should fail
    
    # Test OfferingResponseModel
    response = OfferingResponseModel(suggestions=[
        OfferingSuggestion(suggestion="Suggestion 1"),
        OfferingSuggestion(suggestion="Suggestion 2")
    ])
    assert len(response.suggestions) == 2
    
    with pytest.raises(ValueError):
        OfferingResponseModel(suggestions=[])  # Empty suggestions list should fail 