"""Tests for task engine module."""

import pytest
from datetime import datetime, timezone
from forest_app.modules.task_engine import TaskEngine, _calculate_node_score
from unittest.mock import patch

class MockHTANode:
    """Mock HTA Node for testing."""
    def __init__(self, **kwargs):
        self.id = kwargs.get('id', 'test_id')
        self.title = kwargs.get('title', 'Test Node')
        self.description = kwargs.get('description', 'Test Description')
        self.status = kwargs.get('status', 'pending')
        self.priority = kwargs.get('priority', 0.5)
        self.magnitude = kwargs.get('magnitude', 5.0)
        self.depends_on = kwargs.get('depends_on', [])
        self.estimated_energy = kwargs.get('estimated_energy', 'low')

class MockHTATree:
    """Mock HTA Tree for testing."""
    def __init__(self, nodes=None):
        self.nodes = nodes or {}
        self.root = MockHTANode(id='root')

    def flatten_tree(self):
        """Return list of all nodes."""
        return list(self.nodes.values())

    def get_node_map(self):
        """Return dict of nodes."""
        return self.nodes

    def get_node_depth(self, node_id):
        """Mock depth calculation."""
        return 1 if node_id in self.nodes else -1

@pytest.fixture
def task_engine():
    """Create a TaskEngine instance for testing."""
    return TaskEngine()

@pytest.fixture
def mock_pattern_engine(mocker):
    """Create a mock pattern engine."""
    mock_engine = mocker.Mock()
    mock_engine.identify_patterns.return_value = []
    return mock_engine

@pytest.fixture
def sample_snapshot():
    """Create a sample snapshot for testing."""
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
        "withering_level": 0.2
    }

def test_calculate_node_score():
    """Test node score calculation."""
    node = MockHTANode(priority=0.7)
    snapshot = {"capacity": 0.8, "withering_level": 0.2}
    pattern_score = 0.5

    score = _calculate_node_score(node, snapshot, pattern_score)
    assert isinstance(score, float)
    assert 0 <= score <= 1

def test_calculate_node_score_invalid_priority():
    """Test node score calculation with invalid priority."""
    node = MockHTANode()
    node.priority = "invalid"  # Set invalid priority
    snapshot = {"capacity": 0.8, "withering_level": 0.2}
    
    score = _calculate_node_score(node, snapshot)
    assert isinstance(score, float)
    assert score >= 0  # Should use default priority

def test_task_engine_initialization(task_engine, mock_pattern_engine):
    """Test TaskEngine initialization."""
    engine = TaskEngine(pattern_engine=mock_pattern_engine)
    assert engine.pattern_engine == mock_pattern_engine
    assert engine.logger is not None

def test_get_next_step_no_hta(task_engine):
    """Test get_next_step when no HTA tree is available."""
    snapshot = {"core_state": {}}
    with patch("forest_app.modules.task_engine.is_enabled", return_value=True):
        result = task_engine.get_next_step(snapshot)
        assert "fallback_task" in result
        assert isinstance(result["fallback_task"], dict)
        assert "tasks" in result
        assert len(result["tasks"]) == 0

def test_get_next_step_with_tasks(task_engine, sample_snapshot):
    """Test get_next_step with valid HTA nodes."""
    # Create mock nodes
    nodes = {
        'task1': MockHTANode(id='task1', priority=0.8, magnitude=5.0),
        'task2': MockHTANode(id='task2', priority=0.6, magnitude=4.0)
    }
    tree = MockHTATree(nodes)
    
    # Update snapshot with mock tree
    sample_snapshot["core_state"]["hta_tree"] = {
        "root": {
            "id": "root",
            "title": "Root",
            "children": list(nodes.keys())
        }
    }
    
    def mock_from_dict(data):
        return tree
    
    with patch("forest_app.modules.task_engine.is_enabled", return_value=True):
        with patch("forest_app.modules.hta_tree.HTATree.from_dict", mock_from_dict):
            result = task_engine.get_next_step(sample_snapshot)
            
            assert "tasks" in result
            assert len(result["tasks"]) == 2
            assert "fallback_task" in result
            assert result["fallback_task"] is None

def test_check_dependencies(task_engine):
    """Test dependency checking."""
    # Create mock nodes and tree
    nodes = {
        'task1': MockHTANode(id='task1', status='completed'),
        'task2': MockHTANode(id='task2', depends_on=['task1'], status='pending'),
        'task3': MockHTANode(id='task3', depends_on=['task1'], status='completed')
    }
    tree = MockHTATree(nodes)
    
    # Test node with no dependencies
    assert task_engine._check_dependencies(nodes['task1'], tree) is True
    
    # Test node with incomplete dependency
    assert task_engine._check_dependencies(nodes['task2'], tree) is True
    
    # Test node with completed dependency
    assert task_engine._check_dependencies(nodes['task3'], tree) is True

def test_check_resources(task_engine):
    """Test resource checking."""
    node = MockHTANode(estimated_energy='medium')
    snapshot = {"capacity": 0.8}
    with patch("forest_app.modules.task_engine.is_enabled", return_value=True):
        assert task_engine._check_resources(node, snapshot) is True
    
    # Test with insufficient capacity
    snapshot["capacity"] = 0.2
    with patch("forest_app.modules.task_engine.is_enabled", return_value=True):
        assert task_engine._check_resources(node, snapshot) is False

def test_create_task_from_hta_node(task_engine):
    """Test task creation from HTA node."""
    node = MockHTANode(
        id='test_task',
        title='Test Task',
        description='Test Description',
        priority=0.7,
        magnitude=5.0
    )
    tree = MockHTATree({node.id: node})
    snapshot = {}
    
    task = task_engine._create_task_from_hta_node(snapshot, node, tree)
    
    assert isinstance(task, dict)
    assert task["id"].startswith("hta_")
    assert task["title"] == "Test Task"
    assert task["description"] == "Test Description"
    assert task["magnitude"] == 5.0
    assert isinstance(task["created_at"], str)
    assert "metadata" in task 