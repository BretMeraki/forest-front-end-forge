from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4

from .protocols import HTANodeProtocol, HTATreeProtocol

class HTANode:
    def __init__(
        self,
        title: str,
        description: str,
        parent_id: Optional[UUID] = None,
        node_id: Optional[UUID] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        if not title or not description:
            raise ValueError("Title and description must not be empty")
            
        self.node_id = node_id or uuid4()
        self.parent_id = parent_id
        self.title = title.strip()
        self.description = description.strip()
        self.children: List[HTANode] = []
        self._completion_status = 0.0
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.metadata = metadata or {}

    @property
    def completion_status(self) -> float:
        return self._completion_status

    @completion_status.setter
    def completion_status(self, value: float) -> None:
        if not isinstance(value, (int, float)):
            raise ValueError("Completion status must be a number")
        if value < 0.0 or value > 1.0:
            raise ValueError("Completion status must be between 0.0 and 1.0")
        self._completion_status = float(value)
        self.updated_at = datetime.utcnow()

    def add_child(self, node: 'HTANode') -> None:
        if not isinstance(node, HTANode):
            raise TypeError("Child must be an HTANode instance")
        if node.node_id == self.node_id:
            raise ValueError("Cannot add node as its own child")
        if any(child.node_id == node.node_id for child in self.children):
            raise ValueError("Child node already exists")
            
        node.parent_id = self.node_id
        self.children.append(node)
        self.update_completion()

    def remove_child(self, node_id: UUID) -> None:
        if not isinstance(node_id, UUID):
            raise TypeError("node_id must be a UUID")
            
        original_length = len(self.children)
        self.children = [child for child in self.children if child.node_id != node_id]
        
        if len(self.children) == original_length:
            raise ValueError(f"Child node {node_id} not found")
            
        self.update_completion()

    def update_completion(self) -> None:
        if not self.children:
            return  # Leaf node's completion is set directly
        
        total_weight = len(self.children)
        if total_weight == 0:
            return
        
        try:
            completed_weight = sum(child.completion_status for child in self.children)
            new_status = completed_weight / total_weight
            self.completion_status = new_status  # Use property setter for validation
        except Exception as e:
            raise ValueError(f"Error calculating completion status: {e}")

    def get_frontier_tasks(self) -> List['HTANode']:
        if not self.children:
            return [self]  # This is a frontier task
        
        try:
            frontier_tasks = []
            for child in self.children:
                if not isinstance(child, HTANode):
                    raise TypeError(f"Invalid child type: {type(child)}")
                frontier_tasks.extend(child.get_frontier_tasks())
            return frontier_tasks
        except Exception as e:
            raise ValueError(f"Error getting frontier tasks: {e}")

class HTATree:
    def __init__(self, root: HTANode):
        if not isinstance(root, HTANode):
            raise TypeError("Root must be an HTANode instance")
        self.root = root

    def get_node(self, node_id: UUID, current_node: Optional[HTANode] = None) -> Optional[HTANode]:
        if not isinstance(node_id, UUID):
            raise TypeError("node_id must be a UUID")
            
        current = current_node or self.root
        if not isinstance(current, HTANode):
            raise TypeError("current_node must be an HTANode instance")
        
        if current.node_id == node_id:
            return current
            
        for child in current.children:
            result = self.get_node(node_id, child)
            if result:
                return result
        
        return None

    def update_node(self, node_id: UUID, updates: Dict[str, Any]) -> None:
        if not isinstance(updates, dict):
            raise TypeError("updates must be a dictionary")
            
        node = self.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")
        
        for key, value in updates.items():
            if not hasattr(node, key):
                raise ValueError(f"Invalid attribute: {key}")
            if key == 'completion_status':
                node.completion_status = value  # Use property setter for validation
            else:
                setattr(node, key, value)
        
        node.updated_at = datetime.utcnow()
        if 'completion_status' in updates:
            self.propagate_completion(node_id)

    def propagate_completion(self, node_id: UUID) -> None:
        if not isinstance(node_id, UUID):
            raise TypeError("node_id must be a UUID")
            
        node = self.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")
        
        # Update completion status up the tree
        current = node
        visited = set()  # Prevent infinite loops
        while current.parent_id:
            if current.node_id in visited:
                raise ValueError("Circular reference detected in tree")
            visited.add(current.node_id)
            
            parent = self.get_node(current.parent_id)
            if parent:
                parent.update_completion()
                current = parent
            else:
                raise ValueError(f"Parent node {current.parent_id} not found")

    def get_all_frontier_tasks(self) -> List[HTANode]:
        return self.root.get_frontier_tasks() 