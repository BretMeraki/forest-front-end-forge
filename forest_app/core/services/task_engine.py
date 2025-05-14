from typing import List, Dict, Any, Optional, Union, cast
from uuid import UUID
from datetime import datetime

from ..protocols import TaskEngineProtocol, HTANodeProtocol, SemanticMemoryProtocol
from ..models import HTANode, HTATree

class TaskEngine:
    def __init__(self, tree: HTATree, memory_manager: SemanticMemoryProtocol):
        if not isinstance(tree, HTATree):
            raise TypeError("tree must be an HTATree instance")
        if not isinstance(memory_manager, SemanticMemoryProtocol):
            raise TypeError("memory_manager must implement SemanticMemoryProtocol")
            
        self.tree = tree
        self.memory_manager = memory_manager
        self.current_context: Dict[str, Any] = {}

    def generate_task_batch(self, context: Dict[str, Any]) -> List[HTANode]:
        """Generate a new batch of tasks based on the current context and memory."""
        if not isinstance(context, dict):
            raise TypeError("context must be a dictionary")
            
        self.current_context.update(context)
        self.memory_manager.update_context(context)
        
        # Get all available frontier tasks
        try:
            all_tasks = self.tree.get_all_frontier_tasks()
        except Exception as e:
            raise ValueError(f"Error getting frontier tasks: {e}")
        
        # Filter tasks based on context and completion status
        available_tasks = [
            task for task in all_tasks 
            if isinstance(task, HTANode) and task.completion_status < 1.0
        ]
        
        if not available_tasks:
            return []
        
        # Sort tasks by relevance to current context
        try:
            relevant_memories = self.memory_manager.get_relevant_memories(str(context))
        except Exception as e:
            raise ValueError(f"Error getting relevant memories: {e}")
        
        # Priority scoring based on memories and context
        scored_tasks = []
        for task in available_tasks:
            try:
                score = self._calculate_task_priority(task, relevant_memories)
                scored_tasks.append((score, task))
            except Exception as e:
                raise ValueError(f"Error calculating priority for task {task.node_id}: {e}")
        
        # Sort by priority score and return top tasks
        scored_tasks.sort(reverse=True, key=lambda x: x[0])
        return [task for _, task in scored_tasks[:5]]

    def recommend_next_tasks(self, count: int = 3) -> List[HTANode]:
        """Recommend the next best tasks based on current context and history."""
        if count < 1:
            raise ValueError("Task count must be at least 1")
            
        try:
            all_frontier_tasks = self.tree.get_all_frontier_tasks()
        except Exception as e:
            raise ValueError(f"Error getting frontier tasks: {e}")
        
        # Filter out completed tasks
        available_tasks = [
            task for task in all_frontier_tasks 
            if isinstance(task, HTANode) and task.completion_status < 1.0
        ]
        
        if not available_tasks:
            return []
        
        # Get relevant memories for recommendation
        try:
            memories = self.memory_manager.get_relevant_memories(
                str(self.current_context)
            )
        except Exception as e:
            raise ValueError(f"Error getting relevant memories: {e}")
        
        # Score and sort tasks
        scored_tasks = []
        for task in available_tasks:
            try:
                score = self._calculate_task_priority(task, memories)
                scored_tasks.append((score, task))
            except Exception as e:
                raise ValueError(f"Error calculating priority for task {task.node_id}: {e}")
        
        scored_tasks.sort(reverse=True, key=lambda x: x[0])
        return [task for _, task in scored_tasks[:count]]

    def update_task_status(self, task_id: UUID, completion: float) -> None:
        """Update task completion status and propagate changes."""
        if not isinstance(task_id, UUID):
            raise TypeError("task_id must be a UUID")
        if not isinstance(completion, (int, float)):
            raise TypeError("completion must be a number")
        if completion < 0.0 or completion > 1.0:
            raise ValueError("completion must be between 0.0 and 1.0")
            
        try:
            self.tree.update_node(task_id, {'completion_status': completion})
        except Exception as e:
            raise ValueError(f"Error updating task status: {e}")
        
        # Store milestone if task is completed
        if completion >= 1.0:
            task = self.tree.get_node(task_id)
            if task:
                try:
                    self.memory_manager.store_milestone(
                        task_id,
                        f"Completed task: {task.title}",
                        impact=1.0
                    )
                except Exception as e:
                    raise ValueError(f"Error storing milestone: {e}")

    def _calculate_task_priority(
        self, 
        task: HTANode, 
        memories: List[Dict[str, Any]]
    ) -> float:
        """Calculate priority score for a task based on context and memories."""
        if not isinstance(task, HTANode):
            raise TypeError("task must be an HTANode instance")
        if not isinstance(memories, list):
            raise TypeError("memories must be a list")
            
        base_score = 1.0
        
        # Adjust score based on task metadata
        try:
            if 'priority' in task.metadata:
                priority_value = float(task.metadata['priority'])
                if not 0.0 <= priority_value <= 1.0:
                    raise ValueError("Priority must be between 0.0 and 1.0")
                base_score *= priority_value
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid priority value in metadata: {e}")
        
        # Adjust score based on dependencies
        try:
            if 'dependencies' in task.metadata:
                dependencies = task.metadata['dependencies']
                if not isinstance(dependencies, list):
                    raise TypeError("dependencies must be a list")
                for dep_id in dependencies:
                    dep_node = self.tree.get_node(UUID(str(dep_id)))
                    if dep_node and dep_node.completion_status < 1.0:
                        base_score *= 0.5
        except Exception as e:
            raise ValueError(f"Error processing dependencies: {e}")
        
        # Adjust score based on relevant memories
        try:
            memory_boost = 0.0
            task_title_lower = task.title.lower()
            for memory in memories:
                if not isinstance(memory, dict):
                    continue
                content = str(memory.get('content', '')).lower()
                if task_title_lower in content:
                    memory_boost += 0.2
            return base_score + memory_boost
        except Exception as e:
            raise ValueError(f"Error calculating memory boost: {e}") 