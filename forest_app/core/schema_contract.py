"""
HTA Schema Contract

This module defines the structural contract for HTA trees without templating content.
It ensures data integrity, performance, and flexibility while preserving uniqueness.
The contract acts as a framework that guides the generation of HTA trees without
dictating specific content, aligning with the PRD's vision of personalized experiences.
"""

import logging
from typing import Dict, List, Callable, Any, TypeVar, Set
from uuid import UUID
import re

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Generic type for validators

class HTASchemaContract:
    """
    Defines the structural contract for HTA trees without templating content.
    This ensures data integrity, performance, and flexibility while preserving uniqueness.
    """
    
    # Core structural requirements
    required_fields: Dict[str, List[str]] = {
        "tree": ["id", "user_id", "created_at", "top_node_id"],
        "node": ["id", "tree_id", "user_id", "title", "description", "status"]
    }
    
    # Field validators ensure consistency without dictating content
    validators: Dict[str, Callable[[Any], bool]] = {
        "node.status": lambda s: s in ["pending", "in_progress", "completed", "deferred", "cancelled"],
        "node.title": lambda t: isinstance(t, str) and 3 <= len(t) <= 100,
        "node.description": lambda d: d is None or isinstance(d, str),
    }
    
    # Dynamic relationship rules (these guide structure but not specific content)
    relationship_rules: Dict[str, Dict[str, Any]] = {
        "trunk_nodes": {
            "min_count": 3,
            "max_count": 7,
            "required_metadata": ["phase_type", "expected_duration"]
        },
        "micro_actions": {
            "max_per_parent": 5,
            "required_metadata": ["actionability_score", "joy_factor"]
        }
    }
    
    # Context infusion points - places where user context MUST be incorporated
    context_infusion_points: List[str] = [
        "node.title", "node.description", "micro_action.framing",
        "positive_reinforcement", "branch_triggers"
    ]
    
    # Performance guidelines (not templates)
    performance_guidelines: Dict[str, Any] = {
        "max_tree_depth": 5,
        "optimal_branch_factor": 4,
        "denormalization_fields": ["path", "depth", "ancestor_ids"]
    }

    @classmethod
    def validate_model(cls, model_type: str, data: Dict[str, Any]) -> List[str]:
        """
        Validates that a data dictionary contains all required fields for the given model type.
        
        Args:
            model_type: Type of model to validate (e.g., "tree", "node")
            data: Dictionary of data to validate
            
        Returns:
            List of error messages, empty if validation successful
        """
        errors = []
        
        # Check required fields
        if model_type in cls.required_fields:
            for field in cls.required_fields[model_type]:
                if field not in data:
                    errors.append(f"Missing required field: {field}")
        
        # Run validators for relevant fields
        for field_key, validator in cls.validators.items():
            model, field = field_key.split(".")
            if model == model_type and field in data:
                if not validator(data[field]):
                    errors.append(f"Invalid value for {field}")
        
        return errors

    @classmethod
    def validate_relationship(cls, relationship_type: str, parent_data: Dict[str, Any], children_data: List[Dict[str, Any]]) -> List[str]:
        """
        Validates relationships between nodes according to the defined rules.
        
        Args:
            relationship_type: Type of relationship to validate (e.g., "trunk_nodes", "micro_actions")
            parent_data: Dictionary of parent node data
            children_data: List of dictionaries with child node data
            
        Returns:
            List of error messages, empty if validation successful
        """
        errors = []
        
        if relationship_type in cls.relationship_rules:
            rules = cls.relationship_rules[relationship_type]
            
            # Check count constraints
            if "min_count" in rules and len(children_data) < rules["min_count"]:
                errors.append(f"{relationship_type} requires at least {rules['min_count']} children")
                
            if "max_count" in rules and len(children_data) > rules["max_count"]:
                errors.append(f"{relationship_type} allows at most {rules['max_count']} children")
            
            # Check required metadata
            if "required_metadata" in rules:
                for child in children_data:
                    if "hta_metadata" not in child:
                        errors.append(f"Child node missing hta_metadata")
                        continue
                        
                    for meta_field in rules["required_metadata"]:
                        if meta_field not in child.get("hta_metadata", {}):
                            errors.append(f"Child node missing required metadata: {meta_field}")
        
        return errors

    @classmethod
    def check_context_infusion(cls, model_type: str, field: str, value: Any) -> bool:
        """
        Checks if a field has been properly infused with context.
        This is a heuristic check that looks for signs of personalization.
        
        Args:
            model_type: Type of model to check (e.g., "node", "micro_action")
            field: Field name to check
            value: Field value to check
            
        Returns:
            True if the field appears to be properly contextualized
        """
        field_key = f"{model_type}.{field}"
        
        # Skip check if this isn't a context infusion point
        if field_key not in cls.context_infusion_points:
            return True
            
        if not isinstance(value, str):
            return False
            
        # Simple heuristics to detect template-like content
        template_patterns = [
            r"\[.*?\]",  # Text in square brackets often indicates template placeholder
            r"\{.*?\}",  # Text in curly braces often indicates template placeholder
            r"<.*?>",    # Text in angle brackets often indicates template placeholder
        ]
        
        for pattern in template_patterns:
            if re.search(pattern, value):
                return False
                
        # Check for minimal length as a proxy for content richness
        if field == "title" and len(value) < 10:
            return False
            
        if field == "description" and len(value) < 30:
            return False
            
        return True

    @classmethod
    def optimize_tree_structure(cls, nodes_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyzes nodes data and provides optimization recommendations.
        
        Args:
            nodes_data: List of node data dictionaries
            
        Returns:
            Dictionary with optimization recommendations
        """
        # Count nodes by depth
        depths = {}
        for node in nodes_data:
            depth = node.get("depth", 0)
            depths[depth] = depths.get(depth, 0) + 1
        
        # Analyze branching factor
        branching_factors = []
        parent_child_counts = {}
        
        for node in nodes_data:
            parent_id = node.get("parent_id")
            if parent_id:
                parent_child_counts[parent_id] = parent_child_counts.get(parent_id, 0) + 1
        
        for count in parent_child_counts.values():
            branching_factors.append(count)
        
        avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0
        
        # Generate recommendations
        recommendations = {
            "denormalize_paths": len(nodes_data) > 20,
            "use_bulk_operations": len(nodes_data) > 10,
            "avg_branching_factor": avg_branching,
            "max_depth": max(depths.keys()) if depths else 0,
            "potential_optimization_fields": []
        }
        
        # Recommend denormalization fields based on tree shape
        if recommendations["max_depth"] > 3:
            recommendations["potential_optimization_fields"].append("path")
            recommendations["potential_optimization_fields"].append("depth")
            
        if avg_branching > 5:
            recommendations["potential_optimization_fields"].append("leaf_count")
            recommendations["potential_optimization_fields"].append("child_count")
            
        return recommendations
        
    @classmethod
    def validate_task_dependencies(cls, node_dependencies: Dict[UUID, Set[UUID]]) -> List[str]:
        """
        Validates that task dependencies are valid and don't form cycles.
        
        Args:
            node_dependencies: Dictionary mapping node IDs to sets of dependency IDs
            
        Returns:
            List of error messages, empty if validation successful
        """
        errors = []
        
        # Check for self-dependencies
        for node_id, deps in node_dependencies.items():
            if node_id in deps:
                errors.append(f"Node {node_id} cannot depend on itself")
        
        # Check for cycles using DFS
        visited = set()
        temp_visited = set()
        
        def has_cycle(node_id):
            if node_id in temp_visited:
                return True
                
            if node_id in visited:
                return False
                
            temp_visited.add(node_id)
            
            for dep_id in node_dependencies.get(node_id, set()):
                if has_cycle(dep_id):
                    return True
                    
            temp_visited.remove(node_id)
            visited.add(node_id)
            return False
            
        for node_id in node_dependencies:
            if node_id not in visited:
                if has_cycle(node_id):
                    errors.append(f"Dependency cycle detected involving node {node_id}")
        
        return errors

logger.debug("HTA Schema Contract defined.")
