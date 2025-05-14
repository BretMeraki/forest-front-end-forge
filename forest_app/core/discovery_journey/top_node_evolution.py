"""
Top Node Evolution for Discovery Journey

This module handles the careful evolution of the top node in the HTA tree,
ensuring it remains anchored to the user's original vision while still
incorporating insights from their journey.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from uuid import UUID

from forest_app.core.snapshot import MemorySnapshot
from forest_app.modules.hta_tree import HTATree, HTANode
from forest_app.integrations.llm import LLMClient
from forest_app.core.circuit_breaker import circuit_protected

logger = logging.getLogger(__name__)

class TopNodeEvolutionManager:
    """
    Manages the careful evolution of the top node in the HTA tree.
    
    This ensures that while the top node evolves based on the user's journey,
    it remains fundamentally anchored to their original vision.
    """
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize the top node evolution manager.
        
        Args:
            llm_client: LLM client for generating evolution recommendations
        """
        self.llm_client = llm_client
        
    async def should_evolve_top_node(
        self, 
        journey_data: Dict[str, Any], 
        original_vision: Dict[str, Any],
        evolution_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Determine if the top node should evolve based on journey data.
        
        The recommendation will be careful, focusing on refinement and clarification
        rather than drastic changes to the user's original vision.
        
        Args:
            journey_data: Accumulated data about the user's journey
            original_vision: The user's original goal and vision
            evolution_history: Previous evolutions of the top node
            
        Returns:
            Recommendation dict with 'should_evolve', 'confidence', and 'rationale'
        """
        # If we've already evolved recently, don't evolve again too soon
        if evolution_history:
            last_evolution = evolution_history[-1]
            last_evolution_time = datetime.fromisoformat(last_evolution.get('timestamp', '2020-01-01'))
            time_since_last = (datetime.now(timezone.utc) - last_evolution_time).days
            
            # Don't evolve more than once a week unless there's really compelling evidence
            if time_since_last < 7 and last_evolution.get('confidence', 0) > 0.7:
                return {
                    'should_evolve': False,
                    'confidence': 0.9,
                    'rationale': 'Recent evolution already applied',
                    'cooling_period': True
                }
        
        # Extract key indicators from journey data
        task_completion_count = journey_data.get('task_completion_count', 0)
        reflection_count = journey_data.get('reflection_count', 0)
        
        # Don't evolve until we have sufficient data
        if task_completion_count < 5 or reflection_count < 3:
            return {
                'should_evolve': False,
                'confidence': 0.85,
                'rationale': 'Insufficient journey data for evolution',
                'insufficient_data': True
            }
            
        # Get evolution recommendation from LLM
        recommendation = await self._get_evolution_recommendation(journey_data, original_vision)
        
        # Add safeguards to prevent excessive evolution
        if recommendation.get('should_evolve', False):
            if len(evolution_history) >= 3:
                # Raise the confidence threshold required after multiple evolutions
                confidence_threshold = 0.8 + (len(evolution_history) * 0.05)
                if recommendation.get('confidence', 0) < confidence_threshold:
                    recommendation['should_evolve'] = False
                    recommendation['rationale'] += " (Confidence threshold not met after multiple evolutions)"
        
        return recommendation
        
    @circuit_protected(fallback_function=lambda *args, **kwargs: {'should_evolve': False, 'confidence': 0.0, 'rationale': 'Circuit breaker triggered'})
    async def _get_evolution_recommendation(
        self, 
        journey_data: Dict[str, Any], 
        original_vision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get a recommendation for top node evolution using LLM.
        
        Args:
            journey_data: Accumulated data about the user's journey
            original_vision: The user's original goal and vision
            
        Returns:
            Recommendation dict with evolution details
        """
        try:
            # Create prompt for evolution recommendation
            prompt = self._create_evolution_recommendation_prompt(journey_data, original_vision)
            
            # Get recommendation from LLM
            response = await self.llm_client.generate_text(
                prompt,
                max_tokens=1000,
                temperature=0.5
            )
            
            # Parse the response
            recommendation = self._parse_evolution_recommendation(response)
            return recommendation
            
        except Exception as e:
            logger.error(f"Failed to get evolution recommendation: {e}")
            return {
                'should_evolve': False,
                'confidence': 0.5,
                'rationale': f"Error during recommendation: {e}",
                'error': True
            }
    
    def _create_evolution_recommendation_prompt(
        self, 
        journey_data: Dict[str, Any], 
        original_vision: Dict[str, Any]
    ) -> str:
        """
        Create a prompt for top node evolution recommendation.
        
        Args:
            journey_data: Accumulated data about the user's journey
            original_vision: The user's original goal and vision
            
        Returns:
            Prompt string
        """
        # Extract key data
        original_goal = original_vision.get('goal', 'Unknown goal')
        original_context = original_vision.get('context', 'No additional context')
        
        # Format journey insights
        task_completions = journey_data.get('task_completions', [])
        reflections = journey_data.get('reflections', [])
        patterns = journey_data.get('identified_patterns', [])
        
        # Create the prompt
        prompt = f"""You are helping decide if a user's top goal node in a Hierarchical Task Analysis (HTA) should evolve based on their journey data.

IMPORTANT: The goal should remain SEMI-STATIC, meaning it should stay fundamentally true to the user's original vision while making measured refinements based on their journey.

USER'S ORIGINAL VISION:
Goal: {original_goal}
Context: {original_context}

JOURNEY DATA:
Task Completions: {len(task_completions)}
Reflections: {len(reflections)}
Identified Patterns: {len(patterns)}

TASKS COMPLETED:
{self._format_list_for_prompt(task_completions[:5])}

USER REFLECTIONS:
{self._format_list_for_prompt(reflections[:5])}

IDENTIFIED PATTERNS:
{self._format_list_for_prompt(patterns)}

YOUR TASK: Determine if the top goal node should evolve while remaining fundamentally true to the original vision. Respond with a JSON object with the following fields:
1. "should_evolve": (boolean) Whether the top node should evolve
2. "confidence": (float between 0-1) Your confidence in this recommendation
3. "rationale": (string) Brief explanation of your reasoning
4. "proposed_refinement": (only if should_evolve is true) A careful refinement that stays true to the original vision
5. "original_essence_preserved": (boolean) Whether the proposed refinement preserves the essence of the original vision

VERY IMPORTANT: Do not suggest drastic changes. Refinements should clarify the original vision, not replace it.

Your response should be ONLY a valid JSON object, no other text."""

        return prompt
    
    def _parse_evolution_recommendation(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into a structured recommendation.
        
        Args:
            response: LLM response text
            
        Returns:
            Parsed recommendation dict
        """
        try:
            # Extract JSON from response
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.endswith('```'):
                response = response[:-3]
            response = response.strip()
            
            # Parse JSON
            recommendation = json.loads(response)
            
            # Ensure all required fields are present
            required_fields = ['should_evolve', 'confidence', 'rationale']
            for field in required_fields:
                if field not in recommendation:
                    recommendation[field] = False if field == 'should_evolve' else (0.0 if field == 'confidence' else 'Missing field')
                    
            # If should_evolve is true, ensure we have a proposed refinement
            if recommendation.get('should_evolve', False) and 'proposed_refinement' not in recommendation:
                recommendation['should_evolve'] = False
                recommendation['rationale'] += " (No proposed refinement provided)"
                
            # Ensure original essence is preserved
            if recommendation.get('should_evolve', False) and not recommendation.get('original_essence_preserved', False):
                recommendation['should_evolve'] = False
                recommendation['rationale'] += " (Original essence not preserved)"
                
            return recommendation
            
        except Exception as e:
            logger.error(f"Failed to parse evolution recommendation: {e}")
            return {
                'should_evolve': False,
                'confidence': 0.0,
                'rationale': f"Error parsing recommendation: {e}",
                'error': True
            }
    
    def _format_list_for_prompt(self, items: List[Dict[str, Any]]) -> str:
        """
        Format a list of items for prompt inclusion.
        
        Args:
            items: List of dictionaries to format
            
        Returns:
            Formatted string
        """
        if not items:
            return "None available"
            
        result = []
        for i, item in enumerate(items[:5], 1):  # Limit to 5 items max
            content = item.get('content', item.get('description', 'No content'))
            timestamp = item.get('timestamp', 'Unknown time')
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            result.append(f"{i}. {content} ({timestamp})")
            
        return "\n".join(result)
        
    async def apply_evolution_to_top_node(
        self, 
        tree: HTATree, 
        recommendation: Dict[str, Any],
        user_id: str
    ) -> HTATree:
        """
        Apply the recommended evolution to the top node of the HTA tree.
        
        This carefully refines the top node while preserving its core essence.
        
        Args:
            tree: The current HTA tree
            recommendation: Evolution recommendation
            user_id: User identifier
            
        Returns:
            Updated HTA tree
        """
        if not recommendation.get('should_evolve', False) or not recommendation.get('proposed_refinement'):
            return tree
            
        try:
            # Get the top node
            if not tree.root_id or tree.root_id not in tree.nodes:
                logger.error(f"Tree has no valid root node for user {user_id}")
                return tree
                
            root_node = tree.nodes[tree.root_id]
            
            # Store the original values before evolution
            original_title = root_node.title
            original_description = root_node.description
            
            # Apply the refinement
            refinement = recommendation.get('proposed_refinement', {})
            refined_title = refinement.get('title', original_title)
            refined_description = refinement.get('description', original_description)
            
            # Update the node with careful evolution
            root_node.title = refined_title
            root_node.description = refined_description
            
            # Add evolution metadata
            if not root_node.metadata:
                root_node.metadata = {}
                
            if 'evolution_history' not in root_node.metadata:
                root_node.metadata['evolution_history'] = []
                
            # Record this evolution
            evolution_record = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'original_title': original_title,
                'original_description': original_description,
                'new_title': refined_title,
                'new_description': refined_description,
                'confidence': recommendation.get('confidence', 0.0),
                'rationale': recommendation.get('rationale', 'No rationale provided')
            }
            
            root_node.metadata['evolution_history'].append(evolution_record)
            root_node.metadata['last_evolution'] = evolution_record['timestamp']
            
            # Update the tree
            tree.nodes[tree.root_id] = root_node
            
            logger.info(f"Successfully evolved top node for user {user_id}")
            return tree
            
        except Exception as e:
            logger.error(f"Failed to apply evolution to top node for user {user_id}: {e}")
            return tree
            
    async def get_journey_data_for_evolution(
        self, 
        snapshot: MemorySnapshot,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Extract relevant journey data for top node evolution.
        
        Args:
            snapshot: User's memory snapshot
            user_id: User identifier
            
        Returns:
            Journey data relevant for evolution
        """
        journey_data = {
            'task_completions': [],
            'reflections': [],
            'identified_patterns': [],
            'task_completion_count': 0,
            'reflection_count': 0
        }
        
        try:
            # Extract task completions
            if 'task_footprints' in snapshot.component_state:
                footprints = snapshot.component_state['task_footprints']
                if isinstance(footprints, dict) and 'completed_tasks' in footprints:
                    completed_tasks = footprints['completed_tasks']
                    if isinstance(completed_tasks, list):
                        journey_data['task_completions'] = completed_tasks
                        journey_data['task_completion_count'] = len(completed_tasks)
            
            # Extract reflections
            if 'reflection_logs' in snapshot.component_state:
                reflection_logs = snapshot.component_state['reflection_logs']
                if isinstance(reflection_logs, list):
                    journey_data['reflections'] = reflection_logs
                    journey_data['reflection_count'] = len(reflection_logs)
            
            # Extract patterns
            if 'discovery_journey' in snapshot.component_state:
                discovery_data = snapshot.component_state['discovery_journey']
                if isinstance(discovery_data, dict) and 'identified_patterns' in discovery_data:
                    patterns = discovery_data['identified_patterns']
                    if isinstance(patterns, list):
                        journey_data['identified_patterns'] = patterns
            
            return journey_data
            
        except Exception as e:
            logger.error(f"Failed to extract journey data for user {user_id}: {e}")
            return journey_data
            
    def get_original_vision(self, snapshot: MemorySnapshot) -> Dict[str, Any]:
        """
        Extract the user's original vision from the snapshot.
        
        Args:
            snapshot: User's memory snapshot
            
        Returns:
            Original vision data
        """
        vision = {
            'goal': 'Unknown goal',
            'context': 'No additional context'
        }
        
        try:
            # Try to get the original goal from the component state
            if 'raw_goal_description' in snapshot.component_state:
                vision['goal'] = snapshot.component_state['raw_goal_description']
                
            # Try to get the original context from the component state
            if 'raw_context_reflection' in snapshot.component_state:
                vision['context'] = snapshot.component_state['raw_context_reflection']
                
            # If not found, try to extract from seed manager data
            if vision['goal'] == 'Unknown goal' and 'seed_manager' in snapshot.component_state:
                seed_manager = snapshot.component_state['seed_manager']
                if isinstance(seed_manager, dict) and 'seeds' in seed_manager:
                    seeds = seed_manager['seeds']
                    if isinstance(seeds, dict) and seeds:
                        # Get the first seed
                        first_seed = list(seeds.values())[0]
                        if isinstance(first_seed, dict):
                            if 'seed_name' in first_seed:
                                vision['goal'] = first_seed['seed_name']
                            if 'description' in first_seed:
                                vision['context'] = first_seed['description']
            
            return vision
            
        except Exception as e:
            logger.error(f"Failed to extract original vision: {e}")
            return vision
