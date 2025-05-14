"""Positive feedback and reinforcement logic for Enhanced HTA Service.

This module provides functionality for:
- Generating positive reinforcement messages when tasks are completed
- Customizing feedback based on user preferences and achievements
- Creating encouragement that aligns with the PRD's core vision

These components help create a joyful, encouraging experience that celebrates
user progress and reinforces the value of their journey.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from datetime import datetime, timezone
import random

from forest_app.core.circuit_breaker import circuit_protected, CircuitBreaker
from forest_app.integrations.llm import LLMClient, LLMError
from forest_app.persistence.models import HTANodeModel

logger = logging.getLogger(__name__)


class ReinforcementManager:
    """Manages positive reinforcement generation for the Enhanced HTA service.
    
    This component generates customized, encouraging feedback when users complete tasks,
    creating a sense of joy and accomplishment throughout their journey.
    """
    
    def __init__(self, llm_client: LLMClient):
        """Initialize the reinforcement manager with LLM capabilities.
        
        Args:
            llm_client: LLM client for generating personalized reinforcement content
        """
        self.llm_client = llm_client
    
    @circuit_protected(name="reinforcement_generation", failure_threshold=3, recovery_timeout=60)
    async def generate_reinforcement(self, node: HTANodeModel, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a positive reinforcement message for completing a node.
        
        This method creates personalized, joyful messages that celebrate user
        progress in alignment with the core vision of making life experiences
        beautiful and meaningful.
        
        Args:
            node: HTANodeModel that was completed
            user_context: Optional additional context about the user
            
        Returns:
            String containing the positive reinforcement message
        """
        try:
            # First check if the node already has a pre-defined reinforcement message
            if hasattr(node, 'internal_task_details') and node.internal_task_details:
                if "positive_reinforcement" in node.internal_task_details:
                    return node.internal_task_details["positive_reinforcement"]
            
            # Use more enthusiastic messages for major milestones
            is_major = hasattr(node, 'is_major_phase') and node.is_major_phase
            
            if user_context and self.llm_client:
                # If we have LLM access and user context, generate a personalized message
                try:
                    # Generate personalized message using the LLM
                    response = await self.llm_client.generate_text(
                        prompt=self._build_reinforcement_prompt(node, user_context, is_major),
                        max_tokens=100,
                        temperature=0.7
                    )
                    if response and hasattr(response, 'text'):
                        return response.text.strip()
                except Exception as e:
                    logger.warning(f"LLM reinforcement generation failed: {e}. Using fallback.")
            
            # Fallback to predefined messages if no LLM or if LLM fails
            if is_major:
                messages = [
                    f"Amazing achievement! You've completed '{node.title}' - a major milestone in your journey.",
                    f"Incredible progress! Completing '{node.title}' is a significant step forward.",
                    f"This is a big deal! '{node.title}' complete - you're making remarkable strides."
                ]
            else:
                messages = [
                    f"Well done on completing '{node.title}'!",
                    f"Great job! You've finished '{node.title}'.",
                    f"Success! '{node.title}' is now complete."
                ]
            return random.choice(messages)
            
        except Exception as e:
            logger.error(f"Error generating positive reinforcement: {e}")
            return self.get_fallback_message()
    
    def _build_reinforcement_prompt(self, node: HTANodeModel, user_context: Dict[str, Any], is_major: bool) -> str:
        """Build a prompt for generating personalized reinforcement messages.
        
        Args:
            node: The completed HTANodeModel
            user_context: Additional context about the user
            is_major: Whether this is a major milestone
            
        Returns:
            Formatted prompt string for the LLM
        """
        prompt = f"""Generate a short, positive reinforcement message for a user who just completed a task.
        
        Task details:
        - Title: {node.title}
        - Description: {node.description if hasattr(node, 'description') else 'No description'}
        - This is {'a major milestone' if is_major else 'a regular task'}
        
        User context:
        """
        
        # Add relevant user context
        if 'user_name' in user_context:
            prompt += f"- User name: {user_context['user_name']}\n"
        if 'goal' in user_context:
            prompt += f"- User goal: {user_context['goal']}\n"
        if 'preferences' in user_context:
            prompt += f"- User preferences: {user_context['preferences']}\n"
            
        prompt += "\nThe message should be encouraging, specific to what they accomplished, and align with our vision: 'Remind the user why being alive is a beautiful and precious experience'.\n"
        prompt += "Keep it concise (1-2 sentences), warm, and authentic."
        
        return prompt
    
    def get_fallback_message(self) -> str:
        """Get a fallback reinforcement message when LLM is unavailable.
        
        This provides graceful degradation when external services are down.
        
        Returns:
            String containing a pre-written positive message
        """
        fallback_messages = [
            "Great job completing this step! You're making progress on your journey.",
            "Another step forward! Keep up the momentum.",
            "Well done! Every completed task brings you closer to your goal.",
            "Excellent work! Your commitment is inspiring.",
            "Congratulations on this accomplishment! Your journey continues to unfold beautifully."
        ]
        return random.choice(fallback_messages)
