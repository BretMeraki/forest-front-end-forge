"""
Prompt Augmentation Service for Forest OS LLM Services.

This module implements a prompt augmentation service that enhances LLM prompts
with contextual information, examples, and formatting to improve response quality.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field

# Set up logging
logger = logging.getLogger(__name__)

class AugmentationTemplate(BaseModel):
    """A template for prompt augmentation."""
    name: str = Field(..., description="Template name")
    description: str = Field(..., description="Template description")
    system_prompt: str = Field(..., description="System prompt to provide context")
    prompt_format: str = Field(..., description="Format string for the prompt")
    examples: Optional[List[Dict[str, str]]] = Field(default=None, description="Examples of input/output pairs")
    
    def format_prompt(self, **kwargs) -> Dict[str, Any]:
        """Format the prompt with provided parameters."""
        try:
            formatted_prompt = self.prompt_format.format(**kwargs)
            
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add examples if provided
            if self.examples:
                for example in self.examples:
                    if "input" in example and "output" in example:
                        messages.append({"role": "user", "content": example["input"]})
                        messages.append({"role": "assistant", "content": example["output"]})
            
            # Add the actual prompt
            messages.append({"role": "user", "content": formatted_prompt})
            
            return messages
        except KeyError as e:
            logger.error(f"Missing parameter in prompt format: {e}")
            raise ValueError(f"Missing parameter in prompt format: {e}")
        except Exception as e:
            logger.error(f"Error formatting prompt: {e}")
            raise ValueError(f"Error formatting prompt: {e}")


class PromptAugmentationService:
    """
    Service for augmenting and optimizing prompts sent to LLM services.
    
    This service manages a collection of templates for common prompt patterns
    and provides methods to augment prompts with examples, formatting, and
    contextual information to improve response quality.
    """
    
    def __init__(self):
        """Initialize the PromptAugmentationService with default templates."""
        self.templates: Dict[str, AugmentationTemplate] = {}
        self._initialize_default_templates()
        logger.info(f"PromptAugmentationService initialized with {len(self.templates)} templates")
    
    def _initialize_default_templates(self):
        """Initialize the default set of prompt templates."""
        # General JSON generation template
        self.templates["json_generation"] = AugmentationTemplate(
            name="json_generation",
            description="Template for generating structured JSON data",
            system_prompt=(
                "You are a helpful assistant that generates well-structured JSON data. "
                "Always ensure your response is valid JSON and matches the requested schema."
            ),
            prompt_format=(
                "Generate a JSON object with the following structure:\n\n"
                "{schema_description}\n\n"
                "Based on this information: {input_data}\n\n"
                "Additional requirements: {requirements}"
            ),
            examples=[
                {
                    "input": (
                        "Generate a JSON object with the following structure:\n\n"
                        "- name: string, the person's full name\n"
                        "- age: number, the person's age\n"
                        "- skills: array of strings, the person's skills\n\n"
                        "Based on this information: John Smith is 32 years old and knows Python, JavaScript, and UX design.\n\n"
                        "Additional requirements: Format the skills in lowercase."
                    ),
                    "output": '{\n  "name": "John Smith",\n  "age": 32,\n  "skills": ["python", "javascript", "ux design"]\n}'
                }
            ]
        )
        
        # HTA node generation template
        self.templates["hta_node_generation"] = AugmentationTemplate(
            name="hta_node_generation",
            description="Template for generating HTA nodes",
            system_prompt=(
                "You are Forest OS, an AI system designed to help users achieve their goals through "
                "hierarchical task analysis (HTA). Your mission is to create engaging, clear, and "
                "practical task breakdowns that make achieving goals more joyful and manageable. "
                "Focus on making tasks contextually appropriate and extract as much fun as possible "
                "from the process."
            ),
            prompt_format=(
                "The user has the following goal: {goal}\n\n"
                "Current context: {context}\n\n"
                "Generate {count} specific, actionable tasks that would help the user progress toward "
                "this goal. Each task should be engaging, practical, and appropriate to the user's "
                "current situation.\n\n"
                "For each task, provide:\n"
                "1. A brief, motivating title (max 8 words)\n"
                "2. A clear description of what to do (1-2 sentences)\n"
                "3. An estimated time commitment\n"
                "4. A difficulty rating (1-5)"
            ),
            examples=[
                {
                    "input": (
                        "The user has the following goal: Learn to play the guitar\n\n"
                        "Current context: Complete beginner, has purchased a guitar but hasn't started yet. "
                        "Has 30 minutes available each day to practice.\n\n"
                        "Generate 3 specific, actionable tasks that would help the user progress toward "
                        "this goal. Each task should be engaging, practical, and appropriate to the user's "
                        "current situation.\n\n"
                        "For each task, provide:\n"
                        "1. A brief, motivating title (max 8 words)\n"
                        "2. A clear description of what to do (1-2 sentences)\n"
                        "3. An estimated time commitment\n"
                        "4. A difficulty rating (1-5)"
                    ),
                    "output": (
                        "1. **First Strings Adventure**\n"
                        "   Learn to properly hold the guitar and play the open E and A strings. Practice "
                        "switching between these two strings smoothly.\n"
                        "   Time: 20 minutes\n"
                        "   Difficulty: 2/5\n\n"
                        "2. **Finger Gym Warmup**\n"
                        "   Develop finger strength and dexterity by practicing simple finger exercises. "
                        "Place each finger on different frets and press down firmly.\n"
                        "   Time: 15 minutes\n"
                        "   Difficulty: 1/5\n\n"
                        "3. **Name That String Game**\n"
                        "   Create flashcards to memorize the names of all six strings (E-A-D-G-B-E). "
                        "Test yourself by quickly identifying strings when pointing to them randomly.\n"
                        "   Time: 10 minutes\n"
                        "   Difficulty: 1/5"
                    )
                }
            ]
        )
        
        # Reflection generation template
        self.templates["reflection_generation"] = AugmentationTemplate(
            name="reflection_generation",
            description="Template for generating reflections on completed tasks",
            system_prompt=(
                "You are Forest OS, an AI designed to help users reflect meaningfully on their progress "
                "toward goals. Create thoughtful, encouraging reflections that acknowledge effort, "
                "identify insights, and help users connect their actions to their larger aspirations."
            ),
            prompt_format=(
                "The user has completed the following task related to their goal of {goal}:\n\n"
                "Task: {task_title}\n"
                "Description: {task_description}\n\n"
                "They provided this feedback after completion: {user_feedback}\n\n"
                "Generate a brief, thoughtful reflection that:\n"
                "1. Acknowledges their progress\n"
                "2. Highlights a potential insight or learning\n"
                "3. Connects this task to their broader goal\n"
                "4. Provides gentle encouragement for continued progress"
            ),
            examples=[
                {
                    "input": (
                        "The user has completed the following task related to their goal of learning Spanish:\n\n"
                        "Task: Daily 15-minute vocabulary practice\n"
                        "Description: Review 20 new food-related Spanish words using flashcards\n\n"
                        "They provided this feedback after completion: Found it challenging but managed to "
                        "remember 15 out of 20 words. Still struggling with pronunciation.\n\n"
                        "Generate a brief, thoughtful reflection that:\n"
                        "1. Acknowledges their progress\n"
                        "2. Highlights a potential insight or learning\n"
                        "3. Connects this task to their broader goal\n"
                        "4. Provides gentle encouragement for continued progress"
                    ),
                    "output": (
                        "Great job on your vocabulary practice! Remembering 15 out of 20 words is excellent progress, "
                        "especially when learning food vocabulary which will be immediately useful in conversations. "
                        "Your observation about pronunciation challenges shows good self-awareness - perhaps this "
                        "indicates an opportunity to incorporate more listening exercises to train your ear. "
                        "Each vocabulary session like this builds your mental dictionary, bringing you closer to "
                        "comfortable Spanish conversations during your planned trip. Consider recording yourself "
                        "saying these words and comparing to native speakers - small pronunciation adjustments now "
                        "will make a big difference to your confidence later!"
                    )
                }
            ]
        )
    
    def register_template(self, template: AugmentationTemplate):
        """
        Register a new prompt template.
        
        Args:
            template: The template to register
        """
        self.templates[template.name] = template
        logger.info(f"Registered new template: {template.name}")
    
    def get_template(self, template_name: str) -> Optional[AugmentationTemplate]:
        """
        Get a registered template by name.
        
        Args:
            template_name: The name of the template to retrieve
            
        Returns:
            The template if found, None otherwise
        """
        return self.templates.get(template_name)
    
    def format_with_template(self, template_name: str, **kwargs) -> Dict[str, Any]:
        """
        Format a prompt using a registered template.
        
        Args:
            template_name: The name of the template to use
            **kwargs: Parameters to include in the template
            
        Returns:
            The formatted prompt as a dictionary
            
        Raises:
            ValueError: If the template doesn't exist or there's an error formatting
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        return template.format_prompt(**kwargs)
    
    def augment_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Augment a simple text prompt with contextual information.
        
        Args:
            prompt: The original prompt text
            context: Optional context dictionary with additional information
            
        Returns:
            The augmented prompt text
        """
        if not context:
            return prompt
            
        augmented = prompt
        
        # Add relevant context sections
        if context.get("user_goal"):
            augmented = f"Goal: {context['user_goal']}\n\n{augmented}"
            
        if context.get("recent_tasks"):
            tasks = context["recent_tasks"]
            tasks_summary = "\n".join([f"- {task}" for task in tasks])
            augmented = f"{augmented}\n\nRecent tasks:\n{tasks_summary}"
            
        if context.get("system_instruction"):
            augmented = f"{context['system_instruction']}\n\n{augmented}"
            
        # Add reminders to condition the model for better outputs
        augmented = f"{augmented}\n\nRemember to provide concise, practical, and actionable information."
        
        return augmented
    
    def create_chat_messages(
        self, 
        user_prompt: str, 
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Create a list of chat messages from a user prompt and optional context.
        
        Args:
            user_prompt: The user's prompt
            system_instruction: Optional system instruction to include
            history: Optional chat history to include
            
        Returns:
            A list of chat message dictionaries
        """
        messages = []
        
        # Add system instruction if provided
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
            
        # Add chat history if provided
        if history:
            for msg in history:
                if "role" in msg and "content" in msg:
                    messages.append(msg)
        
        # Add the user prompt
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
