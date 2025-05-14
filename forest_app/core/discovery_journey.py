"""
Discovery Journey Module for Forest App

This module is specifically designed to facilitate the journey from abstract initial
goals to concrete, focused needs and actions. It enables users to start with vague
intentions and naturally discover what they truly need through interaction and reflection.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime, timezone
import json
import uuid

from forest_app.core.snapshot import MemorySnapshot
from forest_app.modules.hta_tree import HTATree, HTANode
from forest_app.core.services.enhanced_hta_service import EnhancedHTAService
from forest_app.core.event_bus import EventBus, EventType, EventData
from forest_app.core.circuit_breaker import circuit_protected
from forest_app.core.cache_service import cacheable
from forest_app.integrations.llm import LLMClient

logger = logging.getLogger(__name__)

class DiscoveryPattern:
    """Models a discovered pattern in the user's journey."""
    
    def __init__(
        self,
        pattern_id: str,
        name: str,
        description: str,
        confidence: float,
        evidence: List[Dict[str, Any]],
        discovered_at: str,
        category: str = "general"
    ):
        """
        Initialize a discovery pattern.
        
        Args:
            pattern_id: Unique identifier for the pattern
            name: Short name for the pattern
            description: Detailed description of the pattern
            confidence: Confidence level (0.0 to 1.0)
            evidence: List of evidence points supporting this pattern
            discovered_at: Timestamp when the pattern was discovered
            category: Category of the pattern (interests, values, needs, etc.)
        """
        self.pattern_id = pattern_id
        self.name = name
        self.description = description
        self.confidence = confidence
        self.evidence = evidence
        self.discovered_at = discovered_at
        self.category = category
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DiscoveryPattern':
        """Create a pattern from a dictionary."""
        return cls(
            pattern_id=data.get('pattern_id', str(uuid.uuid4())),
            name=data.get('name', ''),
            description=data.get('description', ''),
            confidence=data.get('confidence', 0.0),
            evidence=data.get('evidence', []),
            discovered_at=data.get('discovered_at', datetime.now(timezone.utc).isoformat()),
            category=data.get('category', 'general')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pattern_id': self.pattern_id,
            'name': self.name,
            'description': self.description,
            'confidence': self.confidence,
            'evidence': self.evidence,
            'discovered_at': self.discovered_at,
            'category': self.category
        }

class DiscoveryJourneyService:
    """
    Service for managing the user's journey from abstract to concrete goals.
    
    This service builds on the EnhancedHTAService to specifically support the
    discovery journey use case, where users begin with abstract goals and
    gradually discover their true needs through interaction and reflection.
    """
    
    def __init__(
        self,
        hta_service: EnhancedHTAService,
        llm_client: LLMClient,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the discovery journey service.
        
        Args:
            hta_service: Enhanced HTA service for tree operations
            llm_client: LLM client for pattern discovery
            event_bus: Optional event bus for event-driven architecture
        """
        self.hta_service = hta_service
        self.llm_client = llm_client
        self.event_bus = event_bus or EventBus.get_instance()
        
        # Register event listeners
        self._register_event_listeners()
        
        logger.info("DiscoveryJourneyService initialized")
    
    def _register_event_listeners(self):
        """Register event listeners for the discovery journey."""
        # Listen for reflections that might indicate emergent patterns
        self.event_bus.subscribe(EventType.REFLECTION_ADDED, self._handle_reflection_event)
        
        # Listen for task completions to analyze for patterns
        self.event_bus.subscribe(EventType.TASK_COMPLETED, self._handle_task_completed_event)
        
        # Listen for mood recordings to track emotional patterns
        self.event_bus.subscribe(EventType.MOOD_RECORDED, self._handle_mood_event)
        
        logger.debug("Registered event listeners for DiscoveryJourneyService")
    
    async def _handle_reflection_event(self, event: EventData):
        """
        Handle reflection events to identify emergent patterns.
        
        This is triggered when a user adds a reflection, which is a prime
        opportunity to discover their true needs and interests.
        """
        user_id = event.user_id
        if not user_id:
            return
        
        # Schedule background analysis if the reflection seems meaningful
        reflection_content = event.payload.get('content', '')
        if len(reflection_content) > 50:  # Only analyze substantial reflections
            await self._schedule_pattern_analysis(
                user_id=user_id,
                context_type='reflection',
                content=reflection_content,
                metadata=event.metadata
            )
    
    async def _handle_task_completed_event(self, event: EventData):
        """
        Handle task completion events to track patterns in engagement.
        
        This tracks which types of tasks the user engages with most,
        which can reveal their true interests and needs.
        """
        user_id = event.user_id
        if not user_id:
            return
        
        # Extract relevant data
        task_id = event.payload.get('task_id')
        task_title = event.payload.get('title', '')
        has_reflection = event.payload.get('has_reflection', False)
        
        # If task has reflection, it's more meaningful
        if has_reflection and task_id:
            await self._add_engagement_signal(
                user_id=user_id,
                task_id=task_id,
                signal_type='completion_with_reflection',
                signal_strength=0.8,  # Higher weight for tasks with reflection
                metadata={
                    'task_title': task_title,
                    'timestamp': event.payload.get('timestamp', datetime.now(timezone.utc).isoformat())
                }
            )
    
    async def _handle_mood_event(self, event: EventData):
        """
        Handle mood recording events to track emotional responses.
        
        Emotional responses are key indicators of what truly matters to the user,
        and can reveal patterns not evident in explicit content.
        """
        user_id = event.user_id
        if not user_id:
            return
        
        # Extract mood data
        mood = event.payload.get('mood', '')
        context = event.payload.get('context', '')
        
        if mood and context:
            # Add to emotional pattern tracking
            await self._add_emotional_data_point(
                user_id=user_id,
                mood=mood,
                context=context,
                timestamp=event.payload.get('timestamp', datetime.now(timezone.utc).isoformat())
            )
    
    async def _schedule_pattern_analysis(
        self,
        user_id: str,
        context_type: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Schedule a background pattern analysis task.
        
        Args:
            user_id: User identifier
            context_type: Type of context (reflection, task, etc.)
            content: Content to analyze
            metadata: Additional metadata for context
        """
        # Use task queue for background processing
        await self.hta_service.task_queue.enqueue(
            self._analyze_for_emergent_patterns,
            user_id, context_type, content, metadata,
            priority=3,  # Medium priority
            metadata={
                "type": "discovery_analysis",
                "user_id": user_id,
                "context_type": context_type
            }
        )
    
    async def _analyze_for_emergent_patterns(
        self,
        user_id: str,
        context_type: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> Optional[List[DiscoveryPattern]]:
        """
        Analyze content for emergent patterns in the user's journey.
        
        This deep analysis looks for underlying patterns in the user's reflections,
        task completions, and emotional responses to discover their true needs.
        
        Args:
            user_id: User identifier
            context_type: Type of context being analyzed
            content: Content to analyze
            metadata: Additional context metadata
            
        Returns:
            List of discovered patterns or None if analysis failed
        """
        try:
            # Get historical context for the user
            historical_data = await self._get_user_journey_data(user_id)
            
            # Prepare analysis prompt
            prompt = self._create_pattern_discovery_prompt(
                context_type=context_type,
                content=content,
                historical_data=historical_data
            )
            
            # Request analysis from LLM
            analysis_response = await self.llm_client.generate(prompt)
            
            # Parse the response into pattern objects
            patterns = self._parse_pattern_analysis(analysis_response)
            
            # If patterns discovered, store them
            if patterns:
                await self._store_discovered_patterns(user_id, patterns)
                
                # Publish event for discovered patterns
                for pattern in patterns:
                    await self.event_bus.publish({
                        "event_type": EventType.INSIGHT_DISCOVERED,
                        "user_id": user_id,
                        "payload": {
                            "pattern_name": pattern.name,
                            "pattern_description": pattern.description,
                            "confidence": pattern.confidence,
                            "category": pattern.category,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing for emergent patterns: {e}")
            return None
    
    def _create_pattern_discovery_prompt(
        self,
        context_type: str,
        content: str,
        historical_data: Dict[str, Any]
    ) -> str:
        """
        Create a prompt for pattern discovery analysis.
        
        Args:
            context_type: Type of context being analyzed
            content: Content to analyze
            historical_data: Historical user journey data
            
        Returns:
            Prompt string for LLM analysis
        """
        # Extract relevant historical data
        previous_reflections = historical_data.get('reflections', [])
        previous_patterns = historical_data.get('patterns', [])
        emotional_data = historical_data.get('emotional_journey', [])
        
        # Create prompt with context
        prompt = f"""
        As a guide helping someone discover their true needs and interests, analyze this new information:
        
        NEW {context_type.upper()}:
        {content}
        
        PREVIOUS REFLECTIONS (most recent first):
        {self._format_list_for_prompt(previous_reflections[-5:], include_timestamps=True)}
        
        EMOTIONAL JOURNEY (most recent first):
        {self._format_list_for_prompt(emotional_data[-5:], include_timestamps=True)}
        
        PREVIOUSLY IDENTIFIED PATTERNS:
        {self._format_list_for_prompt(previous_patterns, include_confidence=True)}
        
        Based on all this information, identify any emerging patterns, themes, or latent needs that might not be explicitly stated.
        Focus especially on what seems to consistently engage the person emotionally or intellectually.
        
        For each pattern you identify, provide:
        1. A short, descriptive name for the pattern
        2. A detailed description of the pattern
        3. A confidence score (0.0 to 1.0) in this pattern
        4. The category of the pattern (interests, values, needs, etc.)
        5. Evidence from the reflections and emotional journey that supports this pattern
        
        Format your response as JSON with this structure:
        {
          "patterns": [
            {
              "name": "Pattern name",
              "description": "Detailed description",
              "confidence": 0.8,
              "category": "interests",
              "evidence": [
                {"source": "reflection", "content": "Evidence text"}
              ]
            }
          ]
        }
        
        If no clear patterns emerge yet, respond with an empty patterns array.
        """
        
        return prompt
    
    def _format_list_for_prompt(
        self,
        items: List[Dict[str, Any]],
        include_timestamps: bool = False,
        include_confidence: bool = False
    ) -> str:
        """Format a list of items for inclusion in a prompt."""
        if not items:
            return "None available yet."
            
        formatted = []
        for item in items:
            line = f"- {item.get('name') or item.get('content') or item.get('description', 'Item')}"
            
            if include_timestamps and 'timestamp' in item:
                line += f" (at {item['timestamp']})"
                
            if include_confidence and 'confidence' in item:
                line += f" (confidence: {item['confidence']})"
                
            formatted.append(line)
            
        return "\n".join(formatted)
    
    def _parse_pattern_analysis(self, analysis_text: str) -> List[DiscoveryPattern]:
        """
        Parse LLM response into pattern objects.
        
        Args:
            analysis_text: Raw LLM response text
            
        Returns:
            List of DiscoveryPattern objects
        """
        patterns = []
        
        try:
            # Extract JSON from response
            start_idx = analysis_text.find('{')
            end_idx = analysis_text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = analysis_text[start_idx:end_idx]
                data = json.loads(json_str)
                
                # Parse patterns
                for pattern_data in data.get('patterns', []):
                    pattern = DiscoveryPattern(
                        pattern_id=str(uuid.uuid4()),
                        name=pattern_data.get('name', ''),
                        description=pattern_data.get('description', ''),
                        confidence=pattern_data.get('confidence', 0.0),
                        evidence=pattern_data.get('evidence', []),
                        discovered_at=datetime.now(timezone.utc).isoformat(),
                        category=pattern_data.get('category', 'general')
                    )
                    patterns.append(pattern)
        except Exception as e:
            logger.error(f"Error parsing pattern analysis: {e}")
        
        return patterns
    
    async def _store_discovered_patterns(self, user_id: str, patterns: List[DiscoveryPattern]) -> bool:
        """
        Store discovered patterns in the user's journey data.
        
        Args:
            user_id: User identifier
            patterns: List of discovered patterns
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert patterns to dictionaries
            pattern_dicts = [pattern.to_dict() for pattern in patterns]
            
            # Store in semantic memory
            for pattern in patterns:
                await self.hta_service.semantic_memory_manager.store_memory(
                    event_type="discovery_pattern",
                    content=f"Discovered pattern: {pattern.name}",
                    metadata={
                        "pattern": pattern.to_dict(),
                        "user_id": user_id,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    },
                    importance=0.8  # Patterns are highly important
                )
            
            return True
        except Exception as e:
            logger.error(f"Error storing discovered patterns: {e}")
            return False
    
    async def _add_engagement_signal(
        self,
        user_id: str,
        task_id: str,
        signal_type: str,
        signal_strength: float,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Add an engagement signal to track user interest patterns.
        
        Args:
            user_id: User identifier
            task_id: Task identifier
            signal_type: Type of engagement signal
            signal_strength: Strength of the signal (0.0 to 1.0)
            metadata: Additional signal metadata
        """
        try:
            # Store in semantic memory
            await self.hta_service.semantic_memory_manager.store_memory(
                event_type="engagement_signal",
                content=f"Engagement signal: {signal_type} (strength: {signal_strength})",
                metadata={
                    "user_id": user_id,
                    "task_id": task_id,
                    "signal_type": signal_type,
                    "signal_strength": signal_strength,
                    "timestamp": metadata.get('timestamp', datetime.now(timezone.utc).isoformat()),
                    **metadata
                },
                importance=signal_strength  # Importance proportional to signal strength
            )
        except Exception as e:
            logger.error(f"Error adding engagement signal: {e}")
    
    async def _add_emotional_data_point(
        self,
        user_id: str,
        mood: str,
        context: str,
        timestamp: str
    ) -> None:
        """
        Add an emotional data point to track emotional patterns.
        
        Args:
            user_id: User identifier
            mood: Recorded mood
            context: Context in which mood was recorded
            timestamp: When the mood was recorded
        """
        try:
            # Determine importance based on emotional intensity
            importance = 0.5  # Default importance
            intense_moods = ['excited', 'inspired', 'joyful', 'frustrated', 'anxious', 'sad']
            if any(intense_mood in mood.lower() for intense_mood in intense_moods):
                importance = 0.7  # Higher importance for intense emotions
            
            # Store in semantic memory
            await self.hta_service.semantic_memory_manager.store_memory(
                event_type="emotional_data",
                content=f"Emotional data: {mood} while {context}",
                metadata={
                    "user_id": user_id,
                    "mood": mood,
                    "context": context,
                    "timestamp": timestamp
                },
                importance=importance
            )
        except Exception as e:
            logger.error(f"Error adding emotional data point: {e}")
    
    @cacheable(key_pattern="user:{0}:journey_data", ttl=300)  # 5 minutes
    async def _get_user_journey_data(self, user_id: str) -> Dict[str, Any]:
        """
        Get comprehensive user journey data for pattern analysis.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with journey data
        """
        try:
            # Query various types of memories
            reflections = await self.hta_service.semantic_memory_manager.query_memories(
                query=f"Get reflections for user {user_id}",
                k=20,
                event_types=["reflection", "task_completion"]
            )
            
            patterns = await self.hta_service.semantic_memory_manager.query_memories(
                query=f"Get discovered patterns for user {user_id}",
                k=10,
                event_types=["discovery_pattern"]
            )
            
            emotional_journey = await self.hta_service.semantic_memory_manager.query_memories(
                query=f"Get emotional data for user {user_id}",
                k=20,
                event_types=["emotional_data", "mood_recorded"]
            )
            
            engagement_signals = await self.hta_service.semantic_memory_manager.query_memories(
                query=f"Get engagement signals for user {user_id}",
                k=20,
                event_types=["engagement_signal"]
            )
            
            # Extract pattern objects from memories
            extracted_patterns = []
            for pattern_memory in patterns:
                if (
                    isinstance(pattern_memory, dict) and
                    pattern_memory.get('metadata') and
                    pattern_memory['metadata'].get('pattern')
                ):
                    pattern_data = pattern_memory['metadata']['pattern']
                    extracted_patterns.append(pattern_data)
            
            # Compile comprehensive journey data
            journey_data = {
                'reflections': reflections,
                'patterns': extracted_patterns,
                'emotional_journey': emotional_journey,
                'engagement_signals': engagement_signals
            }
            
            return journey_data
            
        except Exception as e:
            logger.error(f"Error getting user journey data: {e}")
            return {
                'reflections': [],
                'patterns': [],
                'emotional_journey': [],
                'engagement_signals': []
            }
    
    async def generate_exploratory_tasks(
        self,
        user_id: str,
        tree: HTATree,
        parent_node_id: str,
        count: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Generate exploratory tasks designed to help users discover their true needs.
        
        This is used when the user's goal is still abstract, to create tasks that
        will help them discover what they truly need or want.
        
        Args:
            user_id: User identifier
            tree: Current HTA tree
            parent_node_id: Parent node under which to add exploratory tasks
            count: Number of tasks to generate
            
        Returns:
            List of task data dictionaries
        """
        try:
            # Get user journey data including discovered patterns
            journey_data = await self._get_user_journey_data(user_id)
            
            # Find parent node
            parent_node = tree.find_node_by_id(parent_node_id)
            if not parent_node:
                logger.error(f"Parent node {parent_node_id} not found in tree")
                return []
            
            # Extract key information
            top_patterns = journey_data.get('patterns', [])[:5]
            recent_reflections = journey_data.get('reflections', [])[:5]
            emotional_data = journey_data.get('emotional_journey', [])[:5]
            
            # Create prompt for LLM
            prompt = f"""
            Generate {count} exploratory tasks to help a user discover their true needs and interests.
            
            Current goal: {parent_node.title}
            
            Emerging patterns in their journey:
            {self._format_list_for_prompt(top_patterns, include_confidence=True)}
            
            Recent reflections:
            {self._format_list_for_prompt(recent_reflections, include_timestamps=True)}
            
            Emotional responses:
            {self._format_list_for_prompt(emotional_data, include_timestamps=True)}
            
            Design tasks that:
            1. Feel engaging and inviting, not clinical or analytical
            2. Are specific enough to take action on, not vague
            3. Help reveal underlying needs, values and interests
            4. Encourage reflection and self-discovery
            5. Build naturally on what seems to resonate with them so far
            
            Format each task as a JSON object with:
            - title: A concise, action-oriented title
            - description: A detailed, inviting description
            - duration: Estimated time in minutes (5-30)
            - reflection_prompt: A thoughtful question to reflect on after completing
            
            Return a JSON array of {count} task objects.
            """
            
            # Generate tasks with LLM
            response = await self.llm_client.generate(prompt)
            
            # Parse response into task objects
            tasks = self._parse_task_generation(response, count)
            
            # Add discovery metadata to tasks
            for task in tasks:
                task['is_exploratory'] = True
                task['discovery_phase'] = 'exploration'
                task['generated_at'] = datetime.now(timezone.utc).isoformat()
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error generating exploratory tasks: {e}")
            return []
    
    def _parse_task_generation(self, response: str, expected_count: int) -> List[Dict[str, Any]]:
        """Parse LLM response into task dictionaries."""
        tasks = []
        
        try:
            # Extract JSON from response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                tasks = json.loads(json_str)
                
                # Validate tasks
                valid_tasks = []
                for task in tasks:
                    if isinstance(task, dict) and 'title' in task and 'description' in task:
                        valid_tasks.append(task)
                
                tasks = valid_tasks[:expected_count]  # Limit to expected count
            
        except Exception as e:
            logger.error(f"Error parsing task generation response: {e}")
        
        # If parsing failed, create fallback tasks
        if not tasks:
            tasks = [
                {
                    "title": "Reflect on a meaningful moment",
                    "description": "Take a few minutes to recall a recent moment when you felt genuinely engaged or fulfilled. What elements of that experience stood out to you?",
                    "duration": 10,
                    "reflection_prompt": "What does this moment reveal about what matters to you?"
                }
            ] * expected_count
        
        return tasks
    
    async def evolve_focus_based_on_patterns(
        self,
        user_id: str,
        tree: HTATree
    ) -> Optional[Dict[str, Any]]:
        """
        Evolve the user's focus based on discovered patterns.
        
        This is called periodically to check if the abstract goal should be
        refined into something more concrete, based on what we've learned
        about the user's true needs and interests.
        
        Args:
            user_id: User identifier
            tree: Current HTA tree
            
        Returns:
            Dictionary with evolution recommendations or None if not ready
        """
        try:
            # Get journey data including discovered patterns
            journey_data = await self._get_user_journey_data(user_id)
            
            # Only proceed if we have substantial patterns
            patterns = journey_data.get('patterns', [])
            if not patterns or len(patterns) < 2:
                return None
            
            # Check confidence levels - only evolve if we have high confidence
            high_confidence_patterns = [p for p in patterns if p.get('confidence', 0) > 0.7]
            if not high_confidence_patterns or len(high_confidence_patterns) < 2:
                return None
            
            # Create prompt for LLM
            prompt = f"""
            Based on the following patterns discovered in a user's journey, determine if it's time
            to evolve their abstract goal into something more concrete and specific.
            
            Current goal: {tree.root.title if tree and tree.root else "Unknown"}
            
            Discovered patterns (with confidence):
            {self._format_list_for_prompt(patterns, include_confidence=True)}
            
            If there's enough confidence and coherence in these patterns to suggest a more concrete focus,
            recommend an evolution of their goal. This should feel like a natural clarification
            rather than a dramatic pivot.
            
            Return a JSON object with:
            - ready_to_evolve: true/false
            - new_focus: The proposed new, more concrete focus (if ready)
            - rationale: Explanation of why this evolution makes sense
            - key_patterns: List of 2-3 key pattern names that inform this evolution
            
            If it's not yet time to evolve (insufficient coherence or confidence),
            set ready_to_evolve to false and explain why in the rationale.
            """
            
            # Generate evolution recommendation
            response = await self.llm_client.generate(prompt)
            
            # Parse recommendation
            recommendation = self._parse_evolution_recommendation(response)
            
            # If ready to evolve, publish event
            if recommendation and recommendation.get('ready_to_evolve', False):
                await self.event_bus.publish({
                    "event_type": EventType.USER_GOAL_UPDATED,
                    "user_id": user_id,
                    "payload": {
                        "old_focus": tree.root.title if tree and tree.root else "Unknown",
                        "new_focus": recommendation.get('new_focus', ''),
                        "rationale": recommendation.get('rationale', ''),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                })
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error evolving focus based on patterns: {e}")
            return None
    
    def _parse_evolution_recommendation(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM response into evolution recommendation."""
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                recommendation = json.loads(json_str)
                
                # Validate required fields
                if 'ready_to_evolve' in recommendation and 'rationale' in recommendation:
                    # If ready to evolve, ensure new focus is present
                    if recommendation.get('ready_to_evolve', False) and 'new_focus' not in recommendation:
                        recommendation['ready_to_evolve'] = False
                        recommendation['rationale'] = "Missing new focus in recommendation"
                    
                    return recommendation
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing evolution recommendation: {e}")
            return None
