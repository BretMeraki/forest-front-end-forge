"""Event and notification publishing for Enhanced HTA Service.

This module provides functionality for:
- Handling task completion events
- Managing tree evolution events
- Processing memory storage events
- Publishing notifications to maintain data consistency

These components enable a reactive architecture that responds to user
actions and keeps all parts of the system synchronized.
"""

import logging
import asyncio
from typing import Any, Dict, List, Optional, Union, Callable, Awaitable
from uuid import UUID
from datetime import datetime, timezone
from enum import Enum

from forest_app.core.event_bus import EventBus, EventType, EventData
from forest_app.core.cache_service import CacheService
from forest_app.core.circuit_breaker import circuit_protected

logger = logging.getLogger(__name__)


class EventManager:
    """Manages event publishing and handling for the Enhanced HTA service.
    
    This component centralizes event handling, providing a consistent way to respond
    to system changes and maintain cache coherence across the application.
    """
    
    def __init__(self, cache_service: Optional[CacheService] = None):
        """Initialize the event manager with required services.
        
        Args:
            cache_service: Optional cache service for invalidation operations
        """
        self.event_bus = EventBus.get_instance()
        self.cache = cache_service or CacheService.get_instance()
        
        # Register event handlers
        self._register_event_handlers()
    
    def _register_event_handlers(self):
        """Register listeners for relevant events."""
        self.event_bus.subscribe(EventType.TASK_COMPLETED, self._handle_task_completed_event)
        self.event_bus.subscribe(EventType.TREE_EVOLVED, self._handle_tree_evolved_event)
        self.event_bus.subscribe(EventType.MEMORY_STORED, self._handle_memory_stored_event)
        logger.debug("Registered event listeners for Enhanced HTA Service")

    async def _handle_task_completed_event(self, event: EventData):
        """Handle task completion events to update caches.
        
        Args:
            event: The event data containing user_id and payload
        """
        user_id = event.user_id
        if not user_id:
            return
        cache_key = f"user:{user_id}:journey"
        await self.cache.delete(cache_key)
        logger.debug(f"Invalidated journey cache for user {user_id} after task completion")

    async def _handle_tree_evolved_event(self, event: EventData):
        """Handle tree evolution events to update caches.
        
        Args:
            event: The event data containing user_id and payload
        """
        user_id = event.user_id
        if not user_id:
            return
        cache_key = f"user:{user_id}:hta_tree"
        await self.cache.delete(cache_key)
        logger.debug(f"Invalidated tree cache for user {user_id} after evolution")

    async def _handle_memory_stored_event(self, event: EventData):
        """Handle memory storage events to update relevant caches.
        
        Args:
            event: The event data containing user_id and payload
        """
        user_id = event.user_id
        if not user_id:
            return
        cache_key = f"user:{user_id}:pattern_analysis"
        await self.cache.delete(cache_key)
        logger.debug(f"Invalidated pattern analysis cache for user {user_id} after new memory")
    
    @circuit_protected(name="event_publishing", failure_threshold=5, recovery_timeout=30)
    async def publish_event(self, event_type: EventType, user_id: UUID, payload: Dict[str, Any], priority: int = 1) -> bool:
        """Publish an event to the event bus.
        
        This method enriches events with timestamp data and handles failures gracefully
        using circuit breaking patterns to prevent cascading failures.
        
        Args:
            event_type: The type of event to publish
            user_id: The UUID of the affected user
            payload: Dictionary containing event data
            priority: Event priority (1-5, with 1 being highest)
            
        Returns:
            Boolean indicating success
        """
        try:
            # Enrich the payload with standard fields
            enriched_payload = {
                **payload,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_id": str(UUID.uuid4()),
                "priority": priority
            }
            
            # Add source information for debugging and auditing
            if "source" not in enriched_payload:
                enriched_payload["source"] = "enhanced_hta_service"
                
            await self.event_bus.publish({
                "event_type": event_type,
                "user_id": str(user_id),
                "payload": enriched_payload
            })
            
            logger.debug(f"Published {event_type} event for user {user_id} with priority {priority}")
            return True
            
        except Exception as e:
            logger.error(f"Error publishing event {event_type}: {e}")
            return False
            
    async def publish_batch_events(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Publish multiple events in a batch operation.
        
        Allows efficient publishing of multiple related events that should be
        processed together, with optional grouping by type.
        
        Args:
            events: List of event dictionaries with event_type, user_id, and payload keys
            
        Returns:
            Dictionary with success and failure counts
        """
        results = {"success": 0, "failure": 0}
        
        try:
            # Group events by type for more efficient processing
            event_groups = {}
            for event in events:
                event_type = event.get("event_type")
                if not event_type:
                    logger.warning("Skipping event with missing event_type")
                    results["failure"] += 1
                    continue
                    
                if event_type not in event_groups:
                    event_groups[event_type] = []
                event_groups[event_type].append(event)
            
            # Process each group concurrently
            tasks = []
            for event_type, group_events in event_groups.items():
                tasks.append(self._publish_event_group(event_type, group_events))
                
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            for group_result in group_results:
                if isinstance(group_result, Exception):
                    # Count exceptions as failures for the whole group
                    results["failure"] += 1
                    logger.error(f"Error in batch event publishing: {group_result}")
                elif isinstance(group_result, dict):
                    results["success"] += group_result.get("success", 0)
                    results["failure"] += group_result.get("failure", 0)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in batch event publishing: {e}")
            results["failure"] = len(events)
            return results
    
    async def _publish_event_group(self, event_type: EventType, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Publish a group of events of the same type.
        
        Args:
            event_type: The common event type for all events in the group
            events: List of event dictionaries
            
        Returns:
            Dictionary with success and failure counts for this group
        """
        results = {"success": 0, "failure": 0}
        
        for event in events:
            try:
                # Ensure events have timestamps
                if "payload" in event and "timestamp" not in event["payload"]:
                    event["payload"]["timestamp"] = datetime.now(timezone.utc).isoformat()
                    
                await self.event_bus.publish(event)
                results["success"] += 1
                
            except Exception as e:
                logger.error(f"Error publishing grouped event {event_type}: {e}")
                results["failure"] += 1
                
        return results
