"""
Event Bus for Forest App

This module implements an event-driven architecture that allows components to
communicate without direct dependencies. This improves modularity, scalability,
and creates a more resilient system while maintaining the intimate, personal
experience for each user.
"""

import asyncio
import logging
import json
import uuid
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union, Awaitable
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

class EventType(str, Enum):
    """Core event types in the system."""
    # Journey events
    TASK_COMPLETED = "task.completed"
    TASK_UPDATED = "task.updated"
    TREE_EVOLVED = "tree.evolved"
    MILESTONE_REACHED = "journey.milestone"
    
    # Emotional/reflection events
    REFLECTION_ADDED = "reflection.added"
    MOOD_RECORDED = "mood.recorded"
    INSIGHT_DISCOVERED = "insight.discovered"
    
    # Memory events
    MEMORY_STORED = "memory.stored"
    MEMORY_RECALLED = "memory.recalled"
    
    # User events
    USER_ONBOARDED = "user.onboarded"
    USER_RETURNED = "user.returned"
    USER_GOAL_UPDATED = "user.goal_updated"
    
    # System events
    LLM_CALL_SUCCEEDED = "system.llm_succeeded"
    LLM_CALL_FAILED = "system.llm_failed"
    DATABASE_OPERATION = "system.database_op"
    SYSTEM_ERROR = "system.error"
    METRICS_RECORDED = "system.metrics"

class EventData(BaseModel):
    """Base model for event data payload."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"
        json_encoders = {
            # Add custom encoders for non-JSON serializable types
            datetime: lambda dt: dt.isoformat(),
            uuid.UUID: lambda id: str(id)
        }
    
    @validator('event_type', pre=True)
    def validate_event_type(cls, v):
        """Validate and convert event_type."""
        if isinstance(v, EventType):
            return v
        if isinstance(v, str):
            try:
                return EventType(v)
            except ValueError:
                pass
        # Allow custom event types
        return v

class EventBus:
    """
    Central event bus for publishing and subscribing to events.
    
    The EventBus enables loose coupling between components by allowing them to
    communicate through events rather than direct method calls. This improves
    modularity, testability, and allows for features like event replay.
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'EventBus':
        """Get the singleton instance of the EventBus."""
        if cls._instance is None:
            cls._instance = EventBus()
        return cls._instance
    
    def __init__(self):
        """Initialize the event bus."""
        # Maps event types to sets of subscribers
        self.subscribers: Dict[str, Set[Callable]] = {}
        # Maps subscriptions to specific event types
        self.subscriber_events: Dict[Callable, Set[str]] = {}
        # For reliable event delivery
        self.event_history: List[EventData] = []
        self.max_history_size = 1000  # Limit history to avoid memory issues
        self.lock = asyncio.Lock()
        
        # Metrics
        self.metrics = {
            "events_published": 0,
            "events_delivered": 0
        }
        
        logger.info("EventBus initialized")
    
    async def publish(self, event: Union[EventData, Dict[str, Any]]) -> str:
        """
        Publish an event to all subscribers.
        
        Args:
            event: The event to publish (EventData or dict that can be converted)
            
        Returns:
            The event ID
        """
        # Convert dict to EventData if needed
        if isinstance(event, dict):
            event = EventData(**event)
        
        # Ensure event_id is set
        if not event.event_id:
            event.event_id = str(uuid.uuid4())
            
        # Get event type as string for subscriber lookup
        event_type = str(event.event_type)
        
        # Store event in history
        async with self.lock:
            self.event_history.append(event)
            # Trim history if needed
            if len(self.event_history) > self.max_history_size:
                self.event_history = self.event_history[-self.max_history_size:]
            self.metrics["events_published"] += 1
        
        # Get subscribers for this event type
        specific_subscribers = self.subscribers.get(event_type, set())
        wildcard_subscribers = self.subscribers.get("*", set())
        all_subscribers = specific_subscribers.union(wildcard_subscribers)
        
        # Notify subscribers
        delivery_tasks = []
        
        for subscriber in all_subscribers:
            # Create task for each subscriber to avoid one blocking others
            delivery_tasks.append(self._deliver_event(subscriber, event))
        
        # Wait for all deliveries to complete
        if delivery_tasks:
            await asyncio.gather(*delivery_tasks, return_exceptions=True)
        
        logger.debug(f"Published event {event.event_id} of type {event_type} to {len(all_subscribers)} subscribers")
        
        return event.event_id
    
    async def _deliver_event(self, subscriber: Callable, event: EventData) -> None:
        """
        Deliver an event to a subscriber with error handling.
        
        Args:
            subscriber: The subscriber callback
            event: The event to deliver
        """
        try:
            if asyncio.iscoroutinefunction(subscriber):
                await subscriber(event)
            else:
                subscriber(event)
            async with self.lock:
                self.metrics["events_delivered"] += 1
        except Exception as e:
            logger.error(f"Error delivering event {event.event_id} to subscriber: {e}")
    
    def subscribe(self, event_type: Union[str, EventType, List[Union[str, EventType]]], 
                 callback: Callable[[EventData], Any]) -> Callable:
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Event type(s) to subscribe to ('*' for all events)
            callback: Function to call when event occurs
            
        Returns:
            Unsubscribe function
        """
        # Convert event_type to list if it's not already
        if not isinstance(event_type, list):
            event_types = [event_type]
        else:
            event_types = event_type
        
        # Convert EventType enums to strings
        event_types = [str(et) for et in event_types]
        
        # Add subscriber to each event type
        for et in event_types:
            if et not in self.subscribers:
                self.subscribers[et] = set()
            self.subscribers[et].add(callback)
            
            # Track events for this subscriber
            if callback not in self.subscriber_events:
                self.subscriber_events[callback] = set()
            self.subscriber_events[callback].add(et)
        
        # Create unsubscribe function
        def unsubscribe():
            self.unsubscribe(callback)
        
        logger.debug(f"Subscribed to event types: {event_types}")
        
        return unsubscribe
    
    def unsubscribe(self, callback: Callable) -> None:
        """
        Unsubscribe a callback from all events.
        
        Args:
            callback: The callback to unsubscribe
        """
        # Get list of event types this callback is subscribed to
        event_types = self.subscriber_events.get(callback, set())
        
        # Remove callback from each event type
        for event_type in event_types:
            if event_type in self.subscribers:
                self.subscribers[event_type].discard(callback)
                # Remove event type entry if no subscribers left
                if not self.subscribers[event_type]:
                    del self.subscribers[event_type]
        
        # Remove callback from tracking
        if callback in self.subscriber_events:
            del self.subscriber_events[callback]
        
        logger.debug(f"Unsubscribed from event types: {event_types}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the event bus.
        
        Returns:
            Dictionary with metrics
        """
        return {
            "subscribers_count": sum(len(subs) for subs in self.subscribers.values()),
            "event_types_count": len(self.subscribers),
            "history_size": len(self.event_history),
            **self.metrics
        }
    
    def get_recent_events(self, 
                         event_type: Optional[Union[str, EventType]] = None,
                         user_id: Optional[str] = None,
                         limit: int = 50) -> List[EventData]:
        """
        Get recent events, optionally filtered.
        
        Args:
            event_type: Optional filter by event type
            user_id: Optional filter by user ID
            limit: Maximum number of events to return
            
        Returns:
            List of events, newest first
        """
        # Convert event_type to string if it's an EventType
        if isinstance(event_type, EventType):
            event_type = str(event_type)
        
        # Start with full history, newest first
        events = list(reversed(self.event_history))
        
        # Apply filters
        if event_type:
            events = [e for e in events if str(e.event_type) == event_type]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        # Apply limit
        return events[:limit]


# Create a decorator for event publishing
def publish_event(event_type: Union[str, EventType], include_result: bool = False):
    """
    Decorator for publishing events before or after function execution.
    
    Args:
        event_type: Type of event to publish
        include_result: Whether to include function result in event payload
        
    Returns:
        Decorated function
    """
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            @asyncio.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Extract user_id from args or kwargs if possible
                user_id = None
                for arg in args:
                    if hasattr(arg, 'user_id'):
                        user_id = arg.user_id
                        break
                if not user_id and 'user_id' in kwargs:
                    user_id = kwargs['user_id']
                
                # Create event bus event
                event_data = {
                    "event_type": event_type,
                    "user_id": user_id,
                    "payload": {
                        "function": func.__name__,
                        "args_summary": f"{len(args)} positional, {len(kwargs)} keyword args"
                    },
                    "metadata": {
                        "source": f"{func.__module__}.{func.__name__}"
                    }
                }
                
                # Get event bus
                event_bus = EventBus.get_instance()
                
                try:
                    # Execute function
                    result = await func(*args, **kwargs)
                    
                    # Include result in payload if requested
                    if include_result:
                        # Try to convert result to JSON-serializable form
                        try:
                            if hasattr(result, 'dict'):
                                # Pydantic model or similar
                                result_dict = result.dict()
                            elif hasattr(result, 'to_dict'):
                                # Object with to_dict method
                                result_dict = result.to_dict()
                            elif isinstance(result, dict):
                                # Already a dict
                                result_dict = result
                            else:
                                # Try to convert to string
                                result_dict = {"value": str(result)}
                                
                            event_data["payload"]["result"] = result_dict
                        except Exception as e:
                            logger.debug(f"Could not include result in event: {e}")
                            event_data["payload"]["result_included"] = False
                    
                    # Add success status
                    event_data["payload"]["success"] = True
                    
                    # Publish event
                    await event_bus.publish(event_data)
                    
                    return result
                    
                except Exception as e:
                    # Add error information to event
                    event_data["payload"]["success"] = False
                    event_data["payload"]["error"] = str(e)
                    event_data["payload"]["error_type"] = type(e).__name__
                    
                    # Publish event
                    await event_bus.publish(event_data)
                    
                    # Re-raise the exception
                    raise
        else:
            # Synchronous function
            def wrapper(*args, **kwargs):
                # We can't do async operations in a sync function,
                # so we'll just log that events would be generated
                logger.info(f"Would publish {event_type} event for {func.__name__} "
                           f"(sync functions can't publish events via event bus)")
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    
    return decorator
