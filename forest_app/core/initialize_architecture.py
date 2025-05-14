"""
Architecture Initialization for The Forest

This module initializes all the enhanced architectural components that create
a sanctuary that can support countless users at scale, while maintaining the
intimate, personal experience that makes The Forest special.
"""

import asyncio
import logging
import os
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI
from dependency_injector import containers, providers

# Import enhanced architectural components
from forest_app.core.task_queue import TaskQueue
from forest_app.core.circuit_breaker import CircuitBreakerConfig, CacheBackend
from forest_app.core.event_bus import EventBus, EventType
from forest_app.core.cache_service import CacheService, CacheConfig
from forest_app.core.services.enhanced_hta_service import EnhancedHTAService
from forest_app.core.discovery_journey import DiscoveryJourneyService

# Core domain imports
from forest_app.integrations.llm import LLMClient
from forest_app.core.services.semantic_base import SemanticMemoryManagerBase

logger = logging.getLogger(__name__)

class ArchitectureContainer(containers.DeclarativeContainer):
    """Dependency injection container for architecture components."""
    
    # Configuration
    config = providers.Configuration()
    
    # External dependencies (will be injected)
    llm_client = providers.Dependency(instance_of=LLMClient)
    semantic_memory_manager = providers.Dependency(instance_of=SemanticMemoryManagerBase)
    
    # Task Queue (Singleton)
    task_queue = providers.Singleton(
        TaskQueue,
        max_workers=config.architecture.task_queue.max_workers,
        result_ttl=config.architecture.task_queue.result_ttl
    )
    
    # Cache Service (Singleton)
    cache_service = providers.Singleton(
        CacheService,
        config=lambda: CacheConfig(
            backend=CacheBackend(config.architecture.cache.backend),
            redis_url=config.architecture.cache.redis_url,
            default_ttl=config.architecture.cache.default_ttl,
            namespace=config.architecture.cache.namespace
        )
    )
    
    # Event Bus (Singleton)
    event_bus = providers.Singleton(
        EventBus.get_instance
    )
    
    # Enhanced HTA Service (with injected dependencies)
    enhanced_hta_service = providers.Factory(
        EnhancedHTAService,
        llm_client=llm_client,
        semantic_memory_manager=semantic_memory_manager
    )
    
    # Discovery Journey Service (with injected dependencies)
    discovery_journey_service = providers.Factory(
        DiscoveryJourneyService,
        hta_service=enhanced_hta_service,
        llm_client=llm_client,
        event_bus=event_bus
    )

async def initialize_architecture(app: FastAPI = None) -> ArchitectureContainer:
    """
    Initialize the scalable architecture components.
    
    Args:
        app: Optional FastAPI app for lifecycle management
        
    Returns:
        Initialized architecture container
    """
    # Create default configuration
    config = {
        "architecture": {
            "task_queue": {
                "max_workers": int(os.environ.get("FOREST_TASK_WORKERS", "10")),
                "result_ttl": int(os.environ.get("FOREST_TASK_RESULT_TTL", "300"))
            },
            "cache": {
                "backend": os.environ.get("FOREST_CACHE_BACKEND", "memory"),
                "redis_url": os.environ.get("FOREST_REDIS_URL", "redis://localhost:6379/0"),
                "default_ttl": int(os.environ.get("FOREST_CACHE_TTL", "3600")),
                "namespace": os.environ.get("FOREST_CACHE_NAMESPACE", "forest:")
            }
        }
    }
    
    # Create container
    container = ArchitectureContainer()
    container.config.from_dict(config)
    
    # Initialize components that need startup
    task_queue = container.task_queue()
    await task_queue.start()
    
    # Register lifecycle management if app provided
    if app:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Starting The Forest architecture components...")
            
            # Yield to FastAPI
            yield
            
            # Shutdown
            logger.info("Shutting down The Forest architecture components...")
            await task_queue.stop()
        
        # Assign lifespan to app
        app.router.lifespan_context = lifespan
    
    logger.info("The Forest's enhanced architecture initialized and ready to support users at scale")
    return container

def inject_enhanced_architecture(app: FastAPI) -> None:
    """
    Inject the enhanced architecture into a FastAPI app.
    
    This sets up the FastAPI app to use the enhanced architecture components,
    making them available throughout the application.
    
    Args:
        app: FastAPI app to enhance
    """
    # Store initialization in startup event
    @app.on_event("startup")
    async def startup_enhanced_architecture():
        container = await initialize_architecture(app)
        app.state.architecture = container
        
        # Log startup
        logger.info("The Forest is now operating with enhanced architecture")
        
        # Publish system startup event
        event_bus = container.event_bus()
        await event_bus.publish({
            "event_type": EventType.SYSTEM_METRICS,
            "payload": {
                "event": "startup",
                "message": "The Forest's enhanced architecture is now active",
                "components": ["task_queue", "circuit_breaker", "event_bus", "cache_service"]
            }
        })
    
    # Log configuration
    logger.info("Enhanced architecture configured for FastAPI app")
