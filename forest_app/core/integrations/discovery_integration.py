"""
Discovery Journey Integration

This module integrates the Discovery Journey capabilities into The Forest's
architecture, ensuring the specialized handling of abstract-to-concrete 
goal discovery is available throughout the system.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from uuid import UUID
from fastapi import Depends, FastAPI
from dependency_injector.wiring import inject, Provide

from forest_app.containers import Container
from forest_app.core.discovery_journey import DiscoveryJourneyService
from forest_app.core.discovery_journey.top_node_evolution import TopNodeEvolutionManager
from forest_app.core.services.enhanced_hta_service import EnhancedHTAService
from forest_app.integrations.llm import LLMClient
from forest_app.core.event_bus import EventBus
from forest_app.core.task_queue import TaskQueue

logger = logging.getLogger(__name__)

def setup_discovery_journey(app: FastAPI) -> None:
    """
    Set up and integrate the Discovery Journey module with the FastAPI app.
    
    Args:
        app: FastAPI application instance
    """
    # Register startup event
    @app.on_event("startup")
    @inject
    async def init_discovery_journey(
        hta_service: EnhancedHTAService = Depends(Provide[Container.enhanced_hta_service]),
        llm_client: LLMClient = Depends(Provide[Container.llm_client]),
        event_bus: EventBus = Depends(Provide[Container.architecture.event_bus]),
    ):
        """Initialize the Discovery Journey service and register it with the app."""
        try:
            # Create Top Node Evolution Manager to ensure semi-static nature
            top_node_manager = TopNodeEvolutionManager(
                llm_client=llm_client
            )
            
            # Create Discovery Journey service with top node manager
            discovery_service = DiscoveryJourneyService(
                hta_service=hta_service,
                llm_client=llm_client,
                event_bus=event_bus,
                top_node_manager=top_node_manager  # Pass the top node manager
            )
            
            # Store on app state for access throughout the application
            app.state.discovery_service = discovery_service
            
            logger.info("Discovery Journey service initialized and integrated")
            
            # Register background task for periodic pattern analysis
            if hasattr(app.state, 'architecture') and hasattr(app.state.architecture, 'task_queue'):
                task_queue = app.state.architecture.task_queue()
                
                # Schedule periodic check for pattern evolution
                async def schedule_periodic_pattern_check():
                    """Schedule periodic checks for pattern evolution."""
                    await task_queue.enqueue(
                        _periodic_pattern_check_task,
                        discovery_service,
                        priority=4,  # Lower priority background task
                        metadata={"type": "periodic_pattern_check"}
                    )
                
                # Schedule initial pattern check
                await schedule_periodic_pattern_check()
                
                logger.info("Scheduled periodic pattern analysis for Discovery Journey")
                
        except Exception as e:
            logger.error(f"Error initializing Discovery Journey service: {e}")

async def _periodic_pattern_check_task(discovery_service: DiscoveryJourneyService) -> None:
    """
    Background task for periodic pattern checking across all users,
    with careful consideration of the semi-static nature of top node.
    
    Args:
        discovery_service: The Discovery Journey service
    """
    try:
        logger.info("Running periodic pattern check for all users")
        
        # In a real implementation, would pull active users from a repository
        # For now, just log that this would happen
        logger.info("Would analyze patterns for all active users")
        
        # Re-schedule this task to run again later (e.g., every 12 hours)
        if hasattr(discovery_service, 'hta_service') and hasattr(discovery_service.hta_service, 'task_queue'):
            task_queue = discovery_service.hta_service.task_queue
            
            # Schedule next run in 12 hours
            # In a real implementation, would use a more robust scheduling system
            logger.info("Re-scheduling next pattern check")
            
    except Exception as e:
        logger.error(f"Error in periodic pattern check: {e}")

def get_discovery_journey_service(app: Optional[FastAPI]) -> Optional[DiscoveryJourneyService]:
    """
    Get the Discovery Journey service from the FastAPI app.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Discovery Journey service instance or None if not found
    """
    if app is None:
        return None
    return getattr(app.state, 'discovery_service', None)
