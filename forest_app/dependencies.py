# forest_app/dependencies.py (Reverted to Absolute Container Import)

import logging
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, status, Depends, Request, Header
from sqlalchemy.orm import Session # Import Session for type hinting

# Import request context
from forest_app.core.request_context import RequestContext

# --- Import the Classes needed for type hinting ---
try:
    from forest_app.core.orchestrator import ForestOrchestrator
    from forest_app.modules.trigger_phrase import TriggerPhraseHandler
    from forest_app.modules.logging_tracking import TaskFootprintLogger, ReflectionLogLogger
    from forest_app.containers import Container
    from forest_app.persistence.database import get_db
    imports_ok = True
except ImportError as e:
    logging.error(f"CRITICAL: dependencies.py failed to import one or more modules: {e}")
    imports_ok = False

logger = logging.getLogger(__name__)

# --- Dependency Provider Functions ---

def get_hta_service(request: Request):
    """Dependency to get the global HTA service instance from the DI container."""
    if not imports_ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: Required modules could not be imported."
        )
    container: Container = getattr(request.app.state, "container", None)
    if container is None or not hasattr(container, "hta_service"):
        logger.critical("CRITICAL: DI container or hta_service not found in app.state!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: HTA service not available."
        )
    return container.hta_service()

# --- Functions to get global singletons from Container ---
def get_orchestrator(request: Request) -> ForestOrchestrator:
    """Dependency to get the global orchestrator instance from the DI container."""
    if not imports_ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: Required modules could not be imported."
        )
    
    container: Container = getattr(request.app.state, "container", None)
    if container is None:
        logger.critical("CRITICAL: DI container not found in app.state!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: Core application container not available."
        )
    try:
        # Get orchestrator instance FROM the container
        orchestrator = container.orchestrator() # Calls the provider in containers.py
        if not isinstance(orchestrator, ForestOrchestrator):
            raise TypeError("Container did not return a valid ForestOrchestrator instance.")
        return orchestrator
    except Exception as e: # Catch potential errors during provider resolution
        logger.exception("CRITICAL: Failed to get orchestrator from DI container: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: Core orchestrator component could not be retrieved."
        )

def get_trigger_handler(request: Request) -> TriggerPhraseHandler:
    """Dependency to get the global trigger_handler instance from the DI container."""
    container: Container = getattr(request.app.state, "container", None)
    if container is None:
        logger.critical("CRITICAL: DI container not found in app.state!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: Core application container not available."
        )
    try:
        # Get trigger_handler instance FROM the container
        trigger_handler = container.trigger_handler() # Assumes provider named 'trigger_handler' exists
        if not isinstance(trigger_handler, TriggerPhraseHandler):
             raise TypeError("Container did not return a valid TriggerPhraseHandler instance.")
        return trigger_handler
    except Exception as e: # Catch potential errors during provider resolution
        logger.exception("CRITICAL: Failed to get trigger_handler from DI container: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error: Core trigger handler component could not be retrieved."
        )

# --- Request Context Dependency ---
async def get_request_context(
    request: Request,
    x_trace_id: Optional[str] = Header(None)
) -> RequestContext:
    """Dependency to get or create a request context with trace_id and user info."""
    # Get feature flags from container if available
    container = getattr(request.app.state, "container", None)
    feature_flags = {}
    
    if container and hasattr(container, "feature_flags"):
        try:
            feature_flags_provider = container.feature_flags()
            if hasattr(feature_flags_provider, "get_all_flags"):
                feature_flags = feature_flags_provider.get_all_flags()
        except Exception as e:
            logger.warning(f"Failed to retrieve feature flags: {e}")
    
    # Try to get user_id from auth - implementation will need to be extended when auth is implemented
    user_id = None
    
    # Use provided trace_id or generate a new one
    trace_id = UUID(x_trace_id) if x_trace_id else None
    
    # Create and return the context
    return RequestContext(
        user_id=user_id,
        trace_id=trace_id,
        feature_flags=feature_flags
    )

# --- Logger Dependencies (UNCHANGED) ---
def get_task_logger(db: Session = Depends(get_db)) -> TaskFootprintLogger:
    """
    Dependency provider that creates a TaskFootprintLogger instance
    with an injected database session.
    """
    try:
        return TaskFootprintLogger(db=db)
    except ValueError as ve: # Catch init errors specifically
         logger.error(f"Failed to create TaskFootprintLogger: {ve}")
         raise HTTPException(status_code=500, detail="Internal server error: Could not initialize task logger.")

def get_reflection_logger(db: Session = Depends(get_db)) -> ReflectionLogLogger:
    """
    Dependency provider that creates a ReflectionLogLogger instance
    with an injected database session.
    """
    try:
        return ReflectionLogLogger(db=db)
    except ValueError as ve: # Catch init errors specifically
         logger.error(f"Failed to create ReflectionLogLogger: {ve}")
         raise HTTPException(status_code=500, detail="Internal server error: Could not initialize reflection logger.") 
