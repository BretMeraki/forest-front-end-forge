"""
Discovery Journey API Router

This module provides API endpoints for the Discovery Journey system, 
facilitating the user's transition from abstract goals to concrete needs
through an adaptive and personalized experience.
"""

import logging
from typing import Dict, Any, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from forest_app.core.integrations.discovery_integration import get_discovery_journey_service
from forest_app.api.dependencies import get_current_user
from forest_app.models.user import UserModel

logger = logging.getLogger(__name__)

# ----- Pydantic Models -----

class ReflectionInput(BaseModel):
    """User reflection input for the discovery journey."""
    content: str
    emotion_level: Optional[int] = None  # 1-10 scale
    context: Optional[Dict[str, Any]] = None
    
class PatternResponse(BaseModel):
    """Response containing discovered patterns in the user's journey."""
    patterns: List[Dict[str, Any]]
    insights: List[str]
    recommended_tasks: List[Dict[str, Any]]
    
class ExploratoryTaskResponse(BaseModel):
    """Response containing exploratory tasks for the user."""
    tasks: List[Dict[str, Any]]
    context: Dict[str, Any]
    emotional_framing: Optional[Dict[str, Any]] = None

# ----- Router -----

router = APIRouter(
    prefix="/discovery",
    tags=["discovery-journey"],
    responses={404: {"description": "Discovery service not found"}},
)

# ----- Endpoints -----

@router.post("/reflection", status_code=status.HTTP_201_CREATED)
async def add_reflection(
    reflection: ReflectionInput,
    request: Request,
    current_user: UserModel = Depends(get_current_user)
):
    """
    Add a new reflection to the user's discovery journey.
    
    This endpoint captures the user's reflections on their journey, including emotional
    responses, which help the system refine its understanding of their needs.
    """
    discovery_service = get_discovery_journey_service(request.app)
    
    if not discovery_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Discovery journey service unavailable"
        )
    
    try:
        result = await discovery_service.process_reflection(
            user_id=current_user.id,
            reflection_content=reflection.content,
            emotion_level=reflection.emotion_level,
            context=reflection.context
        )
        
        return {
            "status": "success",
            "message": "Reflection processed successfully",
            "insights": result.get("insights", []),
            "has_new_pattern": result.get("has_new_pattern", False)
        }
    except Exception as e:
        logger.error(f"Error processing reflection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process reflection: {str(e)}"
        )

@router.get("/patterns", response_model=PatternResponse)
async def get_patterns(
    request: Request,
    current_user: UserModel = Depends(get_current_user)
):
    """
    Get the currently identified patterns in the user's journey.
    
    This endpoint provides insights into patterns detected in the user's reflections
    and interactions, helping them understand their evolving needs and goals.
    """
    discovery_service = get_discovery_journey_service(request.app)
    
    if not discovery_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Discovery journey service unavailable"
        )
    
    try:
        patterns = await discovery_service.get_patterns_for_user(current_user.id)
        return patterns
    except Exception as e:
        logger.error(f"Error retrieving patterns: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve patterns: {str(e)}"
        )

@router.get("/exploratory-tasks", response_model=ExploratoryTaskResponse)
async def get_exploratory_tasks(
    request: Request,
    task_count: int = 3,
    current_user: UserModel = Depends(get_current_user)
):
    """
    Get exploratory tasks to help the user clarify their goals.
    
    This endpoint generates tasks specifically designed to help users explore different
    aspects of their abstract goals, guiding them toward more concrete understanding.
    """
    discovery_service = get_discovery_journey_service(request.app)
    
    if not discovery_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Discovery journey service unavailable"
        )
    
    try:
        tasks = await discovery_service.generate_exploratory_tasks(
            user_id=current_user.id,
            count=task_count
        )
        return tasks
    except Exception as e:
        logger.error(f"Error generating exploratory tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate exploratory tasks: {str(e)}"
        )

@router.post("/task-completion/{task_id}")
async def complete_exploratory_task(
    task_id: str,
    request: Request,
    feedback: Optional[Dict[str, Any]] = None,
    current_user: UserModel = Depends(get_current_user)
):
    """
    Mark an exploratory task as completed and provide feedback.
    
    This endpoint records completion of a discovery task and processes any feedback
    from the user, further refining the system's understanding of their needs.
    """
    discovery_service = get_discovery_journey_service(request.app)
    
    if not discovery_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Discovery journey service unavailable"
        )
    
    try:
        result = await discovery_service.process_task_completion(
            user_id=current_user.id,
            task_id=task_id,
            feedback=feedback or {}
        )
        
        return {
            "status": "success",
            "message": "Task completion processed",
            "next_steps": result.get("next_steps", []),
            "insights_gained": result.get("insights_gained", [])
        }
    except Exception as e:
        logger.error(f"Error processing task completion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process task completion: {str(e)}"
        )

@router.get("/progress-summary")
async def get_journey_progress(
    request: Request,
    current_user: UserModel = Depends(get_current_user)
):
    """
    Get a summary of the user's discovery journey progress.
    
    This endpoint provides a comprehensive view of the user's journey from abstract
    goal to concrete needs, highlighting key insights, patterns, and evolution.
    """
    discovery_service = get_discovery_journey_service(request.app)
    
    if not discovery_service:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Discovery journey service unavailable"
        )
    
    try:
        summary = await discovery_service.get_journey_progress(current_user.id)
        
        return {
            "starting_point": summary.get("starting_point", {}),
            "current_understanding": summary.get("current_understanding", {}),
            "key_insights": summary.get("key_insights", []),
            "progress_metrics": summary.get("progress_metrics", {}),
            "clarity_level": summary.get("clarity_level", 0),
            "journey_highlights": summary.get("journey_highlights", [])
        }
    except Exception as e:
        logger.error(f"Error retrieving journey progress: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve journey progress: {str(e)}"
        )
