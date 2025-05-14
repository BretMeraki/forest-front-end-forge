# forest_app/routers/goals.py (Corrected - Dependency Injection)

import logging
from typing import Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel

# --- Dependencies & Models ---
from forest_app.persistence.database import get_db
from forest_app.persistence.repository import MemorySnapshotRepository
from forest_app.persistence.models import UserModel
from forest_app.core.security import get_current_active_user
from forest_app.core.snapshot import MemorySnapshot
from forest_app.helpers import save_snapshot_with_codename
# --- Import Class needed for Dependency Injection Type Hint ---
from forest_app.core.orchestrator import ForestOrchestrator
# --- Import Dependency Function ---
from forest_app.dependencies import get_orchestrator
# --- REMOVE incorrect import ---
# from forest_app.core.orchestrator import orchestrator
# --- REMOVE main import (no longer needed) ---
# import forest_app.main
try:
    from forest_app.config import constants
except ImportError:
     class ConstantsPlaceholder: SEED_STATUS_COMPLETED="completed"
     constants = ConstantsPlaceholder()

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Pydantic Models DEFINED LOCALLY ---
class MessageResponse(BaseModel):
    message: str
# --- End Pydantic Models ---


@router.post("/confirm_completion/{seed_id}", response_model=MessageResponse, tags=["Goals"])
async def confirm_goal_completion(
    seed_id: str,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user),
    # --- Inject Orchestrator Dependency ---
    orchestrator_i: ForestOrchestrator = Depends(get_orchestrator)
):
    """Marks a specific Seed (goal) as completed."""
    user_id = current_user.id
    logger.info(f"Confirm goal complete user {user_id} seed {seed_id}")
    try:
        repo = MemorySnapshotRepository(db); stored_model = repo.get_latest_snapshot(user_id)
        if not stored_model or not stored_model.snapshot_data: raise HTTPException(status_code=404, detail="Active session not found.")
        try: snapshot = MemorySnapshot.from_dict(stored_model.snapshot_data)
        except Exception as load_err: raise HTTPException(status_code=500, detail=f"Failed load session: {load_err}")
        if not snapshot.activated_state.get("activated"): raise HTTPException(status_code=403, detail="Session not active.")

        try:
            # --- Use injected orchestrator_i ---
            orchestrator_i._load_component_states(snapshot)
            seed_manager = orchestrator_i.seed_manager;
            if not seed_manager: raise HTTPException(status_code=500, detail="Failed goal state load.")
            seed_to_complete = seed_manager.get_seed_by_id(seed_id)
            if not seed_to_complete: raise HTTPException(status_code=404, detail=f"Goal ID '{seed_id}' not found.")
            if not hasattr(seed_to_complete, 'status'): raise HTTPException(status_code=500, detail="Seed data invalid.")
            if seed_to_complete.status == constants.SEED_STATUS_COMPLETED: return MessageResponse(message=f"Goal (ID: {seed_id}) already completed.")
        except HTTPException: raise
        except Exception as state_err: raise HTTPException(status_code=500, detail=f"Internal error accessing goal state: {state_err}")

        seed_to_complete.status = constants.SEED_STATUS_COMPLETED;
        seed_to_complete.updated_at = datetime.now(timezone.utc)
        snapshot.component_state["seed_manager"] = seed_manager.to_dict()

        saved_model = await save_snapshot_with_codename(db, repo, user_id, snapshot, stored_model)
        if not saved_model: raise HTTPException(status_code=500, detail="Failed save session.")

        seed_name = getattr(seed_to_complete, 'seed_name', f'ID: {seed_id}')
        return MessageResponse(message=f"Goal '{seed_name}' marked as completed.")

    except HTTPException: raise
    except (SQLAlchemyError, ValueError, TypeError) as db_val_err:
        logger.exception(f"DB/Data error /confirm_goal user {user_id} seed {seed_id}: {db_val_err}")
        detail = "DB error." if isinstance(db_val_err, SQLAlchemyError) else f"Invalid data: {db_val_err}"
        status_code = status.HTTP_503_SERVICE_UNAVAILABLE if isinstance(db_val_err, SQLAlchemyError) else status.HTTP_400_BAD_REQUEST
        raise HTTPException(status_code=status_code, detail=detail)
    except Exception as e:
        logger.exception(f"Error /confirm_goal user {user_id} seed {seed_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal error.")
