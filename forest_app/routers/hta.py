# forest_app/routers/hta.py

import logging
from typing import Optional, Any, Dict
# <<< --- ADDED IMPORT --- >>>
import json
# <<< --- END ADDED IMPORT --- >>>

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
# --- Pydantic Imports ---
from pydantic import BaseModel # Import base pydantic needs

# --- Dependencies & Models ---
from forest_app.persistence.database import get_db
from forest_app.persistence.repository import MemorySnapshotRepository
from forest_app.persistence.models import UserModel
from forest_app.core.security import get_current_active_user
from forest_app.core.orchestrator import ForestOrchestrator
from forest_app.core.discovery_journey.integration_utils import track_task_completion_for_discovery, infuse_recommendations_into_snapshot
from forest_app.core.integrations.discovery_integration import get_discovery_journey_service
from forest_app.core.snapshot import MemorySnapshot
# --- REMOVED INCORRECT IMPORT ---
# from forest_app.core.pydantic_models import HTAStateResponse
try:
    from forest_app.config import constants
except ImportError:
    class ConstantsPlaceholder: ONBOARDING_STATUS_NEEDS_GOAL="needs_goal"; ONBOARDING_STATUS_NEEDS_CONTEXT="needs_context"; ONBOARDING_STATUS_COMPLETED="completed"
    constants = ConstantsPlaceholder()


logger = logging.getLogger(__name__)
router = APIRouter()

# --- Pydantic Models DEFINED LOCALLY ---
class HTAStateResponse(BaseModel):
    hta_tree: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
# --- End Pydantic Models ---


@router.get("/state", response_model=HTAStateResponse, tags=["HTA"]) # Prefix is in main.py
async def get_hta_state(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user)
):
    user_id = current_user.id
    logger.info(f"Request HTA state user {user_id}")
    try:
        repo = MemorySnapshotRepository(db); stored_model = repo.get_latest_snapshot(user_id);
        if not stored_model: return HTAStateResponse(hta_tree=None, message="No active session found.")
        if not stored_model.snapshot_data: return HTAStateResponse(hta_tree=None, message="Session data missing.")

        try: snapshot = MemorySnapshot.from_dict(stored_model.snapshot_data)
        except Exception as load_err: raise HTTPException(status_code=500, detail=f"Failed load session: {load_err}")

        if not snapshot.activated_state.get("activated", False):
            status_msg = constants.ONBOARDING_STATUS_NEEDS_CONTEXT if snapshot.activated_state.get("goal_set") else constants.ONBOARDING_STATUS_NEEDS_GOAL
            message = "Onboarding incomplete. Provide context." if status_msg == constants.ONBOARDING_STATUS_NEEDS_CONTEXT else "Onboarding incomplete. Set goal."
            return HTAStateResponse(hta_tree=None, message=message)

        # Now run normal processing, orchestrator handles snapshotting if needed
        orchestrator_i = ForestOrchestrator()
        result = await orchestrator_i.process_task_completion(
            user_id=str(current_user.id),
            task_footprint=None
        )
        
        # Invisibly leverage Discovery Journey without creating a separate experience
        discovery_service = get_discovery_journey_service(None)
        if discovery_service:
            try:
                # Silently track task completion to inform the discovery journey
                await track_task_completion_for_discovery(
                    discovery_service=discovery_service,
                    user_id=str(current_user.id),
                    task_id=None,
                    feedback={
                        "emotion": None,
                        "reflection": None,
                        "difficulty": None,
                        "completion_context": {
                            "time": None,
                            "node_data": None
                        }
                    }
                )
                
                # If we have a snapshot, invisibly enrich it with discovery insights
                if result and 'snapshot' in result and result['snapshot']:
                    result['snapshot'] = await infuse_recommendations_into_snapshot(
                        discovery_service=discovery_service,
                        snapshot=result['snapshot'],
                        user_id=str(current_user.id)
                    )
            except Exception as e:
                # Non-critical enhancement - log but don't disrupt the flow
                logger.warning(f"Non-critical: Could not enhance task completion with discovery insights: {e}")

        hta_tree_data = snapshot.core_state.get("hta_tree")

        # <<< --- ADDED LOGGING --- >>>
        try:
            if hta_tree_data:
                log_data_str = json.dumps(hta_tree_data, indent=2, default=str)
                if len(log_data_str) > 1000: log_data_str = log_data_str[:1000] + "... (truncated)"
                logger.debug(f"[ROUTER HTA LOAD] HTA data loaded from core_state to be returned:\n{log_data_str}")
            else:
                logger.debug("[ROUTER HTA LOAD] HTA data loaded from core_state is None or empty.")
        except Exception as log_ex:
            logger.error(f"[ROUTER HTA LOAD] Error logging loaded HTA state: {log_ex}")
        # <<< --- END ADDED LOGGING --- >>>

        if not hta_tree_data or not isinstance(hta_tree_data, dict) or not hta_tree_data.get("root"):
            # Log this specific condition too
            logger.warning(f"[ROUTER HTA LOAD] HTA data is invalid/missing root just before returning 404-like response. Type: {type(hta_tree_data)}")
            return HTAStateResponse(hta_tree=None, message="HTA data not found or invalid.")

        return HTAStateResponse(hta_tree=hta_tree_data, message="HTA structure retrieved.")

    except HTTPException: raise
    except SQLAlchemyError as db_err:
        logger.error("DB error getting HTA state user %d: %s", user_id, db_err, exc_info=True)
        raise HTTPException(status_code=503, detail="DB error.")
    except Exception as e:
        logger.error("Error getting HTA state user %d: %s", user_id, e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error.")
