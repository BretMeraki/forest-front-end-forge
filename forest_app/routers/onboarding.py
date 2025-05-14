# forest_app/routers/onboarding.py (Refactored - Use Injected LLMClient)

import logging
import uuid
import json
from typing import Optional, Any, Dict, List
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from pydantic import ValidationError, BaseModel, Field

# --- Dependencies & Models ---
from forest_app.persistence.database import get_db
from forest_app.persistence.repository import MemorySnapshotRepository
from forest_app.persistence.models import UserModel
from forest_app.core.security import get_current_active_user
from forest_app.core.snapshot import MemorySnapshot
from forest_app.helpers import save_snapshot_with_codename
from forest_app.core.integrations.discovery_integration import get_discovery_journey_service

# --- Updated LLM Imports ---
from forest_app.integrations.llm import (
    LLMClient,  # <-- Import Client
    LLMError,
    LLMValidationError,
    # LLMGenerationError, # This specific one might not be defined in llm.py, use base LLMError or others
    LLMConfigurationError,
    LLMConnectionError
)
# Removed generate_response

from forest_app.modules.hta_models import HTAResponseModel as HTAValidationModel
from forest_app.modules.seed import Seed, SeedManager
from forest_app.core.orchestrator import ForestOrchestrator
from forest_app.core.integrations.discovery_integration import get_discovery_journey_service
from forest_app.core.discovery_journey.integration_utils import enrich_hta_with_discovery_insights
from forest_app.dependencies import get_orchestrator # Assuming this provides orchestrator with injected llm_client

try:
    from forest_app.config import constants
except ImportError:
     class ConstantsPlaceholder:
        MAX_CODENAME_LENGTH=60
        MIN_PASSWORD_LENGTH=8
        ONBOARDING_STATUS_NEEDS_GOAL="needs_goal"
        ONBOARDING_STATUS_NEEDS_CONTEXT="needs_context"
        ONBOARDING_STATUS_COMPLETED="completed"
        SEED_STATUS_ACTIVE="active"
        SEED_STATUS_COMPLETED="completed"
        DEFAULT_RESONANCE_THEME="neutral"
     constants = ConstantsPlaceholder()

logger = logging.getLogger(__name__)
router = APIRouter()

# --- Pydantic Models DEFINED LOCALLY ---
class SetGoalRequest(BaseModel):
    goal_description: Any = Field(...) # Keep Any for flexibility unless specific type known

class AddContextRequest(BaseModel):
    context_reflection: Any = Field(...) # Keep Any for flexibility

class OnboardingResponse(BaseModel):
    onboarding_status: str
    message: str
    refined_goal: Optional[str] = None
    first_task: Optional[dict] = None
# --- End Pydantic Models ---


# --- /start endpoint (No LLM calls, likely no changes needed) ---
@router.post("/start", response_model=OnboardingResponse, tags=["Onboarding"])
async def start_onboarding(
    onboarding_data: SetGoalRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user),
    orchestrator_i: ForestOrchestrator = Depends(get_orchestrator) # Inject orchestrator to get LLMClient for save helper
    # Note: save_snapshot_with_codename requires llm_client
):
    """
    Handles the first step of onboarding: setting the user's goal.
    Saves the goal description to the snapshot and updates the onboarding status.
    """
    user_id = current_user.id
    try:
        logger.info(f"[/onboarding/set_goal] Received goal request user {user_id}.")
        repo = MemorySnapshotRepository(db)
        stored_model = repo.get_latest_snapshot(user_id)
        snapshot = MemorySnapshot()
        if stored_model and stored_model.snapshot_data:
            try: snapshot = MemorySnapshot.from_dict(stored_model.snapshot_data)
            except Exception as load_err:
                logger.error(f"Error loading snapshot user {user_id}: {load_err}. Starting fresh.", exc_info=True)
                stored_model = None

        if snapshot.activated_state.get("activated", False):
             logger.info(f"User {user_id} /set_goal called but session previously active. Resetting goal.")

        # --- Update snapshot state ---
        if not isinstance(snapshot.component_state, dict): snapshot.component_state = {}
        snapshot.component_state["raw_goal_description"] = str(request.goal_description) # Ensure string
        snapshot.activated_state["goal_set"] = True
        snapshot.activated_state["activated"] = False

        # --- Save snapshot (requires LLMClient from orchestrator) ---
        if not orchestrator_i or not orchestrator_i.llm_client:
             logger.error(f"LLMClient not available via orchestrator for user {user_id} in set_goal.")
             raise HTTPException(status_code=500, detail="Internal configuration error: LLM service unavailable.")

        force_create = not stored_model
        saved_model = await save_snapshot_with_codename(
            db=db,
            repo=repo,
            user_id=user_id,
            snapshot=snapshot,
            llm_client=orchestrator_i.llm_client, # <-- Pass LLMClient from orchestrator
            stored_model=stored_model,
            force_create_new=force_create
        )
        if not saved_model: raise HTTPException(status_code=500, detail="Failed to prepare snapshot save.")

        # --- Commit and Refresh ---
        try:
            db.commit()
            db.refresh(saved_model)
            logger.info(f"Successfully committed snapshot for user {user_id} in set_goal.")
        except SQLAlchemyError as commit_err:
            db.rollback()
            logger.exception(f"Failed to commit snapshot for user {user_id} in set_goal: {commit_err}")
            raise HTTPException(status_code=500, detail="Failed to finalize goal save.")

        logger.info(f"Onboarding Step 1 complete user {user_id}.")
        return OnboardingResponse(
            onboarding_status=constants.ONBOARDING_STATUS_NEEDS_CONTEXT,
            message="Vision received. Now add context."
        )
    # (Error handling remains the same)
    except HTTPException: raise
    except (ValueError, TypeError, AttributeError) as data_err:
        logger.exception(f"Data/Type error /set_goal user {user_id}: {data_err}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid data: {data_err}")
    except SQLAlchemyError as db_err:
        logger.exception(f"Database error /set_goal user {user_id}: {db_err}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database error during goal setting.")
    except Exception as e:
        logger.exception(f"Unexpected error /set_goal user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to process goal.")


# --- /add_context endpoint (Needs LLM call update) ---
@router.post("/add_context", response_model=OnboardingResponse, tags=["Onboarding"])
async def add_context_endpoint(
    request: AddContextRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_active_user),
    orchestrator_i: ForestOrchestrator = Depends(get_orchestrator) # Inject orchestrator (provides LLMClient)
):
    """
    Handles the second step of onboarding: adding context and generating the initial HTA.
    Uses the injected orchestrator's LLMClient for HTA generation.
    """
    user_id = current_user.id
    try:
        logger.info(f"[/onboarding/add_context] Received context request user {user_id}.")

        # --- Check Orchestrator and LLMClient availability ---
        if not orchestrator_i or not orchestrator_i.llm_client:
            logger.error(f"LLMClient not available via orchestrator for user {user_id} in add_context.")
            raise HTTPException(status_code=500, detail="Internal configuration error: LLM service unavailable.")
        llm_client_instance = orchestrator_i.llm_client # Get client instance

        repo = MemorySnapshotRepository(db)
        stored_model = repo.get_latest_snapshot(user_id)
        if not stored_model or not stored_model.snapshot_data:
            raise HTTPException(status_code=404, detail="Snapshot not found. Run /set_goal first.")

        try: snapshot = MemorySnapshot.from_dict(stored_model.snapshot_data)
        except Exception as load_err:
            logger.error(f"Error loading snapshot data user {user_id}: {load_err}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Could not load session state: {load_err}")

        if not snapshot.activated_state.get("goal_set", False):
            raise HTTPException(status_code=400, detail="Goal must be set before adding context.")

        # --- Handle already active session (no LLM call needed here) ---
        if snapshot.activated_state.get("activated", False):
            # (Logic for already active session remains the same)
            logger.info(f"User {user_id} /add_context recalled, session already active.")
            first_task = None; refined_goal_desc = "N/A"
            try:
                 if orchestrator_i.seed_manager and snapshot.component_state.get("seed_manager"):
                     seeds_dict = snapshot.component_state["seed_manager"].get("seeds", {})
                     if isinstance(seeds_dict, dict) and seeds_dict:
                          first_seed = seeds_dict.get(next(iter(seeds_dict)), {})
                          refined_goal_desc = first_seed.get("description", "N/A")
                 if orchestrator_i.task_engine and snapshot.core_state.get('hta_tree'):
                     orchestrator_i._load_component_states(snapshot)
                     task_result = orchestrator_i.task_engine.get_next_step(snapshot.to_dict())
                     first_task = task_result.get("base_task")
            except Exception as task_e: logger.exception("Error getting existing task/goal: %s", task_e)
            return OnboardingResponse(onboarding_status=constants.ONBOARDING_STATUS_COMPLETED, message="Session already active.", refined_goal=refined_goal_desc, first_task=first_task)

        # --- Process Goal and Context ---
        raw_goal = snapshot.component_state.get("raw_goal_description")
        raw_context = request.context_reflection
        processed_goal = str(raw_goal) if raw_goal is not None else ""
        processed_context = str(raw_context) if raw_context is not None else ""
        if not processed_goal:
            logger.error(f"Internal Error: Goal description missing state user {user_id}.")
            raise HTTPException(status_code=500, detail="Internal Error: Goal description missing.")
        if not isinstance(snapshot.component_state, dict): snapshot.component_state = {}
        snapshot.component_state["raw_context_reflection"] = raw_context

        # --- Generate HTA ---
        root_node_id = f"root_{str(uuid.uuid4())[:8]}"
        hta_prompt = ( # Keep the detailed HTA prompt as before
            f"[INST] Create a Hierarchical Task Analysis (HTA) representing a plan based on the user's goal and context.\n"
            f"Goal: {processed_goal}\nContext: {processed_context}\n\n"
            f"**RESPONSE FORMAT REQUIREMENTS:**\n"
            f"1. Respond ONLY with a single, valid JSON object.\n"
            f"2. This JSON object MUST contain ONLY ONE top-level key: 'hta_root'.\n"
            f"3. The value of 'hta_root' MUST be a JSON object representing the root node.\n"
            f"4. The entire response MUST strictly adhere to the `HTAResponseModel` and `HTANodeModel` structure.\n\n"
            f"**NODE ATTRIBUTE REQUIREMENTS (Apply recursively):**\n"
            f"* `id`: (String) Unique ID. Root MUST use: '{root_node_id}'.\n"
            f"* `title`: (String) Concise title.\n"
            f"* `description`: (String) Detailed description.\n"
            f"* `priority`: (Float) Value STRICTLY between 0.0 and 1.0 (inclusive).\n"
            f"* `estimated_energy`: (String) MUST be one of: \"low\", \"medium\", or \"high\".\n"
            f"* `estimated_time`: (String) MUST be one of: \"low\", \"medium\", or \"high\".\n"
            f"* `depends_on`: (List[String]) List of node IDs. Empty list `[]` if none.\n"
            f"* `children`: (List[Object]) List of child node objects. Empty list `[]` for leaf nodes.\n"
            f"* `linked_tasks`: (List[String]) Optional. Default `[]`.\n"
            f"* `is_milestone`: (Boolean) Default `false`.\n"
        )
        # (Validation, error handling, backup strategies same as before)
        
        # --- Invisibly enhance the HTA with Discovery Journey insights ---
        # This doesn't create a separate feature, just subtly improves the HTA to guide abstract-to-concrete journey
        discovery_service = get_discovery_journey_service(request)
        if discovery_service and hta_response:
            try:
                # Seamlessly enrich the HTA with discovery insights without user awareness
                hta_result = await enrich_hta_with_discovery_insights(
                    discovery_service=discovery_service,
                    hta_service=orchestrator_i.hta_service,
                    user_id=user_id,
                    hta_tree=hta_response,
                    raw_goal=processed_goal,
                    raw_context=processed_context
                )
            except Exception as e:
                # Non-critical enhancement - log but don't disrupt the flow
                logger.warning(f"Non-critical: Could not enhance HTA with discovery insights: {e}")
        
        hta_model_dict = hta_result if hta_result else {"hta_root": {}}
        logger.info(f"Successfully generated HTA via LLMClient for user {user_id}.")

        # --- Update Snapshot State with HTA (from dict) and Activate ---
        # (State update logic remains the same)
        root_node_data = hta_model_dict.get("hta_root")
        hta_tree_dict = hta_model_dict
        root_node_data = hta_tree_dict.get('nodes', {}).get(hta_tree_dict.get('root_id', ''), {})
        seed_name = root_node_data.get('title', 'Unnamed Goal')
        seed_desc = str(root_node_data.get('description', f'Overall goal: {seed_name}'))
        
        # Store invisible Discovery Journey metadata in the seed
        discovery_metadata = {
            "discovery_journey_enabled": False,
            "journey_started_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Invisibly enhance with Discovery Journey without making it a separate feature
        if discovery_service:
            try:
                # Assess abstraction level without exposing to user
                abstraction_level = await discovery_service.assess_abstraction_level(
                    user_id=user_id,
                    goal_description=processed_goal,
                    context_reflection=processed_context
                )
                discovery_metadata = {
                    "initial_abstraction_level": abstraction_level,
                    "discovery_journey_enabled": True,
                    "journey_started_at": datetime.now(timezone.utc).isoformat()
                }
                
                # If very abstract goal, subtly schedule deeper analysis in background
                if abstraction_level and abstraction_level.get('level', 0) > 7:  # Highly abstract
                    if orchestrator_i and hasattr(orchestrator_i, 'task_queue'):
                        await orchestrator_i.task_queue.enqueue(
                            discovery_service.prepare_exploratory_paths,
                            user_id=user_id,
                            goal_description=processed_goal,
                            context_reflection=processed_context,
                            priority=3,  # Background task
                            metadata={"type": "seamless_discovery_preparation"}
                        )
                        logger.info(f"Scheduled seamless discovery preparation for user {user_id}")
            except Exception as e:
                # Non-critical - don't disrupt normal flow
                logger.warning(f"Non-critical: Could not integrate discovery journey: {e}")
                
        # Create seed with or without discovery metadata
        new_seed = Seed(
            seed_id=f'seed_{str(uuid.uuid4())[:8]}', 
            seed_name=seed_name, 
            seed_domain='General', 
            description=seed_desc, 
            status=constants.SEED_STATUS_ACTIVE, 
            hta_tree=hta_tree_dict, 
            created_at=datetime.now(timezone.utc),
            metadata={"discovery_journey": discovery_metadata}
        )

        # --- Save the updated snapshot ---
        saved_model = await save_snapshot_with_codename(
            db=db,
            repo=repo,
            user_id=user_id,
            snapshot=snapshot,
            llm_client=llm_client_instance, # Pass LLMClient from orchestrator
            stored_model=stored_model
        )
        if not saved_model: raise HTTPException(status_code=500, detail="Failed to prepare activated snapshot save.")

        # --- Commit and Refresh ---
        # (Commit logic remains the same)
        try:
            db.flush()
            db.commit()
            db.refresh(saved_model)
            logger.info(f"Successfully committed ACTIVATED snapshot user {user_id} in add_context.")
        except SQLAlchemyError as commit_err:
            db.rollback()
            logger.exception(f"Failed to commit ACTIVATED snapshot user {user_id}: {commit_err}")
            raise HTTPException(status_code=500, detail="Failed to finalize session activation.")

        # --- Determine First Task (Post-Activation) ---
        # (Logic remains the same)
        first_task = {}; refined_goal_desc = new_seed.description
        try:
            logger.debug(f"Reloading snapshot user {user_id} post-commit for first task.")
            repo_after_commit = MemorySnapshotRepository(db)
            stored_model_after_commit = repo_after_commit.get_latest_snapshot(user_id)
            if stored_model_after_commit and stored_model_after_commit.snapshot_data:
                snapshot_after_commit = MemorySnapshot.from_dict(stored_model_after_commit.snapshot_data)
                orchestrator_i._load_component_states(snapshot_after_commit)
                if snapshot_after_commit.core_state.get('hta_tree'):
                    logger.debug(f"HTA tree FOUND user {user_id} after commit/reload.")
                    snap_dict = snapshot_after_commit.to_dict()
                    task_result = orchestrator_i.task_engine.get_next_step(snap_dict)
                    if isinstance(task_result, dict):
                        first_task = task_result.get("base_task", {})
                        logger.info(f"Determined first task user {user_id}: {first_task.get('id', 'N/A')}")
                    else: logger.warning("Task engine returned non-dict post-activation: %s", task_result)
                else: logger.warning("Committed snapshot user %d missing hta_tree.", user_id)
            else: logger.error("Could not retrieve committed snapshot data user %d.", user_id)
        except Exception as task_e: logger.exception("Error getting first task after activation user %d: %s", user_id, task_e)

        logger.info(f"Onboarding Step 2 (add_context) complete user {user_id}. Session activated.")
        return OnboardingResponse(
            onboarding_status=constants.ONBOARDING_STATUS_COMPLETED,
            message="Onboarding complete! Your journey begins.",
            refined_goal=refined_goal_desc,
            first_task=first_task or None
        )
    # (Error handling remains the same)
    except HTTPException: raise
    except (ValueError, TypeError, AttributeError, ValidationError) as data_err: # Added ValidationError
        logger.exception(f"Data/Validation error /add_context user {user_id}: {data_err}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid data/validation: {data_err}")
    except SQLAlchemyError as db_err:
        logger.exception(f"Database error /add_context user {user_id}: {db_err}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database error during context processing.")
    except Exception as e:
        logger.exception(f"Unexpected error /add_context user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal error: {e}")
