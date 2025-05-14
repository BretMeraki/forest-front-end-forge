# forest_app/core/onboarding.py

"""
Forest OS Onboarding and Session‑Management

This module contains:
  1. `onboard_user`: one‑time initialization of a JSON snapshot
     with NLP‑derived baselines (deep‑copy + persist).
  2. `run_onboarding`: CLI flow to capture a top‑level goal (Seed),
     target date, journey path, reflection, and baseline assessment.
  3. `run_forest_session` / `run_forest_session_async`: ongoing
     heartbeat loops for applying withering updates and persisting state.
"""

from __future__ import annotations

import sys
import time
import copy
import logging
import threading
import asyncio

from datetime import datetime, timedelta
from typing import Callable, Dict, Any, Optional

from forest_app.utils.baseline_loader import load_user_baselines
from forest_app.core.orchestrator import ForestOrchestrator
from forest_app.config.constants import ORCHESTRATOR_HEARTBEAT_SEC

from forest_app.core.snapshot import MemorySnapshot
from forest_app.modules.baseline_assessment import BaselineAssessmentEngine
from forest_app.modules.seed import SeedManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# 1. One‑time, programmatic baseline injection
# -----------------------------------------------------------------------------

def onboard_user(
    snapshot: Dict[str, Any],
    baselines: Dict[str, float],
    save_snapshot: Callable[[Dict[str, Any]], None]
) -> Dict[str, Any]:
    """
    One-time initialization of the user snapshot from NLP-derived baselines.

    - Deep-copies the incoming snapshot to avoid mutating shared state.
    - Injects and validates baseline metrics into component_state.
    - Persists the resulting snapshot as the session's starting state.

    Call this exactly once at session start, before any reflections or task processing.
    """
    new_snapshot = copy.deepcopy(snapshot)
    cs = new_snapshot.setdefault("component_state", {})
    cs["baselines"] = baselines

    # Validate and populate baselines (may raise on invalid data)
    new_snapshot = load_user_baselines(new_snapshot)

    try:
        save_snapshot(new_snapshot)
        logger.info("Initial snapshot persisted during onboarding.")
    except Exception as e:
        logger.error("Failed to save snapshot during onboarding: %s", e, exc_info=True)
        raise

    return new_snapshot


# -----------------------------------------------------------------------------
# 2. CLI‑driven HTA seed onboarding flow
# -----------------------------------------------------------------------------

def _prompt(text: str) -> str:
    sys.stdout.write(f"{text.strip()}\n> ")
    sys.stdout.flush()
    return sys.stdin.readline().strip()


def _parse_date_iso(date_str: str) -> Optional[str]:
    try:
        dt = datetime.fromisoformat(date_str.strip())
        return dt.date().isoformat()
    except ValueError:
        return None


def _recommend_completion_date(hta_scope: int) -> str:
    days = max(hta_scope, 1) * 2
    return (datetime.utcnow() + timedelta(days=days)).date().isoformat()


def run_onboarding(snapshot: MemorySnapshot) -> None:
    """
    Run the entire CLI onboarding flow and mutate `snapshot` in place.

    Steps:
      0. Capture a top-level goal (Seed) & domain
      1. Ask or recommend a completion date
      2. Choose journey path (structured/blended/open)
      3. Collect a "Where are you now?" reflection
      4. Run baseline assessment and seed the FullDevelopmentIndex
    """
    try:
        # 0. Seed details
        goal_title = _prompt(
            "What is the primary goal you wish to cultivate? "
            "(e.g. ‘Run a 5k’, ‘Launch my blog’)"
        )
        seed_domain = _prompt(
            "In one word, which life domain does this goal belong to? "
            "(e.g. health, career, creativity)"
        )
    except (EOFError, KeyboardInterrupt) as e:
        logger.error("Onboarding interrupted during initial prompts: %s", e, exc_info=True)
        sys.exit(1)

    seed_manager = SeedManager()
    seed = seed_manager.plant_seed(goal_title, seed_domain, additional_context=None)
    snapshot.component_state["seed_manager"] = seed_manager.to_dict()

    # Estimate HTA scope for date recommendation
    try:
        tree = getattr(seed, "hta_tree", None)
        if hasattr(tree, "child_count"):
            hta_scope = tree.child_count
        else:
            hta_scope = tree.get("root", {}).get("child_count", 1)
    except Exception:
        hta_scope = 1

    # 1. Completion date
    while True:
        try:
            date_input = _prompt(
                "Enter your target completion date for this goal (YYYY‑MM‑DD) "
                "or type 'recommend' to let the forest suggest one:"
            )
        except (EOFError, KeyboardInterrupt) as e:
            logger.error("Onboarding interrupted during date prompt: %s", e, exc_info=True)
            sys.exit(1)

        if date_input.lower() == "recommend":
            date_iso = _recommend_completion_date(hta_scope)
            print(f"\n🌲 The forest recommends {date_iso} as a gentle target.\n")
            break

        date_iso = _parse_date_iso(date_input)
        if date_iso:
            break
        print("❌ Invalid date format. Please use YYYY‑MM‑DD.")
    snapshot.estimated_completion_date = date_iso

    # 2. Journey path
    options = {"1": "structured", "2": "blended", "3": "open"}
    while True:
        try:
            choice = _prompt(
                "Choose your journey mode:\n"
                "  1) Structured – clear soft deadlines and strong guidance\n"
                "  2) Blended    – guideposts without penalties\n"
                "  3) Open       – no deadlines, introspection‑heavy\n"
                "Enter 1, 2, or 3:"
            )
        except (EOFError, KeyboardInterrupt) as e:
            logger.error("Onboarding interrupted during path selection: %s", e, exc_info=True)
            sys.exit(1)

        if choice in options:
            snapshot.current_path = options[choice]
            break
        print("❌ Please enter 1, 2, or 3.")

    # 3. Reflection
    try:
        where_text = _prompt(
            "Describe where you are right now in relation to this goal. "
            "Feel free to share thoughts, feelings, or context:"
        )
    except (EOFError, KeyboardInterrupt) as e:
        logger.error("Onboarding interrupted during reflection prompt: %s", e, exc_info=True)
        sys.exit(1)
    snapshot.core_state["where_you_are"] = where_text

    # 4. Baseline assessment
    print("\n🌿 Establishing your baseline… this may take a moment.\n")
    assessor = BaselineAssessmentEngine()
    try:
        baseline_data = asyncio.run(assessor.assess_baseline(goal_title, where_text))
    except Exception as e:
        logger.error("Baseline assessment failed: %s", e, exc_info=True)
        raise

    # Initialize dev_index
    snapshot.dev_index.update_from_dict({
        "indexes": baseline_data["development"],
        "adjustment_history": []
    })
    snapshot.component_state["dev_index"] = snapshot.dev_index.to_dict()

    # Populate core metrics
    snapshot.capacity = baseline_data["capacity"]
    snapshot.shadow_score = baseline_data["shadow_score"]
    snapshot.magnitude = baseline_data["magnitude"]
    snapshot.relationship_index = baseline_data["relationship"]

    # Onboarding complete
    # Note: you can also infer this implicitly by checking the above fields,
    # but we set it explicitly here for clarity.
    snapshot.baseline_established = True

    print("\n✅ Onboarding complete! Your journey begins.\n")


# -----------------------------------------------------------------------------
# 3. Blocking & async heartbeat loops for ongoing session maintenance
# -----------------------------------------------------------------------------

def run_forest_session(
    snapshot: Dict[str, Any],
    save_snapshot: Callable[[Dict[str, Any]], None],
    lock: Optional[threading.Lock] = None
) -> None:
    session_id = snapshot.get("user_id", "unknown")
    orch = ForestOrchestrator(saver=save_snapshot)
    logger.info(
        "Starting blocking forest session for session=%s (interval=%s sec)",
        session_id, ORCHESTRATOR_HEARTBEAT_SEC
    )
    try:
        while True:
            start_time = time.monotonic()
            try:
                if lock:
                    with lock:
                        orch._update_withering(snapshot)
                        orch._save_component_states(snapshot)
                else:
                    orch._update_withering(snapshot)
                    orch._save_component_states(snapshot)
            except Exception as tick_err:
                logger.exception(
                    "Error during heartbeat tick for session=%s: %s",
                    session_id, tick_err
                )
            elapsed = time.monotonic() - start_time
            sleep_duration = max(0, ORCHESTRATOR_HEARTBEAT_SEC - elapsed)
            try:
                time.sleep(sleep_duration)
            except KeyboardInterrupt:
                raise
            except Exception as sleep_err:
                logger.error(
                    "Error during heartbeat sleep for session=%s: %s",
                    session_id, sleep_err, exc_info=True
                )
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received; stopping session=%s", session_id)
        try:
            orch._save_component_states(snapshot)
        except Exception as e:
            logger.error(
                "Error saving state at shutdown for session=%s: %s",
                session_id, e, exc_info=True
            )
    finally:
        logger.info("Blocking forest session stopped for session=%s", session_id)


async def run_forest_session_async(
    snapshot: Dict[str, Any],
    save_snapshot: Callable[[Dict[str, Any]], None],
    lock: Optional[threading.Lock] = None
) -> None:
    session_id = snapshot.get("user_id", "unknown")
    orch = ForestOrchestrator(saver=save_snapshot)
    logger.info(
        "Starting async forest session for session=%s (interval=%s sec)",
        session_id, ORCHESTRATOR_HEARTBEAT_SEC
    )
    try:
        while True:
            start_time = asyncio.get_running_loop().time()
            try:
                if lock:
                    with lock:
                        orch._update_withering(snapshot)
                        orch._save_component_states(snapshot)
                else:
                    orch._update_withering(snapshot)
                    orch._save_component_states(snapshot)
            except Exception as tick_err:
                logger.exception(
                    "Error during async heartbeat tick for session=%s: %s",
                    session_id, tick_err
                )
            elapsed = asyncio.get_running_loop().time() - start_time
            sleep_duration = max(0, ORCHESTRATOR_HEARTBEAT_SEC - elapsed)
            try:
                await asyncio.sleep(sleep_duration)
            except asyncio.CancelledError:
                raise
            except Exception as sleep_err:
                logger.error(
                    "Error during async heartbeat sleep for session=%s: %s",
                    session_id, sleep_err, exc_info=True
                )
    except asyncio.CancelledError:
        logger.info("Async session cancelled for session=%s", session_id)
        try:
            orch._save_component_states(snapshot)
        except Exception as e:
            logger.error(
                "Error saving state at cancellation for session=%s: %s",
                session_id, e, exc_info=True
            )
    except Exception as e:
        logger.error(
            "Unhandled error in async session for session=%s: %s",
            session_id, e, exc_info=True
        )
    finally:
        logger.info("Async forest session stopped for session=%s", session_id)
