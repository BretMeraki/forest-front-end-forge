# forest_app/modules/xp_mastery.py

import logging
from datetime import datetime
# Removed unused List, Dict, Any imports for this version

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class XPMastery:
    """
    Handles XP-related state and Mastery Challenges.
    MODIFIED: XP stages are de-emphasized. Progression is now primarily tracked
    via HTA completion. Mastery Challenge triggering is moved to Orchestrator
    based on HTA milestones.
    """

    # --- MODIFICATION START: De-emphasize/Remove XP_STAGES ---
    # The fixed XP stages are no longer the primary driver of progression.
    # Commenting out or removing this dictionary. HTA branch completion defines milestones.
    # XP_STAGES = {
    #     "Awakening": { ... },
    #     "Committing": { ... },
    #     "Deepening": { ... },
    #     "Harmonizing": { ... },
    #     "Becoming": { ... },
    # }
    # --- MODIFICATION END ---

    def __init__(self):
        # No persistent state needed in this modified version.
        pass

    # --- Method to serialize state ---
    def to_dict(self) -> dict:
        """
        Serializes the engine's state (currently stateless).
        """
        # No state to save in this revised version.
        return {}

    # --- Method to load state ---
    def update_from_dict(self, data: dict):
        """
        Updates the engine's state from a dictionary (currently stateless).
        """
        # No state to load in this revised version.
        logger.debug("XPMastery state loaded (currently stateless based on HTA progression).")
        pass

    # --- MODIFICATION START: Adjust get_current_stage ---
    def get_current_stage(self, xp: float = 0.0) -> dict:
        """
        Determines a *descriptive* stage based on HTA progress (future) or provides a default.
        NOTE: This method's significance is reduced. HTA completion defines milestones.
        The 'xp' argument is kept for potential secondary uses but doesn't drive stages.
        """
        # In the future, this could potentially inspect snapshot.core_state.hta_tree
        # to determine a descriptive stage based on completed HTA branches.
        # For now, return a default or placeholder status.
        logger.debug("get_current_stage called (XP: %.2f) - returning generic status as stages are HTA-driven.", xp)
        return {
            "stage": "In Progress (HTA-Driven)", # Generic stage name
            "challenge_type": "HTA Milestone Integration", # Generic challenge type hint
            "min_xp": 0, # No longer relevant for gating
            "max_xp": float("inf"), # No longer relevant for gating
        }
    # --- MODIFICATION END ---

    # --- MODIFICATION START: Keep challenge content generation, but note trigger change ---
    def generate_challenge_content(self, stage_info: dict, snapshot: dict) -> dict:
        """
        Generates a concrete Mastery Challenge based on a *trigger event* (e.g., HTA milestone).
        NOTE: The trigger logic is now external (in Orchestrator). This method just formats content.
        The 'stage_info' dict should ideally contain info about the HTA milestone achieved.
        """
        # Use challenge type from input, default if missing
        ct = stage_info.get("challenge_type", "HTA Milestone Integration")
        stage_name = stage_info.get("stage", "Current Milestone") # Use descriptive name if available

        # Example content based on generic types (can be refined later)
        # This logic could be enhanced to use details from the completed HTA node/branch
        hta_milestone_title = stage_info.get("hta_milestone_title", "your recent progress") # Example key

        if ct == "Naming Desire": # Keep original content as examples if useful
            act = (
                f"Reflect on how completing '{hta_milestone_title}' clarifies your deepest desire. "
                "Capture this refined desire visually or in writing."
            )
        elif ct == "Showing Up":
             act = (
                 f"Building on completing '{hta_milestone_title}', commit to a specific action that demonstrates 'showing up' "
                 "in alignment with your next goal. Schedule it now."
             )
        elif ct == "Softening Shadow":
             act = (
                 f"Consider any shadow aspects revealed or addressed by completing '{hta_milestone_title}'. "
                 "Identify one concrete self-care or boundary-setting action to take this week."
             )
        elif ct == "Harmonizing Seeds":
             act = (
                 f"Now that '{hta_milestone_title}' is complete, how does it connect to other active goals (seeds) in your plan? "
                 "Create a mind map or plan to integrate these efforts."
             )
        elif ct == "Integration Prompt" or ct == "HTA Milestone Integration": # Default/Fallback
             act = (
                 f"Reflect on the journey to complete '{hta_milestone_title}'. "
                 "What have you learned? How does this integrate into your overall vision? Capture your insights."
             )
        else: # Generic fallback
             act = f"Reflect on completing '{hta_milestone_title}' and identify one concrete next step."

        content = (
            f"Integration Challenge: {stage_name}\n"
            f"Milestone Achieved: '{hta_milestone_title}'\n"
            f"Focus: '{ct}'\n"
            f"Suggested Action: {act}\n"
            "Consider how this achievement shapes your path forward."
        )

        challenge = {
            "stage":          stage_name,
            "challenge_type": ct,
            "challenge_content": content,
            "triggered_at":   datetime.utcnow().isoformat(),
        }
        logger.info("Generated Integration Challenge content: %s", challenge["challenge_type"])
        return challenge
    # --- MODIFICATION END ---

    # --- MODIFICATION START: Disable XP-based check ---
    def check_xp_stage(self, snapshot) -> dict:
        """
        DISABLED: This check is no longer driven by XP proximity.
        Challenge triggering now occurs in the Orchestrator based on HTA milestones.
        Returns an empty dictionary.
        """
        # This method is effectively disabled in the HTA-driven model.
        # The logic to check for milestones and trigger challenges moves
        # to core/orchestrator.py -> process_task_completion.
        logger.debug("check_xp_stage called, but XP-based triggers are disabled. Returning {}.")
        return {}
    # --- MODIFICATION END ---
