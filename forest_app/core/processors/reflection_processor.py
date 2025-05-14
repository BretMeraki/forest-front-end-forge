# forest_app/core/processors/reflection_processor.py

import logging
import json
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# --- Core & Module Imports ---
# Import necessary classes for type hints and functionality
from forest_app.core.snapshot import MemorySnapshot
from forest_app.core.utils import clamp01
from forest_app.core.harmonic_framework import SilentScoring, HarmonicRouting
from forest_app.core.services.semantic_memory import SemanticMemoryManager
from forest_app.modules.sentiment import SentimentInput, SentimentOutput, SecretSauceSentimentEngineHybrid, NEUTRAL_SENTIMENT_OUTPUT
from forest_app.modules.practical_consequence import PracticalConsequenceEngine
from forest_app.modules.task_engine import TaskEngine
from forest_app.modules.narrative_modes import NarrativeModesEngine
from forest_app.modules.soft_deadline_manager import schedule_soft_deadlines # Keep if scheduling happens here
# Import LLM Client and specific response models needed
from forest_app.integrations.llm import (
    LLMClient,
    ArbiterStandardResponse,
    LLMError,
    LLMValidationError
)
# --- Feature Flags ---
try:
    from forest_app.core.feature_flags import Feature, is_enabled
except ImportError:
    # Fallback if flags cannot be imported - assume features are off
    def is_enabled(feature): return False
    class Feature:
        SENTIMENT_ANALYSIS = "FEATURE_ENABLE_SENTIMENT_ANALYSIS"
        NARRATIVE_MODES = "FEATURE_ENABLE_NARRATIVE_MODES"
        SOFT_DEADLINES = "FEATURE_ENABLE_SOFT_DEADLINES"
        ENABLE_POETIC_ARBITER_VOICE = "FEATURE_ENABLE_POETIC_ARBITER_VOICE"
        CORE_TASK_ENGINE = "FEATURE_ENABLE_CORE_TASK_ENGINE" # Check usage

# --- Constants (Import or define defaults) ---
from forest_app.config.constants import (
    REFLECTION_CAPACITY_NUDGE_BASE,
    REFLECTION_SHADOW_NUDGE_BASE,
    MAGNITUDE_THRESHOLDS, # Needed for describe_magnitude helper
    DEFAULT_RESONANCE_THEME # Default theme if harmonic routing fails
    # Add FALLBACK_TASK_DETAILS here if it's defined in constants.py
)

# Define fallback task details if not in constants
FALLBACK_TASK_DETAILS = {
    "title": "Reflect on your current focus",
    "description": "Take a moment to consider what feels most important right now.",
    "priority": 5,
    "magnitude": 5,
    "status": "incomplete",
    "soft_deadline": None,
    "parent_id": None,
}


logger = logging.getLogger(__name__)

# --- Helper Functions (Could be moved to utils if used elsewhere) ---

def prune_context(snap_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Minimise prompt size while keeping key info."""
    # NOTE: This function might need access to component state if features like
    # FINANCIAL_READINESS or DESIRE_ENGINE are enabled and used for context pruning.
    # If so, the necessary state might need to be passed in or handled differently.
    # For now, keeping it simple based on direct snapshot attributes.
    ctx = {
        "shadow_score": snap_dict.get("shadow_score", 0.5),
        "capacity": snap_dict.get("capacity", 0.5),
        "magnitude": snap_dict.get("magnitude", 5.0),
        "last_ritual_mode": snap_dict.get("last_ritual_mode") if is_enabled(Feature.NARRATIVE_MODES) else None,
        "current_path": snap_dict.get("current_path"),
    }
    # Filter out None values
    ctx = {k: v for k, v in ctx.items() if v is not None}
    return ctx

def describe_magnitude(value: float) -> str:
    """Describes magnitude based on thresholds."""
    # (Same implementation as in the original orchestrator)
    try:
        float_value = float(value)
        # Ensure MAGNITUDE_THRESHOLDS is accessible (imported constant)
        valid_thresholds = {k: float(v) for k, v in MAGNITUDE_THRESHOLDS.items() if isinstance(v, (int, float))}
        if not valid_thresholds: return "Unknown"
        sorted_thresholds = sorted(valid_thresholds.items(), key=lambda item: item[1], reverse=True)
        for label, thresh in sorted_thresholds:
            if float_value >= thresh: return str(label)
        return str(sorted_thresholds[-1][0]) if sorted_thresholds else "Dormant"
    except (ValueError, TypeError) as e:
        logger.error("Error converting value/threshold for magnitude: %s (Value: %s)", e, value)
        return "Unknown"
    except Exception as e:
        logger.exception("Error describing magnitude for value %s: %s", value, e)
        return "Unknown"

# --- Reflection Processor Class ---

class ReflectionProcessor:
    """Processes user reflections with semantic memory integration."""

    def __init__(self, llm_client, sentiment_engine, pattern_engine):
        self.llm_client = llm_client
        self.sentiment_engine = sentiment_engine
        self.pattern_engine = pattern_engine
        self.logger = logging.getLogger(__name__)

    async def process_reflection(self, 
                               reflection_text: str, 
                               context: Dict[str, Any] = None,
                               snapshot: Any = None) -> Dict[str, Any]:
        """
        Process a reflection with semantic memory context.
        
        Args:
            reflection_text: The user's reflection text
            context: Optional context including relevant memories
            snapshot: Optional snapshot for state management
        """
        try:
            # Extract relevant memories from context
            relevant_memories = context.get("relevant_memories", []) if context else []
            
            # Build memory context string
            memory_context = self._build_memory_context(relevant_memories)
            
            # Analyze reflection with memory context
            sentiment_result = await self.sentiment_engine.analyze(
                text=reflection_text,
                context=memory_context
            )
            
            pattern_result = await self.pattern_engine.identify_patterns(
                text=reflection_text,
                context=memory_context
            )
            
            # Generate insights using LLM with memory context
            insights = await self._generate_insights(
                reflection_text=reflection_text,
                sentiment=sentiment_result,
                patterns=pattern_result,
                memory_context=memory_context
            )
            
            # Update snapshot if provided
            if snapshot:
                self._update_snapshot(
                    snapshot=snapshot,
                    reflection=reflection_text,
                    sentiment=sentiment_result,
                    patterns=pattern_result,
                    insights=insights
                )
            
            return {
                "sentiment": sentiment_result,
                "patterns": pattern_result,
                "insights": insights,
                "relevant_memories": relevant_memories
            }
            
        except Exception as e:
            self.logger.error(f"Error processing reflection: {e}")
            raise

    def _build_memory_context(self, memories: List[Dict[str, Any]]) -> str:
        """Build a context string from relevant memories."""
        if not memories:
            return ""
            
        context_parts = ["Previous relevant experiences:"]
        for memory in memories:
            timestamp = datetime.fromisoformat(memory.get("timestamp", "")).strftime("%Y-%m-%d")
            content = memory.get("content", "")
            context_parts.append(f"- [{timestamp}] {content}")
            
        return "\n".join(context_parts)

    async def _generate_insights(self,
                               reflection_text: str,
                               sentiment: Dict[str, Any],
                               patterns: Dict[str, Any],
                               memory_context: str) -> List[str]:
        """Generate insights using LLM with memory context."""
        prompt = f"""
        Reflection: {reflection_text}
        
        Sentiment Analysis: {json.dumps(sentiment)}
        
        Identified Patterns: {json.dumps(patterns)}
        
        Memory Context:
        {memory_context}
        
        Based on the reflection, analysis, and historical context, generate key insights about:
        1. Progress and growth
        2. Recurring patterns
        3. Potential areas for focus
        """
        
        response = await self.llm_client.generate(prompt)
        
        # Parse insights from response
        insights = [line.strip() for line in response.split("\n") if line.strip()]
        return insights

    def _update_snapshot(self,
                        snapshot: Any,
                        reflection: str,
                        sentiment: Dict[str, Any],
                        patterns: Dict[str, Any],
                        insights: List[str]) -> None:
        """Update snapshot with reflection results."""
        try:
            # Update reflection context
            if hasattr(snapshot, "reflection_context"):
                snapshot.reflection_context["recent_insight"] = insights[0] if insights else ""
                snapshot.reflection_context["themes"] = patterns.get("themes", [])
                
            # Add to reflection log
            if hasattr(snapshot, "reflection_log"):
                snapshot.reflection_log.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "content": reflection,
                    "sentiment": sentiment,
                    "patterns": patterns,
                    "insights": insights
                })
                
        except Exception as e:
            self.logger.error(f"Error updating snapshot: {e}")
            # Continue without snapshot update

    async def process(self, user_input: str, snapshot: MemorySnapshot) -> Dict[str, Any]:
        """
        Processes user reflection, updates state, generates task(s)/narrative.
        NOTE: Assumes component states are already loaded into engines and
              withering is updated before this method is called. It focuses
              on the core reflection processing logic and modifies the snapshot directly.
              It does NOT save component states back to the snapshot.
        """
        logger.info("Processing reflection...")

        # --- 1. Append Reflection to Batch & History (Initial) ---
        # Initialize lists if they don't exist (defensive coding)
        if not hasattr(snapshot, 'current_batch_reflections') or not isinstance(snapshot.current_batch_reflections, list):
            snapshot.current_batch_reflections = []
        if not hasattr(snapshot, 'conversation_history') or not isinstance(snapshot.conversation_history, list):
            snapshot.conversation_history = []
        if not hasattr(snapshot, 'task_backlog') or not isinstance(snapshot.task_backlog, list):
            snapshot.task_backlog = []


        if user_input: # Avoid adding empty reflections
            snapshot.current_batch_reflections.append(user_input)
            snapshot.conversation_history.append({"role": "user", "content": user_input})
            logger.info(f"Appended reflection. Batch size: {len(snapshot.current_batch_reflections)}. History size: {len(snapshot.conversation_history)}.")

        # --- 2. Sentiment Analysis & Metric Nudges ---
        sentiment_score = 0.0
        if isinstance(self.sentiment_engine, SecretSauceSentimentEngineHybrid) and is_enabled(Feature.SENTIMENT_ANALYSIS):
            try:
                if hasattr(self.sentiment_engine, 'analyze_emotional_field'):
                    sentiment_input = SentimentInput(text_to_analyze=user_input)
                    # Assuming analyze_emotional_field might be async now
                    sentiment_output: SentimentOutput = await self.sentiment_engine.analyze_emotional_field(input_data=sentiment_input)
                    if isinstance(sentiment_output, SentimentOutput): sentiment_score = sentiment_output.score
                    else: logger.warning("Sentiment engine returned unexpected type: %s", type(sentiment_output))
                else: logger.error("Injected REAL Sentiment engine lacks analyze_emotional_field method.")
            except Exception as exc: logger.exception("Sentiment analysis step failed: %s", exc)
        else: logger.debug("Sentiment analysis skipped.")

        # Apply nudges directly to snapshot
        try:
            current_capacity = float(getattr(snapshot, 'capacity', 0.5))
            snapshot.capacity = clamp01(current_capacity + REFLECTION_CAPACITY_NUDGE_BASE * sentiment_score)
            current_shadow = float(getattr(snapshot, 'shadow_score', 0.5))
            snapshot.shadow_score = clamp01(current_shadow - REFLECTION_SHADOW_NUDGE_BASE * sentiment_score)
        except Exception as nudge_exc: logger.exception("Error applying metric nudges: %s", nudge_exc)

        # --- 3. Practical Consequence Update ---
        try:
            if isinstance(self.practical_consequence_engine, PracticalConsequenceEngine) and \
               hasattr(self.practical_consequence_engine, 'update_signals_from_reflection'):
                self.practical_consequence_engine.update_signals_from_reflection(user_input)
            elif self.practical_consequence_engine and type(self.practical_consequence_engine).__name__ != 'DummyService':
                 logger.warning("Practical consequence engine lacks update_signals_from_reflection method.")
        except Exception as exc: logger.exception("Practical consequence update step failed: %s", exc)

        # --- 4. Task Generation ---
        generated_tasks: List[Dict[str, Any]] = []
        fallback_task: Optional[Dict[str, Any]] = None
        try:
            snap_dict_for_task_engine = snapshot.to_dict() # Task engine expects dict
            # Ensure HTA is included (TaskEngine likely handles missing HTA gracefully)
            snap_dict_for_task_engine.setdefault("core_state", {})["hta_tree"] = snapshot.core_state.get("hta_tree", {})

            task_bundle = self.task_engine.get_next_step(snap_dict_for_task_engine)

            if isinstance(task_bundle, dict):
                generated_tasks = task_bundle.get("tasks", [])
                fallback_task = task_bundle.get("fallback_task")
                if not generated_tasks and not fallback_task:
                    logger.error("Task engine returned empty bundle. Generating emergency fallback.")
                    fallback_task = self._get_fallback_task("task_engine_empty_bundle")
                elif generated_tasks:
                    logger.info(f"Task engine returned {len(generated_tasks)} frontier tasks.")
                    fallback_task = None # Explicitly clear fallback if HTA tasks found
                else:
                     logger.info("Task engine returned a fallback task.")
                     generated_tasks = [] # Ensure tasks_list is empty if only fallback
            else:
                logger.error("Task engine returned invalid bundle format: %s. Generating fallback.", task_bundle)
                fallback_task = self._get_fallback_task("task_engine_invalid_bundle")
                generated_tasks = []

            # Update snapshot's frontier batch IDs
            frontier_task_ids = [t.get('id') for t in generated_tasks if isinstance(t, dict) and t.get('id')]
            if generated_tasks:
                 snapshot.current_frontier_batch_ids = frontier_task_ids
                 logger.info(f"Set new frontier batch IDs on snapshot: {frontier_task_ids}")
            elif fallback_task and hasattr(snapshot, 'current_frontier_batch_ids'):
                 # Clear batch if only fallback is returned
                 if snapshot.current_frontier_batch_ids:
                     logger.info("Task engine returned fallback, clearing previous frontier batch on snapshot.")
                     snapshot.current_frontier_batch_ids = []

        except Exception as exc:
             logger.exception("Task engine step failed: %s", exc)
             fallback_task = self._get_fallback_task("task_engine_exception")
             generated_tasks = []
             # Clear batch on error
             if hasattr(snapshot, 'current_frontier_batch_ids'):
                 snapshot.current_frontier_batch_ids = []


        # --- 5. Arbiter Narrative Generation ---
        narrative = "(fallback narrative)"
        arbiter_task_data_refined = None # Store refined task data from Arbiter if provided
        try:
            # Prepare context for Arbiter
            primary_task_for_prompt = generated_tasks[0] if generated_tasks else fallback_task if fallback_task else {"id": "error", "title": "Error Task"}
            task_titles_for_prompt = [t.get('title', 'Untitled') for t in generated_tasks] if generated_tasks else [primary_task_for_prompt.get('title', 'Default Task')]
            snap_dict_for_llm = snapshot.to_dict()

            # Get style directive (if applicable)
            style = ""
            if isinstance(self.narrative_engine, NarrativeModesEngine) and \
               is_enabled(Feature.NARRATIVE_MODES) and \
               hasattr(self.narrative_engine, 'determine_narrative_mode'):
                try:
                    context_task = primary_task_for_prompt
                    # Pass snapshot dict as task engine might not have been called with it directly
                    nm = self.narrative_engine.determine_narrative_mode(
                        snap_dict_for_llm, context={"base_task": context_task}
                    )
                    style = nm.get("style_directive", "") if isinstance(nm, dict) else ""
                except Exception as narr_exc: logger.exception("Narrative mode step failed: %s", narr_exc)
            else: logger.debug("Narrative engine skipped.")

            # Construct prompt (using helper logic from original orchestrator)
            arb_prompt = self._construct_arbiter_prompt(
                user_input=user_input,
                snapshot_dict=snap_dict_for_llm,
                conversation_history=snapshot.conversation_history,
                primary_task=primary_task_for_prompt,
                task_titles=task_titles_for_prompt,
                style_directive_input=style
            )

            # Call LLM
            arb_out: Optional[ArbiterStandardResponse] = await self.llm_client.generate(
                prompt_parts=[arb_prompt], response_model=ArbiterStandardResponse
            )

            # Process response
            if isinstance(arb_out, ArbiterStandardResponse):
                arb_data = arb_out.model_dump() # Pydantic v2
                narrative = arb_data.get('narrative', "(Error: Missing narrative)")
                arbiter_task_data_refined = arb_data.get('task') # Get refined task
                logger.info(">>> Successfully received narrative from standard Arbiter LLMClient.")
            else:
                logger.error("generate_response for Arbiter returned unexpected type: %s.", type(arb_out))
                narrative = "(Internal processing error after LLM call)"
        except (LLMError, LLMValidationError) as llm_err:
            logger.warning(f"LLM/Validation error during standard Arbiter call: {llm_err}")
            narrative = "(LLM processing error)"
        except Exception as e:
            logger.exception(f"Unexpected error during Arbiter LLM call: {e}")
            narrative = "(Unexpected internal error)"

        # --- 6. Update Snapshot with Narrative & Processed Task ---
        # Append narrative to history
        if isinstance(narrative, str):
            snapshot.conversation_history.append({"role": "assistant", "content": narrative})
            # Trim history if needed
            max_history = 20
            if len(snapshot.conversation_history) > max_history:
                snapshot.conversation_history = snapshot.conversation_history[-max_history:]

        # Potentially update the task in generated_tasks or fallback_task if Arbiter refined it
        if isinstance(arbiter_task_data_refined, dict):
             refined_id = arbiter_task_data_refined.get('id')
             if generated_tasks and generated_tasks[0].get('id') == refined_id:
                  generated_tasks[0] = arbiter_task_data_refined
                  logger.debug("Updated first generated task with Arbiter refinement.")
             elif fallback_task and fallback_task.get('id') == refined_id:
                  fallback_task = arbiter_task_data_refined
                  logger.debug("Updated fallback task with Arbiter refinement.")


        # --- 7. Soft Deadline Scheduling (Optional) ---
        tasks_for_deadline = generated_tasks + ([fallback_task] if fallback_task else [])
        is_confirmation_task = any(t.get("id") == "completion_confirmation" for t in tasks_for_deadline if isinstance(t, dict))

        if is_enabled(Feature.SOFT_DEADLINES) and not is_confirmation_task:
             current_path = getattr(snapshot, "current_path", "structured")
             if current_path != "open":
                  try:
                       # Filter for valid tasks to schedule
                       valid_tasks_for_deadline = [t for t in tasks_for_deadline if isinstance(t, dict) and t.get("id") and t.get("id") != "fallback"]
                       if valid_tasks_for_deadline:
                           # Pass the snapshot object itself
                           schedule_soft_deadlines(snapshot, valid_tasks_for_deadline, override_existing=False)
                  except ValueError as ve: logger.error("Soft-deadline scheduling error: %s", ve)
                  except Exception as exc: logger.exception("Unexpected soft-deadline scheduling error: %s", exc)
        else: logger.debug("Skipping soft deadline scheduling.")


        # --- 8. Add Generated Tasks to Backlog ---
        # Exclude fallback and confirmation tasks
        tasks_to_add_to_backlog = [
            task for task in generated_tasks
            if isinstance(task, dict) and task.get("id") and task.get("id") != "fallback" and task.get("id") != "completion_confirmation"
        ]
        if isinstance(snapshot.task_backlog, list):
            for task in tasks_to_add_to_backlog:
                 # Avoid duplicates
                 if not any(t.get("id") == task["id"] for t in snapshot.task_backlog if isinstance(t, dict)):
                      snapshot.task_backlog.append(task)
                      logger.debug(f"Appended task {task['id']} to snapshot backlog.")
                 else:
                      logger.warning("Task %s already in snapshot backlog, not adding again.", task["id"])
        else:
            logger.error("snapshot.task_backlog is not a list, cannot append task(s).")

        # --- 9. Calculate Final Response ---
        final_tasks_for_response = generated_tasks if generated_tasks else [fallback_task] if fallback_task else []

        # Calculate magnitude description
        avg_magnitude = 5.0
        if final_tasks_for_response:
             magnitudes = [float(t.get('magnitude', 5.0)) for t in final_tasks_for_response if isinstance(t, dict)]
             if magnitudes: avg_magnitude = sum(magnitudes) / len(magnitudes)
        mag_desc = describe_magnitude(avg_magnitude)

        # Calculate Harmonic Routing
        resonance_info = {"theme": DEFAULT_RESONANCE_THEME, "routing_score": 0.0}
        if isinstance(self.harmonic_router, HarmonicRouting) and isinstance(self.silent_scorer, SilentScoring):
            try:
                snap_dict_for_routing = snapshot.to_dict()
                detailed_scores = self.silent_scorer.compute_detailed_scores(snap_dict_for_routing)
                harmonic_result = self.harmonic_router.route_harmony(snap_dict_for_routing, detailed_scores if isinstance(detailed_scores, dict) else {})
                if isinstance(harmonic_result, dict): resonance_info = harmonic_result
            except Exception as hr_exc: logger.exception("Error getting harmonic routing: %s", hr_exc)
        else: logger.debug("Harmonic routing skipped (components missing or dummy).")

        # Assemble payload
        response_payload = {
            "tasks": final_tasks_for_response,
            "arbiter_response": narrative,
            # Note: Offering/Mastery Challenges are typically generated on completion, not reflection
            "offering": None,
            "mastery_challenge": None,
            "magnitude_description": mag_desc,
            "resonance_theme": str(resonance_info.get("theme", DEFAULT_RESONANCE_THEME)),
            "routing_score": float(resonance_info.get("routing_score", 0.0)),
            # Confirmation logic usually triggered by specific LLM response/task, not standard here
            "action_required": None,
            "confirmation_details": None,
        }

        logger.info("Reflection processing complete by ReflectionProcessor.")
        return response_payload

    # --- Internal Helper Methods ---

    def _get_fallback_task(self, reason: str) -> Dict[str, Any]: # <<< CORRECTED LINE
        """Generates a generic fallback task when primary task generation fails."""
        # Ensure FALLBACK_TASK_DETAILS is accessible (e.g., from constants or defined above)
        task_id = f"fallback_{uuid.uuid4()}"
        logger.warning(f"Generating fallback task {task_id} due to: {reason}")
        task = {
            "id": task_id,
            **FALLBACK_TASK_DETAILS # Use defined details
        }
        return task

    def _construct_arbiter_prompt(
        self,
        user_input: str,
        snapshot_dict: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
        primary_task: Dict[str, Any],
        task_titles: List[str],
        style_directive_input: str = ""
    ) -> str:
        """Constructs the prompt for the Arbiter LLM call."""
        # Context Pruning (Simplified for example)
        pruned_snap_ctx = prune_context(snapshot_dict)
        context_summary = json.dumps(pruned_snap_ctx)

        # Task Representation
        task_summary = f"Primary Task: {primary_task.get('title', 'N/A')}"
        if len(task_titles) > 1:
             task_summary += f" | Other Tasks: {', '.join(task_titles[1:])}"

        # History Formatting (Basic example)
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-5:]]) # Last 5 messages

        # Style Directive
        style_text = f"Style: {style_directive_input}" if style_directive_input else "Style: Default"
        if is_enabled(Feature.ENABLE_POETIC_ARBITER_VOICE):
             style_text = f"Style: Poetic and metaphorical. {style_directive_input}"

        # Prompt Construction (Example structure)
        prompt = f"""
Context Summary: {context_summary}

Recent Conversation:
{history_text}

Current Task Focus: {task_summary}

User's Latest Reflection: {user_input}

Instructions: Respond as the Forest Arbiter. Acknowledge the reflection briefly. Provide a narrative connecting the reflection to the current task(s) and the overall context. Refine the primary task details if necessary based on the reflection. Ensure your response follows the requested '{style_text}'. Your response must be a JSON object matching the ArbiterStandardResponse format, including 'narrative' and optional 'task' fields.
"""
        logger.debug("Constructed Arbiter Prompt:\n%s", prompt[:500] + "..." if len(prompt) > 500 else prompt) # Log truncated prompt
        return prompt
