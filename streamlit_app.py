# front_end/streamlit_app.py (Refactored: Uses api_client, auth_ui, onboarding_ui)

import sys
import os

# --- Add project root to sys.path ---
# Get the directory containing this script (streamlit_app.py, which is the project root)
project_root = os.path.dirname(os.path.abspath(__file__))
# Add the project root to the start of the Python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# Verify the path (optional debug line)
# print(f"--- Added project root to sys.path: {project_root}", file=sys.stderr)
# --- End sys.path modification ---

# --- Import the refactored modules ---
from forest_app.front_end.api_client import call_forest_api # <<< Correct path
from forest_app.front_end.auth_ui import display_auth_sidebar # <<< Correct path
from forest_app.front_end.onboarding_ui import display_onboarding_input # <<< Correct path

import streamlit as st
import json
import uuid
from datetime import datetime
import logging
from typing import Dict, List, Union, Optional, Any, Callable # Added Callable
import graphviz


# Assuming constants are defined in a backend config or a separate constants file
class constants: # Placeholder class if not importing from backend
    ONBOARDING_STATUS_NEEDS_GOAL = "needs_goal"
    ONBOARDING_STATUS_NEEDS_CONTEXT = "needs_context"
    ONBOARDING_STATUS_COMPLETED = "completed"
    MIN_PASSWORD_LENGTH = 8

# --- Configuration ---
BACKEND_URL = st.secrets.get("BACKEND_URL", "http://localhost:8000")
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Constants ---
# (Consider moving these to state_keys.py later)
KEY_STATUS_CODE = "status_code"; KEY_ERROR = "error"; KEY_DETAIL = "detail"; KEY_DATA = "data"
KEY_ACCESS_TOKEN = "access_token"; KEY_ONBOARDING_STATUS = "onboarding_status"
KEY_USER_INFO_EMAIL = "email"; KEY_USER_INFO_ID = "id"; KEY_SNAPSHOT_ID = "id"
KEY_SNAPSHOT_UPDATED_AT = "updated_at"; KEY_SNAPSHOT_CODENAME = "codename"; KEY_MESSAGES = "messages"
KEY_CURRENT_TASK = "current_task"; KEY_HTA_STATE = "hta_state"; KEY_PENDING_CONFIRMATION = "pending_confirmation"
KEY_MILESTONES = "milestones_achieved"; KEY_TASK_TITLE = "title"; KEY_TASK_DESC = "description"
KEY_TASK_MAGNITUDE_DESC = "magnitude_description"; KEY_TASK_INTRO_PROMPT = "introspective_prompt"
KEY_COMMAND_RESPONSE = "arbiter_response"; KEY_COMMAND_OFFERING = "offering"; KEY_COMMAND_MASTERY = "mastery_challenge"
KEY_ERROR_MESSAGE = "error_message";

# --- HTA Node Status Constants ---
STATUS_PENDING = "pending"; STATUS_ACTIVE = "active"; STATUS_COMPLETED = "completed"; STATUS_PRUNED = "pruned"; STATUS_BLOCKED = "blocked";

# --- HTA Fetching Helper ---
def fetch_hta_state():
    """Fetches HTA state from the backend, handles errors, and updates session state."""
    logger.info("Attempting to fetch HTA state...")
    st.session_state[KEY_ERROR_MESSAGE] = None
    hta_response = call_forest_api( # Uses imported function
        endpoint="/hta/state",
        method="GET",
        backend_url=BACKEND_URL,
        api_token=st.session_state.get("token")
    )
    status_code = hta_response.get(KEY_STATUS_CODE)
    error_msg = hta_response.get(KEY_ERROR)
    hta_data = hta_response.get(KEY_DATA)
    if error_msg:
        logger.error(f"Failed fetch HTA: {error_msg} (Status: {status_code})")
        st.session_state[KEY_ERROR_MESSAGE] = f"API Error fetching HTA: {error_msg}"
        st.session_state[KEY_HTA_STATE] = None
    elif status_code == 200:
        if isinstance(hta_data, dict) and 'hta_tree' in hta_data:
            hta_tree_content = hta_data.get('hta_tree')
            if isinstance(hta_tree_content, dict):
                st.session_state[KEY_HTA_STATE] = hta_tree_content
                logger.info("Fetched and stored HTA state.")
            elif hta_tree_content is None:
                st.session_state[KEY_HTA_STATE] = None
                logger.info("Backend indicated no HTA state.")
            else:
                logger.warning(f"HTA 'hta_tree' has unexpected type: {type(hta_tree_content)}")
                st.session_state[KEY_HTA_STATE] = None
                st.session_state[KEY_ERROR_MESSAGE] = "Unexpected HTA format."
        else:
            logger.warning(f"HTA endpoint gave unexpected structure: {type(hta_data)}")
            st.session_state[KEY_HTA_STATE] = None
            st.session_state[KEY_ERROR_MESSAGE] = "Unexpected HTA structure."
    elif status_code == 404:
        st.session_state[KEY_HTA_STATE] = None
        logger.info("Backend returned 404 for HTA state.")
    else:
        logger.error(f"Failed fetch HTA: Status {status_code}. Data: {str(hta_data)[:200]}")
        st.session_state[KEY_ERROR_MESSAGE] = f"Unexpected API status ({status_code}) fetching HTA."
        st.session_state[KEY_HTA_STATE] = None


# --- HTA Visualization Helpers ---
STATUS_COLORS = { STATUS_PENDING: "#E0E0E0", STATUS_ACTIVE: "#ADD8E6", STATUS_COMPLETED: "#90EE90", STATUS_PRUNED: "#A9A9A9", STATUS_BLOCKED: "#FFCCCB", "default": "#FFFFFF" }
def build_hta_dot_string(node_data: Dict[str, Any], dot: graphviz.Digraph):
    node_id = node_data.get("id")
    if not node_id: logger.warning("Skip HTA node missing 'id'."); return
    node_title = node_data.get("title", "Untitled"); node_status = str(node_data.get("status", "default")).lower()
    node_color = STATUS_COLORS.get(node_status, STATUS_COLORS["default"]); node_label = f"{node_title}\n(Status: {node_status.capitalize()})"
    dot.node(name=str(node_id), label=node_label, shape="box", style="filled", fillcolor=node_color)
    children = node_data.get("children", [])
    if isinstance(children, list):
        for child_data in children:
            if isinstance(child_data, dict):
                child_id = child_data.get("id")
                if child_id: dot.edge(str(node_id), str(child_id)); build_hta_dot_string(child_data, dot)
            else: logger.warning(f"Skip non-dict child under node {node_id}.")

def display_hta_visualization(hta_tree_root: Optional[Dict]):
    if not hta_tree_root or not isinstance(hta_tree_root, dict):
        st.info("🌱 Your skill tree (HTA) is being cultivated..."); return
    try:
        dot = graphviz.Digraph(comment='HTA Skill Tree'); dot.attr(rankdir='TB')
        build_hta_dot_string(hta_tree_root, dot)
        st.graphviz_chart(dot); st.caption("Current Skill Tree Visualization")
    except Exception as e: logger.exception("HTA viz render exception!"); st.error(f"Error generating HTA viz: {e}")


# --- Completion Confirmation Helper ---
def handle_completion_confirmation():
    pending_conf = st.session_state.get(KEY_PENDING_CONFIRMATION)
    if not isinstance(pending_conf, dict): return
    prompt_text = pending_conf.get("prompt", "Confirm?"); node_id_to_confirm = pending_conf.get("hta_node_id")
    if not node_id_to_confirm: logger.error("Confirm missing 'hta_node_id'."); st.error("Confirm prompt missing ID."); st.session_state[KEY_PENDING_CONFIRMATION] = None; st.rerun(); return
    st.info(f"**Confirmation Needed:** {prompt_text}"); col_confirm, col_deny = st.columns(2)
    with col_confirm:
        if st.button("✅ Yes", key=f"confirm_yes_{node_id_to_confirm}"):
            st.session_state[KEY_ERROR_MESSAGE] = None; logger.info(f"User confirm node: {node_id_to_confirm}")
            payload = {"task_id": node_id_to_confirm, "success": True}
            with st.spinner("Confirming..."):
                response = call_forest_api("/core/complete_task", method="POST", data=payload, backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
            if response.get(KEY_ERROR): error_msg = response.get(KEY_ERROR, "Unknown"); logger.error(f"API error confirm task {node_id_to_confirm}: {error_msg}"); st.error(f"Confirm Fail: {error_msg}"); st.session_state[KEY_ERROR_MESSAGE] = f"API Error: {error_msg}"
            elif response.get(KEY_STATUS_CODE) == 200:
                st.success("Confirmed!"); logger.info(f"Success confirm node {node_id_to_confirm}."); st.session_state[KEY_PENDING_CONFIRMATION] = None
                resp_data = response.get(KEY_DATA, {});
                if isinstance(resp_data, dict):
                    completion_message = resp_data.get("detail", resp_data.get("message"))
                    if completion_message:
                        if not isinstance(st.session_state.get(KEY_MESSAGES), list): st.session_state[KEY_MESSAGES] = []
                        st.session_state.messages.append({"role": "assistant", "content": str(completion_message)})
                    challenge_data = resp_data.get("result", {}).get(KEY_COMMAND_MASTERY)
                    if isinstance(challenge_data, dict):
                        challenge_content = challenge_data.get("challenge_content", "Consider next steps."); challenge_type = challenge_data.get("challenge_type", "Reflect")
                        logger.info(f"Mastery challenge ({challenge_type}) received.");
                        if not isinstance(st.session_state.get(KEY_MESSAGES), list): st.session_state[KEY_MESSAGES] = []
                        st.session_state.messages.append({"role": "assistant", "content": f"✨ Mastery Challenge ({challenge_type}):\n{challenge_content}"})
                fetch_hta_state(); st.rerun()
            else: logger.error(f"Unexpected status {response.get(KEY_STATUS_CODE)} confirm task {node_id_to_confirm}."); st.error(f"Unexpected confirm status ({response.get(KEY_STATUS_CODE)})."); st.session_state[KEY_ERROR_MESSAGE] = f"API Error: Confirm status {response.get(KEY_STATUS_CODE)}"
    with col_deny:
        if st.button("❌ No", key=f"confirm_no_{node_id_to_confirm}"):
            st.session_state[KEY_ERROR_MESSAGE] = None; logger.info(f"User denied node: {node_id_to_confirm}")
            st.info("Okay, task not marked complete."); st.session_state[KEY_PENDING_CONFIRMATION] = None
            if not isinstance(st.session_state.get(KEY_MESSAGES), list): st.session_state[KEY_MESSAGES] = []
            st.session_state.messages.append({"role": "assistant", "content": "Okay, let me know when you're ready."}); st.rerun()


# --- REMOVED Onboarding Handler Function Definitions ---
# handle_set_goal and handle_add_context moved to onboarding_ui.py


# --- Streamlit App Layout ---
st.set_page_config(page_title="Forest OS", layout="wide", initial_sidebar_state="expanded")
st.title("🌳 Forest OS - Your Growth Companion")

# --- Initialize Session State ---
st.session_state.setdefault("authenticated", False); st.session_state.setdefault("token", None)
st.session_state.setdefault("user_info", None); st.session_state.setdefault(KEY_MESSAGES, [])
st.session_state.setdefault(KEY_CURRENT_TASK, None); st.session_state.setdefault(KEY_ONBOARDING_STATUS, None)
st.session_state.setdefault("snapshots", []); st.session_state.setdefault(KEY_ERROR_MESSAGE, None)
st.session_state.setdefault(KEY_HTA_STATE, None); st.session_state.setdefault(KEY_PENDING_CONFIRMATION, None)
st.session_state.setdefault(KEY_MILESTONES, [])

# --- Authentication UI (Sidebar) ---
with st.sidebar:
    # Call the display function from the auth_ui module
    auth_action_taken = display_auth_sidebar(backend_url=BACKEND_URL)

    # --- Snapshot Management (Keep here for now, refactor later) ---
    st.divider()
    if st.session_state.get("authenticated"):
        with st.expander("Snapshot Management", expanded=False):
            # (Snapshot code remains here for now, uses call_forest_api)
            st.info("Snapshots allow saving and loading session states (experimental).")
            if st.button("Refresh Snapshot List"):
                st.session_state[KEY_ERROR_MESSAGE] = None; st.session_state.snapshots = []
                with st.spinner("Fetching snapshot list..."):
                    response = call_forest_api("/snapshots/list", method="GET", backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
                    if response.get(KEY_ERROR): st.error(f"Fetch Snap Fail: {response.get(KEY_ERROR)}"); st.session_state[KEY_ERROR_MESSAGE] = f"API Error: {response.get(KEY_ERROR)}"
                    elif response.get(KEY_STATUS_CODE) == 200 and isinstance(response.get(KEY_DATA), list):
                        snapshot_list_data = response.get(KEY_DATA, [])
                        valid_snapshots = [item for item in snapshot_list_data if isinstance(item, dict) and KEY_SNAPSHOT_ID in item and KEY_SNAPSHOT_UPDATED_AT in item]
                        try:
                            st.session_state.snapshots = sorted(valid_snapshots, key=lambda x: datetime.fromisoformat(str(x[KEY_SNAPSHOT_UPDATED_AT]).replace('Z', '+00:00')), reverse=True)
                            if not st.session_state.snapshots and snapshot_list_data: st.warning("Snapshot list format unexpected.")
                            elif not snapshot_list_data: st.info("No snapshots found.")
                        except Exception as sort_e: logger.error(f"Snapshot sort fail: {sort_e}"); st.session_state.snapshots = valid_snapshots; st.warning("Snapshots loaded, sort fail.")
                    else: status = response.get(KEY_STATUS_CODE, 'N/A'); data_type = type(response.get(KEY_DATA)).__name__; st.error(f"Fetch Snap Fail: Status {status}, Format: {data_type}"); st.session_state[KEY_ERROR_MESSAGE] = f"Snapshot API Error: Status {status}, Format {data_type}"

            if st.session_state.get("snapshots"):
                snapshot_options = {}
                for s in st.session_state.snapshots:
                    snap_id = s.get(KEY_SNAPSHOT_ID);
                    if not snap_id: continue
                    codename = s.get(KEY_SNAPSHOT_CODENAME, 'Untitled'); updated_at_raw = s.get(KEY_SNAPSHOT_UPDATED_AT); dt_str = 'Date N/A'
                    if updated_at_raw:
                        try: dt_obj = datetime.fromisoformat(str(updated_at_raw).replace('Z', '+00:00')); dt_str = dt_obj.strftime('%Y-%m-%d %H:%M UTC')
                        except (ValueError, TypeError): logger.warning(f"Parse date fail: {updated_at_raw}")
                    display_key = f"'{codename}' ({dt_str}) - ID: ...{str(snap_id)[-6:]}"; snapshot_options[display_key] = snap_id
                if snapshot_options:
                    selected_snapshot_display = st.selectbox("Select Snapshot:", options=list(snapshot_options.keys()), key="snap_select", index=0)
                    snapshot_id_selected = snapshot_options.get(selected_snapshot_display)
                    col_load, col_delete = st.columns(2)
                    with col_load:
                        if st.button("Load Selected", key="load_snap", help="Load session state"):
                            st.session_state[KEY_ERROR_MESSAGE] = None
                            if snapshot_id_selected:
                                with st.spinner(f"Loading '{selected_snapshot_display}'..."):
                                    load_payload = {"snapshot_id": snapshot_id_selected}
                                    response = call_forest_api("/core/session/load", method="POST", data=load_payload, backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
                                    if response.get(KEY_ERROR): st.error(f"Load Fail: {response.get(KEY_ERROR)}"); st.session_state[KEY_ERROR_MESSAGE] = f"API Error: {response.get(KEY_ERROR)}"
                                    elif response.get(KEY_STATUS_CODE) == 200 and isinstance(response.get(KEY_DATA), dict):
                                        load_data = response.get(KEY_DATA, {}); st.success(load_data.get("message", "Loaded!"))
                                        logger.info(f"Snapshot {snapshot_id_selected} loaded."); st.session_state[KEY_MESSAGES] = load_data.get(KEY_MESSAGES, []) if isinstance(load_data.get(KEY_MESSAGES), list) else []
                                        st.session_state[KEY_CURRENT_TASK] = load_data.get(KEY_CURRENT_TASK) if isinstance(load_data.get(KEY_CURRENT_TASK), dict) else None
                                        st.session_state[KEY_MILESTONES] = load_data.get(KEY_MILESTONES, []) if isinstance(load_data.get(KEY_MILESTONES), list) else []
                                        st.session_state[KEY_ERROR_MESSAGE] = None; st.session_state[KEY_ONBOARDING_STATUS] = constants.ONBOARDING_STATUS_COMPLETED
                                        st.session_state[KEY_HTA_STATE] = None; st.session_state[KEY_PENDING_CONFIRMATION] = None
                                        fetch_hta_state(); st.rerun()
                                    else: status = response.get(KEY_STATUS_CODE, 'N/A'); st.error(f"Load Fail: Status {status}"); st.session_state[KEY_ERROR_MESSAGE] = f"Snapshot Load API Error: Status {status}"
                            else: st.warning("No snapshot selected.")
                    with col_delete:
                        if st.button("Delete Selected", type="secondary", key="delete_snap", help="Permanently delete"):
                            st.session_state[KEY_ERROR_MESSAGE] = None
                            if snapshot_id_selected:
                                with st.spinner(f"Deleting '{selected_snapshot_display}'..."):
                                    delete_endpoint = f"/snapshots/{snapshot_id_selected}"
                                    response = call_forest_api(delete_endpoint, method="DELETE", backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
                                    if response.get(KEY_ERROR): st.error(f"Delete Fail: {response.get(KEY_ERROR)}"); st.session_state[KEY_ERROR_MESSAGE] = f"API Error: {response.get(KEY_ERROR)}"
                                    elif response.get(KEY_STATUS_CODE) in [200, 204]:
                                        st.success("Deleted."); logger.info(f"Snapshot {snapshot_id_selected} deleted.")
                                        if isinstance(st.session_state.get("snapshots"), list): st.session_state.snapshots = [s for s in st.session_state.snapshots if s.get(KEY_SNAPSHOT_ID) != snapshot_id_selected]
                                        st.rerun()
                                    else: status = response.get(KEY_STATUS_CODE, 'N/A'); st.error(f"Delete Fail: Status {status}"); st.session_state[KEY_ERROR_MESSAGE] = f"Snapshot Delete API Error: Status {status}"
                            else: st.warning("No snapshot selected.")
                else:
                    if "snapshots" in st.session_state and not st.session_state.snapshots: pass
                    else: st.caption("Click 'Refresh' to load snapshots.")


    # Display global errors
    global_error = st.session_state.get(KEY_ERROR_MESSAGE)
    if global_error:
        st.sidebar.error(f"⚠️ Error: {global_error}")

# --- Post-Auth Action Handling ---
if auth_action_taken:
    if st.session_state.get("authenticated") and \
       st.session_state.get(KEY_ONBOARDING_STATUS) == constants.ONBOARDING_STATUS_COMPLETED and \
       not st.session_state.get(KEY_HTA_STATE):
           logger.info("Login/Register ok, onboarding complete, fetching initial HTA...")
           fetch_hta_state()
    st.rerun()

# --- Main Application Area ---
if not st.session_state.get("authenticated"):
    st.warning("Please log in or register using the sidebar to begin.")
    st.image("https://placehold.co/800x400/334455/FFFFFF?text=Welcome+to+Forest+OS", caption="Visualize your growth journey")
else:
    # Check/Fetch User Status if missing
    if st.session_state.get(KEY_ONBOARDING_STATUS) is None and st.session_state.get("token"):
         logger.warning("Onboarding status missing, refreshing...")
         with st.spinner("Checking status..."):
            user_details_response = call_forest_api("/users/me", method="GET", backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
            if user_details_response.get(KEY_ERROR) or not isinstance(user_details_response.get(KEY_DATA), dict):
                logger.error("Failed refresh user status."); st.error("Status retrieval failed. Try logout/login."); st.session_state[KEY_ONBOARDING_STATUS] = "error"
            else:
                user_data = user_details_response[KEY_DATA]; st.session_state.user_info = user_data
                user_onboarding_status = user_data.get(KEY_ONBOARDING_STATUS); valid_statuses = [constants.ONBOARDING_STATUS_NEEDS_GOAL, constants.ONBOARDING_STATUS_NEEDS_CONTEXT, constants.ONBOARDING_STATUS_COMPLETED]
                if user_onboarding_status in valid_statuses:
                    st.session_state[KEY_ONBOARDING_STATUS] = user_onboarding_status; logger.info(f"Refreshed status: {st.session_state[KEY_ONBOARDING_STATUS]}")
                    if st.session_state[KEY_ONBOARDING_STATUS] == constants.ONBOARDING_STATUS_COMPLETED and not st.session_state.get(KEY_HTA_STATE): fetch_hta_state()
                    st.rerun()
                else: logger.error(f"Invalid status received: {user_onboarding_status}"); st.error("Invalid status from backend."); st.session_state[KEY_ONBOARDING_STATUS] = "error"

    # Main Content Area Layout
    col_hta, col_chat = st.columns([1, 1])

    # HTA Visualization Column
    with col_hta:
        st.header("Skill Tree (HTA)"); hta_viz_enabled = True; onboarding_status = st.session_state.get(KEY_ONBOARDING_STATUS)
        if onboarding_status == constants.ONBOARDING_STATUS_COMPLETED and hta_viz_enabled:
            if st.button("🔄 Refresh Skill Tree", key="refresh_hta"):
                with st.spinner("Refreshing..."): fetch_hta_state(); st.rerun()
            hta_data_to_display = st.session_state.get(KEY_HTA_STATE)
            display_hta_visualization(hta_data_to_display)
        elif not hta_viz_enabled: st.info("Viz disabled.")
        elif onboarding_status in [constants.ONBOARDING_STATUS_NEEDS_GOAL, constants.ONBOARDING_STATUS_NEEDS_CONTEXT]: st.info("Complete onboarding to view Skill Tree.")
        elif onboarding_status == "error": st.warning("Cannot display Skill Tree due to status error.")

    # Chat / Interaction Column
    with col_chat:
        st.header("Conversation & Actions")
        # Display Chat History
        messages_list = st.session_state.get(KEY_MESSAGES, [])
        if isinstance(messages_list, list):
            for message in messages_list:
                if isinstance(message, dict) and "role" in message and "content" in message:
                    with st.chat_message(message["role"]): st.markdown(str(message["content"]))
                else: logger.warning(f"Skip invalid message: {message}")
        elif messages_list: logger.error(f"Chat history not list: {type(messages_list)}"); st.error("Chat history corrupt.")

        # Handle Pending Confirmation
        handle_completion_confirmation()

        # Chat Input Logic
        current_status = st.session_state.get(KEY_ONBOARDING_STATUS)
        chat_disabled = st.session_state.get(KEY_PENDING_CONFIRMATION) is not None
        onboarding_action_taken = False

        # --- Onboarding Input (Uses imported function) --- ### MODIFIED ###
        if current_status in [constants.ONBOARDING_STATUS_NEEDS_GOAL, constants.ONBOARDING_STATUS_NEEDS_CONTEXT]:
            onboarding_action_taken = display_onboarding_input(
                current_status=current_status,
                backend_url=BACKEND_URL,
                fetch_hta_state_func=fetch_hta_state # Pass the HTA fetch function
            )
            if onboarding_action_taken:
                st.rerun() # Rerun if onboarding step was processed

        # --- Main Interaction Input ---
        elif current_status == constants.ONBOARDING_STATUS_COMPLETED:
            input_placeholder = "Enter reflection, command..."
            if prompt := st.chat_input(input_placeholder, disabled=chat_disabled, key="main_chat"):
                st.session_state[KEY_ERROR_MESSAGE] = None
                if not isinstance(st.session_state.get(KEY_MESSAGES), list): st.session_state[KEY_MESSAGES] = []
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.markdown(prompt)
                with st.chat_message("assistant"):
                    message_placeholder = st.empty(); message_placeholder.markdown("🌳 Thinking...")
                    api_endpoint = "/core/command"; payload = {"command": prompt}
                    response = call_forest_api(api_endpoint, method="POST", data=payload, backend_url=BACKEND_URL, api_token=st.session_state.get("token"))
                    # ... (process /core/command response logic - remains the same) ...
                    assistant_response_content = ""; confirmation_details = None # Default values
                    if response.get(KEY_ERROR):
                        error_msg = response.get(KEY_ERROR, "Command failed"); logger.warning(f"Error {api_endpoint}: {error_msg} (Status: {response.get(KEY_STATUS_CODE)})")
                        assistant_response_content = f"⚠️ Error: {error_msg}"
                        if response.get(KEY_STATUS_CODE) == 403: assistant_response_content += "\nSession issue? Refreshing status..."; st.session_state[KEY_ONBOARDING_STATUS] = None
                        st.session_state[KEY_ERROR_MESSAGE] = assistant_response_content
                    elif response.get(KEY_STATUS_CODE) in [200, 201] and isinstance(response.get(KEY_DATA), dict):
                        resp_data = response.get(KEY_DATA, {}); logger.info(f"Success from {api_endpoint}.")
                        assistant_response_content = resp_data.get(KEY_COMMAND_RESPONSE, resp_data.get("message", ""))
                        new_task_data = resp_data.get(KEY_CURRENT_TASK); st.session_state[KEY_CURRENT_TASK] = new_task_data if isinstance(new_task_data, dict) else None
                        action_required = resp_data.get("action_required"); confirmation_details = resp_data.get("confirmation_details")
                        milestone_feedback = resp_data.get("milestone_feedback")
                        if milestone_feedback: logger.info(f"Milestone: {milestone_feedback}");
                        if not isinstance(st.session_state.get(KEY_MILESTONES), list): st.session_state[KEY_MILESTONES] = []
                        st.session_state.milestones_achieved.append(str(milestone_feedback)); assistant_response_content += f"\n\n🎉 *Milestone: {milestone_feedback}*"
                        if action_required == "confirm_completion" and isinstance(confirmation_details, dict):
                            logger.info(f"Confirm requested: {confirmation_details.get('hta_node_id')}")
                            st.session_state[KEY_PENDING_CONFIRMATION] = confirmation_details
                            if not assistant_response_content: assistant_response_content = confirmation_details.get("prompt", "Confirm?")
                        else:
                            if st.session_state.get(KEY_PENDING_CONFIRMATION): logger.debug("Clearing stale confirm."); st.session_state[KEY_PENDING_CONFIRMATION] = None
                        if not assistant_response_content: assistant_response_content = "Okay, processed."
                        if new_task_data or milestone_feedback: fetch_hta_state()
                    else:
                        status_code = response.get(KEY_STATUS_CODE, "N/A"); data_type = type(response.get(KEY_DATA)).__name__
                        logger.error(f"Unexpected response {api_endpoint}: Status {status_code}, Type: {data_type}")
                        assistant_response_content = f"Unexpected server response (Status: {status_code})."
                        st.session_state[KEY_ERROR_MESSAGE] = assistant_response_content

                    message_placeholder.markdown(assistant_response_content)
                    if assistant_response_content:
                        if not isinstance(st.session_state.get(KEY_MESSAGES), list): st.session_state[KEY_MESSAGES] = []
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response_content})
                    st.rerun()

        # --- Handle Other/Error States ---
        elif current_status is None: st.info("Checking status...")
        elif current_status == "error": st.error("Status error. Try logout/login.")
        else: st.warning(f"Unknown state ('{current_status}'). Try logout/login.")

# --- End of App ---
