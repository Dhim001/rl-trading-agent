from __future__ import annotations

import streamlit as st


def ensure_state_defaults() -> None:
    if "refresh_enabled" not in st.session_state:
        st.session_state["refresh_enabled"] = False
    if "refresh_seconds" not in st.session_state:
        st.session_state["refresh_seconds"] = 15
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "Overview"
    if "operation_run_mode" not in st.session_state:
        st.session_state["operation_run_mode"] = "background"
    if "artifact_watch_enabled" not in st.session_state:
        st.session_state["artifact_watch_enabled"] = True
    if "artifact_snapshot" not in st.session_state:
        st.session_state["artifact_snapshot"] = {}
    if "notifications" not in st.session_state:
        st.session_state["notifications"] = []
    if "job_status_cache" not in st.session_state:
        st.session_state["job_status_cache"] = {}
