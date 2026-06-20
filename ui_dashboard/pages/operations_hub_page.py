from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ui_dashboard.components import ui as ui_components
from ui_dashboard.services.job_service import JobService, WORKFLOW_LABELS


def _render_banner(level: str, message: str) -> None:
    renderer = getattr(ui_components, "render_banner", None)
    if callable(renderer):
        renderer(level, message)
        return
    if level.lower() == "error":
        st.error(message)
    elif level.lower() == "warning":
        st.warning(message)
    elif level.lower() == "success":
        st.success(message)
    else:
        st.info(message)


def _state_defaults() -> None:
    if "op_selected_workflow" not in st.session_state:
        st.session_state["op_selected_workflow"] = "training"
    if "op_selected_job_id" not in st.session_state:
        st.session_state["op_selected_job_id"] = ""
    if "op_poll_seconds" not in st.session_state:
        st.session_state["op_poll_seconds"] = 5
    if "op_auto_poll" not in st.session_state:
        st.session_state["op_auto_poll"] = True
    if "op_last_launch_error" not in st.session_state:
        st.session_state["op_last_launch_error"] = ""


def _handle_launch(job_service: JobService, workflow: str, run_mode: str) -> None:
    try:
        record = job_service.launch_workflow(
            workflow=workflow,
            run_mode=run_mode,
            metadata={"triggered_from": "operations_hub"},
        )
        st.session_state["op_last_launch_error"] = ""
        st.toast(f"{WORKFLOW_LABELS[workflow]} started", icon="✅")
        st.session_state["op_selected_job_id"] = str(record.get("id", ""))
        _render_banner("success", f"{WORKFLOW_LABELS[workflow]} started: `{record.get('id', 'unknown')}`")
    except Exception as exc:
        st.session_state["op_last_launch_error"] = str(exc)
        ui_components.render_section_error("Failed to start workflow.", str(exc))


def _refresh_jobs(job_service: JobService) -> list[dict[str, Any]]:
    try:
        return job_service.list_jobs()
    except Exception as exc:
        ui_components.render_section_error("Could not refresh jobs.", str(exc))
        return []


def _workflow_card(job_service: JobService, workflow: dict[str, Any], run_mode: str, disabled: bool) -> None:
    workflow_id = str(workflow["id"])
    st.markdown(f"**{workflow['label']}**")
    artifact_summary = job_service.validate_workflow_artifacts(workflow_id)
    if artifact_summary["all_present"]:
        st.caption("Artifacts: available")
    else:
        st.caption("Artifacts: partial/missing")
    if st.button(
        f"Run {workflow['label']}",
        key=f"op_run_{workflow_id}",
        width="stretch",
        disabled=disabled,
    ):
        _handle_launch(job_service, workflow_id, run_mode)


def _render_controls(job_service: JobService, active_jobs: list[dict[str, Any]]) -> None:
    run_mode_choice = st.radio(
        "Execution mode",
        options=["Background", "Foreground"],
        index=0,
        horizontal=True,
        help="Foreground waits and captures immediate output. Background returns immediately and tracks job status.",
        key="operations_run_mode",
    )
    run_mode = "background" if run_mode_choice == "Background" else "foreground"

    has_foreground_blocker = run_mode == "foreground" and any(
        str(job.get("status", "")) in {"running", "queued"} for job in active_jobs
    )
    if has_foreground_blocker:
        _render_banner("warning", "Foreground mode is blocked while another job is active. Stop running jobs or switch to background mode.")

    workflows = job_service.get_workflow_catalog()
    rows = [workflows[i : i + 3] for i in range(0, len(workflows), 3)]
    for row in rows:
        cols = st.columns(3)
        for idx, workflow in enumerate(row):
            with cols[idx]:
                with st.container(border=True):
                    _workflow_card(job_service, workflow, run_mode, has_foreground_blocker)


def _render_jobs_table(job_service: JobService, jobs: list[dict[str, Any]]) -> None:
    if not jobs:
        ui_components.render_empty_state("No jobs launched yet.", "Use the controls above to start trading, training, or tuning workflows.")
        return

    jobs_df = pd.DataFrame(jobs).copy()
    jobs_df["status"] = jobs_df.get("status", "unknown")
    jobs_df["workflow"] = jobs_df.get("workflow", "unknown")

    c1, c2, c3 = st.columns(3)
    c1.metric("Active Jobs", int((jobs_df["status"].isin(["running", "queued"])).sum()))
    c2.metric("Completed Jobs", int((jobs_df["status"] == "completed").sum()))
    c3.metric("Failed Jobs", int((jobs_df["status"] == "failed").sum()))

    filter_col, _ = st.columns([1, 3])
    with filter_col:
        selected_status = st.selectbox(
            "Filter status",
            options=["all", "running", "queued", "completed", "failed", "stopped"],
            index=0,
            key="op_status_filter",
        )
    if selected_status != "all":
        jobs_df = jobs_df[jobs_df["status"] == selected_status]

    preferred_cols = [
        "id",
        "workflow",
        "status",
        "pid",
        "started_at",
        "ended_at",
        "return_code",
        "source",
    ]
    cols = [c for c in preferred_cols if c in jobs_df.columns]
    if jobs_df.empty:
        ui_components.render_empty_state("No jobs match the selected filter.")
    else:
        st.dataframe(jobs_df[cols], width="stretch", hide_index=True)

    running_df = jobs_df[jobs_df["status"].isin(["running", "queued"])] if "status" in jobs_df.columns else pd.DataFrame()
    if not running_df.empty:
        choices = [f"{row['id']} | {row['workflow']} | {row.get('status', '')}" for _, row in running_df.iterrows()]
        selected = st.selectbox("Stop running job", choices, key="op_stop_select")
        if st.button("Stop Selected Job", type="secondary", key="op_stop_btn"):
            job_id = selected.split("|")[0].strip()
            result = job_service.stop_job(job_id)
            if result.get("ok"):
                st.success(f"Stopped job `{job_id}`.")
            else:
                st.error(f"Could not stop `{job_id}`: {result.get('error', 'unknown error')}")
    else:
        st.caption("No running jobs detected.")

    if jobs_df.empty:
        return

    selected_log_job = st.selectbox(
        "View job log tail",
        jobs_df["id"].tolist(),
        key="op_log_select",
        index=0,
    )
    st.session_state["op_selected_job_id"] = selected_log_job
    if selected_log_job:
        try:
            tail = job_service.get_job_log_tail(selected_log_job, lines=80)
            if tail:
                st.code(tail, language="text")
            else:
                st.caption("No logs available yet.")
        except Exception as exc:
            ui_components.render_section_error("Could not load job logs.", str(exc))

    selected_job = next((job for job in jobs if str(job.get("id")) == selected_log_job), None)
    if selected_job:
        progress = job_service.get_job_progress(selected_job)
        st.markdown(f"**Progress: {progress['phase']}**")
        st.progress(int(progress["progress_pct"]))
        with st.expander("Validate expected artifacts"):
            summary = job_service.validate_workflow_artifacts(str(selected_job.get("workflow", "")))
            st.dataframe(pd.DataFrame(summary["artifacts"]), width="stretch", hide_index=True)
            if summary["all_present"]:
                _render_banner("success", "All expected artifacts are present.")
            else:
                _render_banner("warning", "One or more expected artifacts are missing or stale.")


def render(project_root: Path, job_service: JobService) -> None:
    _state_defaults()
    st.subheader("Operations Control Center")
    st.caption(
        "Launch, monitor, and control training, trading, and tuning workflows with near real-time status, logs, and artifact validation."
    )

    poll_col1, poll_col2 = st.columns([2, 3])
    with poll_col1:
        st.session_state["op_auto_poll"] = st.toggle("Auto poll jobs", value=st.session_state["op_auto_poll"])
    with poll_col2:
        st.session_state["op_poll_seconds"] = st.slider(
            "Poll interval (seconds)",
            min_value=2,
            max_value=30,
            value=int(st.session_state["op_poll_seconds"]),
        )

    if st.session_state["op_last_launch_error"]:
        _render_banner("error", st.session_state["op_last_launch_error"])

    jobs_snapshot = _refresh_jobs(job_service)
    _render_controls(job_service, jobs_snapshot)
    ui_components.render_retry_hint("Refresh Jobs")

    refresh_col, _ = st.columns([1, 4])
    with refresh_col:
        if st.button("Refresh Jobs", width="stretch", key="op_refresh"):
            st.rerun()

    # Use Streamlit fragment for efficient polling if available, fallback to normal render.
    if st.session_state["op_auto_poll"] and hasattr(st, "fragment"):
        @st.fragment(run_every=f"{int(st.session_state['op_poll_seconds'])}s")
        def _live_jobs_fragment() -> None:
            with st.status("Monitoring jobs...", expanded=False):
                jobs = _refresh_jobs(job_service)
                _render_jobs_table(job_service, jobs)

        _live_jobs_fragment()
    else:
        if st.session_state["op_auto_poll"]:
            st.caption("Auto polling fallback active. Use Refresh Jobs if updates lag.")
        jobs = _refresh_jobs(job_service)
        _render_jobs_table(job_service, jobs)


if __name__ == "__main__":
    from ui_dashboard.core.config import load_config

    cfg = load_config()
    render(cfg.project_root, JobService(cfg))
