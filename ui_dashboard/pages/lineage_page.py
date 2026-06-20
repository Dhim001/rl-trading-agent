from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui_dashboard.components import ui as ui_components
from ui_dashboard.services.job_service import JobService
from ui_dashboard.services.lineage_service import (
    build_dependency_graph_dot,
    build_provenance,
    diff_dicts,
    list_trial_param_files,
    load_trial_params,
)


def _render_banner(level: str, message: str) -> None:
    renderer = getattr(ui_components, "render_banner", None)
    if callable(renderer):
        renderer(level, message)
        return
    # Safe fallback if an old module instance lacks render_banner.
    if level.lower() == "error":
        st.error(message)
    elif level.lower() == "warning":
        st.warning(message)
    elif level.lower() == "success":
        st.success(message)
    else:
        st.info(message)


def render(project_root: Path, job_service: JobService) -> None:
    st.subheader("Artifact Lineage and Rollback")
    st.caption("Trace data-to-model-to-backtest relationships, compare trial parameter deltas, and restore prior model versions safely.")

    provenance_df = build_provenance(project_root)
    if provenance_df.empty:
        ui_components.render_empty_state(
            "No lineage rows available yet.",
            "Run training/backtest/fine-tuning workflows to populate artifact relationships.",
        )
    else:
        st.markdown("**Data Provenance**")
        st.dataframe(provenance_df, width="stretch", hide_index=True)

        st.markdown("**Dependency Graph**")
        st.graphviz_chart(build_dependency_graph_dot(provenance_df), width="stretch")

    st.markdown("---")
    st.markdown("**Tuning Trial Parameter Diff Viewer**")
    trials = list_trial_param_files(project_root)
    if len(trials) < 2:
        ui_components.render_empty_state(
            "Need at least two trial parameter files.",
            "Re-run fine-tuning so each trial writes `tuning/trial_x/params.json`.",
        )
    else:
        names = sorted(trials.keys())
        col1, col2 = st.columns(2)
        left_trial = col1.selectbox("Left trial", names, index=0, key="lineage_diff_left")
        right_trial = col2.selectbox("Right trial", names, index=min(1, len(names) - 1), key="lineage_diff_right")
        if left_trial == right_trial:
            st.caption("Pick two different trials to compare.")
        else:
            left_params = load_trial_params(project_root, left_trial) or {}
            right_params = load_trial_params(project_root, right_trial) or {}
            diff_df = diff_dicts(left_params, right_params)
            st.dataframe(diff_df, width="stretch", hide_index=True)
            changed_count = int(diff_df["changed"].sum()) if "changed" in diff_df.columns else 0
            st.caption(f"Changed parameters: {changed_count}")

    st.markdown("---")
    st.markdown("**Rollback Model Version**")
    versions = job_service.list_model_versions()
    if not versions:
        ui_components.render_empty_state("No model versions found.")
        return
    options = [v["path"] for v in versions]
    selected = st.selectbox("Select model artifact to activate as final model", options, key="lineage_rollback_target")
    if st.button("Rollback to Selected Model", type="secondary", key="lineage_rollback_btn"):
        result = job_service.rollback_model_version(selected)
        if result.get("ok"):
            _render_banner("success", f"Rollback complete. Active model set to `{result.get('active_model')}`.")
        else:
            _render_banner("error", f"Rollback failed: {result.get('error', 'unknown error')}")


if __name__ == "__main__":
    from ui_dashboard.core.config import load_config

    cfg = load_config()
    render(cfg.project_root, JobService(cfg))
