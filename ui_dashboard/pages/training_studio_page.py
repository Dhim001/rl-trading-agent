from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import plotly.express as px
import streamlit as st

from ui_dashboard.components import ui as ui_components
from ui_dashboard.services.job_service import JobService
from ui_dashboard.services.training_studio_service import (
    TRAINING_PRESETS,
    apply_preset,
    build_training_run_table,
    collect_resource_snapshot,
    estimate_training_time,
    infer_step_rate_from_jobs,
    list_checkpoints,
    load_default_config,
    parse_training_progress,
    write_runtime_training_config,
)


def _state_defaults(base_cfg: dict[str, Any]) -> None:
    training_cfg = (base_cfg.get("training") or {}).copy()
    env_cfg = (base_cfg.get("environment") or {}).copy()
    st.session_state.setdefault("ts_step", 1)
    st.session_state.setdefault("ts_preset", "balanced")
    st.session_state.setdefault("ts_algorithm", training_cfg.get("algorithm", "PPO"))
    st.session_state.setdefault("ts_timesteps", int(training_cfg.get("total_timesteps", 200000)))
    st.session_state.setdefault("ts_train_split", float(training_cfg.get("train_split", 0.8)))
    st.session_state.setdefault("ts_window_size", int(env_cfg.get("window_size", 30)))
    st.session_state.setdefault("ts_learning_rate", float(training_cfg.get("learning_rate", 3e-4)))
    st.session_state.setdefault("ts_gamma", float(training_cfg.get("gamma", 0.99)))
    st.session_state.setdefault("ts_batch_size", int(training_cfg.get("batch_size", 64)))
    st.session_state.setdefault("ts_eval_freq", int(training_cfg.get("eval_freq", 10000)))
    st.session_state.setdefault("ts_save_freq", int(training_cfg.get("save_freq", 25000)))
    st.session_state.setdefault("ts_run_mode", "background")


def _render_wizard(base_cfg: dict[str, Any], project_root: Path, job_service: JobService) -> None:
    st.markdown("### Model Configuration Wizard")
    nav_l, nav_c, nav_r = st.columns([1, 2, 1])
    with nav_l:
        if st.button("Previous Step", key="ts_prev", disabled=st.session_state["ts_step"] <= 1):
            st.session_state["ts_step"] -= 1
    with nav_c:
        st.caption(f"Step {st.session_state['ts_step']} of 3")
    with nav_r:
        if st.button("Next Step", key="ts_next", disabled=st.session_state["ts_step"] >= 3):
            st.session_state["ts_step"] += 1

    step = int(st.session_state["ts_step"])
    if step == 1:
        c1, c2, c3 = st.columns(3)
        st.session_state["ts_algorithm"] = c1.selectbox("Algorithm", options=["PPO", "DQN"], index=0 if st.session_state["ts_algorithm"] == "PPO" else 1)
        st.session_state["ts_timesteps"] = c2.number_input("Total Timesteps", min_value=10000, step=10000, value=int(st.session_state["ts_timesteps"]))
        st.session_state["ts_window_size"] = c3.number_input("Window Size", min_value=10, step=5, value=int(st.session_state["ts_window_size"]))
        st.session_state["ts_train_split"] = st.slider("Train Split", min_value=0.5, max_value=0.95, step=0.01, value=float(st.session_state["ts_train_split"]))
    elif step == 2:
        preset = st.selectbox("Hyperparameter Preset", options=list(TRAINING_PRESETS.keys()), index=list(TRAINING_PRESETS.keys()).index(st.session_state["ts_preset"]))
        st.session_state["ts_preset"] = preset
        if st.button("Apply Preset", key="ts_apply_preset"):
            merged = apply_preset({}, preset)
            st.session_state["ts_learning_rate"] = float(merged["learning_rate"])
            st.session_state["ts_gamma"] = float(merged["gamma"])
            st.session_state["ts_batch_size"] = int(merged["batch_size"])
            st.session_state["ts_eval_freq"] = int(merged["eval_freq"])
            st.session_state["ts_save_freq"] = int(merged["save_freq"])
        col1, col2, col3 = st.columns(3)
        st.session_state["ts_learning_rate"] = col1.number_input("Learning Rate", min_value=0.00001, max_value=0.01, format="%.5f", value=float(st.session_state["ts_learning_rate"]))
        st.session_state["ts_gamma"] = col2.number_input("Gamma", min_value=0.90, max_value=0.9999, format="%.4f", value=float(st.session_state["ts_gamma"]))
        st.session_state["ts_batch_size"] = col3.selectbox("Batch Size", options=[32, 64, 128, 256], index=[32, 64, 128, 256].index(int(st.session_state["ts_batch_size"])) if int(st.session_state["ts_batch_size"]) in [32, 64, 128, 256] else 1)
        col4, col5 = st.columns(2)
        st.session_state["ts_eval_freq"] = col4.number_input("Eval Frequency", min_value=1000, step=1000, value=int(st.session_state["ts_eval_freq"]))
        st.session_state["ts_save_freq"] = col5.number_input("Checkpoint Save Frequency", min_value=1000, step=1000, value=int(st.session_state["ts_save_freq"]))
    else:
        effective_cfg = copy.deepcopy(base_cfg)
        effective_cfg.setdefault("training", {})
        effective_cfg.setdefault("environment", {})
        effective_cfg["training"].update(
            {
                "algorithm": st.session_state["ts_algorithm"],
                "total_timesteps": int(st.session_state["ts_timesteps"]),
                "learning_rate": float(st.session_state["ts_learning_rate"]),
                "gamma": float(st.session_state["ts_gamma"]),
                "batch_size": int(st.session_state["ts_batch_size"]),
                "train_split": float(st.session_state["ts_train_split"]),
                "eval_freq": int(st.session_state["ts_eval_freq"]),
                "save_freq": int(st.session_state["ts_save_freq"]),
            }
        )
        effective_cfg["environment"]["window_size"] = int(st.session_state["ts_window_size"])
        st.markdown("**Review Configuration**")
        st.json({"training": effective_cfg["training"], "environment": {"window_size": effective_cfg["environment"]["window_size"]}})

        jobs = job_service.list_jobs()
        inferred_rate = infer_step_rate_from_jobs(jobs)
        eta = estimate_training_time(int(st.session_state["ts_timesteps"]), inferred_rate)
        st.caption(f"Estimated Training Time: `{eta['eta_hms']}` at ~`{eta['steps_per_sec']:.0f}` steps/sec")
        st.session_state["ts_run_mode"] = st.radio("Execution Mode", options=["background", "foreground"], horizontal=True, index=0 if st.session_state["ts_run_mode"] == "background" else 1)
        if st.button("Launch Training from Wizard", key="ts_launch_training", type="primary"):
            runtime_cfg_path = write_runtime_training_config(project_root, effective_cfg)
            rel_cfg = str(runtime_cfg_path.relative_to(project_root)).replace("\\", "/")
            record = job_service.launch_training_with_config(
                config_rel_path=rel_cfg,
                run_mode=st.session_state["ts_run_mode"],
                metadata={
                    "preset": st.session_state["ts_preset"],
                    "total_timesteps": int(st.session_state["ts_timesteps"]),
                    "window_size": int(st.session_state["ts_window_size"]),
                },
            )
            ui_components.render_banner("success", f"Training launched: `{record.get('id', 'unknown')}`")


def _render_progress_and_checkpoints(project_root: Path, job_service: JobService) -> None:
    st.markdown("### Training Progress and Checkpoints")
    jobs = [j for j in job_service.list_jobs() if str(j.get("workflow")) == "training"]
    if not jobs:
        ui_components.render_empty_state("No training jobs found yet.", "Launch a training run from the wizard or Operations Hub.")
    else:
        options = [f"{j.get('id')} | {j.get('status')} | {j.get('started_at')}" for j in jobs]
        selected = st.selectbox("Inspect training job", options=options, key="ts_inspect_job")
        selected_id = selected.split("|")[0].strip()
        log_tail = job_service.get_job_log_tail(selected_id, lines=500)
        progress_df = parse_training_progress(log_tail)
        if progress_df.empty:
            st.caption("No parseable SB3 metrics in logs yet. Training may still be warming up.")
        else:
            c1, c2 = st.columns(2)
            if progress_df["reward"].notna().any():
                rew_df = progress_df.dropna(subset=["reward"])
                fig_reward = px.line(rew_df, x="step", y="reward", title="Reward Progression")
                fig_reward.update_layout(hovermode="x unified")
                c1.plotly_chart(fig_reward, width="stretch")
            if progress_df["loss"].notna().any():
                loss_df = progress_df.dropna(subset=["loss"])
                fig_loss = px.line(loss_df, x="step", y="loss", title="Loss Curve")
                fig_loss.update_layout(hovermode="x unified")
                c2.plotly_chart(fig_loss, width="stretch")
            with st.expander("Raw Log Tail"):
                st.code(log_tail or "No logs yet.", language="text")

    checkpoints_df = list_checkpoints(project_root)
    st.markdown("**Checkpoint Browser**")
    if checkpoints_df.empty:
        ui_components.render_empty_state("No checkpoint artifacts found.", "Checkpoints appear after training saves intermediate models.")
    else:
        st.dataframe(checkpoints_df, width="stretch", hide_index=True)


def _render_model_comparison(project_root: Path, job_service: JobService) -> None:
    st.markdown("### Model Comparison")
    jobs = job_service.list_jobs()
    run_table = build_training_run_table(project_root, jobs)
    if run_table.empty:
        ui_components.render_empty_state("No completed training runs to compare yet.")
        return
    st.dataframe(run_table, width="stretch", hide_index=True)


def _render_resource_monitor(job_service: JobService) -> None:
    st.markdown("### GPU/CPU Resource Monitor and ETA Calculator")
    snapshot = collect_resource_snapshot()
    c1, c2, c3 = st.columns(3)
    c1.metric("CPU %", f"{snapshot['cpu_percent']:.1f}%" if snapshot["cpu_percent"] is not None else "n/a")
    memory_label = "n/a"
    if snapshot["memory_percent"] is not None:
        memory_label = f"{snapshot['memory_percent']:.1f}% ({snapshot['memory_used_gb']} / {snapshot['memory_total_gb']} GB)"
    c2.metric("RAM", memory_label)
    gpu_label = snapshot["gpu_name"] or "GPU not detected"
    if snapshot["gpu_memory_allocated_gb"] is not None:
        gpu_label = f"{gpu_label} | alloc {snapshot['gpu_memory_allocated_gb']} GB"
    c3.metric("GPU", gpu_label)

    jobs = job_service.list_jobs()
    inferred_rate = infer_step_rate_from_jobs(jobs)
    est_steps = st.number_input("Timesteps for ETA", min_value=10000, step=10000, value=200000, key="ts_eta_steps")
    manual_rate = st.number_input("Assumed Steps/Sec", min_value=1, step=50, value=int(inferred_rate), key="ts_eta_rate")
    eta = estimate_training_time(int(est_steps), float(manual_rate))
    st.caption(f"Estimated duration: `{eta['eta_hms']}`")


def _render_clone_and_retrain(project_root: Path, job_service: JobService) -> None:
    st.markdown("### Clone and Retrain")
    versions = job_service.list_model_versions()
    if not versions:
        ui_components.render_empty_state("No model artifacts found for cloning.")
        return
    options = [v["path"] for v in versions]
    source = st.selectbox("Source model", options=options, key="ts_clone_source")
    auto_retrain = st.toggle("Launch retraining after clone", value=False, key="ts_clone_auto_retrain")
    if st.button("Clone Model", key="ts_clone_btn"):
        clone = job_service.clone_model_artifact(source)
        if not clone.get("ok"):
            ui_components.render_banner("error", f"Clone failed: {clone.get('error', 'unknown error')}")
            return
        ui_components.render_banner("success", f"Model cloned to `{clone.get('clone_model')}`.")
        if auto_retrain:
            cfg = load_default_config(project_root)
            runtime_cfg_path = write_runtime_training_config(project_root, cfg)
            rel_cfg = str(runtime_cfg_path.relative_to(project_root)).replace("\\", "/")
            record = job_service.launch_training_with_config(
                config_rel_path=rel_cfg,
                run_mode="background",
                metadata={
                    "preset": st.session_state.get("ts_preset", "balanced"),
                    "total_timesteps": st.session_state.get("ts_timesteps", 200000),
                    "clone_source_model": source,
                    "clone_model_artifact": clone.get("clone_model"),
                },
            )
            ui_components.render_banner("success", f"Retraining launched from cloned context: `{record.get('id', 'unknown')}`")


def render(project_root: Path, job_service: JobService) -> None:
    st.subheader("Training Studio")
    st.caption(
        "Plan, launch, monitor, and compare training runs with guided presets, progress telemetry, "
        "checkpoint visibility, and clone/retrain workflows."
    )
    base_cfg = load_default_config(project_root)
    _state_defaults(base_cfg)

    _render_wizard(base_cfg, project_root, job_service)
    st.markdown("---")
    _render_progress_and_checkpoints(project_root, job_service)
    st.markdown("---")
    _render_model_comparison(project_root, job_service)
    st.markdown("---")
    _render_resource_monitor(job_service)
    st.markdown("---")
    _render_clone_and_retrain(project_root, job_service)


if __name__ == "__main__":
    from ui_dashboard.core.config import load_config

    cfg = load_config()
    render(cfg.project_root, JobService(cfg))
