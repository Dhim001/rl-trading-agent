from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from rl_trading_agent.data.features import FEATURE_COLUMNS
from rl_trading_agent.data.pipeline import load_dataset
from ui_dashboard.components import ui as ui_components
from ui_dashboard.services.job_service import JobService
from ui_dashboard.services.strategy_studio_service import (
    choose_symbol_frame,
    indicator_correlation_matrix,
    list_strategy_preset_versions,
    list_strategy_presets,
    load_strategy_preset_version,
    load_project_config,
    load_strategy_preset,
    parameter_sensitivity,
    run_backtest_with_overrides,
    save_strategy_preset,
    shap_feature_importance,
    walk_forward_analysis,
)


def _model_options(job_service: JobService) -> list[str]:
    options = [item["path"] for item in job_service.list_model_versions()]
    if "models/final_model.zip" not in options:
        options.insert(0, "models/final_model.zip")
    # preserve order and uniqueness
    seen: set[str] = set()
    deduped: list[str] = []
    for opt in options:
        if opt in seen:
            continue
        seen.add(opt)
        deduped.append(opt)
    return deduped


def _default_builder_conditions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"order": 1, "indicator": "rsi", "operator": "<", "threshold": 30.0, "logical": "AND"},
            {"order": 2, "indicator": "macd", "operator": ">", "threshold": 0.0, "logical": "END"},
        ]
    )


def _render_strategy_builder(project_root: Path, available_indicators: list[str]) -> None:
    st.markdown("### Visual Strategy Builder")
    st.caption("Compose indicator conditions visually. Reorder rows by editing `order` (drag-drop style workflow in table form).")

    st.session_state.setdefault("ss_selected_indicators", FEATURE_COLUMNS[:5])
    st.session_state.setdefault("ss_conditions_df", _default_builder_conditions())
    st.session_state.setdefault("ss_condition_order", st.session_state["ss_conditions_df"]["indicator"].astype(str).tolist())

    selected = st.multiselect(
        "Indicators",
        options=available_indicators,
        default=[c for c in st.session_state["ss_selected_indicators"] if c in available_indicators],
        key="ss_indicator_multiselect",
    )
    st.session_state["ss_selected_indicators"] = selected

    edited_df = st.data_editor(
        st.session_state["ss_conditions_df"],
        width="stretch",
        num_rows="dynamic",
        column_config={
            "order": st.column_config.NumberColumn("order", min_value=1, step=1),
            "indicator": st.column_config.SelectboxColumn("indicator", options=available_indicators),
            "operator": st.column_config.SelectboxColumn("operator", options=[">", "<", ">=", "<=", "=="]),
            "logical": st.column_config.SelectboxColumn("logical", options=["AND", "OR", "END"]),
        },
        key="ss_conditions_editor",
    )
    edited_df = edited_df.sort_values("order").reset_index(drop=True)
    st.session_state["ss_conditions_df"] = edited_df

    st.markdown("**Condition Ordering (Drag-Drop)**")
    indicator_order_options = edited_df["indicator"].astype(str).tolist()
    try:
        from streamlit_sortables import sort_items  # type: ignore

        ordered = sort_items(indicator_order_options, direction="vertical")
        if ordered and set(ordered) == set(indicator_order_options):
            st.session_state["ss_condition_order"] = ordered
            rank = {name: idx for idx, name in enumerate(ordered, start=1)}
            st.session_state["ss_conditions_df"]["order"] = st.session_state["ss_conditions_df"]["indicator"].map(rank)
            st.session_state["ss_conditions_df"] = st.session_state["ss_conditions_df"].sort_values("order").reset_index(drop=True)
    except Exception:
        st.caption("Install `streamlit-sortables` for drag-drop ordering. Fallback: edit numeric `order` column above.")

    preset_name = st.text_input("Preset Name", value="my_strategy", key="ss_preset_name")
    c1, c2 = st.columns(2)
    if c1.button("Save Strategy Preset", key="ss_save_preset"):
        payload = {
            "indicators": st.session_state["ss_selected_indicators"],
            "conditions": st.session_state["ss_conditions_df"].to_dict(orient="records"),
        }
        saved_path = save_strategy_preset(project_root, preset_name, payload)
        ui_components.render_banner("success", f"Saved preset to `{saved_path}`")
    preset_list = list_strategy_presets(project_root)
    selected_preset = c2.selectbox("Load Preset", options=["<none>", *preset_list], index=0, key="ss_load_preset_name")
    if c2.button("Apply Preset", key="ss_apply_preset", disabled=selected_preset == "<none>"):
        payload = load_strategy_preset(project_root, selected_preset)
        if not payload:
            ui_components.render_banner("error", "Could not load selected preset.")
        else:
            st.session_state["ss_selected_indicators"] = payload.get("indicators", [])
            st.session_state["ss_conditions_df"] = pd.DataFrame(payload.get("conditions", []))
            ui_components.render_banner("success", f"Applied preset `{selected_preset}`.")

    st.markdown("**Preset Version History**")
    version_options = list_strategy_preset_versions(project_root, preset_name)
    if not version_options:
        st.caption("No version snapshots yet for this preset name.")
    else:
        v1, v2 = st.columns([3, 1])
        selected_version = v1.selectbox("Saved versions", options=version_options, index=0, key="ss_preset_version")
        if v2.button("Restore Version", key="ss_restore_version"):
            payload = load_strategy_preset_version(project_root, selected_version)
            if not payload:
                ui_components.render_banner("error", "Could not load selected version.")
            else:
                st.session_state["ss_selected_indicators"] = payload.get("indicators", [])
                st.session_state["ss_conditions_df"] = pd.DataFrame(payload.get("conditions", []))
                ui_components.render_banner("success", f"Restored preset version `{selected_version}`.")


def _render_backtest_on_demand(project_root: Path, job_service: JobService) -> None:
    st.markdown("### Backtest On Demand")
    models = _model_options(job_service)
    model_rel = st.selectbox("Model Artifact", options=models, index=0, key="ss_backtest_model")

    col1, col2, col3, col4 = st.columns(4)
    window_size = int(col1.number_input("window_size", min_value=10, max_value=120, value=30, step=5, key="ss_window_size"))
    tx_cost = float(col2.number_input("transaction_cost_pct", min_value=0.0, max_value=0.02, value=0.001, format="%.4f", key="ss_tx_cost"))
    stop_loss = float(col3.number_input("stop_loss_pct", min_value=0.0, max_value=0.5, value=0.05, format="%.3f", key="ss_stop_loss"))
    position_size = float(col4.number_input("position_size_pct", min_value=0.05, max_value=1.0, value=0.95, format="%.2f", key="ss_position_size"))

    if st.button("Run Custom Backtest", key="ss_run_backtest", type="primary"):
        with st.spinner("Running on-demand backtest..."):
            result = run_backtest_with_overrides(
                project_root=project_root,
                model_rel_path=model_rel,
                overrides={
                    "environment": {"window_size": window_size, "transaction_cost_pct": tx_cost},
                    "risk": {"stop_loss_pct": stop_loss, "position_size_pct": position_size},
                },
            )
        metrics = result["metrics"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Return", f"{metrics.get('total_return', 0.0) * 100:.2f}%")
        m2.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0.0) * 100:.2f}%")
        m3.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0.0):.3f}")
        curve_df = result.get("equity_curve", pd.DataFrame())
        if not curve_df.empty:
            fig = px.line(curve_df, x="step", y="equity", title="On-Demand Backtest Equity Curve")
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, width="stretch")
        st.caption(f"Output dir: `{result.get('output_dir', '-')}`")


def _render_walk_forward(project_root: Path, job_service: JobService) -> None:
    st.markdown("### Walk-Forward Analysis Wizard")
    models = _model_options(job_service)
    model_rel = st.selectbox("Model for Walk-Forward", options=models, index=0, key="ss_wf_model")
    c1, c2 = st.columns(2)
    n_splits = int(c1.slider("Folds", min_value=2, max_value=10, value=4, key="ss_wf_folds"))
    train_ratio = float(c2.slider("Train/Test Split", min_value=0.5, max_value=0.9, value=0.8, step=0.05, key="ss_wf_split"))
    if st.button("Run Walk-Forward", key="ss_run_wf"):
        with st.spinner("Running walk-forward folds..."):
            wf_df = walk_forward_analysis(project_root, model_rel_path=model_rel, n_splits=n_splits, train_ratio=train_ratio)
        if wf_df.empty:
            ui_components.render_empty_state("Walk-forward did not produce any rows.")
            return
        st.dataframe(wf_df, width="stretch", hide_index=True)
        fig = px.line(wf_df, x="fold", y="sharpe_ratio", markers=True, title="Walk-Forward Sharpe by Fold")
        st.plotly_chart(fig, width="stretch")


def _render_indicator_correlation(frame: pd.DataFrame) -> None:
    st.markdown("### Indicator Correlation Heatmap")
    options = [c for c in FEATURE_COLUMNS if c in frame.columns]
    chosen = st.multiselect("Indicators for Correlation", options=options, default=options[: min(8, len(options))], key="ss_corr_indicators")
    corr = indicator_correlation_matrix(frame, chosen)
    if corr.empty:
        ui_components.render_empty_state("Select at least 2 valid indicators.")
        return
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(title="Indicator Correlation Heatmap")
    st.plotly_chart(fig, width="stretch")


def _render_shap_importance(frame: pd.DataFrame) -> None:
    st.markdown("### Feature Importance Analyzer (SHAP)")
    st.caption("Uses a surrogate tree model on next-period returns to estimate SHAP importance.")
    if st.button("Run SHAP Analysis", key="ss_run_shap"):
        with st.spinner("Computing SHAP importances..."):
            shap_df, status = shap_feature_importance(frame)
        if shap_df.empty:
            ui_components.render_banner("warning", status)
            return
        st.dataframe(shap_df, width="stretch", hide_index=True)
        fig = px.bar(shap_df.head(12), x="mean_abs_shap", y="feature", orientation="h", title="Mean |SHAP| by Feature")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")


def _render_parameter_sensitivity(project_root: Path, job_service: JobService) -> None:
    st.markdown("### Parameter Sensitivity Charts")
    models = _model_options(job_service)
    model_rel = st.selectbox("Model for Sensitivity", options=models, index=0, key="ss_sens_model")
    param = st.selectbox(
        "Parameter",
        options=["window_size", "transaction_cost_pct", "stop_loss_pct", "position_size_pct"],
        index=0,
        key="ss_sens_param",
    )
    c1, c2, c3 = st.columns(3)
    if param == "window_size":
        vmin = c1.number_input("Min", min_value=10, max_value=100, value=20, step=5, key="ss_sens_min_i")
        vmax = c2.number_input("Max", min_value=15, max_value=150, value=60, step=5, key="ss_sens_max_i")
        points = int(c3.slider("Points", min_value=3, max_value=8, value=5, key="ss_sens_pts"))
        values = np.linspace(float(vmin), float(vmax), points)
        values = [int(round(v / 5) * 5) for v in values]
    else:
        default_bounds = {
            "transaction_cost_pct": (0.0005, 0.005),
            "stop_loss_pct": (0.02, 0.12),
            "position_size_pct": (0.4, 1.0),
        }
        dmin, dmax = default_bounds[param]
        vmin = c1.number_input("Min", min_value=0.0, max_value=2.0, value=float(dmin), format="%.4f", key="ss_sens_min_f")
        vmax = c2.number_input("Max", min_value=0.0, max_value=2.0, value=float(dmax), format="%.4f", key="ss_sens_max_f")
        points = int(c3.slider("Points", min_value=3, max_value=8, value=5, key="ss_sens_pts_f"))
        values = np.linspace(float(vmin), float(vmax), points).tolist()
    if st.button("Run Sensitivity", key="ss_run_sens"):
        with st.spinner("Running parameter sweep backtests..."):
            sens_df = parameter_sensitivity(project_root, model_rel, parameter=param, values=[float(v) for v in values])
        if sens_df.empty:
            ui_components.render_empty_state("No sensitivity outputs produced.")
            return
        st.dataframe(sens_df, width="stretch", hide_index=True)
        fig = px.line(sens_df, x="value", y="sharpe_ratio", markers=True, title=f"Sensitivity: {param} vs Sharpe")
        st.plotly_chart(fig, width="stretch")


@st.cache_data(show_spinner=False)
def _load_dataset_bundle(project_root: Path):
    cfg = load_project_config(project_root)
    data, _, _ = load_dataset(cfg, project_root)
    if isinstance(data, dict):
        symbols = sorted(data.keys())
        return data, symbols
    return data, [cfg["data"].get("symbol", "AAPL")]


def render(project_root: Path, job_service: JobService) -> None:
    st.subheader("Strategy Studio")
    st.caption(
        "Build and iterate rule-based strategies with backtesting, walk-forward checks, indicator analytics, and reusable preset management."
    )

    cfg = load_project_config(project_root)
    cfg_symbols = cfg["data"].get("symbols") or [cfg["data"].get("symbol", "AAPL")]
    symbols = [str(s) for s in cfg_symbols]
    symbol = st.selectbox("Dataset Symbol Context", options=symbols, index=0, key="ss_symbol_context")

    st.session_state.setdefault("ss_dataset_ready", False)
    st.session_state.setdefault("ss_last_dataset_symbol", "")
    st.session_state.setdefault("ss_dataset_error", "")

    c_load, c_reset = st.columns([2, 1])
    with c_load:
        if st.button("Load Indicator Dataset", key="ss_load_dataset", width="stretch"):
            st.session_state["ss_dataset_ready"] = False
            st.session_state["ss_dataset_error"] = ""
            try:
                data_bundle, _ = _load_dataset_bundle(project_root)
                _ = choose_symbol_frame(data_bundle, symbol)
                st.session_state["ss_dataset_ready"] = True
                st.session_state["ss_last_dataset_symbol"] = symbol
            except Exception as exc:
                st.session_state["ss_dataset_error"] = str(exc)
                st.session_state["ss_dataset_ready"] = False
    with c_reset:
        if st.button("Unload", key="ss_unload_dataset", width="stretch"):
            st.session_state["ss_dataset_ready"] = False
            st.session_state["ss_dataset_error"] = ""

    if st.session_state["ss_dataset_error"]:
        ui_components.render_section_error("Could not load strategy dataset.", st.session_state["ss_dataset_error"])

    frame: pd.DataFrame | None = None
    if st.session_state["ss_dataset_ready"]:
        data_bundle, _ = _load_dataset_bundle(project_root)
        frame = choose_symbol_frame(data_bundle, symbol)
        st.caption(f"Dataset loaded for symbol context: `{symbol}`")
    else:
        st.caption("Dataset not loaded yet. Strategy analytics panels stay lightweight until you click Load Indicator Dataset.")
    available_indicators = [c for c in FEATURE_COLUMNS if frame is None or c in frame.columns]

    ui_components.render_component_guard(
        "Visual Strategy Builder",
        lambda: _render_strategy_builder(project_root, available_indicators),
        key_prefix="ss_guard_builder",
    )
    st.markdown("---")
    ui_components.render_component_guard(
        "Backtest On Demand",
        lambda: _render_backtest_on_demand(project_root, job_service),
        key_prefix="ss_guard_backtest",
    )
    st.markdown("---")
    ui_components.render_component_guard(
        "Walk-Forward Analysis",
        lambda: _render_walk_forward(project_root, job_service),
        key_prefix="ss_guard_wf",
    )
    st.markdown("---")
    if frame is None:
        ui_components.render_empty_state("Correlation Heatmap requires loaded indicator dataset.", "Click **Load Indicator Dataset** above.")
    else:
        ui_components.render_component_guard(
            "Correlation Heatmap",
            lambda: _render_indicator_correlation(frame),
            key_prefix="ss_guard_corr",
        )
    st.markdown("---")
    if frame is None:
        ui_components.render_empty_state("SHAP analysis requires loaded indicator dataset.", "Click **Load Indicator Dataset** above.")
    else:
        ui_components.render_component_guard(
            "SHAP Feature Importance",
            lambda: _render_shap_importance(frame),
            key_prefix="ss_guard_shap",
        )
    st.markdown("---")
    ui_components.render_component_guard(
        "Parameter Sensitivity",
        lambda: _render_parameter_sensitivity(project_root, job_service),
        key_prefix="ss_guard_sens",
    )


if __name__ == "__main__":
    from ui_dashboard.core.config import load_config
    from ui_dashboard.services.job_service import JobService

    cfg = load_config()
    render(cfg.project_root, JobService(cfg))
