from __future__ import annotations

from pathlib import Path

import plotly.express as px
import streamlit as st

from ui_dashboard.components.ui import render_empty_state, render_freshness_row
from ui_dashboard.services.data_service import discover_tuning_trials, load_equity_curve, load_json


def render(project_root: Path) -> None:
    st.subheader("Fine-tuning Insights")
    st.caption("Compare Optuna trials, inspect best hyperparameters, and drill into trial-level equity curves.")
    tuning_dir = project_root / "tuning"
    best_params_path = tuning_dir / "best_params.json"
    render_freshness_row([("Best Params Source", best_params_path)])

    with st.spinner("Loading tuning artifacts..."):
        best_params = load_json(best_params_path)
        trials_df = discover_tuning_trials(tuning_dir)

    if best_params:
        st.markdown("**Best Parameters**")
        st.json(best_params)

    if trials_df.empty:
        render_empty_state(
            "No tuning trial equity curves found.",
            "Run `scripts/tune.py` from Operations Hub to generate trial outputs.",
        )
        return

    st.dataframe(
        trials_df[["trial", "total_return", "max_drawdown", "final_equity", "steps"]],
        width="stretch",
        hide_index=True,
    )

    bar_fig = px.bar(
        trials_df,
        x="trial",
        y="total_return",
        title="Trial Total Return",
        labels={"total_return": "Total Return", "trial": "Trial"},
    )
    st.plotly_chart(bar_fig, width="stretch")

    trial_options = trials_df["trial"].tolist()
    selected_trial = st.selectbox("View trial equity curve", trial_options, index=len(trial_options) - 1)
    selected_row = trials_df[trials_df["trial"] == selected_trial].iloc[0]
    curve_df = load_equity_curve(Path(selected_row["path"]))
    if curve_df is not None:
        fig = px.line(curve_df, x="step", y="equity", title=f"Trial {selected_trial} Equity Curve")
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    from ui_dashboard.core.config import load_config

    cfg = load_config()
    render(cfg.project_root)
