from __future__ import annotations

from pathlib import Path

import plotly.express as px
import streamlit as st

from ui_dashboard.components.ui import render_empty_state, render_freshness_row
from ui_dashboard.services.cache_service import get_backtest_curve_cached
from ui_dashboard.services.data_service import summarize_equity


def render(project_root: Path) -> None:
    st.subheader("Backtest Analytics")
    st.caption("Evaluate strategy performance, drawdown risk, and equity trajectory from historical simulation results.")
    curve_path = project_root / "results" / "equity_curve.csv"
    render_freshness_row([("Backtest Equity Source", curve_path)])

    with st.spinner("Loading backtest data..."):
        curve_df = get_backtest_curve_cached(curve_path)
    if curve_df is None:
        render_empty_state(
            "No backtest equity curve found.",
            "Run `scripts/backtest.py` from Operations Hub and refresh.",
        )
        return

    summary = summarize_equity(curve_df["equity"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Return", f"{summary['total_return'] * 100:.2f}%")
    c2.metric("Max Drawdown", f"{summary['max_drawdown'] * 100:.2f}%")
    c3.metric("Final Equity", f"{summary['final_equity']:,.2f}")

    fig = px.line(curve_df, x="step", y="equity", title="Backtest Equity Curve")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    from ui_dashboard.core.config import load_config

    cfg = load_config()
    render(cfg.project_root)
