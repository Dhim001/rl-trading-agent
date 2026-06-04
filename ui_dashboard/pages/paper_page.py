from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ui_dashboard.components.ui import render_empty_state, render_freshness_row
from ui_dashboard.services.cache_service import get_paper_log_cached, get_paper_state_cached
from ui_dashboard.services.realtime_service import read_websocket_event


def render(project_root: Path) -> None:
    st.subheader("Paper Trading")
    state_path = project_root / "paper" / "portfolio_state.json"
    log_path = project_root / "paper" / "trades.log"
    render_freshness_row(
        [
            ("Portfolio State", state_path),
            ("Trade Log", log_path),
        ]
    )

    if "paper_live_events" not in st.session_state:
        st.session_state["paper_live_events"] = []

    ws_url = os.getenv("RL_DASHBOARD_PAPER_WS_URL", "").strip()
    col_a, col_b = st.columns([2, 3])
    with col_a:
        ws_enabled = st.toggle(
            "Enable WebSocket live sync",
            value=False,
            disabled=not bool(ws_url),
            help="Reads live paper-trading updates from RL_DASHBOARD_PAPER_WS_URL.",
        )
    with col_b:
        if ws_url:
            st.caption(f"Live source: `{ws_url}`")
        else:
            st.caption("Live source not configured. Set `RL_DASHBOARD_PAPER_WS_URL` to enable.")

    if ws_enabled and ws_url and hasattr(st, "fragment"):
        @st.fragment(run_every="2s")
        def _ws_fragment() -> None:
            event = read_websocket_event(ws_url, timeout_seconds=0.5)
            if event:
                events = st.session_state.get("paper_live_events", [])
                events.append(event)
                st.session_state["paper_live_events"] = events[-300:]
            st.caption(f"Buffered live events: {len(st.session_state.get('paper_live_events', []))}")

        _ws_fragment()

    with st.spinner("Loading paper-trading state..."):
        state = get_paper_state_cached(state_path)
        log_rows = get_paper_log_cached(log_path)

    if state:
        st.markdown("**Current Portfolio State**")
        c1, c2 = st.columns(2)
        c1.metric("Cash", f"{state.get('cash', 0.0):,.2f}")
        equity = float(state.get("cash", 0.0)) + sum(
            state.get("shares", {}).get(s, 0) * state.get("last_prices", {}).get(s, 0.0)
            for s in state.get("shares", {})
        )
        c2.metric("Estimated Equity", f"{equity:,.2f}")
        with st.expander("Portfolio JSON"):
            st.json(state)
    else:
        render_empty_state("No portfolio state found.", "Run paper trading once to initialize state.")

    exec_rows = [r for r in log_rows if "equity" in r and "timestamp" in r]
    live_rows = [
        r for r in st.session_state.get("paper_live_events", []) if "equity" in r and "timestamp" in r
    ]
    if live_rows:
        exec_rows.extend(live_rows)
    if not exec_rows:
        render_empty_state("No paper-trading execution rows found.", "Start paper trading from Operations Hub.")
        return

    df = pd.DataFrame(exec_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    fig = px.line(df, x="timestamp", y="equity", title="Paper Trading Equity Over Time (Live Synced)")
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, width="stretch")

    last_n = st.slider("Recent trade rows", min_value=5, max_value=200, value=20, key="paper_recent_rows")
    st.dataframe(df.tail(last_n)[["timestamp", "equity", "cash", "action", "message"]], width="stretch")


if __name__ == "__main__":
    from ui_dashboard.core.config import load_config

    cfg = load_config()
    render(cfg.project_root)
