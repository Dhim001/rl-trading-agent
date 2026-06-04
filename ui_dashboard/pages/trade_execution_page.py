from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from ui_dashboard.components import ui as ui_components
from ui_dashboard.services.trade_execution_service import (
    cancel_order,
    create_order,
    execute_order,
    export_history_csv_bytes,
    export_history_pdf_bytes,
    get_available_symbols,
    load_pending_orders,
    load_trade_history,
    run_paper_to_live_what_if,
    save_pending_orders,
)


def _load_risk_defaults(project_root: Path) -> dict[str, float]:
    cfg_path = project_root / "config" / "default.yaml"
    defaults = {
        "stop_loss_pct": 0.05,
        "position_size_pct": 0.95,
        "max_drawdown_pct": 0.15,
    }
    if not cfg_path.exists():
        return defaults
    try:
        import yaml

        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        risk_cfg = cfg.get("risk") or {}
        return {
            "stop_loss_pct": float(risk_cfg.get("stop_loss_pct", defaults["stop_loss_pct"])),
            "position_size_pct": float(risk_cfg.get("position_size_pct", defaults["position_size_pct"])),
            "max_drawdown_pct": float(risk_cfg.get("max_drawdown_pct", defaults["max_drawdown_pct"])),
        }
    except Exception:
        return defaults


def _load_cash(project_root: Path) -> float:
    state_path = project_root / "paper" / "portfolio_state.json"
    if not state_path.exists():
        return 0.0
    try:
        import json

        with open(state_path, encoding="utf-8") as f:
            payload = json.load(f)
        return float(payload.get("cash", 0.0))
    except Exception:
        return 0.0


def _render_manual_override_and_ticket(project_root: Path, risk_defaults: dict[str, float]) -> None:
    st.markdown("### Manual Trade Override")
    st.caption("Submit manual orders that bypass model actions for selected trades.")
    symbols = get_available_symbols(project_root)
    available_cash = _load_cash(project_root)
    default_stop = float(risk_defaults["stop_loss_pct"])

    col1, col2, col3, col4 = st.columns(4)
    symbol = col1.selectbox("Symbol", options=symbols, key="te_symbol")
    side = col2.selectbox("Side", options=["buy", "sell"], key="te_side")
    price = float(col3.number_input("Order Price", min_value=0.01, value=100.0, step=0.5, key="te_price"))
    qty = float(col4.number_input("Quantity", min_value=1.0, value=10.0, step=1.0, key="te_qty"))

    st.markdown("**Order Ticket (risk pre-filled)**")
    c1, c2, c3 = st.columns(3)
    stop_loss_pct = float(c1.number_input("Stop Loss %", min_value=0.0, max_value=1.0, value=default_stop, step=0.005))
    position_size_pct = float(
        c2.number_input(
            "Max Position Size %",
            min_value=0.0,
            max_value=1.0,
            value=float(risk_defaults["position_size_pct"]),
            step=0.01,
        )
    )
    max_drawdown_pct = float(
        c3.number_input(
            "Max Drawdown %",
            min_value=0.0,
            max_value=1.0,
            value=float(risk_defaults["max_drawdown_pct"]),
            step=0.01,
        )
    )

    st.markdown("### Position Sizing Calculator")
    notional = price * qty
    max_notional = max(0.0, available_cash * position_size_pct)
    risk_per_trade = notional * stop_loss_pct
    size_col1, size_col2, size_col3 = st.columns(3)
    size_col1.metric("Order Notional", f"{notional:,.2f}")
    size_col2.metric("Max Allowed Notional", f"{max_notional:,.2f}")
    size_col3.metric("Approx. Risk Amount", f"{risk_per_trade:,.2f}")

    warnings: list[str] = []
    if notional > max_notional > 0:
        warnings.append("Order notional exceeds configured position-size limit.")
    if stop_loss_pct <= 0:
        warnings.append("Stop loss is 0%; risk control is disabled for this order.")
    if max_drawdown_pct < 0.05:
        warnings.append("Configured max drawdown is very tight and may block strategy behavior.")

    if warnings:
        ui_components.render_banner("warning", "Risk warnings detected for this order.")
        for w in warnings:
            st.caption(f"- {w}")

    confirm_label = "I confirm this manual override trade."
    confirmation = st.checkbox(confirm_label, key="te_confirm")
    if st.button("Add Pending Order", key="te_add_order", type="primary"):
        if not confirmation:
            ui_components.render_banner("error", "Please confirm the trade before submitting.")
            return
        order = create_order(
            symbol=symbol,
            side=side,
            quantity=qty,
            limit_price=price,
            stop_loss_pct=stop_loss_pct,
            risk_note="; ".join(warnings),
            source="manual_override",
        )
        orders = load_pending_orders(project_root)
        orders.append(order)
        save_pending_orders(project_root, orders)
        ui_components.render_banner("success", f"Pending order created: `{order['id']}`")


def _render_batch_upload(project_root: Path, risk_defaults: dict[str, float]) -> None:
    st.markdown("### Batch Trade Upload (CSV)")
    st.caption("Upload CSV with columns: symbol, side, quantity, limit_price, optional stop_loss_pct")
    file = st.file_uploader("Batch orders CSV", type=["csv"], key="te_batch_csv")
    if file is None:
        return
    try:
        batch_df = pd.read_csv(io.BytesIO(file.getvalue()))
    except Exception as exc:
        ui_components.render_banner("error", f"Could not parse CSV: {exc}")
        return

    required = {"symbol", "side", "quantity", "limit_price"}
    if not required.issubset(set(batch_df.columns)):
        ui_components.render_banner("error", f"Missing required columns: {sorted(required)}")
        return

    st.dataframe(batch_df.head(20), width="stretch", hide_index=True)
    if st.button("Create Pending Orders from CSV", key="te_batch_apply"):
        orders = load_pending_orders(project_root)
        for _, row in batch_df.iterrows():
            stop_loss = float(row.get("stop_loss_pct", risk_defaults["stop_loss_pct"]))
            orders.append(
                create_order(
                    symbol=str(row["symbol"]),
                    side=str(row["side"]),
                    quantity=float(row["quantity"]),
                    limit_price=float(row["limit_price"]),
                    stop_loss_pct=stop_loss,
                    source="batch_upload",
                )
            )
        save_pending_orders(project_root, orders)
        ui_components.render_banner("success", f"Created {len(batch_df)} pending orders from batch upload.")


def _render_pending_orders(project_root: Path) -> None:
    st.markdown("### Pending Orders Dashboard")
    orders = load_pending_orders(project_root)
    if not orders:
        ui_components.render_empty_state("No pending orders.")
        return
    df = pd.DataFrame(orders)
    st.dataframe(df, width="stretch", hide_index=True)

    pending = [o for o in orders if str(o.get("status")) == "pending"]
    if not pending:
        st.caption("No pending status orders to execute/cancel.")
        return

    options = [f"{o['id']} | {o['symbol']} | {o['side']} | qty={o['quantity']}" for o in pending]
    selected = st.selectbox("Select pending order", options=options, key="te_pending_select")
    target_id = selected.split("|")[0].strip()
    selected_order = next((o for o in orders if o.get("id") == target_id), None)
    if selected_order is None:
        return

    c1, c2 = st.columns(2)
    with c1:
        exec_price = st.number_input(
            "Execution Price",
            min_value=0.01,
            value=float(selected_order.get("limit_price", 0.01)),
            step=0.1,
            key="te_exec_price",
        )
        if st.button("Execute Selected Order", key="te_exec_btn"):
            execute_order(project_root, selected_order, execution_price=float(exec_price))
            save_pending_orders(project_root, orders)
            ui_components.render_banner("success", f"Order executed: `{target_id}`")
    with c2:
        if st.button("Cancel Selected Order", key="te_cancel_btn"):
            cancel_order(selected_order)
            save_pending_orders(project_root, orders)
            ui_components.render_banner("warning", f"Order cancelled: `{target_id}`")


def _render_trade_history_and_export(project_root: Path) -> pd.DataFrame:
    st.markdown("### Trade History")
    history = load_trade_history(project_root)
    if history.empty:
        ui_components.render_empty_state("No trade history entries found.")
        return history

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    min_dt = history["timestamp"].min().date() if "timestamp" in history.columns and history["timestamp"].notna().any() else datetime.utcnow().date()
    max_dt = history["timestamp"].max().date() if "timestamp" in history.columns and history["timestamp"].notna().any() else datetime.utcnow().date()
    date_range = filter_col1.date_input("Date Range", value=(min_dt, max_dt), key="te_hist_date")
    symbols = sorted(set(str(x).upper() for x in history["symbol"].fillna("").tolist() if str(x).strip()))
    selected_symbols = filter_col2.multiselect("Symbols", options=symbols, default=symbols, key="te_hist_symbol")
    outcomes = sorted(set(str(x) for x in history["outcome"].fillna("unknown").tolist()))
    selected_outcomes = filter_col3.multiselect("Outcome", options=outcomes, default=outcomes, key="te_hist_outcome")

    filtered = history.copy()
    if isinstance(date_range, tuple) and len(date_range) == 2 and "timestamp" in filtered.columns:
        start_dt = pd.Timestamp(date_range[0]).tz_localize("UTC")
        end_dt = pd.Timestamp(date_range[1]).tz_localize("UTC") + pd.Timedelta(days=1)
        filtered = filtered[(filtered["timestamp"] >= start_dt) & (filtered["timestamp"] < end_dt)]
    if selected_symbols:
        filtered = filtered[filtered["symbol"].astype(str).str.upper().isin([s.upper() for s in selected_symbols])]
    if selected_outcomes:
        filtered = filtered[filtered["outcome"].astype(str).isin(selected_outcomes)]

    st.dataframe(filtered, width="stretch", hide_index=True)

    st.markdown("**Export Trade Log**")
    csv_bytes = export_history_csv_bytes(filtered)
    pdf_bytes = export_history_pdf_bytes(filtered)
    c1, c2 = st.columns(2)
    c1.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="trade_history_export.csv",
        mime="text/csv",
        disabled=not bool(csv_bytes),
        key="te_dl_csv",
    )
    c2.download_button(
        "Download PDF",
        data=pdf_bytes,
        file_name="trade_history_export.pdf",
        mime="application/pdf",
        disabled=not bool(pdf_bytes),
        key="te_dl_pdf",
    )
    return filtered


def _render_paper_to_live_simulation(filtered_history: pd.DataFrame) -> None:
    st.markdown("### Paper to Live Simulation (What-If)")
    if filtered_history.empty:
        ui_components.render_empty_state("No filtered trades to simulate.")
        return

    c1, c2 = st.columns(2)
    slippage_bps = c1.number_input("Assumed Slippage (bps)", min_value=0.0, value=8.0, step=1.0, key="te_slip_bps")
    fee_bps = c2.number_input("Assumed Fees (bps)", min_value=0.0, value=2.0, step=1.0, key="te_fee_bps")
    sim_df = run_paper_to_live_what_if(filtered_history, slippage_bps=float(slippage_bps), fee_bps=float(fee_bps))
    if sim_df.empty:
        ui_components.render_empty_state("Unable to compute what-if analysis from current data.")
        return

    total_live_cost = float(sim_df["estimated_live_cost"].sum())
    st.metric("Estimated Additional Live Execution Cost", f"{total_live_cost:,.2f}")
    if "timestamp" in sim_df.columns and sim_df["timestamp"].notna().any():
        plot_df = sim_df.sort_values("timestamp")
        fig = px.line(plot_df, x="timestamp", y="cumulative_live_cost", title="Cumulative Estimated Live Cost")
    else:
        plot_df = sim_df.reset_index(drop=True)
        plot_df["idx"] = range(len(plot_df))
        fig = px.line(plot_df, x="idx", y="cumulative_live_cost", title="Cumulative Estimated Live Cost")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, width="stretch")


def render(project_root: Path) -> None:
    st.subheader("Trade Execution")
    st.caption("Manual override trading controls, risk-aware order ticketing, pending order management, and what-if live execution analysis.")
    risk_defaults = _load_risk_defaults(project_root)

    ui_components.render_component_guard(
        "Manual Override Panel",
        lambda: _render_manual_override_and_ticket(project_root, risk_defaults),
        key_prefix="te_guard_override",
    )
    st.markdown("---")
    ui_components.render_component_guard(
        "Batch Upload",
        lambda: _render_batch_upload(project_root, risk_defaults),
        key_prefix="te_guard_batch",
    )
    st.markdown("---")
    ui_components.render_component_guard(
        "Pending Orders Dashboard",
        lambda: _render_pending_orders(project_root),
        key_prefix="te_guard_pending",
    )
    st.markdown("---")
    filtered: pd.DataFrame = pd.DataFrame()
    def _history_section() -> None:
        nonlocal filtered
        filtered = _render_trade_history_and_export(project_root)

    ui_components.render_component_guard(
        "Trade History",
        _history_section,
        key_prefix="te_guard_history",
    )
    st.markdown("---")
    ui_components.render_component_guard(
        "Paper-to-Live Simulation",
        lambda: _render_paper_to_live_simulation(filtered),
        key_prefix="te_guard_sim",
    )


if __name__ == "__main__":
    from ui_dashboard.core.config import load_config

    cfg = load_config()
    render(cfg.project_root)
