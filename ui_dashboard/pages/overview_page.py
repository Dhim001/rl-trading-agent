from __future__ import annotations

import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ui_dashboard.components.ui import render_freshness_row
from ui_dashboard.services.data_service import discover_tuning_trials, load_equity_curve, load_json, load_jsonl, summarize_equity


def _day_floor(ts: datetime) -> datetime:
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)


def _count_series_from_paths(paths: list[Path], days: int = 35) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    start = _day_floor(now - timedelta(days=days - 1))
    idx = [start + timedelta(days=i) for i in range(days)]
    series = pd.Series(0, index=pd.to_datetime(idx), dtype="int64")
    for path in paths:
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            day = _day_floor(mtime)
            if day in series.index:
                series.loc[day] += 1
        except Exception:
            continue
    cumulative = series.cumsum()
    return pd.DataFrame({"ts": cumulative.index, "value": cumulative.values})


def _paper_equity_series(project_root: Path, days: int = 35) -> pd.DataFrame:
    rows = load_jsonl(project_root / "paper" / "trades.log")
    now = datetime.now(timezone.utc)
    start = _day_floor(now - timedelta(days=days - 1))
    idx = pd.to_datetime([start + timedelta(days=i) for i in range(days)], utc=True)
    if not rows:
        return pd.DataFrame({"ts": idx, "value": [0.0] * len(idx)})

    df = pd.DataFrame(rows)
    if "timestamp" not in df.columns or "equity" not in df.columns:
        return pd.DataFrame({"ts": idx, "value": [0.0] * len(idx)})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["equity"] = pd.to_numeric(df["equity"], errors="coerce")
    df = df.dropna(subset=["timestamp", "equity"]).sort_values("timestamp")
    if df.empty:
        return pd.DataFrame({"ts": idx, "value": [0.0] * len(idx)})

    df["day"] = df["timestamp"].dt.floor("D")
    daily = df.groupby("day")["equity"].last()
    aligned = pd.Series(index=idx, dtype="float64")
    aligned.loc[daily.index.intersection(idx)] = daily.loc[daily.index.intersection(idx)]
    aligned = aligned.ffill().fillna(float(df["equity"].iloc[0]))
    return pd.DataFrame({"ts": idx, "value": aligned.values})


def _value_at_or_before(series_df: pd.DataFrame, at_time: datetime) -> float:
    if series_df.empty:
        return 0.0
    ts_series = pd.to_datetime(series_df["ts"], utc=True, errors="coerce")
    values = pd.to_numeric(series_df["value"], errors="coerce").fillna(0.0)
    mask = ts_series <= at_time
    if not mask.any():
        return float(values.iloc[0]) if not values.empty else 0.0
    return float(values[mask].iloc[-1])


def _delta_for_window(series_df: pd.DataFrame, days: int) -> float:
    now = datetime.now(timezone.utc)
    current = _value_at_or_before(series_df, now)
    prior = _value_at_or_before(series_df, now - timedelta(days=days))
    return current - prior


def _render_sparkline(title: str, series_df: pd.DataFrame) -> None:
    fig = px.line(series_df, x="ts", y="value", title=title)
    fig.update_layout(
        height=120,
        margin=dict(l=8, r=8, t=28, b=8),
        showlegend=False,
        xaxis_title=None,
        yaxis_title=None,
    )
    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch")


def _render_system_health(project_root: Path) -> None:
    st.markdown("### System Health")
    disk = shutil.disk_usage(project_root)
    used_pct = 0.0 if disk.total == 0 else (disk.used / disk.total) * 100.0

    cpu_pct = None
    mem_pct = None
    mem_used = None
    mem_total = None
    try:
        import psutil

        cpu_pct = float(psutil.cpu_percent(interval=0.1))
        vm = psutil.virtual_memory()
        mem_pct = float(vm.percent)
        mem_used = vm.used / (1024**3)
        mem_total = vm.total / (1024**3)
    except Exception:
        pass

    c1, c2, c3 = st.columns(3)
    c1.metric("Disk Used", f"{used_pct:.1f}%")
    c1.caption(f"{disk.used / (1024**3):.1f} GB / {disk.total / (1024**3):.1f} GB")
    c2.metric("CPU", f"{cpu_pct:.1f}%" if cpu_pct is not None else "n/a")
    if mem_pct is None:
        c3.metric("Memory", "n/a")
    else:
        c3.metric("Memory", f"{mem_pct:.1f}%")
        c3.caption(f"{mem_used:.1f} GB / {mem_total:.1f} GB")


def _collect_recent_activities(project_root: Path) -> list[dict[str, str]]:
    activities: list[dict[str, str]] = []
    jobs_path = project_root / "ui_dashboard" / ".runtime" / "jobs.json"
    jobs = load_json(jobs_path) or []
    for row in jobs:
        ts = row.get("ended_at") or row.get("started_at")
        if not ts:
            continue
        activities.append(
            {
                "timestamp": str(ts),
                "title": f"Workflow: {row.get('workflow', 'unknown')}",
                "detail": f"Status={row.get('status', 'unknown')}",
            }
        )

    for row in load_jsonl(project_root / "paper" / "trades.log")[-80:]:
        ts = row.get("timestamp")
        if not ts:
            continue
        activities.append(
            {
                "timestamp": str(ts),
                "title": str(row.get("message", "Trade activity")),
                "detail": f"Symbol={row.get('symbol', '-')}, Action={row.get('action', '-')}",
            }
        )

    def _parse(ts: str) -> datetime:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    activities.sort(key=lambda r: _parse(r["timestamp"]), reverse=True)
    return activities[:5]


def _render_recent_activity_quick_actions(project_root: Path) -> None:
    st.markdown("### Last 5 Activities")
    rows = _collect_recent_activities(project_root)
    if not rows:
        st.caption("No activity yet.")
    else:
        for row in rows:
            ts = row["timestamp"].replace("T", " ").replace("+00:00", " UTC")
            st.markdown(f"**{row['title']}**")
            st.caption(f"{row['detail']} | {ts}")

    st.markdown("**Quick Actions**")
    a1, a2, a3 = st.columns(3)
    if a1.button("Go to Operations Hub", key="ov_quick_ops", width="stretch"):
        st.session_state["selected_page"] = "Operations Hub"
        st.rerun()
    if a2.button("Go to Trade Execution", key="ov_quick_trade", width="stretch"):
        st.session_state["selected_page"] = "Trade Execution"
        st.rerun()
    if a3.button("Refresh Overview", key="ov_quick_refresh", width="stretch"):
        st.rerun()


def render(project_root: Path) -> None:
    st.subheader("Executive Overview")
    st.caption("Monitor portfolio health, model output trends, artifact freshness, and recent operational activity from one summary view.")
    model_dir = project_root / "models"
    runs_dir = project_root / "runs"
    tuning_dir = project_root / "tuning"
    paper_state_path = project_root / "paper" / "portfolio_state.json"
    backtest_curve_path = project_root / "results" / "equity_curve.csv"

    backtest_curve = load_equity_curve(backtest_curve_path)
    tuning_trials = discover_tuning_trials(tuning_dir)
    paper_state = load_json(paper_state_path) or {}

    model_paths = [p for p in model_dir.glob("*.zip")] if model_dir.exists() else []
    run_paths = [p for p in runs_dir.glob("*") if p.is_dir()] if runs_dir.exists() else []
    trial_paths = list(tuning_dir.glob("trial_*")) if tuning_dir.exists() else []

    model_series = _count_series_from_paths(model_paths)
    run_series = _count_series_from_paths(run_paths)
    tuning_series = _count_series_from_paths(trial_paths)
    paper_equity_series = _paper_equity_series(project_root)

    week_label = "vs last week"
    month_label = "vs last month"
    card1, card2, card3, card4 = st.columns(4)

    with card1:
        models_now = int(model_series["value"].iloc[-1]) if not model_series.empty else 0
        models_week = _delta_for_window(model_series, 7)
        models_month = _delta_for_window(model_series, 30)
        st.metric("Models", models_now, delta=f"{models_week:+.0f} {week_label}")
        st.caption(f"{models_month:+.0f} {month_label}")
        _render_sparkline("Model Trend", model_series)

    with card2:
        runs_now = int(run_series["value"].iloc[-1]) if not run_series.empty else 0
        runs_week = _delta_for_window(run_series, 7)
        runs_month = _delta_for_window(run_series, 30)
        st.metric("Run Folders", runs_now, delta=f"{runs_week:+.0f} {week_label}")
        st.caption(f"{runs_month:+.0f} {month_label}")
        _render_sparkline("Run Trend", run_series)

    with card3:
        tuning_now = int(tuning_series["value"].iloc[-1]) if not tuning_series.empty else len(tuning_trials)
        tuning_week = _delta_for_window(tuning_series, 7)
        tuning_month = _delta_for_window(tuning_series, 30)
        st.metric("Tuning Trials", tuning_now, delta=f"{tuning_week:+.0f} {week_label}")
        st.caption(f"{tuning_month:+.0f} {month_label}")
        _render_sparkline("Tuning Trend", tuning_series)

    with card4:
        cash_now = float(paper_state.get("cash", 0.0))
        equity_week = _delta_for_window(paper_equity_series, 7)
        equity_month = _delta_for_window(paper_equity_series, 30)
        st.metric("Paper Cash", f"{cash_now:,.2f}", delta=f"{equity_week:+,.2f} {week_label}")
        st.caption(f"{equity_month:+,.2f} {month_label}")
        _render_sparkline("Paper Equity Trend", paper_equity_series)

    if backtest_curve is not None:
        summary = summarize_equity(backtest_curve["equity"])
        st.markdown("### Latest Backtest Snapshot")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Return", f"{summary['total_return'] * 100:.2f}%")
        c2.metric("Max Drawdown", f"{summary['max_drawdown'] * 100:.2f}%")
        c3.metric("Final Equity", f"{summary['final_equity']:,.2f}")

    render_freshness_row(
        [
            ("Backtest", backtest_curve_path),
            ("Paper State", paper_state_path),
            ("Best Params", tuning_dir / "best_params.json"),
        ]
    )

    st.markdown("---")
    _render_system_health(project_root)
    st.markdown("---")
    _render_recent_activity_quick_actions(project_root)


if __name__ == "__main__":
    from ui_dashboard.core.config import load_config

    cfg = load_config()
    render(cfg.project_root)
