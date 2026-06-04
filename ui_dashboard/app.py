from __future__ import annotations

import importlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure package imports work when launched as a Streamlit script path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ui_dashboard.core.config import load_config
from ui_dashboard.core.state import ensure_state_defaults
from ui_dashboard.components import ui as ui_components
from ui_dashboard.services.cache_service import invalidate_dashboard_caches
from ui_dashboard.services.job_service import JobService
from ui_dashboard.services.realtime_service import detect_artifact_changes, snapshot_artifacts


PAGES = {
    "Overview": "ui_dashboard.pages.overview_page",
    "Strategy Studio": "ui_dashboard.pages.strategy_studio_page",
    "Training Studio": "ui_dashboard.pages.training_studio_page",
    "Trade Execution": "ui_dashboard.pages.trade_execution_page",
    "Backtest": "ui_dashboard.pages.backtest_page",
    "Fine-tuning": "ui_dashboard.pages.tuning_page",
    "Paper Trading": "ui_dashboard.pages.paper_page",
    "Lineage": "ui_dashboard.pages.lineage_page",
    "Operations Hub": "ui_dashboard.pages.operations_hub_page",
    "Artifacts": "ui_dashboard.pages.artifacts_page",
}

PAGE_META: dict[str, dict[str, str]] = {
    "Overview": {"group": "Dashboard", "icon": "🏠", "desc": "KPI snapshots and system health"},
    "Strategy Studio": {"group": "Research", "icon": "🧠", "desc": "Design and evaluate strategies"},
    "Training Studio": {"group": "Research", "icon": "🧪", "desc": "Configure and monitor model training"},
    "Backtest": {"group": "Analysis", "icon": "📈", "desc": "Backtest performance and risk"},
    "Fine-tuning": {"group": "Analysis", "icon": "🎯", "desc": "Hyperparameter tuning insights"},
    "Paper Trading": {"group": "Execution", "icon": "📝", "desc": "Live paper-trading telemetry"},
    "Trade Execution": {"group": "Execution", "icon": "⚡", "desc": "Manual order overrides and audit"},
    "Lineage": {"group": "Operations", "icon": "🧬", "desc": "Artifact lineage and rollback"},
    "Operations Hub": {"group": "Operations", "icon": "🛠️", "desc": "Run and manage workflows"},
    "Artifacts": {"group": "Operations", "icon": "📦", "desc": "Generated artifact inventory"},
}

PAGE_GROUP_ORDER = ["Dashboard", "Research", "Analysis", "Execution", "Operations"]


def _inject_sidebar_spacing_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] .block-container {
            padding-top: 0.8rem;
            padding-bottom: 1rem;
        }
        [data-testid="stSidebar"] h3 {
            margin-top: 0.55rem;
            margin-bottom: 0.35rem;
            letter-spacing: 0.01em;
        }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            margin-top: 0.08rem;
            margin-bottom: 0.45rem;
            line-height: 1.25;
        }
        [data-testid="stSidebar"] .stButton {
            margin-top: 0.16rem;
            margin-bottom: 0.16rem;
        }
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stTextInput,
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stCheckbox {
            margin-bottom: 0.35rem;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            margin-top: 0.2rem;
            margin-bottom: 0.25rem;
        }
        .sidebar-section-divider {
            margin: 0.55rem 0 0.35rem 0;
            border-top: 1px solid rgba(128, 128, 128, 0.22);
        }
        .sidebar-nav-item-spacer {
            margin-bottom: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _touch_recent_page(page_name: str) -> None:
    if page_name not in PAGES:
        return
    rows = [p for p in st.session_state.get("recent_pages", []) if p in PAGES and p != page_name]
    rows.insert(0, page_name)
    st.session_state["recent_pages"] = rows[:5]


def _render_navigation_menu() -> None:
    st.sidebar.markdown("### Navigation")
    selected = st.session_state.get("selected_page", "Overview")
    if selected not in PAGES:
        selected = "Overview"
        st.session_state["selected_page"] = selected
    _touch_recent_page(selected)

    search = st.sidebar.text_input(
        "Search pages",
        value=st.session_state.get("nav_search", ""),
        placeholder="Type page name...",
        key="nav_search",
    ).strip().lower()

    st.sidebar.caption(f"Active: **{selected}**")
    recent = [p for p in st.session_state.get("recent_pages", []) if p in PAGES]
    if recent:
        quick = st.sidebar.selectbox("Recent pages", options=["(none)", *recent], index=0, key="recent_page_select")
        if quick != "(none)" and quick != st.session_state["selected_page"]:
            st.session_state["selected_page"] = quick
            _touch_recent_page(quick)
            st.rerun()

    grouped: dict[str, list[str]] = {k: [] for k in PAGE_GROUP_ORDER}
    for page in PAGES:
        grouped.setdefault(PAGE_META.get(page, {}).get("group", "Other"), []).append(page)

    rendered_any = False
    for group in PAGE_GROUP_ORDER:
        pages = grouped.get(group, [])
        filtered = [p for p in pages if not search or search in p.lower() or search in PAGE_META.get(p, {}).get("desc", "").lower()]
        if not filtered:
            continue
        rendered_any = True
        expanded = st.session_state["selected_page"] in filtered or bool(search)
        with st.sidebar.expander(group, expanded=expanded):
            for page in filtered:
                meta = PAGE_META.get(page, {})
                icon = meta.get("icon", "•")
                desc = meta.get("desc", "")
                is_active = page == st.session_state["selected_page"]
                button_label = f"{icon} {page}"
                if st.button(
                    button_label,
                    key=f"nav_btn_{page}",
                    width="stretch",
                    type="primary" if is_active else "secondary",
                    disabled=is_active,
                ):
                    st.session_state["selected_page"] = page
                    _touch_recent_page(page)
                    st.rerun()
                if desc:
                    st.caption(desc)
                st.markdown("<div class='sidebar-nav-item-spacer'></div>", unsafe_allow_html=True)

    if not rendered_any:
        st.sidebar.caption("No pages matched your search.")


def _push_notification(level: str, message: str) -> None:
    rows = st.session_state.get("notifications", [])
    rows.insert(
        0,
        {
            "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": level,
            "message": message,
        },
    )
    st.session_state["notifications"] = rows[:25]


def _track_job_completion_notifications(job_service: JobService) -> None:
    try:
        jobs = job_service.list_jobs()
    except Exception:
        return
    old_cache: dict[str, str] = st.session_state.get("job_status_cache", {})
    new_cache: dict[str, str] = {}
    for job in jobs:
        job_id = str(job.get("id", ""))
        status = str(job.get("status", "unknown"))
        if job_id:
            new_cache[job_id] = status
        prior = old_cache.get(job_id)
        if prior in {"running", "queued"} and status in {"completed", "failed", "stopped"}:
            workflow = str(job.get("workflow", "workflow"))
            msg = f"{workflow} job {status}: {job_id}"
            st.toast(msg, icon="🔔")
            _push_notification("info", msg)
    st.session_state["job_status_cache"] = new_cache


def _check_artifact_file_watcher(project_root: Path, rerun_on_change: bool = False) -> None:
    if not st.session_state.get("artifact_watch_enabled", True):
        return
    now = datetime.now().timestamp()
    min_interval_s = float(st.session_state.get("artifact_watch_interval_s", 20.0))
    last_check = float(st.session_state.get("artifact_watch_last_check_s", 0.0))
    if now - last_check < min_interval_s:
        return
    st.session_state["artifact_watch_last_check_s"] = now

    previous = st.session_state.get("artifact_snapshot", {})
    current = snapshot_artifacts(project_root)
    changes = detect_artifact_changes(previous, current)
    st.session_state["artifact_snapshot"] = current
    if not previous or not changes:
        return
    sample = ", ".join(changes[:3])
    overflow = "" if len(changes) <= 3 else f" (+{len(changes) - 3} more)"
    msg = f"Artifact watcher detected changes: {sample}{overflow}"
    st.toast(msg, icon="📁")
    _push_notification("info", msg)
    invalidate_dashboard_caches(changes)
    if rerun_on_change:
        st.rerun()


def _render_sidebar(job_service: JobService, project_root: Path) -> None:
    # Defensive fallback: Streamlit reruns can occasionally evaluate widgets before
    # cross-module initialization settles, so ensure critical keys exist locally.
    st.session_state.setdefault("refresh_enabled", False)
    st.session_state.setdefault("refresh_seconds", 15)
    st.session_state.setdefault("selected_page", "Overview")
    st.session_state.setdefault("artifact_watch_enabled", False)
    st.session_state.setdefault("artifact_snapshot", {})
    st.session_state.setdefault("notifications", [])
    st.session_state.setdefault("job_status_cache", {})
    st.session_state.setdefault("recent_pages", ["Overview"])
    st.session_state.setdefault("nav_search", "")
    st.session_state.setdefault("artifact_watch_last_check_s", 0.0)
    st.session_state.setdefault("artifact_watch_interval_s", 20.0)
    st.session_state.setdefault("background_monitors_enabled", False)

    st.sidebar.header("Control Panel")
    st.sidebar.caption("Fast navigation and runtime controls")
    if st.sidebar.button("Refresh Data"):
        st.rerun()
    st.session_state["refresh_enabled"] = st.sidebar.checkbox(
        "Enable auto-refresh",
        value=st.session_state["refresh_enabled"],
    )
    st.session_state["refresh_seconds"] = st.sidebar.slider(
        "Auto-refresh interval (seconds)",
        min_value=5,
        max_value=120,
        value=st.session_state["refresh_seconds"],
    )
    st.session_state["background_monitors_enabled"] = st.sidebar.checkbox(
        "Enable background monitors",
        value=st.session_state["background_monitors_enabled"],
        help="Runs periodic job and artifact checks. Disable if UI feels laggy.",
    )
    st.sidebar.markdown("<div class='sidebar-section-divider'></div>", unsafe_allow_html=True)
    st.sidebar.caption(f"Last refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.sidebar.caption("Use grouped navigation for focused workflow views.")
    st.session_state["artifact_watch_enabled"] = st.sidebar.checkbox(
        "Enable artifact file watcher",
        value=st.session_state["artifact_watch_enabled"],
        help="Auto-detect new or changed artifacts and surface notifications.",
        disabled=not st.session_state["background_monitors_enabled"],
    )
    st.session_state["artifact_watch_interval_s"] = st.sidebar.slider(
        "Watcher interval (seconds)",
        min_value=10,
        max_value=120,
        value=int(st.session_state["artifact_watch_interval_s"]),
        help="Higher intervals reduce background overhead.",
    )
    _render_navigation_menu()

    st.sidebar.markdown("<div class='sidebar-section-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("### Notification Center")
    notes: list[dict[str, Any]] = st.session_state.get("notifications", [])
    if st.sidebar.button("Clear notifications", key="clear_notifications"):
        st.session_state["notifications"] = []
        notes = []
    if not notes:
        st.sidebar.caption("No notifications yet.")
    else:
        for row in notes[:8]:
            st.sidebar.caption(f"[{row['at']}] {row['message']}")

    # Keep notification polling in the background fragment to avoid duplicate list_jobs calls.


def _sync_page_query_params_from_state() -> None:
    return


def _apply_page_query_params_to_state() -> None:
    return


def _render_page(page_name: str, project_root, job_service: JobService) -> None:
    try:
        module_name = PAGES[page_name]
        module = importlib.import_module(module_name)

        # Route-level lazy loading: page module is imported only on demand above.
        if page_name in {"Operations Hub", "Lineage", "Training Studio", "Strategy Studio"}:
            module.render(project_root, job_service)
        else:
            module.render(project_root)
    except Exception as exc:
        ui_components.render_section_error(f"Page rendering failed: {page_name}", str(exc))
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Go to Overview", key="page_error_to_overview", width="stretch"):
                st.session_state["selected_page"] = "Overview"
                st.rerun()
        with c2:
            if st.button("Retry this page", key="page_error_retry", width="stretch"):
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="RL Trading Dashboard", layout="wide")
    ensure_state_defaults()
    config = load_config()
    job_service = JobService(config)

    # Hide Streamlit's auto multipage nav to avoid duplicated navigation UI.
    st.markdown(
        "<style>[data-testid='stSidebarNav']{display:none;}</style>",
        unsafe_allow_html=True,
    )
    _inject_sidebar_spacing_styles()

    st.title("RL Trading Agent - UI Dashboard")
    st.caption("Data-intensive control center for trading, training, fine-tuning, and artifact observability.")
    st.caption(f"Project root: `{config.project_root}`")
    if config.api_base_url:
        st.caption(f"Backend API mode enabled: `{config.api_base_url}`")
    else:
        st.caption("Backend API mode disabled: using local process adapter.")

    _render_sidebar(job_service, config.project_root)

    monitors_enabled = bool(st.session_state.get("background_monitors_enabled", False))
    selected_page = str(st.session_state.get("selected_page", "Overview"))

    if monitors_enabled and selected_page in {"Overview", "Operations Hub"}:
        if hasattr(st, "fragment"):
            @st.fragment(run_every="15s")
            def _job_notifications_fragment() -> None:
                _track_job_completion_notifications(job_service)

            _job_notifications_fragment()
        else:
            _track_job_completion_notifications(job_service)

    if monitors_enabled and st.session_state.get("artifact_watch_enabled", False) and selected_page in {"Overview", "Artifacts"}:
        if hasattr(st, "fragment"):
            @st.fragment(run_every="20s")
            def _artifact_watch_fragment() -> None:
                _check_artifact_file_watcher(config.project_root, rerun_on_change=False)

            _artifact_watch_fragment()
        else:
            _check_artifact_file_watcher(config.project_root)

    if st.session_state["refresh_enabled"]:
        st.markdown(
            f"<meta http-equiv='refresh' content='{st.session_state['refresh_seconds']}'>",
            unsafe_allow_html=True,
        )

    _render_page(st.session_state["selected_page"], config.project_root, job_service)


if __name__ == "__main__":
    main()
