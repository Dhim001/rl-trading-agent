from __future__ import annotations

import html
import importlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

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
    "Metadata Registry": "ui_dashboard.pages.metadata_registry_page",
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
    "Metadata Registry": {"group": "Operations", "icon": "🗃️", "desc": "SQLite run/artifact/lineage registry"},
    "Operations Hub": {"group": "Operations", "icon": "🛠️", "desc": "Run and manage workflows"},
    "Artifacts": {"group": "Operations", "icon": "📦", "desc": "Generated artifact inventory"},
}

PAGE_GROUP_ORDER = ["Dashboard", "Research", "Analysis", "Execution", "Operations"]


def _inject_sidebar_spacing_styles() -> None:
    st.markdown(
        """
        <style>
        .skip-link {
            position: absolute;
            left: -9999px;
            top: 0.5rem;
            z-index: 99999;
            background: #111;
            color: #fff !important;
            padding: 0.5rem 0.75rem;
            border-radius: 0.5rem;
            text-decoration: none;
        }
        .skip-link:focus {
            left: 0.75rem;
        }
        .sr-only {
            position: absolute !important;
            width: 1px !important;
            height: 1px !important;
            padding: 0 !important;
            margin: -1px !important;
            overflow: hidden !important;
            clip: rect(0, 0, 0, 0) !important;
            white-space: nowrap !important;
            border: 0 !important;
        }
        html.a11y-strong-focus *:focus-visible {
            outline: 3px solid #ffbf47 !important;
            outline-offset: 2px !important;
            border-radius: 4px;
        }
        body.a11y-high-contrast {
            background: #000 !important;
            color: #fff !important;
        }
        body.a11y-high-contrast [data-testid="stAppViewContainer"],
        body.a11y-high-contrast [data-testid="stHeader"],
        body.a11y-high-contrast [data-testid="stSidebar"] {
            background: #000 !important;
            color: #fff !important;
        }
        body.a11y-high-contrast button,
        body.a11y-high-contrast input,
        body.a11y-high-contrast select,
        body.a11y-high-contrast textarea {
            background: #000 !important;
            color: #fff !important;
            border: 2px solid #fff !important;
        }
        body.a11y-high-contrast a,
        body.a11y-high-contrast p,
        body.a11y-high-contrast span,
        body.a11y-high-contrast label,
        body.a11y-high-contrast div {
            color: #fff !important;
        }
        body.a11y-high-contrast [data-testid="stMetricValue"],
        body.a11y-high-contrast [data-testid="stMetricLabel"] {
            color: #fff !important;
        }
        [data-testid="stAppViewContainer"] .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1.5rem;
            max-width: 1360px;
        }
        [data-testid="stAppViewContainer"] .main h1 {
            margin-bottom: 0.35rem;
        }
        [data-testid="stAppViewContainer"] .main [data-testid="stCaptionContainer"] {
            margin-top: 0.1rem;
            margin-bottom: 0.55rem;
            line-height: 1.35;
        }
        [data-testid="stAppViewContainer"] .main [data-testid="stHorizontalBlock"] {
            gap: 0.75rem;
        }
        [data-testid="stAppViewContainer"] .main [data-testid="stMetric"] {
            padding-top: 0.15rem;
            padding-bottom: 0.2rem;
        }
        [data-testid="stAppViewContainer"] .main [data-testid="stPlotlyChart"],
        [data-testid="stAppViewContainer"] .main [data-testid="stDataFrame"] {
            margin-top: 0.35rem;
            margin-bottom: 0.35rem;
        }
        [data-testid="stAppViewContainer"] .main [data-testid="stExpander"] {
            margin-top: 0.35rem;
            margin-bottom: 0.35rem;
        }
        [data-testid="stAppViewContainer"] .main .stButton {
            margin-top: 0.2rem;
            margin-bottom: 0.2rem;
        }
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


def _queue_accessibility_announcement(message: str) -> None:
    if not st.session_state.get("a11y_screen_reader_announcements", True):
        return
    stamp = datetime.now().strftime("%H:%M:%S")
    st.session_state["a11y_live_announcement"] = f"{stamp} - {message}"


def _render_screen_reader_live_region() -> None:
    if not st.session_state.get("a11y_screen_reader_announcements", True):
        return
    msg = str(st.session_state.get("a11y_live_announcement", "")).strip()
    if not msg:
        return
    st.markdown(
        f"<div class='sr-only' role='status' aria-live='polite' aria-atomic='true'>{html.escape(msg)}</div>",
        unsafe_allow_html=True,
    )


def _inject_accessibility_runtime() -> None:
    high_contrast = "true" if st.session_state.get("a11y_high_contrast", False) else "false"
    strong_focus = "true" if st.session_state.get("a11y_strong_focus", True) else "false"
    script = """
    <script>
    (function () {
      const root = window.parent && window.parent.document ? window.parent.document : document;
      const body = root.body;
      if (body) {
        body.classList.toggle("a11y-high-contrast", __HIGH_CONTRAST__);
      }
      if (root.documentElement) {
        root.documentElement.classList.toggle("a11y-strong-focus", __STRONG_FOCUS__);
      }

      const main = root.querySelector('[data-testid="stAppViewContainer"] .main');
      if (main && !main.id) {
        main.id = "main-content-anchor";
      }

      const selector = 'button, [role="button"], input, select, textarea, a[href]';
      const deriveLabel = (el) => {
        const title = (el.getAttribute("title") || "").trim();
        const placeholder = (el.getAttribute("placeholder") || "").trim();
        const ownText = ((el.innerText || el.textContent || "") + "").trim();
        let widgetLabel = "";
        const widgetRoot = el.closest('[data-testid="stWidget"], [data-testid="stButton"], [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"]');
        if (widgetRoot) {
          const labelNode = widgetRoot.querySelector('label, [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p');
          if (labelNode) {
            widgetLabel = ((labelNode.innerText || labelNode.textContent || "") + "").trim();
          }
        }
        return widgetLabel || ownText || placeholder || title || "Interactive control";
      };

      const patchInteractiveA11y = () => {
        const nodes = root.querySelectorAll(selector);
        nodes.forEach((el) => {
          if (el.hasAttribute("disabled") || el.getAttribute("aria-hidden") === "true") {
            return;
          }
          if (!el.getAttribute("aria-label") && !el.getAttribute("aria-labelledby")) {
            el.setAttribute("aria-label", deriveLabel(el));
          }
          if (!el.hasAttribute("tabindex")) {
            el.setAttribute("tabindex", "0");
          }
        });
      };

      patchInteractiveA11y();
      if (!root.__rlA11yObserver) {
        const observer = new MutationObserver(() => patchInteractiveA11y());
        observer.observe(root.body, { childList: true, subtree: true });
        root.__rlA11yObserver = observer;
      }

      if (!root.__rlEnterActivationHandler) {
        root.addEventListener("keydown", (event) => {
          if (event.key !== "Enter") return;
          const target = event.target;
          if (!target) return;
          const isNativeControl = ["BUTTON", "INPUT", "SELECT", "TEXTAREA", "A"].includes(target.tagName);
          if (isNativeControl) return;
          if (target.getAttribute("role") === "button") {
            event.preventDefault();
            target.click();
          }
        }, true);
        root.__rlEnterActivationHandler = true;
      }
    })();
    </script>
    """
    script = script.replace("__HIGH_CONTRAST__", high_contrast).replace("__STRONG_FOCUS__", strong_focus)
    components.html(script, height=0, width=0)


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
            _queue_accessibility_announcement(f"Navigated to {quick}.")
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
                    _queue_accessibility_announcement(f"Navigated to {page}.")
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
            _queue_accessibility_announcement(msg)
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
    _queue_accessibility_announcement(msg)
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
    st.session_state.setdefault("a11y_high_contrast", False)
    st.session_state.setdefault("a11y_strong_focus", True)
    st.session_state.setdefault("a11y_screen_reader_announcements", True)
    st.session_state.setdefault("a11y_live_announcement", "")

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
    st.sidebar.markdown("### Accessibility")
    st.session_state["a11y_high_contrast"] = st.sidebar.checkbox(
        "High contrast mode",
        value=st.session_state["a11y_high_contrast"],
        help="Apply high-contrast colors across the dashboard.",
    )
    st.session_state["a11y_strong_focus"] = st.sidebar.checkbox(
        "Strong focus indicators",
        value=st.session_state["a11y_strong_focus"],
        help="Show strong visible outlines while tabbing through controls.",
    )
    st.session_state["a11y_screen_reader_announcements"] = st.sidebar.checkbox(
        "Screen reader announcements",
        value=st.session_state["a11y_screen_reader_announcements"],
        help="Announce page navigation and workflow status changes.",
    )
    st.sidebar.caption("Keyboard: use Tab / Shift+Tab to move, Enter to activate focused controls.")

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
    st.markdown("<a class='skip-link' href='#main-content-anchor'>Skip to main content</a>", unsafe_allow_html=True)

    st.title("RL Trading Agent Dashboard")
    st.caption("Unified workspace for strategy research, model operations, execution oversight, and artifact observability.")
    if config.api_base_url:
        st.caption(f"Backend API mode enabled: `{config.api_base_url}`")
    else:
        st.caption("Backend API mode disabled: using local process adapter.")

    _render_sidebar(job_service, config.project_root)
    _inject_accessibility_runtime()
    _render_screen_reader_live_region()

    monitors_enabled = bool(st.session_state.get("background_monitors_enabled", False))
    selected_page = str(st.session_state.get("selected_page", "Overview"))
    if st.session_state.get("a11y_last_page", "") != selected_page:
        st.session_state["a11y_last_page"] = selected_page
        _queue_accessibility_announcement(f"{selected_page} page loaded.")

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
