from __future__ import annotations

from pathlib import Path
from typing import Callable

import streamlit as st

from ui_dashboard.services.data_service import age_level, format_age


def freshness_badge(label: str, path: Path) -> str:
    level = age_level(path)
    color_map = {
        "fresh": "#1B5E20",
        "stale": "#8A6D1D",
        "old": "#8B1A1A",
        "missing": "#5A5A5A",
    }
    color = color_map.get(level, "#5A5A5A")
    return (
        f"<span><b>{label}:</b> "
        f"<span style='color:{color};font-weight:600'>{format_age(path)}</span></span>"
    )


def render_freshness_row(items: list[tuple[str, Path]]) -> None:
    st.markdown(" | ".join([freshness_badge(label, path) for label, path in items]), unsafe_allow_html=True)


def render_section_error(message: str, details: str | None = None) -> None:
    st.error(message)
    if details:
        with st.expander("Error details"):
            st.code(details, language="text")


def render_empty_state(message: str, hint: str | None = None) -> None:
    st.info(message)
    if hint:
        st.caption(hint)


def render_banner(level: str, message: str) -> None:
    """Render a consistent semantic banner."""
    normalized = level.lower().strip()
    if normalized == "success":
        st.success(message)
    elif normalized == "warning":
        st.warning(message)
    elif normalized == "error":
        st.error(message)
    else:
        st.info(message)


def render_loading_state(label: str, body: Callable[[], None]) -> None:
    """Wrap a UI block with a consistent loading spinner."""
    with st.spinner(label):
        body()


def render_retry_hint(action_label: str = "Refresh Jobs") -> None:
    st.caption(
        f"If this view looks stale or incomplete, use **{action_label}** or enable auto-refresh in the sidebar."
    )


def render_component_guard(
    title: str,
    body: Callable[[], None],
    key_prefix: str,
) -> None:
    try:
        body()
    except Exception as exc:
        render_section_error(f"{title} failed to render.", str(exc))
        c1, c2 = st.columns(2)
        with c1:
            if st.button(f"Retry {title}", key=f"{key_prefix}_retry", width="stretch"):
                st.rerun()
        with c2:
            st.button("Dismiss", key=f"{key_prefix}_dismiss", width="stretch", disabled=True)
