from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import streamlit as st

from ui_dashboard.services.data_service import load_equity_curve, load_json, load_jsonl


def _file_mtime_ns(path: Path) -> int:
    try:
        return int(path.stat().st_mtime_ns)
    except Exception:
        return -1


@st.cache_resource(show_spinner=False)
def get_redis_client():
    url = os.getenv("RL_DASHBOARD_REDIS_URL", "").strip()
    if not url:
        return None
    try:
        import redis

        client = redis.Redis.from_url(
            url,
            decode_responses=False,
            socket_connect_timeout=0.25,
            socket_timeout=0.25,
            retry_on_timeout=False,
        )
        client.ping()
        return client
    except Exception:
        return None


def _redis_get(key: str) -> Any | None:
    client = get_redis_client()
    if client is None:
        return None
    try:
        raw = client.get(key)
        if raw is None:
            return None
        return pickle.loads(raw)
    except Exception:
        return None


def _redis_set(key: str, value: Any, ttl_seconds: int) -> None:
    client = get_redis_client()
    if client is None:
        return
    try:
        client.setex(key, ttl_seconds, pickle.dumps(value))
    except Exception:
        return


@st.cache_data(ttl=3600, show_spinner=False)
def _load_backtest_curve_local(path_str: str, mtime_ns: int):
    _ = mtime_ns
    return load_equity_curve(Path(path_str))


def get_backtest_curve_cached(path: Path):
    mtime = _file_mtime_ns(path)
    key = f"dash:backtest_curve:{path.as_posix()}:{mtime}"
    redis_cached = _redis_get(key)
    if redis_cached is not None:
        return redis_cached
    df = _load_backtest_curve_local(str(path), mtime)
    _redis_set(key, df, ttl_seconds=3600)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def _load_paper_state_local(path_str: str, mtime_ns: int):
    _ = mtime_ns
    return load_json(Path(path_str))


def get_paper_state_cached(path: Path):
    mtime = _file_mtime_ns(path)
    key = f"dash:paper_state:{path.as_posix()}:{mtime}"
    redis_cached = _redis_get(key)
    if redis_cached is not None:
        return redis_cached
    payload = _load_paper_state_local(str(path), mtime)
    _redis_set(key, payload, ttl_seconds=300)
    return payload


@st.cache_data(ttl=300, show_spinner=False)
def _load_paper_log_local(path_str: str, mtime_ns: int):
    _ = mtime_ns
    return load_jsonl(Path(path_str))


def get_paper_log_cached(path: Path):
    mtime = _file_mtime_ns(path)
    key = f"dash:paper_log:{path.as_posix()}:{mtime}"
    redis_cached = _redis_get(key)
    if redis_cached is not None:
        return redis_cached
    rows = _load_paper_log_local(str(path), mtime)
    _redis_set(key, rows, ttl_seconds=300)
    return rows


@st.cache_resource(show_spinner=False)
def load_rl_model_cached(model_path: str):
    from stable_baselines3 import DQN, PPO

    lower = model_path.lower()
    if "dqn" in lower:
        return DQN.load(model_path)
    return PPO.load(model_path)


def _redis_delete_prefix(prefix: str) -> None:
    client = get_redis_client()
    if client is None:
        return
    try:
        keys = list(client.scan_iter(match=f"{prefix}*"))
        if keys:
            client.delete(*keys)
    except Exception:
        return


def invalidate_dashboard_caches(changes: list[str]) -> None:
    if not changes:
        return
    st.cache_data.clear()
    for item in changes:
        rel = item.split(":", 1)[-1].strip().lower()
        if rel.startswith("paper/"):
            _redis_delete_prefix("dash:paper_")
        if rel.startswith("results/"):
            _redis_delete_prefix("dash:backtest_curve:")
        if rel.startswith("models/") or "/models/" in rel:
            st.cache_resource.clear()
