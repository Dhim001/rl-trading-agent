from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from websocket import create_connection
except Exception:  # pragma: no cover - optional dependency at runtime
    create_connection = None


WATCHED_ARTIFACT_DIRS = ("models", "results", "tuning", "paper", "runs", "data")


def snapshot_artifacts(project_root: Path) -> dict[str, tuple[int, int]]:
    """Return a lightweight file watcher snapshot: relpath -> (mtime_ns, size)."""
    snapshot: dict[str, tuple[int, int]] = {}
    for folder in WATCHED_ARTIFACT_DIRS:
        base = project_root / folder
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            stat = path.stat()
            snapshot[str(path.relative_to(project_root))] = (int(stat.st_mtime_ns), int(stat.st_size))
    return snapshot


def detect_artifact_changes(
    previous: dict[str, tuple[int, int]],
    current: dict[str, tuple[int, int]],
) -> list[str]:
    changed: list[str] = []
    for rel_path, stamp in current.items():
        if rel_path not in previous:
            changed.append(f"created:{rel_path}")
            continue
        if previous[rel_path] != stamp:
            changed.append(f"updated:{rel_path}")
    for rel_path in previous:
        if rel_path not in current:
            changed.append(f"deleted:{rel_path}")
    return changed


def read_websocket_event(ws_url: str, timeout_seconds: float = 0.5) -> dict[str, Any] | None:
    """
    Best-effort read of a single websocket payload.
    Expects JSON message shape for paper updates.
    """
    if not ws_url or create_connection is None:
        return None
    ws = None
    try:
        ws = create_connection(ws_url, timeout=timeout_seconds)
        payload = ws.recv()
        if not payload:
            return None
        data = json.loads(payload)
        if isinstance(data, dict):
            return data
        return {"value": data}
    except Exception:
        return None
    finally:
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
