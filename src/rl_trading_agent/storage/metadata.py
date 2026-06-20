from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_run_id(workflow: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{workflow}_{stamp}_{uuid.uuid4().hex[:8]}"


def get_run_dir(project_root: Path, workflow: str, run_id: str) -> Path:
    path = project_root / "artifacts" / workflow / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _db_path(project_root: Path) -> Path:
    path = project_root / "artifacts" / "metadata.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect(project_root: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(project_root))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            workflow TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            config_path TEXT,
            params_json TEXT,
            metrics_json TEXT,
            error TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            path TEXT NOT NULL,
            format TEXT,
            size_bytes INTEGER,
            created_at TEXT NOT NULL,
            metadata_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lineage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            input_path TEXT,
            output_path TEXT,
            relation TEXT NOT NULL,
            recorded_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def start_run(
    project_root: Path,
    run_id: str,
    workflow: str,
    config_path: str | None = None,
    params: dict[str, Any] | None = None,
) -> None:
    with _connect(project_root) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO runs (
                run_id, workflow, status, started_at, ended_at, config_path, params_json, metrics_json, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                workflow,
                "running",
                utc_now_iso(),
                None,
                config_path,
                json.dumps(params or {}),
                None,
                None,
            ),
        )
        conn.commit()


def complete_run(
    project_root: Path,
    run_id: str,
    metrics: dict[str, Any] | None = None,
    status: str = "completed",
    error: str | None = None,
) -> None:
    with _connect(project_root) as conn:
        conn.execute(
            """
            UPDATE runs
            SET status = ?, ended_at = ?, metrics_json = ?, error = ?
            WHERE run_id = ?
            """,
            (status, utc_now_iso(), json.dumps(metrics or {}), error, run_id),
        )
        conn.commit()


def record_artifact(
    project_root: Path,
    run_id: str,
    kind: str,
    path: str | Path,
    fmt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    full = Path(path)
    if not full.is_absolute():
        full = project_root / full
    rel = str(full.relative_to(project_root)).replace("\\", "/")
    size_bytes = int(full.stat().st_size) if full.exists() and full.is_file() else None
    with _connect(project_root) as conn:
        conn.execute(
            """
            INSERT INTO artifacts (run_id, kind, path, format, size_bytes, created_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                kind,
                rel,
                fmt,
                size_bytes,
                utc_now_iso(),
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()


def record_lineage(
    project_root: Path,
    run_id: str,
    input_path: str | None,
    output_path: str | None,
    relation: str,
) -> None:
    with _connect(project_root) as conn:
        conn.execute(
            """
            INSERT INTO lineage (run_id, input_path, output_path, relation, recorded_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, input_path, output_path, relation, utc_now_iso()),
        )
        conn.commit()


def copy_latest_pointer(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)

