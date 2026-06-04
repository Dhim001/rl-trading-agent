from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    runtime_dir: Path
    jobs_dir: Path
    jobs_db_path: Path
    api_base_url: str | None


def load_config() -> AppConfig:
    project_root = Path(__file__).resolve().parents[2]
    runtime_dir = project_root / "ui_dashboard" / ".runtime"
    jobs_dir = runtime_dir / "jobs"
    jobs_db_path = runtime_dir / "jobs.json"
    api_base_url = os.getenv("RL_DASHBOARD_API_BASE")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return AppConfig(
        project_root=project_root,
        runtime_dir=runtime_dir,
        jobs_dir=jobs_dir,
        jobs_db_path=jobs_db_path,
        api_base_url=api_base_url,
    )
