from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import uuid
import csv
import io
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ui_dashboard.core.config import AppConfig
from ui_dashboard.core.types import JobRecord
from ui_dashboard.services.api_adapter import RemoteApiAdapter
from rl_trading_agent.dashboard.workflows import WORKFLOW_COMMANDS, WORKFLOW_DEFINITIONS, WORKFLOW_LABELS


FINAL_STATUSES = {"completed", "failed", "stopped"}
ACTIVE_STATUSES = {"running", "queued"}


WORKFLOW_ARTIFACTS: dict[str, list[str]] = {
    "download_data": ["data/feature_stats.json"],
    "training": ["models/final_model.zip"],
    "backtest": ["results/equity_curve.csv"],
    "fine_tuning": ["tuning/best_params.json"],
    "paper_once": ["paper/portfolio_state.json", "paper/trades.log"],
    "paper_loop": ["paper/portfolio_state.json", "paper/trades.log"],
}

WORKFLOW_INPUT_ARTIFACTS: dict[str, list[str]] = {
    "download_data": [],
    "training": ["data/feature_stats.json"],
    "backtest": ["data/feature_stats.json", "models/final_model.zip"],
    "fine_tuning": ["data/feature_stats.json"],
    "paper_once": ["data/feature_stats.json", "models/final_model.zip"],
    "paper_loop": ["data/feature_stats.json", "models/final_model.zip"],
}


class JobService:
    """Operational job service for dashboard workflow control and monitoring."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.api = RemoteApiAdapter(config.api_base_url) if config.api_base_url else None
        self.lineage_db_path = self.config.runtime_dir / "lineage_events.jsonl"

    def _python_executable(self) -> str:
        venv_python = self.config.project_root / ".venv" / "Scripts" / "python.exe"
        if venv_python.exists():
            return str(venv_python)
        return sys.executable

    def _load_jobs_local(self) -> list[JobRecord]:
        if not self.config.jobs_db_path.exists():
            return []
        with open(self.config.jobs_db_path, encoding="utf-8") as f:
            rows = json.load(f)
        return [JobRecord.from_dict(row) for row in rows]

    def _save_jobs_local(self, jobs: list[JobRecord]) -> None:
        with open(self.config.jobs_db_path, "w", encoding="utf-8") as f:
            json.dump([job.to_dict() for job in jobs], f, indent=2)

    def _tasklist_running(self, pid: int) -> bool:
        completed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (completed.stdout or "").strip()
        return bool(output and not output.startswith("INFO: No tasks"))

    def _running_pids(self) -> set[int]:
        completed = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (completed.stdout or "").strip()
        if not output:
            return set()
        reader = csv.reader(io.StringIO(output))
        pids: set[int] = set()
        for row in reader:
            if len(row) < 2:
                continue
            try:
                pids.add(int(str(row[1]).replace('"', "").strip()))
            except Exception:
                continue
        return pids

    def _safe_read_log_tail(self, path: Path, lines: int = 80) -> str:
        """Efficiently read the last N lines from a potentially large log file."""
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            block_size = 2048
            data = b""
            while file_size > 0 and data.count(b"\n") <= lines:
                read_size = min(block_size, file_size)
                file_size -= read_size
                f.seek(file_size)
                data = f.read(read_size) + data
            try:
                text = data.decode("utf-8", errors="replace")
            except Exception:
                text = data.decode(errors="replace")
        return "\n".join(text.splitlines()[-lines:])

    def _get_local_job(self, job_id: str) -> JobRecord | None:
        jobs = self._load_jobs_local()
        return next((job for job in jobs if job.id == job_id), None)

    def _refresh_local_statuses(self, jobs: list[JobRecord]) -> list[JobRecord]:
        changed = False
        now_iso = datetime.now(timezone.utc).isoformat()
        running_pids = self._running_pids()
        for job in jobs:
            if job.status not in ACTIVE_STATUSES:
                continue
            if job.pid is None:
                continue
            if job.pid in running_pids:
                job.metadata["last_checked_at"] = now_iso
                continue

            # Process is no longer active. Determine status from return code if known.
            if job.return_code is None:
                job.status = "completed"
                job.return_code = 0
            else:
                job.status = "completed" if job.return_code == 0 else "failed"
            if job.ended_at is None:
                job.ended_at = now_iso
            job.metadata["last_checked_at"] = now_iso
            self._capture_lineage_for_job(job)
            changed = True
        if changed:
            self._save_jobs_local(jobs)
        return jobs

    def _sha256(self, path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1024 * 64)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()[:12]

    def _artifact_fingerprint(self, rel_path: str) -> dict[str, Any] | None:
        path = self.config.project_root / rel_path
        if not path.exists() or not path.is_file():
            return None
        stat = path.stat()
        return {
            "path": rel_path,
            "size_bytes": int(stat.st_size),
            "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            "version": f"{int(stat.st_mtime)}-{stat.st_size}-{self._sha256(path)}",
        }

    def _resolve_workflow_input_versions(self, workflow: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rel in WORKFLOW_INPUT_ARTIFACTS.get(workflow, []):
            fp = self._artifact_fingerprint(rel)
            if fp is not None:
                rows.append(fp)
        return rows

    def _resolve_workflow_output_versions(self, workflow: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for rel in WORKFLOW_ARTIFACTS.get(workflow, []):
            fp = self._artifact_fingerprint(rel)
            if fp is not None:
                rows.append(fp)
        return rows

    def _append_lineage_event(self, payload: dict[str, Any]) -> None:
        self.lineage_db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lineage_db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def _capture_lineage_for_job(self, job: JobRecord) -> None:
        if job.metadata.get("lineage_recorded"):
            return
        if job.status not in FINAL_STATUSES:
            return
        event = {
            "event": "job_finalized",
            "job_id": job.id,
            "workflow": job.workflow,
            "status": job.status,
            "started_at": job.started_at,
            "ended_at": job.ended_at,
            "inputs": job.metadata.get("input_artifacts", []),
            "outputs": self._resolve_workflow_output_versions(job.workflow),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self._append_lineage_event(event)
        job.metadata["lineage_recorded"] = True

    def get_workflow_catalog(self) -> list[dict[str, Any]]:
        """Return canonical workflow metadata for the Operations Hub."""
        return WORKFLOW_DEFINITIONS

    def validate_workflow_artifacts(self, workflow: str) -> dict[str, Any]:
        """
        Validate expected outputs for a workflow.

        Returns a summary with freshness and existence information for each artifact.
        """
        expected = WORKFLOW_ARTIFACTS.get(workflow, [])
        rows: list[dict[str, Any]] = []
        all_present = True
        for rel in expected:
            path = self.config.project_root / rel
            exists = path.exists()
            all_present = all_present and exists
            modified_at = (
                datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat() if exists else None
            )
            rows.append(
                {
                    "artifact": rel,
                    "exists": exists,
                    "modified_at": modified_at,
                }
            )
        return {
            "workflow": workflow,
            "all_present": all_present,
            "artifacts": rows,
        }

    def get_job_progress(self, job: dict[str, Any]) -> dict[str, Any]:
        """Return coarse progress signals for display in the UI."""
        status = str(job.get("status", "unknown"))
        if status in FINAL_STATUSES:
            return {"progress_pct": 100, "phase": status}
        if status in ACTIVE_STATUSES:
            return {"progress_pct": 50, "phase": "running"}
        return {"progress_pct": 0, "phase": "unknown"}

    def list_jobs(self) -> list[dict[str, Any]]:
        if self.api is not None:
            return self.api.list_jobs()
        jobs = self._refresh_local_statuses(self._load_jobs_local())
        return [job.to_dict() for job in sorted(jobs, key=lambda j: j.started_at, reverse=True)]

    def launch_workflow(self, workflow: str, run_mode: str = "background", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        command = WORKFLOW_COMMANDS[workflow]
        if self.api is not None:
            return self.api.launch_job(workflow=workflow, command=command, metadata=metadata)
        return self._launch_workflow_local(workflow, command, run_mode, metadata)

    def _launch_workflow_local(
        self,
        workflow: str,
        command: list[str],
        run_mode: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        python_exe = self._python_executable()
        full_command = [python_exe, *command]
        job_id = str(uuid.uuid4())
        log_path = self.config.jobs_dir / f"{job_id}.log"
        jobs = self._load_jobs_local()

        if run_mode == "foreground":
            completed = subprocess.run(
                full_command,
                cwd=self.config.project_root,
                capture_output=True,
                text=True,
                check=False,
            )
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(completed.stdout or "")
                if completed.stderr:
                    f.write("\n[stderr]\n")
                    f.write(completed.stderr)

            record = JobRecord(
                id=job_id,
                workflow=workflow,
                command=full_command,
                status="completed" if completed.returncode == 0 else "failed",
                pid=None,
                started_at=datetime.now(timezone.utc).isoformat(),
                ended_at=datetime.now(timezone.utc).isoformat(),
                return_code=completed.returncode,
                log_path=str(log_path),
                source="local",
                error=None if completed.returncode == 0 else "Foreground command failed",
                metadata={
                    **(metadata or {}),
                    "last_checked_at": datetime.now(timezone.utc).isoformat(),
                    "input_artifacts": self._resolve_workflow_input_versions(workflow),
                    "lineage_recorded": False,
                },
            )
            self._capture_lineage_for_job(record)
            jobs.append(record)
            self._save_jobs_local(jobs)
            return record.to_dict()

        creationflags = 0
        if hasattr(subprocess, "DETACHED_PROCESS"):
            creationflags |= subprocess.DETACHED_PROCESS
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creationflags |= subprocess.CREATE_NEW_PROCESS_GROUP

        with open(log_path, "w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                full_command,
                cwd=self.config.project_root,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )

        record = JobRecord(
            id=job_id,
            workflow=workflow,
            command=full_command,
            status="running",
            pid=process.pid,
            started_at=datetime.now(timezone.utc).isoformat(),
            ended_at=None,
            return_code=None,
            log_path=str(log_path),
            source="local",
            error=None,
            metadata={
                **(metadata or {}),
                "last_checked_at": datetime.now(timezone.utc).isoformat(),
                "input_artifacts": self._resolve_workflow_input_versions(workflow),
                "lineage_recorded": False,
            },
        )
        jobs.append(record)
        self._save_jobs_local(jobs)
        return record.to_dict()

    def stop_job(self, job_id: str) -> dict[str, Any]:
        if self.api is not None:
            return self.api.stop_job(job_id)
        jobs = self._load_jobs_local()
        target = next((j for j in jobs if j.id == job_id), None)
        if target is None:
            return {"ok": False, "error": "Job not found", "job_id": job_id}

        if target.pid and target.status in {"running", "queued"}:
            completed = subprocess.run(
                ["taskkill", "/PID", str(target.pid), "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode == 0:
                target.status = "stopped"
                target.ended_at = datetime.now(timezone.utc).isoformat()
                target.return_code = -1
                self._save_jobs_local(jobs)
                return {"ok": True, "job_id": job_id}
            return {"ok": False, "error": completed.stderr or completed.stdout, "job_id": job_id}
        return {"ok": False, "error": "Job is not running", "job_id": job_id}

    def get_job_log_tail(self, job_id: str, lines: int = 80) -> str:
        if self.api is not None:
            return self.api.get_job_log_tail(job_id, lines=lines)
        target = self._get_local_job(job_id)
        if target is None or not target.log_path:
            return ""
        path = Path(target.log_path)
        return self._safe_read_log_tail(path, lines=lines)

    def list_model_versions(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        model_dir = self.config.project_root / "models"
        tuning_dir = self.config.project_root / "tuning"
        for path in sorted(model_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
            rows.append(
                {
                    "path": str(path.relative_to(self.config.project_root)),
                    "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "size_bytes": int(path.stat().st_size),
                }
            )
        for path in sorted(tuning_dir.glob("trial_*/models/**/*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
            rows.append(
                {
                    "path": str(path.relative_to(self.config.project_root)),
                    "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                    "size_bytes": int(path.stat().st_size),
                }
            )
        return rows

    def rollback_model_version(self, model_rel_path: str) -> dict[str, Any]:
        source = self.config.project_root / model_rel_path
        if not source.exists():
            return {"ok": False, "error": f"Model not found: {model_rel_path}"}
        models_dir = self.config.project_root / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        target = models_dir / "final_model.zip"

        if target.exists():
            backup_name = f"final_model.backup.{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.zip"
            shutil.copy2(target, models_dir / backup_name)

        shutil.copy2(source, target)
        self._append_lineage_event(
            {
                "event": "model_rollback",
                "source_model": model_rel_path,
                "active_model": "models/final_model.zip",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return {"ok": True, "active_model": "models/final_model.zip", "source_model": model_rel_path}

    def launch_training_with_config(
        self,
        config_rel_path: str,
        run_mode: str = "background",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        command = ["scripts/train.py", "--config", config_rel_path]
        payload = {
            **(metadata or {}),
            "config_path": config_rel_path,
            "triggered_from": "training_studio",
        }
        if self.api is not None:
            return self.api.launch_job(workflow="training", command=command, metadata=payload)
        return self._launch_workflow_local("training", command, run_mode, payload)

    def clone_model_artifact(self, model_rel_path: str) -> dict[str, Any]:
        source = self.config.project_root / model_rel_path
        if not source.exists():
            return {"ok": False, "error": f"Model not found: {model_rel_path}"}
        clones_dir = self.config.project_root / "models" / "clones"
        clones_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        suffix = source.suffix or ".zip"
        clone_name = f"{source.stem}.clone.{stamp}{suffix}"
        target = clones_dir / clone_name
        shutil.copy2(source, target)
        rel_target = str(target.relative_to(self.config.project_root)).replace("\\", "/")
        self._append_lineage_event(
            {
                "event": "model_cloned",
                "source_model": model_rel_path,
                "clone_model": rel_target,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return {"ok": True, "source_model": model_rel_path, "clone_model": rel_target}
