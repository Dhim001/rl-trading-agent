from __future__ import annotations

from typing import Any

import requests


class RemoteApiAdapter:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def launch_job(self, workflow: str, command: list[str], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/jobs",
            json={"workflow": workflow, "command": command, "metadata": metadata or {}},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def list_jobs(self) -> list[dict[str, Any]]:
        response = requests.get(f"{self.base_url}/jobs", timeout=30)
        response.raise_for_status()
        return response.json()

    def stop_job(self, job_id: str) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/jobs/{job_id}/stop", timeout=30)
        response.raise_for_status()
        return response.json()

    def get_job_log_tail(self, job_id: str, lines: int = 80) -> str:
        response = requests.get(f"{self.base_url}/jobs/{job_id}/logs", params={"lines": lines}, timeout=30)
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("tail", ""))
