from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    id: str
    workflow: str
    command: list[str]
    status: str
    pid: int | None = None
    started_at: str = field(default_factory=utc_now_iso)
    ended_at: str | None = None
    return_code: int | None = None
    log_path: str | None = None
    source: str = "local"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobRecord":
        return cls(**data)
