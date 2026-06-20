from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Any


@contextmanager
def file_lock(target_path: str | Path, timeout_seconds: float = 8.0) -> Iterator[None]:
    """Simple cross-process lock using sidecar .lock file.

    If a stale lock remains from a crashed process, it is evicted after timeout.
    """
    target = Path(target_path)
    lock_path = target.with_suffix(target.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    acquired = False
    while not acquired:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
        except FileExistsError:
            if (time.time() - start) > timeout_seconds:
                try:
                    lock_age = time.time() - lock_path.stat().st_mtime
                except Exception:
                    lock_age = timeout_seconds + 1.0
                if lock_age > timeout_seconds:
                    try:
                        lock_path.unlink(missing_ok=True)
                    except Exception:
                        raise TimeoutError(f"Timed out waiting for lock: {lock_path}") from None
                    continue
                raise TimeoutError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(0.05)

    try:
        yield
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2))

