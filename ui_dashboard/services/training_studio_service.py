from __future__ import annotations

import copy
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ui_dashboard.services.data_service import human_size


TRAINING_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {"learning_rate": 3e-4, "gamma": 0.99, "batch_size": 64, "eval_freq": 10000, "save_freq": 25000},
    "conservative": {"learning_rate": 1e-4, "gamma": 0.995, "batch_size": 128, "eval_freq": 8000, "save_freq": 20000},
    "aggressive": {"learning_rate": 7e-4, "gamma": 0.97, "batch_size": 32, "eval_freq": 12000, "save_freq": 30000},
}


def load_default_config(project_root: Path) -> dict[str, Any]:
    cfg_path = project_root / "config" / "default.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_runtime_training_config(project_root: Path, config: dict[str, Any]) -> Path:
    runtime_dir = project_root / "ui_dashboard" / ".runtime" / "training_studio"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = runtime_dir / f"training_config_{stamp}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return out_path


def apply_preset(base_training_cfg: dict[str, Any], preset_name: str) -> dict[str, Any]:
    out = copy.deepcopy(base_training_cfg)
    updates = TRAINING_PRESETS.get(preset_name, TRAINING_PRESETS["balanced"])
    out.update(updates)
    return out


def list_checkpoints(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    patterns = [
        project_root / "models" / "checkpoints",
        project_root / "tuning",
    ]
    for base in patterns:
        if not base.exists():
            continue
        for path in base.rglob("*.zip"):
            rel = str(path.relative_to(project_root)).replace("\\", "/")
            if "checkpoint" not in rel and "rl_trader" not in rel:
                continue
            stat = path.stat()
            rows.append(
                {
                    "path": rel,
                    "size": human_size(stat.st_size),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("modified_at", ascending=False).reset_index(drop=True)


def parse_training_progress(log_text: str) -> pd.DataFrame:
    """
    Parse common Stable-Baselines3 console metrics from logs.
    """
    if not log_text.strip():
        return pd.DataFrame()
    step_pattern = re.compile(r"total_timesteps\s*\|\s*([0-9]+)")
    reward_pattern = re.compile(r"ep_rew_mean\s*\|\s*([-+0-9.eE]+)")
    loss_pattern = re.compile(r"(?:train/loss|loss)\s*\|\s*([-+0-9.eE]+)")
    fps_pattern = re.compile(r"fps\s*\|\s*([0-9]+)")

    rows: list[dict[str, Any]] = []
    current_step: int | None = None
    current_reward: float | None = None
    current_loss: float | None = None
    current_fps: float | None = None
    for line in log_text.splitlines():
        step_match = step_pattern.search(line)
        if step_match:
            current_step = int(step_match.group(1))
        reward_match = reward_pattern.search(line)
        if reward_match:
            current_reward = float(reward_match.group(1))
        loss_match = loss_pattern.search(line)
        if loss_match:
            current_loss = float(loss_match.group(1))
        fps_match = fps_pattern.search(line)
        if fps_match:
            current_fps = float(fps_match.group(1))
        if current_step is not None and (
            current_reward is not None or current_loss is not None or current_fps is not None
        ):
            rows.append(
                {
                    "step": current_step,
                    "reward": current_reward,
                    "loss": current_loss,
                    "fps": current_fps,
                }
            )
            current_reward = None
            current_loss = None

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset=["step"], keep="last").sort_values("step")
    return df.reset_index(drop=True)


def build_training_run_table(project_root: Path, jobs: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    final_model = project_root / "models" / "final_model.zip"
    final_model_stat = final_model.stat() if final_model.exists() else None
    for job in jobs:
        if str(job.get("workflow")) != "training":
            continue
        started_at = str(job.get("started_at") or "")
        ended_at = str(job.get("ended_at") or "")
        duration_min = None
        if started_at and ended_at:
            try:
                start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
                duration_min = max(0.0, (end_dt - start_dt).total_seconds() / 60.0)
            except Exception:
                duration_min = None
        metadata = job.get("metadata") or {}
        rows.append(
            {
                "job_id": job.get("id"),
                "status": job.get("status"),
                "started_at": started_at,
                "ended_at": ended_at or "-",
                "duration_min": round(duration_min, 2) if duration_min is not None else "-",
                "preset": metadata.get("preset", "-"),
                "total_timesteps": metadata.get("total_timesteps", "-"),
                "config_path": metadata.get("config_path", "-"),
                "final_model_size": human_size(final_model_stat.st_size) if final_model_stat else "-",
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def infer_step_rate_from_jobs(jobs: list[dict[str, Any]], fallback_steps_per_sec: float = 1200.0) -> float:
    samples: list[float] = []
    for job in jobs:
        if str(job.get("workflow")) != "training":
            continue
        metadata = job.get("metadata") or {}
        timesteps = metadata.get("total_timesteps")
        started_at = str(job.get("started_at") or "")
        ended_at = str(job.get("ended_at") or "")
        if not timesteps or not started_at or not ended_at:
            continue
        try:
            start_dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
            elapsed = (end_dt - start_dt).total_seconds()
            if elapsed > 0:
                samples.append(float(timesteps) / elapsed)
        except Exception:
            continue
    if not samples:
        return fallback_steps_per_sec
    return float(sum(samples) / len(samples))


def estimate_training_time(total_timesteps: int, steps_per_sec: float) -> dict[str, Any]:
    rate = max(1.0, float(steps_per_sec))
    total_seconds = int(total_timesteps / rate)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return {
        "steps_per_sec": rate,
        "eta_seconds": total_seconds,
        "eta_hms": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
    }


def collect_resource_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "cpu_percent": None,
        "memory_percent": None,
        "memory_used_gb": None,
        "memory_total_gb": None,
        "gpu_name": None,
        "gpu_memory_allocated_gb": None,
        "gpu_memory_reserved_gb": None,
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        snapshot["cpu_percent"] = float(psutil.cpu_percent(interval=0.1))
        snapshot["memory_percent"] = float(vm.percent)
        snapshot["memory_used_gb"] = round(vm.used / (1024**3), 2)
        snapshot["memory_total_gb"] = round(vm.total / (1024**3), 2)
    except Exception:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            snapshot["gpu_name"] = str(torch.cuda.get_device_name(0))
            snapshot["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
            snapshot["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
    except Exception:
        pass
    return snapshot
