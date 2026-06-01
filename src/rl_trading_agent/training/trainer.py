from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_trading_agent.env.factory import make_trading_env


def split_train_test(
    data: pd.DataFrame | dict[str, pd.DataFrame],
    train_split: float,
) -> tuple[Any, Any]:
    if isinstance(data, dict):
        symbols = sorted(data.keys())
        length = len(data[symbols[0]])
        split_idx = int(length * train_split)
        train = {s: data[s].iloc[:split_idx].copy() for s in symbols}
        test = {s: data[s].iloc[split_idx:].copy() for s in symbols}
        return train, test

    split_idx = int(len(data) * train_split)
    return data.iloc[:split_idx].copy(), data.iloc[split_idx:].copy()


def build_model(algorithm: str, env, training_cfg: dict[str, Any]):
    common = {
        "learning_rate": training_cfg["learning_rate"],
        "gamma": training_cfg["gamma"],
        "verbose": 0,
        "seed": training_cfg["seed"],
        "tensorboard_log": training_cfg.get("log_dir"),
    }
    if algorithm.upper() == "PPO":
        return PPO("MlpPolicy", env, batch_size=training_cfg["batch_size"], **common)
    if algorithm.upper() == "DQN":
        return DQN("MlpPolicy", env, batch_size=training_cfg["batch_size"], **common)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def train_agent(
    train_data: pd.DataFrame | dict[str, pd.DataFrame],
    eval_data: pd.DataFrame | dict[str, pd.DataFrame],
    mode: str,
    env_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    training_cfg: dict[str, Any],
) -> Path:
    model_dir = Path(training_cfg["model_dir"])
    log_dir = Path(training_cfg["log_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([
        lambda: Monitor(make_trading_env(train_data, mode, env_cfg, risk_cfg))
    ])
    eval_env = DummyVecEnv([
        lambda: Monitor(make_trading_env(eval_data, mode, env_cfg, risk_cfg))
    ])

    model = build_model(training_cfg["algorithm"], train_env, training_cfg)
    model.verbose = 1

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best"),
            log_path=str(log_dir / "eval") if log_dir else None,
            eval_freq=training_cfg["eval_freq"],
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=training_cfg["save_freq"],
            save_path=str(model_dir / "checkpoints"),
            name_prefix="rl_trader",
        ),
    ]

    model.learn(total_timesteps=training_cfg["total_timesteps"], callback=callbacks)

    final_path = model_dir / "final_model.zip"
    model.save(final_path)
    return final_path
