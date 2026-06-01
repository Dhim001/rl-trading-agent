from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from stable_baselines3 import DQN, PPO

from rl_trading_agent.env.factory import make_trading_env
from rl_trading_agent.evaluation.metrics import save_equity_curve, summarize_backtest


def load_model(model_path: str | Path, env):
    path = str(model_path)
    if "dqn" in path.lower():
        return DQN.load(path, env=env)
    return PPO.load(path, env=env)


def run_backtest(
    model_path: str | Path,
    test_data: pd.DataFrame | dict[str, pd.DataFrame],
    mode: str,
    env_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    output_dir: str | Path = "results",
) -> dict[str, float]:
    env = make_trading_env(test_data, mode, env_cfg, risk_cfg)
    model = load_model(model_path, env)

    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        if mode == "multi":
            action = np.asarray(action, dtype=int)
        else:
            action = int(action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    metrics = summarize_backtest(env.equity_curve)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_equity_curve(env.equity_curve, out / "equity_curve.csv")
    return metrics
