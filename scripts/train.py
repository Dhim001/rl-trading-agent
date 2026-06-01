#!/usr/bin/env python3
"""Train a reinforcement learning trading agent."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_trading_agent.config import load_config
from rl_trading_agent.data.pipeline import load_dataset, save_feature_stats
from rl_trading_agent.training.trainer import split_train_test, train_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data, stats, mode = load_dataset(cfg, ROOT)
    save_feature_stats(stats, ROOT / cfg["data"]["stats_file"])

    train_data, eval_data = split_train_test(data, cfg["training"]["train_split"])
    model_path = train_agent(
        train_data=train_data,
        eval_data=eval_data,
        mode=mode,
        env_cfg=cfg["environment"],
        risk_cfg=cfg["risk"],
        training_cfg={
            **cfg["training"],
            "log_dir": str(ROOT / cfg["training"]["log_dir"]),
            "model_dir": str(ROOT / cfg["training"]["model_dir"]),
        },
    )
    print(f"Training complete ({mode}-stock mode). Model saved to {model_path}")


if __name__ == "__main__":
    main()
