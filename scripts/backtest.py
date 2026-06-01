#!/usr/bin/env python3
"""Backtest a trained RL trading agent."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_trading_agent.config import load_config
from rl_trading_agent.data.pipeline import load_dataset
from rl_trading_agent.evaluation.backtest import run_backtest
from rl_trading_agent.training.trainer import split_train_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest RL trading agent")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default="models/final_model.zip")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data, _, mode = load_dataset(cfg, ROOT)
    _, test_data = split_train_test(data, cfg["training"]["train_split"])

    metrics = run_backtest(
        model_path=ROOT / args.model,
        test_data=test_data,
        mode=mode,
        env_cfg=cfg["environment"],
        risk_cfg=cfg["risk"],
        output_dir=ROOT / "results",
    )
    print(f"Backtest results ({mode}-stock mode):")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
