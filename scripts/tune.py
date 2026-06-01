#!/usr/bin/env python3
"""Hyperparameter tuning with Optuna."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_trading_agent.config import load_config
from rl_trading_agent.data.pipeline import load_dataset, save_feature_stats
from rl_trading_agent.training.trainer import split_train_test
from rl_trading_agent.training.tuner import run_hyperparameter_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune RL trading hyperparameters")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    tuning_cfg = cfg["tuning"]
    if args.trials:
        tuning_cfg = {**tuning_cfg, "n_trials": args.trials}

    data, stats, mode = load_dataset(cfg, ROOT)
    save_feature_stats(stats, ROOT / cfg["data"]["stats_file"])

    train_data, remainder = split_train_test(data, cfg["training"]["train_split"])
    eval_data, test_data = split_train_test(remainder, 0.5)

    study = run_hyperparameter_search(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        mode=mode,
        env_cfg=cfg["environment"],
        risk_cfg=cfg["risk"],
        training_cfg={
            **cfg["training"],
            "log_dir": str(ROOT / cfg["training"]["log_dir"]),
            "model_dir": str(ROOT / cfg["training"]["model_dir"]),
        },
        tuning_cfg=tuning_cfg,
        root=ROOT,
    )

    print("Best trial:")
    print(f"  value (Sharpe): {study.best_value:.4f}")
    print(f"  params: {study.best_params}")


if __name__ == "__main__":
    main()
