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
from rl_trading_agent.storage import (
    complete_run,
    copy_latest_pointer,
    create_run_id,
    get_run_dir,
    record_artifact,
    start_run,
)
from rl_trading_agent.training.trainer import split_train_test
from rl_trading_agent.training.tuner import run_hyperparameter_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune RL trading hyperparameters")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    run_id = create_run_id("fine_tuning")
    start_run(ROOT, run_id, "fine_tuning", config_path=args.config, params={"trials": args.trials})
    try:
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

        run_dir = get_run_dir(ROOT, "fine_tuning", run_id)
        trials_df = study.trials_dataframe()
        trials_df.to_parquet(run_dir / "optuna_trials.parquet", index=False)
        best_params_path = ROOT / tuning_cfg.get("study_dir", "tuning") / "best_params.json"
        if best_params_path.exists():
            copy_latest_pointer(best_params_path, run_dir / "best_params.json")
            record_artifact(ROOT, run_id, "best_params", best_params_path, "json")
            record_artifact(ROOT, run_id, "best_params_snapshot", run_dir / "best_params.json", "json")
        record_artifact(ROOT, run_id, "optuna_trials", run_dir / "optuna_trials.parquet", "parquet")

        complete_run(
            ROOT,
            run_id,
            metrics={"best_value": float(study.best_value), "best_params": study.best_params},
            status="completed",
        )
        print("Best trial:")
        print(f"  value (Sharpe): {study.best_value:.4f}")
        print(f"  params: {study.best_params}")
    except Exception as exc:
        complete_run(ROOT, run_id, status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
