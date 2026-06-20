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
from rl_trading_agent.storage import (
    complete_run,
    copy_latest_pointer,
    create_run_id,
    get_run_dir,
    record_artifact,
    record_lineage,
    start_run,
)
from rl_trading_agent.training.trainer import split_train_test, train_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    run_id = create_run_id("training")
    start_run(ROOT, run_id, "training", config_path=args.config)
    try:
        cfg = load_config(ROOT / args.config)
        data, stats, mode = load_dataset(cfg, ROOT)
        stats_path = ROOT / cfg["data"]["stats_file"]
        save_feature_stats(stats, stats_path)

        run_dir = get_run_dir(ROOT, "training", run_id)
        run_model_dir = run_dir / "models"
        run_log_dir = run_dir / "runs"

        train_data, eval_data = split_train_test(data, cfg["training"]["train_split"])
        model_path = train_agent(
            train_data=train_data,
            eval_data=eval_data,
            mode=mode,
            env_cfg=cfg["environment"],
            risk_cfg=cfg["risk"],
            training_cfg={
                **cfg["training"],
                "log_dir": str(run_log_dir),
                "model_dir": str(run_model_dir),
            },
        )
        latest_model = ROOT / cfg["training"]["model_dir"] / "final_model.zip"
        copy_latest_pointer(model_path, latest_model)

        record_artifact(ROOT, run_id, "model", model_path, "zip")
        record_artifact(ROOT, run_id, "model_latest", latest_model, "zip")
        record_lineage(
            ROOT,
            run_id,
            input_path=str(stats_path.relative_to(ROOT)).replace("\\", "/"),
            output_path=str(model_path.relative_to(ROOT)).replace("\\", "/"),
            relation="training_input_to_model",
        )
        complete_run(ROOT, run_id, metrics={"mode": mode}, status="completed")
        print(f"Training complete ({mode}-stock mode). Model saved to {model_path}")
    except Exception as exc:
        complete_run(ROOT, run_id, status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
