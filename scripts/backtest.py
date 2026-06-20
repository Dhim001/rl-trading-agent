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
from rl_trading_agent.storage import (
    complete_run,
    copy_latest_pointer,
    create_run_id,
    get_run_dir,
    record_artifact,
    record_lineage,
    start_run,
)
from rl_trading_agent.evaluation.backtest import run_backtest
from rl_trading_agent.training.trainer import split_train_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest RL trading agent")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default="models/final_model.zip")
    args = parser.parse_args()

    run_id = create_run_id("backtest")
    start_run(ROOT, run_id, "backtest", config_path=args.config, params={"model": args.model})
    try:
        cfg = load_config(ROOT / args.config)
        data, _, mode = load_dataset(cfg, ROOT)
        _, test_data = split_train_test(data, cfg["training"]["train_split"])
        run_dir = get_run_dir(ROOT, "backtest", run_id)
        run_results_dir = run_dir / "results"

        metrics = run_backtest(
            model_path=ROOT / args.model,
            test_data=test_data,
            mode=mode,
            env_cfg=cfg["environment"],
            risk_cfg=cfg["risk"],
            output_dir=run_results_dir,
        )
        latest_csv = ROOT / "results" / "equity_curve.csv"
        latest_parquet = ROOT / "results" / "equity_curve.parquet"
        copy_latest_pointer(run_results_dir / "equity_curve.csv", latest_csv)
        copy_latest_pointer(run_results_dir / "equity_curve.parquet", latest_parquet)

        record_artifact(ROOT, run_id, "equity_curve", run_results_dir / "equity_curve.csv", "csv")
        record_artifact(ROOT, run_id, "equity_curve", run_results_dir / "equity_curve.parquet", "parquet")
        record_artifact(ROOT, run_id, "equity_curve_latest", latest_csv, "csv")
        record_lineage(
            ROOT,
            run_id,
            input_path=args.model,
            output_path=str((run_results_dir / "equity_curve.parquet").relative_to(ROOT)).replace("\\", "/"),
            relation="model_to_backtest_curve",
        )

        complete_run(ROOT, run_id, metrics=metrics, status="completed")
        print(f"Backtest results ({mode}-stock mode):")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    except Exception as exc:
        complete_run(ROOT, run_id, status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
