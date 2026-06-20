#!/usr/bin/env python3
"""Run live paper-trading loop with a trained model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_trading_agent.config import load_config
from rl_trading_agent.data.pipeline import is_multi_stock, load_feature_stats
from rl_trading_agent.paper.trader import PaperTrader
from rl_trading_agent.storage import complete_run, create_run_id, record_artifact, start_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trade with RL agent")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default="models/final_model.zip")
    parser.add_argument("--once", action="store_true", help="Run a single iteration")
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()

    workflow = "paper_once" if args.once else "paper_loop"
    run_id = create_run_id(workflow)
    start_run(ROOT, run_id, workflow, config_path=args.config, params={"model": args.model})
    try:
        cfg = load_config(ROOT / args.config)
        data_cfg = cfg["data"]
        paper_cfg = cfg["paper_trading"]
        symbols = data_cfg["symbols"] if is_multi_stock(cfg) else [data_cfg["symbol"]]
        stats = load_feature_stats(ROOT / data_cfg["stats_file"])

        trader = PaperTrader(
            model_path=ROOT / args.model,
            symbols=symbols,
            stats=stats,
            env_cfg=cfg["environment"],
            risk_cfg=cfg["risk"],
            paper_cfg=paper_cfg,
            root=ROOT,
        )

        if args.once:
            result = trader.run_once()
            record_artifact(ROOT, run_id, "paper_state", ROOT / "paper" / "portfolio_state.json", "json")
            record_artifact(ROOT, run_id, "paper_trades_log", ROOT / "paper" / "trades.parquet", "parquet")
            complete_run(ROOT, run_id, metrics={"equity": float(result["equity"])}, status="completed")
            print(f"Equity: {result['equity']:.2f} | Action: {result['action']}")
            return

        max_iters = args.max_iterations or paper_cfg.get("max_iterations")
        trader.run_loop(max_iterations=max_iters)
        record_artifact(ROOT, run_id, "paper_state", ROOT / "paper" / "portfolio_state.json", "json")
        record_artifact(ROOT, run_id, "paper_trades_log", ROOT / "paper" / "trades.parquet", "parquet")
        complete_run(ROOT, run_id, status="completed")
    except Exception as exc:
        complete_run(ROOT, run_id, status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
