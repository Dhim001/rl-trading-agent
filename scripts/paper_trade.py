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


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper trade with RL agent")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default="models/final_model.zip")
    parser.add_argument("--once", action="store_true", help="Run a single iteration")
    parser.add_argument("--max-iterations", type=int, default=None)
    args = parser.parse_args()

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
        print(f"Equity: {result['equity']:.2f} | Action: {result['action']}")
        return

    max_iters = args.max_iterations or paper_cfg.get("max_iterations")
    trader.run_loop(max_iterations=max_iters)


if __name__ == "__main__":
    main()
