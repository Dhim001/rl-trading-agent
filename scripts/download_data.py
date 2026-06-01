#!/usr/bin/env python3
"""Download OHLCV data for configured symbol(s)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_trading_agent.config import load_config
from rl_trading_agent.data.pipeline import load_dataset, save_feature_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Download market data")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    data, stats, mode = load_dataset(cfg, ROOT)
    save_feature_stats(stats, ROOT / cfg["data"]["stats_file"])

    if mode == "multi":
        symbols = cfg["data"]["symbols"]
        length = len(data[symbols[0]])
        print(f"Downloaded aligned panel for {len(symbols)} symbols, {length} rows each")
    else:
        print(f"Downloaded {len(data)} rows for {cfg['data']['symbol']}")


if __name__ == "__main__":
    main()
