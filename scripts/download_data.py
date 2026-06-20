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
from rl_trading_agent.storage import (
    complete_run,
    copy_latest_pointer,
    create_run_id,
    get_run_dir,
    record_artifact,
    start_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download market data")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()

    run_id = create_run_id("download_data")
    start_run(ROOT, run_id, "download_data", config_path=args.config)
    try:
        cfg = load_config(ROOT / args.config)
        data, stats, mode = load_dataset(cfg, ROOT)
        stats_path = ROOT / cfg["data"]["stats_file"]
        save_feature_stats(stats, stats_path)

        run_dir = get_run_dir(ROOT, "download_data", run_id)
        versioned_stats = run_dir / "feature_stats.json"
        copy_latest_pointer(stats_path, versioned_stats)
        record_artifact(ROOT, run_id, "feature_stats", stats_path, "json")
        record_artifact(ROOT, run_id, "feature_stats_snapshot", versioned_stats, "json")

        if mode == "multi":
            symbols = cfg["data"]["symbols"]
            length = len(data[symbols[0]])
            print(f"Downloaded aligned panel for {len(symbols)} symbols, {length} rows each")
        else:
            print(f"Downloaded {len(data)} rows for {cfg['data']['symbol']}")
        complete_run(ROOT, run_id, metrics={"mode": mode}, status="completed")
    except Exception as exc:
        complete_run(ROOT, run_id, status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
