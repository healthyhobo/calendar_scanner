#!/usr/bin/env python3
"""
Fetch historical data from ORATS (summaries + cores + earnings).

Usage:
    python run_fetch.py
    python run_fetch.py --config path/to/config.yaml

Total API calls ≈ 3 × number of tickers (~45 for 15 ETFs).
Data is cached as Parquet in data/raw/ — re-running skips cached tickers.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.orats_data import fetch_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fetch ORATS historical data")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Refresh all cached ticker files")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if cfg["orats"]["token"] == "YOUR_ORATS_API_TOKEN":
        logger.error("Set your ORATS API token in config.yaml before running.")
        sys.exit(1)

    raw_dir = Path("data/raw")

    logger.info("Fetching data for %d tickers via summaries + cores endpoints",
                len(cfg["universe"]))
    logger.info("Expected API calls: ~%d", len(cfg["universe"]) * 3)

    data = fetch_all(cfg, raw_dir, force_refresh=args.force)

    logger.info("Summaries rows: %d", len(data["summaries"]))
    logger.info("Cores rows:     %d", len(data["cores"]))
    logger.info("Earnings rows:  %d", len(data["earnings"]))
    logger.info("Data cached in %s", raw_dir.resolve())


if __name__ == "__main__":
    main()
