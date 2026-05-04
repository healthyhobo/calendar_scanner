#!/usr/bin/env python3
"""
Run a curated grid of high-conviction calendar spread backtests.

Usage:
    python run_grid_backtest.py
    python run_grid_backtest.py --config path/to/config.yaml
"""
import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["MPLBACKEND"] = "Agg"

import pandas as pd
from tqdm.auto import tqdm

from src.backtest_data import ensure_backtest_data
from src.config import load_config
from src.features import compute_features
from src.grid_backtest import (
    build_configured_focused_grid,
    build_high_conviction_grid,
    run_backtest_grid,
)
from src.universe_logging import log_universe_coverage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run grid backtests for calendar spreads")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--grid",
        choices=["focused", "legacy"],
        default="focused",
        help="Scenario family to run. 'focused' centers on z2_pct90.",
    )
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument(
        "--slippage-pcts",
        type=float,
        nargs="+",
        default=None,
        help="Override grid_ranges.slippage_pcts, e.g. --slippage-pcts 0.03 0.05 0.07",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.slippage_pcts is not None:
        cfg.setdefault("grid_ranges", {})["slippage_pcts"] = args.slippage_pcts
    results_dir = Path("data/results/grid")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Checking configured universe data ...")
    data = ensure_backtest_data(cfg, logger)
    summaries = data["summaries"]
    cores = data["cores"]
    earnings = data.get("earnings", pd.DataFrame(columns=["ticker", "earnings_date"]))
    if summaries.empty or cores.empty:
        logger.error("No summaries/cores data available for configured universe.")
        sys.exit(1)

    logger.info("Computing shared feature table ...")
    features = compute_features(summaries, cores, cfg)
    if features.empty:
        logger.error("No features computed.")
        sys.exit(1)
    log_universe_coverage(features, cfg, logger, label="grid")

    scenarios = (
        build_configured_focused_grid(cfg)
        if args.grid == "focused"
        else build_high_conviction_grid()
    )
    logger.info("Running %d scenarios ...", len(scenarios))
    scenario_bar = tqdm(total=len(scenarios), desc="Grid scenarios", unit="scenario")
    last_completed = 0

    def update_progress(completed, total, name):
        nonlocal last_completed
        if scenario_bar.total != total:
            scenario_bar.reset(total=total)
        scenario_bar.update(max(completed - last_completed, 0))
        scenario_bar.set_postfix_str(name[:40])
        last_completed = completed

    try:
        grid_results = run_backtest_grid(
            features,
            earnings,
            cfg,
            scenarios,
            results_dir,
            max_workers=args.max_workers,
            progress_callback=update_progress,
        )
    finally:
        scenario_bar.close()

    summary = grid_results["summary"]
    if summary.empty:
        logger.warning("Grid run produced no scenario summary rows.")
        return

    display_cols = [
        "scenario",
        "profit_target_pct",
        "stop_loss_pct",
        "max_hold_days_param",
        "slippage_pct",
        "zscore_entry_min",
        "percentile_entry_min",
        "total_trades",
        "trades_per_year",
        "win_rate_pct",
        "avg_pnl_pct",
        "total_pnl_usd",
        "profit_factor",
        "gross_profit_factor",
        "expectancy_return_on_capital_pct",
        "positive_train_subwindows",
        "sharpe_ratio",
        "max_drawdown_pct",
    ]
    cols = [c for c in display_cols if c in summary.columns]
    print("\n" + "=" * 100)
    print("  GRID BACKTEST COMPARISON")
    print("=" * 100)
    print(summary[cols].to_string(index=False))
    print(f"\nSaved outputs to {results_dir.resolve()}\n")


if __name__ == "__main__":
    main()
