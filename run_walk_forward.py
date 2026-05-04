#!/usr/bin/env python3
"""
Run walk-forward optimisation for the calendar spread scanner.

Usage:
    python run_walk_forward.py
    python run_walk_forward.py --train-months 24 --test-months 6
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
from src.grid_backtest import build_configured_focused_grid, build_high_conviction_grid
from src.universe_logging import log_universe_coverage
from src.walk_forward import run_walk_forward_optimisation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run walk-forward optimisation for calendar spreads")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--train-months", type=int, default=18)
    parser.add_argument("--test-months", type=int, default=6)
    parser.add_argument("--min-train-trades", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument(
        "--grid",
        choices=["focused", "legacy"],
        default="focused",
        help="Scenario family to evaluate in each training window.",
    )
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
    results_dir = Path("data/results/walk_forward")
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
    log_universe_coverage(features, cfg, logger, label="walk-forward")

    scenarios = (
        build_configured_focused_grid(cfg)
        if args.grid == "focused"
        else build_high_conviction_grid()
    )
    logger.info(
        "Running walk-forward optimisation with %d scenarios, %dM train, %dM test ...",
        len(scenarios),
        args.train_months,
        args.test_months,
    )
    fold_bar = tqdm(desc="Walk-forward folds", unit="fold", position=0)
    scenario_bar = None
    last_fold_completed = 0
    last_scenario_completed = 0
    current_scenario_fold = None

    def update_fold_progress(fold_no, total_folds, message):
        nonlocal last_fold_completed
        if fold_bar.total != total_folds:
            fold_bar.reset(total=total_folds)
        fold_bar.update(max(fold_no - last_fold_completed, 0))
        fold_bar.set_postfix_str(message[:80])
        last_fold_completed = fold_no

    def update_scenario_progress(fold_no, completed, total, name):
        nonlocal scenario_bar, last_scenario_completed, current_scenario_fold
        if scenario_bar is None or current_scenario_fold != fold_no:
            if scenario_bar is not None:
                scenario_bar.close()
            scenario_bar = tqdm(
                total=total,
                desc=f"Fold {fold_no} train grid",
                unit="scenario",
                position=1,
                leave=False,
            )
            last_scenario_completed = 0
            current_scenario_fold = fold_no
        scenario_bar.update(max(completed - last_scenario_completed, 0))
        scenario_bar.set_postfix_str(name[:40])
        last_scenario_completed = completed
        if completed >= total:
            scenario_bar.close()
            scenario_bar = None

    try:
        results = run_walk_forward_optimisation(
            features,
            earnings,
            cfg,
            scenarios,
            results_dir,
            train_months=args.train_months,
            test_months=args.test_months,
            min_train_trades=args.min_train_trades,
            max_workers=args.max_workers,
            progress_callback=update_fold_progress,
            scenario_progress_callback=update_scenario_progress,
        )
    finally:
        if scenario_bar is not None:
            scenario_bar.close()
        fold_bar.close()

    folds = results["folds"]
    metrics = results["metrics"]
    summary = results["summary"]
    if not folds.empty:
        print("\n" + "=" * 100)
        print("  WALK-FORWARD FOLD SUMMARY")
        print("=" * 100)
        print(folds.to_string(index=False))

    print("\n" + "=" * 100)
    print("  WALK-FORWARD OOS METRICS")
    print("=" * 100)
    for key in [
        "total_trades",
        "win_rate_pct",
        "avg_pnl_pct",
        "total_pnl_usd",
        "profit_factor",
        "sharpe_ratio",
        "max_drawdown_pct",
    ]:
        if key in metrics:
            print(f"{key:20s} {metrics[key]}")
    if summary:
        print("\n" + "=" * 100)
        print("  WALK-FORWARD SUMMARY")
        print("=" * 100)
        for key, value in summary.items():
            print(f"{key:35s} {value}")
    print(f"\nSaved outputs to {results_dir.resolve()}\n")


if __name__ == "__main__":
    main()
