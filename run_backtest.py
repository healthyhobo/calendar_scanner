#!/usr/bin/env python3
"""
Run the calendar spread backtest.

Usage:
    python run_backtest.py
    python run_backtest.py --config path/to/config.yaml

Prerequisites:
    Run `python run_fetch.py` first to download ORATS data.

Outputs:
    data/results/trade_log.csv
    data/results/equity_curve.png
    data/results/pnl_distribution.png
    data/results/monthly_heatmap.png
    data/results/features.parquet
    data/results/metrics.json
    data/results/regimes.json
    stdout: summary metrics + regime breakdown
"""
import argparse
import logging
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from tqdm.auto import tqdm
from src.backtest_data import ensure_backtest_data
from src.config import load_config
from src.features import compute_features
from src.backtest import run_backtest
from src.llm_report import save_llm_backtest_report
from src.signals import audit_entry_filters
from src.universe_logging import log_universe_coverage
from src.metrics import (
    compute_metrics, regime_analysis,
    print_report, save_plots, save_trade_log,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run calendar spread backtest")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────
    logger.info("Checking configured universe data ...")
    data = ensure_backtest_data(cfg, logger)
    summaries = data["summaries"]
    cores     = data["cores"]
    earnings  = data.get("earnings", pd.DataFrame(columns=["ticker", "earnings_date"]))

    if summaries.empty or cores.empty:
        logger.error("No summaries/cores data available for configured universe.")
        sys.exit(1)

    logger.info("  Summaries: %d rows", len(summaries))
    logger.info("  Cores:     %d rows", len(cores))
    logger.info("  Earnings:  %d rows", len(earnings))

    # ── Compute features ──────────────────────────────────────
    logger.info("Computing features ...")
    t0 = time.time()
    features = compute_features(summaries, cores, cfg)
    logger.info("  %d feature rows in %.1fs", len(features), time.time() - t0)

    if features.empty:
        logger.error("No features computed — check data and config.")
        sys.exit(1)
    log_universe_coverage(features, cfg, logger, label="single-backtest")

    features.to_parquet(results_dir / "features.parquet", index=False)
    audit = audit_entry_filters(features, earnings, cfg)
    if not audit.empty:
        logger.info("Entry filter audit by ticker:")
        logger.info("\n%s", audit.to_string(index=False))

    # ── Run backtest ──────────────────────────────────────────
    logger.info("Running backtest ...")
    t0 = time.time()
    backtest_bar = tqdm(desc="Backtest trading days", unit="day")
    last_completed = 0

    def update_backtest_progress(completed, total, trade_date, open_positions, closed_trades):
        nonlocal last_completed
        if backtest_bar.total != total:
            backtest_bar.reset(total=total)
        backtest_bar.update(max(completed - last_completed, 0))
        backtest_bar.set_postfix_str(
            f"{trade_date} open={open_positions} closed={closed_trades}"
        )
        last_completed = completed

    try:
        results = run_backtest(
            features,
            earnings,
            cfg,
            progress_callback=update_backtest_progress,
        )
    finally:
        backtest_bar.close()
    logger.info("  Backtest complete in %.1fs", time.time() - t0)

    trades     = results["trades"]
    portfolio  = results["portfolio"]
    still_open = results["open"]

    logger.info("  Closed trades: %d", len(trades))
    logger.info("  Still open:    %d", len(still_open))

    # ── Metrics ───────────────────────────────────────────────
    metrics = compute_metrics(trades, portfolio)
    regimes = regime_analysis(trades, features)

    print_report(metrics, regimes)

    # ── Save outputs ──────────────────────────────────────────
    save_trade_log(trades, results_dir)
    save_plots(portfolio, trades, results_dir)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(results_dir / "regimes.json", "w") as f:
        json.dump(regimes, f, indent=2, default=str)

    save_llm_backtest_report(
        results_dir,
        cfg,
        metrics,
        regimes,
        trades,
        portfolio,
        features,
        audit,
    )

    logger.info("Results saved to %s", results_dir.resolve())
    logger.info("LLM report saved to %s", (results_dir / "llm_backtest_overview.json").resolve())
    logger.info("LLM prompt saved to %s", (results_dir / "llm_analysis_prompt.txt").resolve())


if __name__ == "__main__":
    main()
