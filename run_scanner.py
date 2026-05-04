#!/usr/bin/env python3
"""
Live calendar spread scanner via IBKR.

Usage:
    python run_scanner.py
    python run_scanner.py --config path/to/config.yaml --with-history

Requires TWS or IB Gateway running with API enabled on the configured port.

Options:
    --with-history   Load historical features from data/results/features.parquet
                     to compute z-scores. Without this flag, candidates are
                     ranked by raw IV spread only.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Live calendar spread scanner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--with-history", action="store_true",
                        help="Load historical features for z-score computation")
    parser.add_argument("--output-dir", default="data/results/live_signals",
                        help="Directory for live signal snapshot files")
    args = parser.parse_args()

    cfg = load_config(args.config)

    try:
        from src.ibkr_scanner import run_live_scan
    except ImportError as e:
        logger.error("ib_insync required: pip install ib_insync")
        sys.exit(1)

    historical = None
    if args.with_history:
        feat_file = Path("data/results/features.parquet")
        if feat_file.exists():
            import pandas as pd
            historical = pd.read_parquet(feat_file)
            logger.info("Loaded %d historical feature rows", len(historical))
        else:
            logger.warning("No historical features found at %s — proceeding without", feat_file)

    logger.info("Scanning %d tickers via IBKR ...", len(cfg["universe"]))
    result = run_live_scan(
        cfg,
        historical_features=historical,
        output_dir=Path(args.output_dir),
    )

    entry_signals = result["entry_signals"]
    close_signals = result["close_signals"]
    positions = result["positions"]
    snapshot = result["snapshot"]

    print("\n" + "=" * 80)
    print("  LIVE CALENDAR STRATEGY SNAPSHOT")
    print("=" * 80)
    print(f"Snapshot time: {snapshot['snapshot_time']}")
    print(f"Output dir:    {Path(args.output_dir).resolve()}")
    for key, value in snapshot["counts"].items():
        print(f"{key:24s} {value}")

    if not close_signals.empty:
        close_cols = [
            "ticker", "recommendation", "close_reason", "triggered_checks",
            "strike", "right", "front_expiry", "back_expiry", "front_dte",
            "estimated_pnl_pct", "live_spread_zscore",
        ]
        show = [c for c in close_cols if c in close_signals.columns]
        print("\n" + "=" * 80)
        print("  CURRENT POSITION CLOSE/HOLD RECOMMENDATIONS")
        print("=" * 80)
        print(close_signals[show].to_string(index=False, float_format="%.4f"))
    elif not positions.empty:
        print("\nNo calendar close recommendations generated, but current positions were saved.")
    else:
        print("\nNo configured-universe IBKR positions found.")

    if entry_signals.empty:
        logger.info("No entry signals found.")
        return

    # Display results
    display_cols = [
        "ticker", "stock_price", "atm_strike",
        "front_expir", "back_expir", "front_dte", "back_dte",
        "front_iv", "back_iv", "iv_spread",
        "spread_zscore", "spread_pctile",
        "calendar_debit_bs", "signal_rank",
    ]
    show = [c for c in display_cols if c in entry_signals.columns]
    print("\n" + "=" * 80)
    print("  NEW ENTRY SIGNALS")
    print("=" * 80)
    print(entry_signals[show].to_string(index=False, float_format="%.4f"))
    print(f"\n  {len(entry_signals)} entry signals found\n")


if __name__ == "__main__":
    main()
