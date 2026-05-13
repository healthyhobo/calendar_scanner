#!/usr/bin/env python3
"""
Live calendar spread scanner via IBKR.

Usage:
    python run_scanner.py
    python run_scanner.py --config path/to/config.yaml --with-history

Requires TWS or IB Gateway running with API enabled on the configured port.

Options:
    --with-history   Load/refresh historical features from data/results/features.parquet
                     to compute z-scores.
"""
import argparse
import copy
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.backtest_data import ensure_backtest_data
from src.config import load_config
from src.features import compute_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("ib_insync.wrapper").setLevel(logging.WARNING)


def _previous_business_day() -> pd.Timestamp:
    """Use the latest likely complete historical data date for live scans."""
    return pd.Timestamp.today().normalize() - pd.tseries.offsets.BDay(1)


def _feature_file_max_date(path: Path) -> pd.Timestamp | None:
    if not path.exists():
        return None
    try:
        features = pd.read_parquet(path, columns=["tradeDate"])
    except Exception:
        features = pd.read_parquet(path)
    if features.empty or "tradeDate" not in features.columns:
        return None
    dates = pd.to_datetime(features["tradeDate"], errors="coerce").dropna()
    return dates.max().normalize() if not dates.empty else None


def _load_earnings_from_processed() -> pd.DataFrame:
    path = Path("data/processed/all_earnings.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["ticker", "earnings_date"])


def _orats_token_configured(cfg: dict) -> bool:
    token = cfg.get("orats", {}).get("token")
    return bool(token and token != "YOUR_ORATS_API_TOKEN")


def load_or_refresh_history(cfg: dict, skip_refresh: bool = False) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """
    Load historical features for z-scores, refreshing ORATS-backed features if stale.

    The scanner targets the previous business day because ORATS history often
    does not have a complete same-day row while the live market is open.
    """
    feat_file = Path("data/results/features.parquet")
    target_date = _previous_business_day()
    current_max = _feature_file_max_date(feat_file)

    needs_refresh = current_max is None or current_max < target_date
    if skip_refresh:
        needs_refresh = False

    if needs_refresh:
        if not _orats_token_configured(cfg):
            logger.warning(
                "Historical features are stale or missing, but no ORATS token is configured. "
                "Using existing features if available."
            )
        else:
            logger.info(
                "Historical features need refresh: current=%s target=%s",
                current_max.date().isoformat() if current_max is not None else "missing",
                target_date.date().isoformat(),
            )
            hist_cfg = copy.deepcopy(cfg)
            hist_cfg.setdefault("backtest", {})["end_date"] = target_date.date().isoformat()
            data = ensure_backtest_data(hist_cfg, logger)
            features = compute_features(data["summaries"], data["cores"], hist_cfg)
            if not features.empty:
                feat_file.parent.mkdir(parents=True, exist_ok=True)
                features.to_parquet(feat_file, index=False)
                refreshed_max = pd.to_datetime(features["tradeDate"], errors="coerce").max()
                logger.info(
                    "Refreshed historical features: %d rows through %s",
                    len(features),
                    refreshed_max.date().isoformat() if pd.notna(refreshed_max) else "unknown",
                )
                return features, data.get("earnings", pd.DataFrame(columns=["ticker", "earnings_date"]))
            logger.warning("Historical feature refresh produced no rows; using existing features if available.")

    if feat_file.exists():
        features = pd.read_parquet(feat_file)
        logger.info("Loaded %d historical feature rows from %s", len(features), feat_file)
        return features, _load_earnings_from_processed()

    logger.warning("No historical features found at %s; proceeding without history", feat_file)
    return None, _load_earnings_from_processed()


def main():
    parser = argparse.ArgumentParser(description="Live calendar spread scanner")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--with-history", action="store_true",
                        help="Load historical features for z-score computation")
    parser.add_argument("--skip-history-refresh", action="store_true",
                        help="Use existing data/results/features.parquet without refreshing stale ORATS history")
    parser.add_argument("--output-dir", default="data/results/live_signals",
                        help="Directory for live signal snapshot files")
    args = parser.parse_args()

    cfg = load_config(args.config)

    try:
        from src.ibkr_scanner import run_live_scan
    except ImportError:
        logger.error("ib_insync required: pip install ib_insync")
        sys.exit(1)

    historical = None
    earnings = _load_earnings_from_processed()
    if args.with_history:
        historical, earnings = load_or_refresh_history(cfg, skip_refresh=args.skip_history_refresh)

    logger.info("Scanning %d tickers via IBKR ...", len(cfg["universe"]))
    result = run_live_scan(
        cfg,
        historical_features=historical,
        earnings=earnings,
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
