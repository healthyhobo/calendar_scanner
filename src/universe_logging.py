"""Logging helpers for universe coverage before backtests."""
from __future__ import annotations

import logging

import pandas as pd


def _fmt_tickers(tickers: list[str], max_items: int = 80) -> str:
    """Return a compact ticker list for log output."""
    if not tickers:
        return "(none)"
    shown = tickers[:max_items]
    suffix = "" if len(tickers) <= max_items else f" ... +{len(tickers) - max_items} more"
    return ", ".join(shown) + suffix


def log_universe_coverage(
    features: pd.DataFrame,
    cfg: dict,
    logger: logging.Logger,
    label: str = "backtest",
) -> None:
    """
    Log configured universe and tickers present in the feature table.

    This runs before simulation starts, so it tells you which tickers are
    eligible to be considered by the backtest. It does not mean every ticker
    will enter a trade; signal and portfolio filters can still reject them.
    """
    configured = sorted({str(t).upper() for t in cfg.get("universe", [])})
    feature_tickers = []
    if not features.empty and "ticker" in features.columns:
        feature_tickers = sorted({str(t).upper() for t in features["ticker"].dropna().unique()})

    missing = sorted(set(configured) - set(feature_tickers))
    extra = sorted(set(feature_tickers) - set(configured))

    logger.info("%s universe configured: %d tickers", label, len(configured))
    logger.info("%s configured tickers: %s", label, _fmt_tickers(configured))
    logger.info("%s feature-table tickers: %d tickers", label, len(feature_tickers))
    logger.info("%s backtestable feature tickers: %s", label, _fmt_tickers(feature_tickers))

    if missing:
        logger.warning(
            "%s configured tickers missing from feature table: %s",
            label,
            _fmt_tickers(missing),
        )
    if extra:
        logger.info(
            "%s feature tickers not listed in config universe: %s",
            label,
            _fmt_tickers(extra),
        )

    if not features.empty and {"ticker", "tradeDate"}.issubset(features.columns):
        counts = (
            features.groupby("ticker")["tradeDate"]
            .count()
            .sort_values(ascending=False)
        )
        logger.info("%s feature rows by ticker:\n%s", label, counts.to_string())
