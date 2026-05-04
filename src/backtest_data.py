"""Data preparation helpers for backtest entry points."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .orats_data import (
    _safe_to_parquet,
    fetch_all,
    get_ticker_cache_status,
)


def _fmt_tickers(tickers: list[str]) -> str:
    if not tickers:
        return "(none)"
    return ", ".join(tickers)


def _load_raw_for_universe(raw_dir: Path, universe: list[str]) -> dict[str, pd.DataFrame]:
    """Load cached per-ticker parquet files only for the configured universe."""
    frames: dict[str, list[pd.DataFrame]] = {
        "summaries": [],
        "cores": [],
        "earnings": [],
    }
    for ticker in universe:
        ticker_dir = raw_dir / ticker
        files = {
            "summaries": ticker_dir / "summaries_hist.parquet",
            "cores": ticker_dir / "cores_hist.parquet",
            "earnings": ticker_dir / "earnings.parquet",
        }
        for key, path in files.items():
            if path.exists():
                frames[key].append(pd.read_parquet(path))

    return {
        key: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for key, parts in frames.items()
    }


def _filter_to_universe(df: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    """Return rows whose ticker is in the configured universe."""
    if df.empty or "ticker" not in df.columns:
        return df.copy()
    allowed = {ticker.upper() for ticker in universe}
    out = df[df["ticker"].astype(str).str.upper().isin(allowed)].copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out


def _write_processed(data: dict[str, pd.DataFrame], proc_dir: Path) -> None:
    proc_dir.mkdir(parents=True, exist_ok=True)
    for key, df in data.items():
        if key == "earnings" and df.empty:
            df = pd.DataFrame(columns=["ticker", "earnings_date"])
        _safe_to_parquet(df, proc_dir / f"all_{key}.parquet")


def ensure_backtest_data(
    cfg: dict,
    logger: logging.Logger,
    raw_dir: Path | None = None,
    proc_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Ensure backtests use exactly the configured universe.

    This checks local per-ticker caches for the configured universe, fetches
    missing summaries/cores when needed, rebuilds data/processed/all_*.parquet
    from the configured universe only, and returns filtered DataFrames.
    """
    raw_dir = raw_dir or Path("data/raw")
    proc_dir = proc_dir or Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    universe = [str(ticker).upper() for ticker in cfg.get("universe", [])]
    required_start = cfg.get("backtest", {}).get("start_date")
    required_end = cfg.get("backtest", {}).get("end_date")

    logger.info("Backtest universe from config: %d tickers", len(universe))
    logger.info("Backtest universe tickers: %s", _fmt_tickers(universe))

    status_rows = [
        get_ticker_cache_status(ticker, raw_dir, required_start, required_end)
        for ticker in universe
    ]
    missing = [
        row["ticker"]
        for row in status_rows
        if not (row["summaries_ok"] and row["cores_ok"])
    ]
    cached = [row["ticker"] for row in status_rows if row["summaries_ok"] and row["cores_ok"]]

    if cached:
        logger.info("Local summaries/cores cache OK for: %s", _fmt_tickers(cached))
    if missing:
        logger.info(
            "Missing or incomplete local summaries/cores cache for: %s",
            _fmt_tickers(missing),
        )
        logger.info(
            "Fetching missing/incomplete data and rebuilding processed files for configured universe ..."
        )
        data = fetch_all(cfg, raw_dir, force_refresh=False)
    else:
        logger.info(
            "All configured tickers have local summaries/cores cache; rebuilding processed files from local cache ..."
        )
        data = _load_raw_for_universe(raw_dir, universe)

    data = {key: _filter_to_universe(df, universe) for key, df in data.items()}
    _write_processed(data, proc_dir)

    for key, df in data.items():
        tickers = (
            sorted(df["ticker"].dropna().astype(str).str.upper().unique())
            if not df.empty and "ticker" in df.columns
            else []
        )
        logger.info(
            "Prepared %s data: %d rows, %d tickers: %s",
            key,
            len(df),
            len(tickers),
            _fmt_tickers(tickers),
        )

    return data
