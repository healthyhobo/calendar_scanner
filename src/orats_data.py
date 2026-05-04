"""
ORATS Data API client — uses summaries + cores endpoints for efficient
historical data fetching.

Instead of fetching raw per-strike data (one API call per date = ~19k calls
for 5yr × 15 tickers), we use:
  /datav2/hist/summaries  → ATM IV by tenor, forward rates, contango
  /datav2/hist/cores      → ATM IV per monthly expiry, percentiles, HV, earnings
  /datav2/hist/earnings   → earnings dates
  /datav2/hist/strikes    → single-date lookups for P&L MTM only

Total API calls ≈ 3 × len(universe) for initial fetch (~45 for 15 ETFs).

ORATS docs: https://orats.com/docs
"""
import time
import logging
from pathlib import Path
from datetime import date

import pandas as pd
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _safe_to_parquet(df: pd.DataFrame, path: Path):
    """Write DataFrame to parquet while avoiding pyarrow conversion errors.

    Converts object-dtype columns to pandas string dtype before writing.
    """
    if df is None or df.empty:
        # let pandas handle empty frames
        df.to_parquet(path, index=False)
        return
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols):
        df = df.copy()
        for col in obj_cols:
            df[col] = df[col].astype("string")
    df.to_parquet(path, index=False)


def _normalize_date(value) -> date | None:
    """Convert config/date-like values to Python dates."""
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _trade_date_bounds(path: Path) -> tuple[date, date, int] | None:
    """Return min/max tradeDate coverage for a cached parquet file."""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["tradeDate"])
    except Exception:
        df = pd.read_parquet(path)
    if df.empty or "tradeDate" not in df.columns:
        return None
    trade_dates = pd.to_datetime(df["tradeDate"], errors="coerce").dropna()
    if trade_dates.empty:
        return None
    return trade_dates.min().date(), trade_dates.max().date(), len(df)


def cache_covers_window(path: Path, required_start=None, required_end=None) -> bool:
    """Check whether a cached file fully spans the requested date window."""
    bounds = _trade_date_bounds(path)
    if bounds is None:
        return False
    cached_start, cached_end, _ = bounds
    required_start = _normalize_date(required_start)
    required_end = _normalize_date(required_end)
    if required_start is not None and cached_start > required_start:
        return False
    if required_end is not None and cached_end < required_end:
        return False
    return True


def cache_is_usable_full_history(path: Path, required_start=None, required_end=None) -> bool:
    """Check whether a full-history cache is good enough to reuse.

    The summaries/cores endpoints fetch full available history for a ticker,
    not a user-selected slice. Some symbols legitimately start after the
    requested backtest start (IPO/SPAC/rebrand behavior), so repeatedly
    forcing a refresh is wasteful if the cached file already reaches the
    requested end date.
    """
    bounds = _trade_date_bounds(path)
    if bounds is None:
        return False
    _, cached_end, _ = bounds
    required_end = _normalize_date(required_end)
    if required_end is not None and cached_end < required_end:
        return False
    return True


def get_ticker_cache_status(ticker: str, raw_dir: Path,
                            required_start=None, required_end=None) -> dict:
    """Summarize local cache coverage for notebook/CLI progress reporting."""
    ticker_dir = raw_dir / ticker
    summaries_path = ticker_dir / "summaries_hist.parquet"
    cores_path = ticker_dir / "cores_hist.parquet"
    earnings_path = ticker_dir / "earnings.parquet"

    summaries_bounds = _trade_date_bounds(summaries_path)
    cores_bounds = _trade_date_bounds(cores_path)
    return {
        "ticker": ticker,
        "summaries_path": summaries_path,
        "cores_path": cores_path,
        "earnings_path": earnings_path,
        "summaries_bounds": summaries_bounds,
        "cores_bounds": cores_bounds,
        "summaries_ok": cache_covers_window(summaries_path, required_start, required_end),
        "cores_ok": cache_covers_window(cores_path, required_start, required_end),
        "earnings_cached": earnings_path.exists(),
    }


def _orats_get(base_url: str, endpoint: str, token: str,
               params: dict, pause: float = 1.1) -> list[dict]:
    """Single ORATS API GET request; returns the 'data' array."""
    url = f"{base_url}/{endpoint}"
    params["token"] = token
    resp = requests.get(url, params=params, timeout=120)
    if resp.status_code != 200:
        logger.error("ORATS API %d for %s: %s",
                     resp.status_code, endpoint, resp.text[:500])
        resp.raise_for_status()
    body = resp.json()
    if "data" not in body:
        logger.warning("No 'data' key in response for %s %s", endpoint, params)
        return []
    time.sleep(pause)
    return body["data"]


# ── Summaries history ───────────────────────────────────────────

def fetch_summaries_history(ticker: str, cfg: dict, raw_dir: Path,
                            force_refresh: bool = False,
                            required_start=None,
                            required_end=None) -> pd.DataFrame:
    """
    Full historical summaries for one ticker.

    Key fields from ORATS summaries:
      iv30d, iv60d, iv90d, iv6m, iv1y        — ATM IV by constant maturity
      exErnIv30d, exErnIv60d, ...            — IV ex earnings effect
      fwd30_20, fwd60_30, fwd90_30, fwd90_60 — forward IV rates
      fexErn60_30, fexErn90_30, fexErn90_60  — forward rates ex-earnings
      contango                                — term structure indicator
      rVol30, rVol2y                          — realized vol
      confidence                              — fit quality
    """
    cache_file = raw_dir / ticker / "summaries_hist.parquet"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not force_refresh:
        if cache_is_usable_full_history(cache_file, required_start, required_end):
            logger.info("  %s summaries: using cached history", ticker)
            return pd.read_parquet(cache_file)
        logger.info(
            "  %s summaries: cache does not cover requested window %s to %s; refreshing full history",
            ticker, required_start, required_end,
        )

    rows = _orats_get(
        cfg["orats"]["base_url"], "hist/summaries", cfg["orats"]["token"],
        {"ticker": ticker},
        pause=cfg["orats"].get("rate_limit_pause", 1.1),
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "tradeDate" in df.columns:
        df["tradeDate"] = pd.to_datetime(df["tradeDate"]).dt.date
    _safe_to_parquet(df, cache_file)
    logger.info("  %s summaries: %d rows", ticker, len(df))
    return df


# ── Cores history ───────────────────────────────────────────────

def fetch_cores_history(ticker: str, cfg: dict, raw_dir: Path,
                        force_refresh: bool = False,
                        required_start=None,
                        required_end=None) -> pd.DataFrame:
    """
    Full historical cores data for one ticker.

    Key fields:
      atmIvM1..M4, atmFitIvM1..M4, dtExM1..M4 — ATM IV per monthly expiry
      ivPctile1m, ivPctile1y                   — IV percentile ranks
      orHv20d, orHv60d, clsHv20d, clsHv60d    — realized vol
      nextErn, lastErn, daysToNextErn          — earnings proximity
      stkVolu, avgOptVolu20d, cVolu, pVolu     — volume/liquidity
      straPxM1, straPxM2                       — ATM straddle prices
      contango                                 — term structure shape
    """
    cache_file = raw_dir / ticker / "cores_hist.parquet"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not force_refresh:
        if cache_is_usable_full_history(cache_file, required_start, required_end):
            logger.info("  %s cores: using cached history", ticker)
            return pd.read_parquet(cache_file)
        logger.info(
            "  %s cores: cache does not cover requested window %s to %s; refreshing full history",
            ticker, required_start, required_end,
        )

    rows = _orats_get(
        cfg["orats"]["base_url"], "hist/cores", cfg["orats"]["token"],
        {"ticker": ticker},
        pause=cfg["orats"].get("rate_limit_pause", 1.1),
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "tradeDate" in df.columns:
        df["tradeDate"] = pd.to_datetime(df["tradeDate"]).dt.date
    _safe_to_parquet(df, cache_file)
    logger.info("  %s cores: %d rows", ticker, len(df))
    return df


# ── Earnings history ────────────────────────────────────────────

def fetch_earnings(ticker: str, cfg: dict, raw_dir: Path,
                   force_refresh: bool = False) -> pd.DataFrame:
    cache_file = raw_dir / ticker / "earnings.parquet"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists() and not force_refresh:
        return pd.read_parquet(cache_file)

    rows = _orats_get(
        cfg["orats"]["base_url"], "hist/earnings", cfg["orats"]["token"],
        {"ticker": ticker},
        pause=cfg["orats"].get("rate_limit_pause", 1.1),
    )
    if not rows:
        return pd.DataFrame(columns=["ticker", "earnings_date"])

    df = pd.DataFrame(rows)
    date_col = None
    for candidate in ("earnDate", "earningsDate", "tradeDate"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        logger.warning("No earnings date column for %s. Cols: %s",
                       ticker, df.columns.tolist())
        return pd.DataFrame(columns=["ticker", "earnings_date"])

    out = pd.DataFrame({
        "ticker": ticker,
        "earnings_date": pd.to_datetime(df[date_col]).dt.date,
    }).drop_duplicates()
    _safe_to_parquet(out, cache_file)
    return out


# ── Single-date strikes (for MTM only) ─────────────────────────

def fetch_strikes_single_date(ticker: str, trade_date: str,
                               dte_min: int, dte_max: int,
                               cfg: dict, raw_dir: Path) -> pd.DataFrame:
    """
    Fetch strikes for one ticker on ONE specific date.
    Use sparingly — each call counts against the 20k/month limit.
    """
    cache_file = raw_dir / ticker / f"strikes_{trade_date}.parquet"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    rows = _orats_get(
        cfg["orats"]["base_url"], "hist/strikes", cfg["orats"]["token"],
        {"ticker": ticker, "tradeDate": trade_date, "dte": f"{dte_min},{dte_max}"},
        pause=cfg["orats"].get("rate_limit_pause", 1.1),
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in ("tradeDate", "expirDate"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col]).dt.date
    _safe_to_parquet(df, cache_file)
    return df


# ── Bulk fetch ──────────────────────────────────────────────────

def fetch_all(cfg: dict, raw_dir: Path = None, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """
    Fetch summaries + cores + earnings for all tickers.
    ~3 API calls per ticker = ~45 calls for 15 ETFs.
    """
    if raw_dir is None:
        raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    universe = cfg["universe"]
    required_start = cfg.get("backtest", {}).get("start_date")
    required_end = cfg.get("backtest", {}).get("end_date")
    all_summaries, all_cores, all_earnings = [], [], []

    for tkr in tqdm(universe, desc="Fetching ORATS data"):
        status = get_ticker_cache_status(tkr, raw_dir, required_start, required_end)
        if force_refresh:
            logger.info("Refreshing %s due to force_refresh=True", tkr)
        elif status["summaries_ok"] and status["cores_ok"]:
            logger.info("Skipping %s fetch; local summaries/cores already cover requested window", tkr)

        s = fetch_summaries_history(
            tkr, cfg, raw_dir,
            force_refresh=force_refresh,
            required_start=required_start,
            required_end=required_end,
        )
        if not s.empty:
            all_summaries.append(s)
        c = fetch_cores_history(
            tkr, cfg, raw_dir,
            force_refresh=force_refresh,
            required_start=required_start,
            required_end=required_end,
        )
        if not c.empty:
            all_cores.append(c)
        e = fetch_earnings(tkr, cfg, raw_dir, force_refresh=force_refresh)
        if not e.empty:
            all_earnings.append(e)

    result = {
        "summaries": pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame(),
        "cores":     pd.concat(all_cores,     ignore_index=True) if all_cores     else pd.DataFrame(),
        "earnings":  pd.concat(all_earnings,  ignore_index=True) if all_earnings  else pd.DataFrame(),
    }
    # Save combined files
    proc_dir = raw_dir.parent / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)
    for key, df in result.items():
        if not df.empty:
            _safe_to_parquet(df, proc_dir / f"all_{key}.parquet")

    return result


def load_cached(raw_dir: Path = None) -> dict[str, pd.DataFrame]:
    if raw_dir is None:
        raw_dir = Path("data/raw")
    def _load(pattern):
        frames = [pd.read_parquet(p) for p in sorted(raw_dir.rglob(pattern))]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return {
        "summaries": _load("summaries_hist.parquet"),
        "cores":     _load("cores_hist.parquet"),
        "earnings":  _load("earnings.parquet"),
    }
