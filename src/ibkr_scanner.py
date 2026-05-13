"""
Live calendar spread scanner using IBKR via ib_insync.

The live scanner produces a timestamped snapshot under
data/results/live_signals/ with:
 - live_features.csv/parquet: one row per ticker live calendar setup
 - entry_signals.csv: rows passing the configured entry rules
 - current_positions.csv: IBKR option positions for configured tickers
 - close_signals.csv: open-position close/hold recommendations
 - snapshot.json: paths and headline counts for notebooks/LLM review
"""
from __future__ import annotations

from datetime import date, datetime
import json
import logging
from pathlib import Path
from typing import Any
import asyncio

import numpy as np
import pandas as pd

from .orats_data import _safe_to_parquet
from .signals import check_exit, screen_entries

logger = logging.getLogger(__name__)

try:
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())
    from ib_insync import IB, Stock, Option
    HAS_IB = True
except ImportError:
    HAS_IB = False
    logger.warning("ib_insync not installed - IBKR scanner unavailable")


def _connect(cfg: dict) -> "IB":
    ib = IB()
    ib.connect(
        cfg["ibkr"]["host"],
        cfg["ibkr"]["port"],
        clientId=cfg["ibkr"]["client_id"],
    )
    market_data_type = int(cfg.get("ibkr", {}).get("market_data_type", 2))
    if market_data_type in {1, 2, 3, 4}:
        ib.reqMarketDataType(market_data_type)
        logger.info("Requested IBKR market data type %s", market_data_type)
    return ib


def _safe_float(value, default=np.nan) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _mid(bid, ask, last=np.nan, close=np.nan, market=np.nan) -> float:
    bid = _safe_float(bid)
    ask = _safe_float(ask)
    if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    for value in (last, close, market):
        value = _safe_float(value)
        if np.isfinite(value) and value > 0:
            return value
    return np.nan


def _greek_value(tick, field: str) -> float:
    """Read a Greek from model/bid/ask/last greeks, preferring model values."""
    model = getattr(tick, "modelGreeks", None)
    if model is not None:
        value = _safe_float(getattr(model, field, np.nan))
        if np.isfinite(value):
            return value

    bid_greeks = getattr(tick, "bidGreeks", None)
    ask_greeks = getattr(tick, "askGreeks", None)
    bid_value = _safe_float(getattr(bid_greeks, field, np.nan)) if bid_greeks is not None else np.nan
    ask_value = _safe_float(getattr(ask_greeks, field, np.nan)) if ask_greeks is not None else np.nan
    if np.isfinite(bid_value) and np.isfinite(ask_value):
        return (bid_value + ask_value) / 2.0
    if np.isfinite(bid_value):
        return bid_value
    if np.isfinite(ask_value):
        return ask_value

    last_greeks = getattr(tick, "lastGreeks", None)
    return _safe_float(getattr(last_greeks, field, np.nan)) if last_greeks is not None else np.nan


def _dte(expiry: str | date, today: date) -> int:
    if isinstance(expiry, date):
        exp_date = expiry
    else:
        exp_date = datetime.strptime(str(expiry), "%Y%m%d").date()
    return (exp_date - today).days


def _expiry_date(expiry: str | date) -> date:
    if isinstance(expiry, date):
        return expiry
    return datetime.strptime(str(expiry), "%Y%m%d").date()


def _is_standard_monthly_expiry(expiry: str | date) -> bool:
    """Return True for standard US monthly option expiries.

    Most expire on the third Friday. When that Friday is an exchange holiday,
    the listed last trading date is commonly the preceding Thursday.
    """
    exp_date = _expiry_date(expiry)
    return exp_date.weekday() in (3, 4) and 15 <= exp_date.day <= 21


def _candidate_chains(chains: list[Any], ticker: str) -> list[Any]:
    """Prefer standard SMART option chains for the underlying symbol."""
    ticker = ticker.upper()

    def score(chain) -> tuple[int, int, int, str]:
        exchange = str(getattr(chain, "exchange", "") or "")
        trading_class = str(getattr(chain, "tradingClass", "") or "").upper()
        multiplier = str(getattr(chain, "multiplier", "") or "")
        return (
            0 if exchange == "SMART" else 1,
            0 if trading_class in ("", ticker) else 1,
            0 if multiplier in ("", "100") else 1,
            trading_class,
        )

    usable = [
        c for c in chains
        if getattr(c, "expirations", None) and getattr(c, "strikes", None)
    ]
    return sorted(usable, key=score)


def _expiry_pairs(chain: Any, cfg: dict, today: date) -> list[tuple[tuple[str, int], tuple[str, int]]]:
    """Build front/back expiry pairs ordered by closeness to configured DTE targets."""
    expiry_cfg = cfg.get("expiry", {})
    front_min = int(expiry_cfg.get("front_dte_min", 10))
    front_max = int(expiry_cfg.get("front_dte_max", 45))
    front_target = int(expiry_cfg.get("front_dte_target", 30))
    back_min = int(expiry_cfg.get("back_dte_min", 30))
    back_max = int(expiry_cfg.get("back_dte_max", 80))
    back_target = int(expiry_cfg.get("back_dte_target", 60))
    min_gap = int(expiry_cfg.get("min_gap_days", 10))
    monthlies_only = bool(expiry_cfg.get("use_monthlies_only", False))

    expiries = []
    for exp in sorted(chain.expirations):
        try:
            if monthlies_only and not _is_standard_monthly_expiry(exp):
                continue
            dte = _dte(exp, today)
        except Exception:
            continue
        expiries.append((exp, dte))

    front_choices = [(exp, dte) for exp, dte in expiries if front_min <= dte <= front_max]
    back_choices = [(exp, dte) for exp, dte in expiries if back_min <= dte <= back_max]
    pairs = [
        (front, back)
        for front in front_choices
        for back in back_choices
        if back[1] > front[1] and back[1] - front[1] >= min_gap
    ]
    pairs.sort(key=lambda p: (abs(p[0][1] - front_target) + abs(p[1][1] - back_target), p[0][1], p[1][1]))
    return pairs


def _make_option(ticker: str, expiry: str, strike: float, right: str, chain: Any) -> "Option":
    opt = Option(ticker, expiry, strike, right, "SMART", currency="USD")
    trading_class = getattr(chain, "tradingClass", None)
    multiplier = getattr(chain, "multiplier", None)
    if trading_class:
        opt.tradingClass = trading_class
    if multiplier:
        opt.multiplier = str(multiplier)
    return opt


def _qualify_option(ib: "IB", ticker: str, expiry: str, strike: float, right: str, chain: Any):
    opt = _make_option(ticker, expiry, strike, right, chain)
    try:
        qualified = ib.qualifyContracts(opt)
    except Exception as exc:
        logger.debug("%s %s %s %s qualify failed: %s", ticker, expiry, strike, right, exc)
        return None
    return qualified[0] if qualified else None


def _listed_options_for_expiry(ib: "IB", ticker: str, expiry: str, right: str, chain: Any) -> dict[float, Any]:
    """Return IBKR-listed contracts for one expiry/right keyed by strike."""
    template = _make_option(ticker, expiry, 0.0, right, chain)
    try:
        details = ib.reqContractDetails(template)
    except Exception as exc:
        logger.debug("%s %s %s contract-details lookup failed: %s", ticker, expiry, right, exc)
        return {}

    contracts: dict[float, Any] = {}
    for detail in details:
        contract = getattr(detail, "contract", None)
        if contract is None:
            continue
        if str(getattr(contract, "right", "")).upper() != right:
            continue
        if str(getattr(contract, "lastTradeDateOrContractMonth", "")) != str(expiry):
            continue
        strike = _safe_float(getattr(contract, "strike", np.nan))
        if np.isfinite(strike) and strike > 0:
            contracts[float(strike)] = contract
    return contracts


def _find_valid_calendar_contracts(
    ib: "IB",
    ticker: str,
    chain: Any,
    stock_price: float,
    cfg: dict,
    today: date,
) -> tuple[float, tuple[str, int], tuple[str, int], list[tuple[Any, str, int, str]]] | None:
    """
    Find a same-strike call calendar that IBKR can actually qualify.

    IBKR's option-chain strike list is not guaranteed to contain only valid
    strike/expiry combinations. Searching nearby strikes prevents one invalid
    nominal ATM strike from dropping the ticker.
    """
    strike_limit = int(cfg.get("ibkr", {}).get("strike_search_count", 12))
    expiry_pair_limit = int(cfg.get("ibkr", {}).get("expiry_pair_search_count", 8))

    pairs = _expiry_pairs(chain, cfg, today)
    if not pairs:
        return None

    strikes = sorted(float(s) for s in chain.strikes if s and float(s) > 0)
    strike_candidates = sorted(strikes, key=lambda s: abs(s - stock_price))[:strike_limit]
    if not strike_candidates:
        return None

    for front, back in pairs[:expiry_pair_limit]:
        front_calls = _listed_options_for_expiry(ib, ticker, front[0], "C", chain)
        back_calls = _listed_options_for_expiry(ib, ticker, back[0], "C", chain)
        common_call_strikes = set(front_calls) & set(back_calls)
        if not common_call_strikes:
            continue

        valid_strikes = [s for s in strike_candidates if s in common_call_strikes]
        if not valid_strikes:
            valid_strikes = sorted(common_call_strikes, key=lambda s: abs(s - stock_price))[:strike_limit]
        if not valid_strikes:
            continue

        strike = valid_strikes[0]
        contracts = [
            (front_calls[strike], front[0], front[1], "C"),
            (back_calls[strike], back[0], back[1], "C"),
        ]

        front_puts = _listed_options_for_expiry(ib, ticker, front[0], "P", chain)
        back_puts = _listed_options_for_expiry(ib, ticker, back[0], "P", chain)
        if strike in front_puts:
            contracts.append((front_puts[strike], front[0], front[1], "P"))
        if strike in back_puts:
            contracts.append((back_puts[strike], back[0], back[1], "P"))
        return strike, front, back, contracts

    return None


def _get_stock_price(ib: "IB", ticker: str) -> tuple[Any, float]:
    stock = Stock(ticker, "SMART", "USD")
    ib.qualifyContracts(stock)
    [tick] = ib.reqTickers(stock)
    price = _safe_float(tick.marketPrice())
    if not np.isfinite(price) or price <= 0:
        price = _safe_float(tick.close)
    return stock, price


def _get_live_calendar_feature(ib: "IB", ticker: str, cfg: dict, today: date) -> dict[str, Any] | None:
    """
    Build one live ATM calendar setup for a ticker using configured DTE targets.

    Uses the average of call/put model IVs for the ATM strike when available,
    and call mid prices for a concrete long-call-calendar debit estimate.
    """
    stock, stock_price = _get_stock_price(ib, ticker)
    if not np.isfinite(stock_price) or stock_price <= 0:
        logger.warning("%s: no valid stock price", ticker)
        return None

    chains = ib.reqSecDefOptParams(stock.symbol, "", stock.secType, stock.conId)
    if not chains:
        logger.warning("%s: no option chains", ticker)
        return None

    calendar = None
    selected_chain = None
    for chain in _candidate_chains(chains, ticker):
        calendar = _find_valid_calendar_contracts(ib, ticker, chain, stock_price, cfg, today)
        if calendar is not None:
            selected_chain = chain
            break

    if calendar is None:
        logger.warning("%s: no valid same-strike calendar found in configured DTE/monthly ranges", ticker)
        return None

    atm_strike, front, back, contracts = calendar
    rows = []
    ticks = ib.reqTickers(*[c[0] for c in contracts])
    for (contract, exp, dte, right), tick in zip(contracts, ticks):
        rows.append({
            "ticker": ticker,
            "expiry": exp,
            "dte": dte,
            "right": right,
            "strike": atm_strike,
            "bid": _safe_float(tick.bid),
            "ask": _safe_float(tick.ask),
            "last": _safe_float(tick.last),
            "close": _safe_float(tick.close),
            "market_price": _safe_float(tick.marketPrice()),
            "mid": _mid(tick.bid, tick.ask, tick.last, tick.close, tick.marketPrice()),
            "iv": _greek_value(tick, "impliedVol"),
            "delta": _greek_value(tick, "delta"),
            "gamma": _greek_value(tick, "gamma"),
            "theta": _greek_value(tick, "theta"),
            "vega": _greek_value(tick, "vega"),
        })

    quotes = pd.DataFrame(rows)
    if quotes.empty:
        return None

    def leg_value(exp, field, right=None):
        q = quotes[quotes["expiry"].eq(exp)]
        if right is not None:
            q = q[q["right"].eq(right)]
        vals = pd.to_numeric(q[field], errors="coerce").dropna()
        return float(vals.mean()) if not vals.empty else np.nan

    front_iv = leg_value(front[0], "iv")
    back_iv = leg_value(back[0], "iv")
    # IB modelGreeks IV is decimal; convert to percentage points for signal parity.
    if np.isfinite(front_iv) and front_iv <= 2.0:
        front_iv *= 100.0
    if np.isfinite(back_iv) and back_iv <= 2.0:
        back_iv *= 100.0

    front_call_mid = leg_value(front[0], "mid", right="C")
    back_call_mid = leg_value(back[0], "mid", right="C")
    debit = back_call_mid - front_call_mid if np.isfinite(back_call_mid) and np.isfinite(front_call_mid) else np.nan

    return {
        "ticker": ticker,
        "tradeDate": today,
        "stock_price": stock_price,
        "atm_strike": atm_strike,
        "front_expir": front[0],
        "back_expir": back[0],
        "front_dte": front[1],
        "back_dte": back[1],
        "front_iv": front_iv,
        "back_iv": back_iv,
        "iv_spread": front_iv - back_iv if np.isfinite(front_iv) and np.isfinite(back_iv) else np.nan,
        "calendar_debit_bs": debit,
        "calendar_debit_proxy": debit,
        "front_call_mid": front_call_mid,
        "back_call_mid": back_call_mid,
        "qualified_contract_count": len(contracts),
        "trading_class": getattr(selected_chain, "tradingClass", None),
        "total_opt_volume": np.nan,
        "avgOptVolu20d": np.nan,
        "rv_20": np.nan,
        "rv_iv_ratio": np.nan,
        "daily_gamma_drag": np.nan,
    }


def _attach_historical_stats(live_features: pd.DataFrame, historical_features: pd.DataFrame | None, cfg: dict) -> pd.DataFrame:
    if live_features.empty:
        return live_features

    out = live_features.copy()
    out["tradeDate"] = pd.to_datetime(out["tradeDate"]).dt.date
    if historical_features is None or historical_features.empty:
        out["spread_zscore"] = np.nan
        out["spread_pctile"] = np.nan
        out["back_iv_pctile"] = np.nan
        out["signal_rank"] = np.nan
        return out

    lookback = int(cfg.get("features", {}).get("lookback_days", 252))
    min_hist = int(cfg.get("features", {}).get("min_history_days", 60))
    hist = historical_features.copy()
    hist["ticker"] = hist["ticker"].astype(str).str.upper()
    hist["tradeDate"] = pd.to_datetime(hist["tradeDate"]).dt.date

    stats = []
    for _, row in out.iterrows():
        ticker = str(row["ticker"]).upper()
        h = hist[hist["ticker"].eq(ticker)].sort_values("tradeDate").tail(lookback)
        spread_hist = pd.to_numeric(h.get("iv_spread", pd.Series(dtype=float)), errors="coerce").dropna()
        back_hist = pd.to_numeric(h.get("back_iv", pd.Series(dtype=float)), errors="coerce").dropna()
        spread = _safe_float(row.get("iv_spread"))
        back_iv = _safe_float(row.get("back_iv"))

        if len(spread_hist) >= min_hist and np.isfinite(spread):
            mean = spread_hist.mean()
            std = spread_hist.std()
            z = (spread - mean) / std if std and np.isfinite(std) else np.nan
            pctile = (spread_hist.le(spread).mean() * 100.0)
        else:
            z = np.nan
            pctile = np.nan

        if len(back_hist) >= min_hist and np.isfinite(back_iv):
            back_pctile = back_hist.le(back_iv).mean() * 100.0
        else:
            back_pctile = np.nan

        stats.append({
            "ticker": ticker,
            "spread_zscore": z,
            "spread_pctile": pctile,
            "back_iv_pctile": back_pctile,
        })

    stat_df = pd.DataFrame(stats)
    out = out.drop(columns=["spread_zscore", "spread_pctile", "back_iv_pctile"], errors="ignore")
    out = out.merge(stat_df, on="ticker", how="left")
    return out


def _portfolio_items_to_positions(ib: "IB", cfg: dict) -> pd.DataFrame:
    universe = {str(t).upper() for t in cfg.get("universe", [])}
    rows = []
    for item in ib.portfolio():
        c = item.contract
        symbol = str(getattr(c, "symbol", "")).upper()
        if universe and symbol not in universe:
            continue
        rows.append({
            "ticker": symbol,
            "conId": getattr(c, "conId", None),
            "secType": getattr(c, "secType", None),
            "localSymbol": getattr(c, "localSymbol", None),
            "expiry": getattr(c, "lastTradeDateOrContractMonth", None),
            "strike": getattr(c, "strike", np.nan),
            "right": getattr(c, "right", None),
            "position": item.position,
            "market_price": item.marketPrice,
            "market_value": item.marketValue,
            "average_cost": item.averageCost,
            "unrealized_pnl": item.unrealizedPNL,
            "realized_pnl": item.realizedPNL,
        })
    return pd.DataFrame(rows)


def _calendar_position_recommendations(
    positions: pd.DataFrame,
    live_features: pd.DataFrame,
    cfg: dict,
    today: date,
) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame()

    options = positions[positions["secType"].astype(str).str.upper().eq("OPT")].copy()
    if options.empty:
        return pd.DataFrame()
    options["abs_qty"] = pd.to_numeric(options["position"], errors="coerce").abs()
    options["expiry_date"] = pd.to_datetime(options["expiry"], errors="coerce").dt.date

    live_by_ticker = {
        str(r["ticker"]).upper(): r
        for _, r in live_features.iterrows()
    } if not live_features.empty else {}

    recs = []
    group_cols = ["ticker", "strike", "right"]
    for (ticker, strike, right), grp in options.groupby(group_cols, dropna=False):
        if len(grp) < 2:
            continue
        grp = grp.sort_values("expiry_date")
        shorts = grp[pd.to_numeric(grp["position"], errors="coerce") < 0]
        longs = grp[pd.to_numeric(grp["position"], errors="coerce") > 0]
        if shorts.empty or longs.empty:
            continue

        front = shorts.iloc[0]
        back = longs.iloc[-1]
        front_dte = (front["expiry_date"] - today).days if pd.notna(front["expiry_date"]) else np.nan
        current_value = (
            _safe_float(back["market_price"]) - _safe_float(front["market_price"])
        )
        average_debit = (
            _safe_float(back["average_cost"]) - _safe_float(front["average_cost"])
        ) / 100.0
        pnl_pct = (
            (current_value - average_debit) / average_debit
            if np.isfinite(current_value) and np.isfinite(average_debit) and average_debit > 0
            else 0.0
        )

        live = live_by_ticker.get(str(ticker).upper(), {})
        reason = check_exit(
            {
                "entry_date": today,
                "entry_iv_spread": np.nan,
            },
            today,
            pnl_pct,
            _safe_float(live.get("spread_zscore")),
            front_dte,
            cfg,
            current_front_iv=_safe_float(live.get("front_iv")),
            current_back_iv=_safe_float(live.get("back_iv")),
        )

        checks = []
        exit_cfg = cfg.get("exit", {})
        sig_cfg = cfg.get("signals", {})
        if pnl_pct <= exit_cfg.get("stop_loss_pct", -0.30):
            checks.append("stop_loss")
        if pnl_pct >= exit_cfg.get("profit_target_pct", 0.25):
            checks.append("profit_target")
        z = _safe_float(live.get("spread_zscore"))
        if np.isfinite(z) and z <= sig_cfg.get("zscore_exit", 0.0):
            checks.append("zscore_normalized")
        if np.isfinite(front_dte) and front_dte < exit_cfg.get("time_stop_dte", 5):
            checks.append("time_stop_dte")
        heuristic_reason = reason
        if heuristic_reason is None and checks:
            heuristic_reason = checks[0]

        recs.append({
            "ticker": ticker,
            "strategy_type": "calendar_candidate",
            "strike": strike,
            "right": right,
            "front_expiry": front["expiry"],
            "back_expiry": back["expiry"],
            "front_dte": front_dte,
            "short_front_qty": front["position"],
            "long_back_qty": back["position"],
            "current_calendar_value": current_value,
            "estimated_average_debit": average_debit,
            "estimated_pnl_pct": pnl_pct * 100.0,
            "live_spread_zscore": live.get("spread_zscore", np.nan),
            "live_spread_pctile": live.get("spread_pctile", np.nan),
            "live_front_iv": live.get("front_iv", np.nan),
            "live_back_iv": live.get("back_iv", np.nan),
            "recommendation": "CLOSE" if heuristic_reason is not None else "HOLD",
            "close_reason": heuristic_reason or "",
            "triggered_checks": ", ".join(checks),
            "note": (
                "Uses IBKR averageCost/marketPrice; verify manually before trading. "
                "Entry date/spread are unknown for externally opened positions."
            ),
        })

    return pd.DataFrame(recs)


def run_live_scan(
    cfg: dict,
    historical_features: pd.DataFrame | None = None,
    earnings: pd.DataFrame | None = None,
    output_dir: Path | str = Path("data/results/live_signals"),
) -> dict[str, pd.DataFrame | dict]:
    """Run live scan, persist snapshot files, and return all tables."""
    if not HAS_IB:
        raise ImportError("ib_insync required for live scanning")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().date()
    snapshot_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    ib = _connect(cfg)
    logger.info("Connected to IBKR")
    try:
        live_rows = []
        for ticker in [str(t).upper() for t in cfg.get("universe", [])]:
            logger.info("Scanning %s ...", ticker)
            try:
                row = _get_live_calendar_feature(ib, ticker, cfg, today)
                if row is not None:
                    live_rows.append(row)
            except Exception as exc:
                logger.exception("Failed to scan %s: %s", ticker, exc)

        live_features = pd.DataFrame(live_rows)
        if not live_features.empty:
            live_features = _attach_historical_stats(live_features, historical_features, cfg)
            if "signal_rank" not in live_features.columns:
                live_features["signal_rank"] = np.nan
            live_features.sort_values(
                ["signal_rank", "iv_spread"],
                ascending=[False, False],
                inplace=True,
                na_position="last",
            )

        earnings_df = earnings if earnings is not None else pd.DataFrame(columns=["ticker", "earnings_date"])
        entry_required_cols = [
            "front_iv", "back_iv", "iv_spread", "calendar_debit_bs",
            "spread_zscore", "spread_pctile",
        ]
        entry_features = (
            live_features.dropna(subset=entry_required_cols)
            if not live_features.empty and all(c in live_features.columns for c in entry_required_cols)
            else pd.DataFrame()
        )
        entry_signals = screen_entries(entry_features, earnings_df, cfg) if not entry_features.empty else pd.DataFrame()

        positions = _portfolio_items_to_positions(ib, cfg)
        close_signals = _calendar_position_recommendations(positions, live_features, cfg, today)
    finally:
        ib.disconnect()

    files = {
        "live_features_csv": output_dir / "live_features.csv",
        "entry_signals_csv": output_dir / "entry_signals.csv",
        "current_positions_csv": output_dir / "current_positions.csv",
        "close_signals_csv": output_dir / "close_signals.csv",
        "snapshot_json": output_dir / "snapshot.json",
    }

    live_features.to_csv(files["live_features_csv"], index=False)
    entry_signals.to_csv(files["entry_signals_csv"], index=False)
    positions.to_csv(files["current_positions_csv"], index=False)
    close_signals.to_csv(files["close_signals_csv"], index=False)
    _safe_to_parquet(live_features, output_dir / "live_features.parquet")
    _safe_to_parquet(entry_signals, output_dir / "entry_signals.parquet")
    _safe_to_parquet(positions, output_dir / "current_positions.parquet")
    _safe_to_parquet(close_signals, output_dir / "close_signals.parquet")

    universe = [str(t).upper() for t in cfg.get("universe", [])]
    scanned_tickers = (
        sorted(live_features["ticker"].dropna().astype(str).str.upper().unique())
        if not live_features.empty and "ticker" in live_features.columns else []
    )
    missing_tickers = [ticker for ticker in universe if ticker not in set(scanned_tickers)]
    critical_cols = ["front_iv", "back_iv", "iv_spread", "calendar_debit_bs"]
    incomplete_rows = int(live_features[critical_cols].isna().any(axis=1).sum()) if not live_features.empty else 0

    snapshot = {
        "snapshot_ts": snapshot_ts,
        "snapshot_time": datetime.now().isoformat(timespec="seconds"),
        "trade_date": today.isoformat(),
        "universe": universe,
        "counts": {
            "live_feature_rows": int(len(live_features)),
            "missing_universe_tickers": int(len(missing_tickers)),
            "incomplete_live_feature_rows": incomplete_rows,
            "entry_signal_rows": int(len(entry_signals)),
            "position_rows": int(len(positions)),
            "close_signal_rows": int(len(close_signals)),
            "close_recommendations": int((close_signals.get("recommendation", pd.Series(dtype=str)) == "CLOSE").sum())
            if not close_signals.empty else 0,
        },
        "missing_tickers": missing_tickers,
        "files": {key: str(path) for key, path in files.items()},
        "strategy_config_snapshot": {
            "signals": cfg.get("signals", {}),
            "expiry": cfg.get("expiry", {}),
            "entry": cfg.get("entry", {}),
            "exit": cfg.get("exit", {}),
            "costs": cfg.get("costs", {}),
        },
        "notes": [
            "Entry signals use live IBKR option quotes plus historical backtest features for z-score/percentile when provided.",
            "Close signals for existing positions are recommendations only; verify contracts and market prices before trading.",
            "Externally opened positions may not have known strategy entry date or entry IV spread, so close checks rely on available IBKR average cost, live z-score, profit/stop, and DTE.",
            "Live contract selection searches nearby strikes and expiry pairs because IBKR strike lists can include invalid strike/expiry combinations.",
        ],
    }
    files["snapshot_json"].write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

    # Timestamped archival copies for later comparison.
    archive_dir = output_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    live_features.to_csv(archive_dir / f"live_features_{snapshot_ts}.csv", index=False)
    entry_signals.to_csv(archive_dir / f"entry_signals_{snapshot_ts}.csv", index=False)
    positions.to_csv(archive_dir / f"current_positions_{snapshot_ts}.csv", index=False)
    close_signals.to_csv(archive_dir / f"close_signals_{snapshot_ts}.csv", index=False)
    (archive_dir / f"snapshot_{snapshot_ts}.json").write_text(
        json.dumps(snapshot, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "live_features": live_features,
        "entry_signals": entry_signals,
        "positions": positions,
        "close_signals": close_signals,
        "snapshot": snapshot,
    }
