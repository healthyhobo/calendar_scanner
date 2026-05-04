"""
Backtest engine for calendar spread scanner.

P&L estimation approach
-----------------------
Two pricing methods are available, in order of accuracy:

1. Black-Scholes pricing (calendar_debit_bs):
   Computes the calendar debit as C_back - C_front using ATM Black-Scholes
   prices with ORATS smoothed IVs (atmFitIvM1/M2) and remaining DTEs.
   This correctly accounts for DTE-weighted vega differences between legs
   and reprices accurately as the underlying moves and IVs change day-to-day.
   Used when 'calendar_debit_bs' is present in the feature table.

2. Straddle proxy (calendar_debit_proxy):
   The fast approximation: (straPxM2 - straPxM1) / 2
   Assumes calendar value scales proportionally with straddle price
   differentials. Inaccurate when (a) spot has moved from ATM, (b) the
   two legs have non-proportional vol changes. Used as fallback.

For production-grade P&L validation, use fetch_strikes_single_date() to
pull actual bid/ask on entry and exit dates (costs ~2 API calls per trade).
The backtest will systematically overstate P&L if the proxy overstates
exit values — verify against actual market prices on a sample of trades.

Gamma drag note
---------------
A long calendar is net short gamma (front Γ > back Γ). Large daily spot
moves hurt the position regardless of direction. The daily_gamma_drag
feature estimates this cost. The entry filter (max_gamma_drag_pct_of_debit)
screens out entries where this cost is unsustainable, but positions entered
during moderate vol can still experience elevated gamma drag if vol spikes
during the hold period. The BS repricing partially captures this — as
realized vol rises, the straddle prices rise and the calendar's mark-to-
market changes accordingly.
"""
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .signals import check_exit, screen_entries

logger = logging.getLogger(__name__)


@dataclass
class Position:
    ticker:               str
    entry_date:           date
    entry_value:          float   # calendar debit at entry (mid, pre-slippage)
    entry_cost:           float   # actual cost after slippage + commission
    entry_front_iv:       float
    entry_back_iv:        float
    entry_iv_spread:      float
    entry_spread_zscore:  float
    entry_spread_pctile:  float
    entry_front_dte:      float
    contracts:            int
    current_value:        float = 0.0
    current_pnl:          float = 0.0
    current_pnl_pct:      float = 0.0
    exit_date:            date | None = None
    exit_reason:          str = ""


@dataclass
class PortfolioState:
    date:            date
    cash:            float
    positions_value: float
    equity:          float
    n_positions:     int
    trades_today:    int = 0


def _get_calendar_value(row: Mapping | pd.Series) -> float:
    """
    Return the best available calendar spread value for a given row.

    Priority order:
    1. calendar_debit_bs    — Black-Scholes ATM pricing (most accurate)
    2. calendar_debit_proxy — straddle-price approximation (fast fallback)
    3. Direct straddle computation from smoothed/raw straddle columns

    The BS value correctly reprices the calendar as spot moves and as
    the ORATS fitted IVs evolve day-to-day. Use it for MTM when available.
    """
    # 1. BS debit (preferred)
    bs_val = row.get("calendar_debit_bs", np.nan)
    if not np.isnan(bs_val) and bs_val > 0:
        return bs_val

    # 2. Pre-computed straddle proxy
    proxy = row.get("calendar_debit_proxy", np.nan)
    if not np.isnan(proxy) and proxy > 0:
        return proxy

    # 3. Compute from raw straddle columns directly
    for m2_col, m1_col in [("smoothStrPxM2", "smoothStraPxM1"),
                            ("straPxM2",      "straPxM1")]:
        if m2_col in row and m1_col in row:
            v2, v1 = row[m2_col], row[m1_col]
            if not np.isnan(v2) and not np.isnan(v1) and v2 > v1:
                return (v2 - v1) / 2.0

    return np.nan


def _apply_entry_slippage(value: float, cfg: dict) -> float:
    """
    Adjust calendar value for entry costs.

    You pay above mid to buy (buy side of the bid-ask spread) plus
    per-leg commissions. Slippage is applied as a percentage of
    calendar value; commission is per contract (2 contracts per calendar).

    At 3% slippage: a $2.00 calendar costs $2.06 + commissions.
    Conservative test: run at 7% to check if edge survives wider spreads.
    """
    slip = cfg["costs"]["slippage_pct"]
    comm = cfg["costs"]["commission_per_contract"] * 2   # 2 legs per calendar
    return value * (1 + slip) + comm / 100.0


def _apply_exit_slippage(value: float, cfg: dict) -> float:
    """
    Adjust calendar value for exit costs.

    You receive below mid when closing (sell side of bid-ask) plus
    per-leg commissions. Value is floored at 0 (can't receive negative).
    """
    slip = cfg["costs"]["slippage_pct"]
    comm = cfg["costs"]["commission_per_contract"] * 2
    return max(value * (1 - slip) - comm / 100.0, 0.0)


def _position_size(entry_cost: float, cash: float, cfg: dict) -> int:
    """
    Determine number of contracts based on risk-per-trade sizing.

    Risk per trade = risk_per_trade_pct × portfolio equity.
    Each contract represents 100 shares, so cost_per_contract = debit × 100.
    Returns 0 if less than 1 contract is affordable.
    """
    risk_pct  = cfg["sizing"]["risk_per_trade_pct"]
    max_risk  = cash * risk_pct
    cost_100  = entry_cost * 100   # cost per 1 contract (100-share multiplier)
    if cost_100 <= 0:
        return 0
    contracts = int(max_risk / cost_100)
    return max(contracts, 1) if contracts >= 1 else 0


def run_backtest(
    features: pd.DataFrame,
    earnings_df: pd.DataFrame,
    cfg: dict,
    progress_callback=None,
) -> dict:
    """
    Run a full walk-through backtest on the pre-computed feature table.

    Algorithm per trading day:
    1. MTM all open positions using that day's calendar value
    2. Check exits for all open positions
    3. Close positions that triggered an exit rule
    4. Screen new entry candidates from today's features
    5. Open new positions within portfolio limits
    6. Record portfolio state (cash + positions)

    The simulation is sequential and deterministic — no lookahead.
    Each position is sized independently based on cash available at entry.

    Returns
    -------
    dict with keys:
        'trades'    : list of closed trade dicts
        'portfolio' : list of PortfolioState objects (daily equity curve)
        'open'      : list of positions still open at end of period
    """
    initial_capital = cfg["backtest"]["initial_capital"]
    max_positions   = cfg["entry"]["max_concurrent_positions"]
    max_per_ticker  = cfg["entry"]["max_positions_per_ticker"]

    features = features.copy()
    features["tradeDate"] = pd.to_datetime(features["tradeDate"]).dt.date

    # Pre-build date-indexed structures for O(1) daily lookups
    all_dates = sorted(features["tradeDate"].unique())
    if progress_callback is not None:
        progress_callback(0, len(all_dates), "pre-screening entries", 0, 0)
    # Plain dicts are much faster than pandas .loc/iterrows in the simulation loop.
    row_lookup = {
        td: {
            row["ticker"]: row
            for row in grp.drop_duplicates(subset=["ticker"], keep="first").to_dict("records")
        }
        for td, grp in features.groupby("tradeDate", sort=True)
    }
    # Pre-screen all entry candidates once (avoids re-filtering every day)
    screened = screen_entries(features, earnings_df, cfg)
    candidates_by_date = {
        td: grp.to_dict("records")
        for td, grp in screened.groupby("tradeDate", sort=True)
    }

    open_positions:   list[Position]     = []
    closed_trades:    list[dict]         = []
    portfolio_history: list[PortfolioState] = []
    cash = initial_capital

    total_dates = len(all_dates)
    for idx, td in enumerate(all_dates, start=1):
        trades_today  = 0
        today_lookup  = row_lookup.get(td)

        # ── Step 1: MTM all open positions ────────────────────────────
        # Re-price each position using today's calendar value.
        # The BS pricing naturally reflects spot movement and IV changes,
        # so the MTM correctly shrinks when the trade is going against us.
        for pos in open_positions:
            row = None
            if today_lookup is not None:
                row = today_lookup.get(pos.ticker)
            if row is not None:
                val = _get_calendar_value(row)
                if not np.isnan(val):
                    pos.current_value   = val
                    pos.current_pnl     = val - pos.entry_value
                    pos.current_pnl_pct = (
                        pos.current_pnl / pos.entry_value if pos.entry_value > 0 else 0.0
                    )

        # ── Step 2 & 3: Check exits and close positions ───────────────
        still_open = []
        ticker_counts = {}
        for pos in open_positions:
            row = None
            if today_lookup is not None:
                row = today_lookup.get(pos.ticker)

            current_zscore  = row["spread_zscore"] if row is not None else np.nan
            current_front_iv = row.get("front_iv", np.nan) if row is not None else np.nan
            current_back_iv  = row.get("back_iv", np.nan) if row is not None else np.nan
            current_iv_spread = (
                current_front_iv - current_back_iv
                if not pd.isna(current_front_iv) and not pd.isna(current_back_iv)
                else np.nan
            )
            front_dte       = (
                row["front_dte"]
                if (row is not None and "front_dte" in row)
                else np.nan
            )

            # Estimate current front DTE from days held if ORATS data missing
            days_held = (td - pos.entry_date).days
            est_front_dte = max(pos.entry_front_dte - days_held, 0)
            if not np.isnan(front_dte):
                est_front_dte = front_dte   # use actual when available

            reason = check_exit(
                {
                    "entry_date": pos.entry_date,
                    "entry_iv_spread": pos.entry_iv_spread,
                },
                td,
                pos.current_pnl_pct,
                current_zscore,
                est_front_dte,
                cfg,
                current_front_iv=current_front_iv,
                current_back_iv=current_back_iv,
            )

            if reason is not None:
                # Exit: compute net and gross P&L
                exit_val            = _apply_exit_slippage(max(pos.current_value, 0), cfg)
                gross_exit_value    = max(pos.current_value, 0)
                gross_pnl_per_spread = (gross_exit_value - pos.entry_value) * 100
                pnl_per_spread       = (exit_val - pos.entry_cost) * 100
                pnl                  = pnl_per_spread * pos.contracts
                gross_pnl            = gross_pnl_per_spread * pos.contracts
                transaction_costs    = gross_pnl - pnl
                capital_deployed     = pos.entry_cost  * 100 * pos.contracts
                gross_capital        = pos.entry_value * 100 * pos.contracts

                closed_trades.append({
                    "ticker":                    pos.ticker,
                    "entry_date":                pos.entry_date,
                    "exit_date":                 td,
                    "entry_value":               pos.entry_value,
                    "entry_cost":                pos.entry_cost,
                    "gross_exit_value":          gross_exit_value,
                    "exit_value":                exit_val,
                    "gross_pnl":                 gross_pnl,
                    "pnl":                       pnl,
                    "pnl_pct":                   pos.current_pnl_pct,
                    "transaction_costs":         transaction_costs,
                    "capital_deployed":          capital_deployed,
                    "gross_capital_deployed":    gross_capital,
                    "return_on_capital_pct":     (pnl / capital_deployed * 100) if capital_deployed > 0 else 0.0,
                    "gross_return_on_capital_pct": (gross_pnl / gross_capital * 100) if gross_capital > 0 else 0.0,
                    "contracts":                 pos.contracts,
                    "hold_days":                 (td - pos.entry_date).days,
                    "exit_reason":               reason,
                    "entry_zscore":              pos.entry_spread_zscore,
                    "entry_pctile":              pos.entry_spread_pctile,
                    "entry_front_iv":            pos.entry_front_iv,
                    "entry_back_iv":             pos.entry_back_iv,
                    "entry_iv_spread":           pos.entry_iv_spread,
                    "exit_front_iv":             current_front_iv,
                    "exit_back_iv":              current_back_iv,
                    "exit_iv_spread":            current_iv_spread,
                    "iv_spread_compression":     (
                        current_iv_spread / pos.entry_iv_spread
                        if pos.entry_iv_spread and not np.isnan(pos.entry_iv_spread)
                        and not np.isnan(current_iv_spread)
                        else np.nan
                    ),
                })
                cash += exit_val * 100 * pos.contracts
                trades_today += 1
            else:
                still_open.append(pos)
                ticker_counts[pos.ticker] = ticker_counts.get(pos.ticker, 0) + 1

        open_positions = still_open

        # ── Step 4 & 5: Screen and open new entries ───────────────────
        candidates = candidates_by_date.get(td, ())
        for cand in candidates:
            if len(open_positions) >= max_positions:
                break   # portfolio at capacity

            ticker = cand["ticker"]
            if ticker_counts.get(ticker, 0) >= max_per_ticker:
                continue

            entry_val = _get_calendar_value(cand)
            if np.isnan(entry_val) or entry_val <= 0:
                continue

            entry_cost = _apply_entry_slippage(entry_val, cfg)
            contracts  = _position_size(entry_cost, cash, cfg)
            if contracts <= 0:
                continue

            total_cost = entry_cost * 100 * contracts
            # Hard cap: single trade cannot use more than 20% of cash
            if total_cost > cash * 0.20:
                contracts = max(int(cash * 0.20 / (entry_cost * 100)), 0)
                if contracts <= 0:
                    continue
                total_cost = entry_cost * 100 * contracts

            cash -= total_cost

            pos = Position(
                ticker              = ticker,
                entry_date          = td,
                entry_value         = entry_val,
                entry_cost          = entry_cost,
                entry_front_iv      = cand.get("front_iv",          np.nan),
                entry_back_iv       = cand.get("back_iv",           np.nan),
                entry_iv_spread     = cand.get("iv_spread",         np.nan),
                entry_spread_zscore = cand.get("spread_zscore",     np.nan),
                entry_spread_pctile = cand.get("spread_pctile",     np.nan),
                entry_front_dte     = cand.get("front_dte",         30),
                contracts           = contracts,
                current_value       = entry_val,
            )
            open_positions.append(pos)
            ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
            trades_today += 1

        # ── Step 6: Record portfolio state ────────────────────────────
        pos_value = sum(p.current_value * 100 * p.contracts for p in open_positions)
        portfolio_history.append(PortfolioState(
            date            = td,
            cash            = cash,
            positions_value = pos_value,
            equity          = cash + pos_value,
            n_positions     = len(open_positions),
            trades_today    = trades_today,
        ))

        if progress_callback is not None:
            progress_callback(idx, total_dates, td, len(open_positions), len(closed_trades))

    return {
        "trades":    closed_trades,
        "portfolio": portfolio_history,
        "open":      open_positions,
    }
