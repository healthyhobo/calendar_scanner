"""
Entry and exit signal rules for calendar spread scanner.

Key design decisions
--------------------
1. Entry signal uses iv30d - iv60d (constant-maturity spread), not monthly
   contract labels, to avoid roll artifacts.

2. min_hold_days gate: suppresses normalization/time-stop exits until a
   minimum hold period, preventing whipsaw exits on noise.

3. Gamma drag filter (new): rejects entries where the expected daily gamma
   cost exceeds a configurable fraction of the calendar debit. This excludes
   the highest-vol regimes where short-gamma losses consume the trade edge.
   See config.yaml: signals.max_gamma_drag_pct_of_debit

4. back_iv_percentile_max (now enabled in default config): rejects entries
   when back-month IV is already in the top X% of its history. High absolute
   vol at entry means the short-gamma drag will be severe even if the spread
   normalizes.

Why these filters help win rate
--------------------------------
Entering when z-score is high but absolute vol is also high is a double-edged
setup: the spread is dislocated (good signal) but gamma drag and realized vol
are maximally unfavourable. Filtering on back_iv_pctile and gamma_drag screens
out the cases where the edge is most likely to be consumed by costs before
the spread reverts.
"""
import numpy as np
import pandas as pd
from datetime import timedelta


def _coerce_to_date(value):
    """Normalise strings, timestamps, and date-like objects to Python dates."""
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


# ── Individual row-level filters (used by audit) ────────────────────────

def passes_liquidity(row: pd.Series, cfg: dict) -> bool:
    """
    Check that the underlying and its options are liquid enough to trade.

    Uses aggregate option volume from ORATS cores — not per-strike OI.
    Both current-day volume (total_opt_volume) and 20-day average are
    checked so a single unusual day doesn't pass/fail a name.
    """
    liq = cfg["liquidity"]

    spot = row.get("stock_price", np.nan)
    if np.isnan(spot) or spot < liq["min_stock_price"]:
        return False

    opt_vol  = row.get("total_opt_volume", np.nan)
    avg_vol  = row.get("avgOptVolu20d", np.nan)
    avg_floor = liq.get("min_avg_opt_volume_20d", liq["min_option_volume"])

    if not np.isnan(opt_vol) and opt_vol < liq["min_option_volume"]:
        return False
    if not np.isnan(avg_vol) and avg_vol < avg_floor:
        return False

    return True


def passes_expiry_structure(row: pd.Series, cfg: dict) -> bool:
    """
    Enforce the calendar's DTE structure at entry.

    We need:
      - Front DTE ≥ 10 (hard floor regardless of config)
      - Back DTE > Front DTE (back must be further out)
      - Minimum gap between front and back expiry

    This ensures we're trading a genuine calendar (time spread) and not
    something collapsing into a short-term trade accidentally.
    """
    expiry    = cfg.get("expiry", {})
    front_dte = row.get("front_dte", np.nan)
    back_dte  = row.get("back_dte", np.nan)

    if np.isnan(front_dte):
        return False

    front_min = max(int(expiry.get("front_dte_min", 10)), 10)
    if front_dte < front_min:
        return False

    front_max = expiry.get("front_dte_max")
    if front_max is not None and not np.isnan(front_max) and front_dte > front_max:
        return False

    if not np.isnan(back_dte):
        if back_dte <= front_dte:
            return False
        back_min = expiry.get("back_dte_min")
        if back_min is not None and back_dte < back_min:
            return False
        back_max = expiry.get("back_dte_max")
        if back_max is not None and back_dte > back_max:
            return False
        gap_min = expiry.get("min_gap_days")
        if gap_min is not None and (back_dte - front_dte) < gap_min:
            return False

    return True


def passes_signal(row: pd.Series, cfg: dict) -> bool:
    """
    Check all term-structure signal thresholds.

    Filters applied in order:
    1. z-score band: must be in [zscore_entry_min, zscore_entry_max]
       - Min: ensures spread is meaningfully dislocated (edge exists)
       - Max: extreme z-scores often reflect genuine event risk (e.g.
         earnings, central bank meetings) — these don't mean-revert quickly.
    2. Percentile: must be ≥ percentile_entry_min
       - Confirms the dislocation is historically unusual, not just noisy.
    3. back_iv_percentile_max (enabled at 60 by default):
       - Rejects entries where back-month IV is already in the top 40%
         of its own history. High absolute vol = severe gamma drag.
    4. rv_iv_ratio_max (enabled at 0.9 by default):
       - Rejects entries where realized vol ≥ 90% of implied vol.
         When RV ≥ IV, the front month is not truly "rich" — it's just
         correctly pricing the realized daily moves you're taking on.
    5. max_gamma_drag_pct_of_debit:
       - Rejects entries where the estimated daily gamma drag exceeds
         a fraction of the calendar debit. Prevents entering in high-vol
         regimes where gamma cost will consume the trade P&L.
    """
    sig      = cfg["signals"]
    zs       = row.get("spread_zscore", np.nan)
    pc       = row.get("spread_pctile", np.nan)
    biv_pc   = row.get("back_iv_pctile", np.nan)
    rv_iv    = row.get("rv_iv_ratio", np.nan)
    gamma_dr = row.get("daily_gamma_drag", np.nan)
    debit    = row.get("calendar_debit_bs",
               row.get("calendar_debit_proxy", np.nan))

    # Basic z-score and percentile gates
    if np.isnan(zs) or np.isnan(pc):
        return False
    if zs < sig["zscore_entry_min"]:
        return False
    if zs > sig["zscore_entry_max"]:
        return False
    if pc < sig["percentile_entry_min"]:
        return False

    # Back-IV absolute level filter: reject if already in elevated vol regime
    # Default: back_iv_percentile_max = 60 (reject top 40% of vol history)
    back_iv_pctile_max = sig.get("back_iv_percentile_max")
    if back_iv_pctile_max is not None and not np.isnan(biv_pc):
        if biv_pc > back_iv_pctile_max:
            return False

    # RV/IV filter: front IV must be genuinely rich vs. realized vol
    rv_iv_ratio_max = sig.get("rv_iv_ratio_max")
    if rv_iv_ratio_max is not None and not np.isnan(rv_iv):
        if rv_iv > rv_iv_ratio_max:
            return False

    # Gamma drag filter: skip entries where daily gamma cost is too high
    # relative to the calendar debit (the total risk capital in the trade).
    # At 15% per day, a 10-day hold costs ~150% of debit in gamma drag alone.
    max_drag_frac = sig.get("max_gamma_drag_pct_of_debit")
    if max_drag_frac is not None and not np.isnan(gamma_dr) and not np.isnan(debit):
        if debit > 0 and (gamma_dr / debit) > max_drag_frac:
            return False

    return True


def passes_debit_cap(row: pd.Series, cfg: dict) -> bool:
    """
    Calendar debit must not exceed threshold % of spot price.

    Prevents entering extremely expensive calendars on high-priced names
    where the debit would represent outsized capital risk. Prefer BS debit
    when available, fall back to straddle proxy.
    """
    max_pct = cfg["entry"]["max_debit_pct_of_spot"]
    # Prefer more accurate BS debit; fall back to proxy
    debit   = row.get("calendar_debit_bs", row.get("calendar_debit_proxy", np.nan))
    spot    = row.get("stock_price", np.nan)
    if np.isnan(debit) or np.isnan(spot) or spot <= 0:
        return True  # can't compute — don't filter
    if debit <= 0:
        return False
    return (debit / spot) <= max_pct


def no_earnings_before_front(row: pd.Series, earnings_dates: set) -> bool:
    """
    Reject entries where an earnings release falls before front expiry.

    Earnings create an IV crush in the front month and a volatility event
    in the underlying — both of which destroy the calendar spread's P&L
    before the term structure can normalize. We estimate front expiry as
    tradeDate + front_dte days.
    """
    td = _coerce_to_date(row.get("tradeDate"))
    if td is None:
        return True

    front_dte = row.get("front_dte", np.nan)
    if np.isnan(front_dte) or front_dte <= 0:
        return True

    est_front_expiry = td + timedelta(days=int(front_dte))

    for ed in earnings_dates:
        ed = _coerce_to_date(ed)
        if ed is None:
            continue
        if td <= ed <= est_front_expiry:
            return False
    return True


def _build_earnings_lookup(earnings_df: pd.DataFrame) -> dict[str, set]:
    """Pre-build a normalised ticker → earnings date set for fast row lookups."""
    earnings_by_ticker = {}
    if earnings_df.empty or "earnings_date" not in earnings_df.columns:
        return earnings_by_ticker
    for tkr, grp in earnings_df.groupby("ticker"):
        earnings_by_ticker[tkr] = {
            ed for ed in (_coerce_to_date(v) for v in grp["earnings_date"])
            if ed is not None
        }
    return earnings_by_ticker


EARNINGS_OK_COL = "__earnings_ok"


def _earnings_ok_mask(features: pd.DataFrame, earnings_by_ticker: dict[str, set]) -> pd.Series:
    """Return a boolean Series: True where there are no upcoming earnings in the front window."""
    if features.empty:
        return pd.Series(dtype=bool)
    if EARNINGS_OK_COL in features.columns:
        return features[EARNINGS_OK_COL].fillna(True).astype(bool)

    trade_dates = pd.to_datetime(features["tradeDate"], errors="coerce").dt.date
    front_dtes = pd.to_numeric(features.get("front_dte", np.nan), errors="coerce")
    mask = []
    for ticker, td, front_dte in zip(features["ticker"], trade_dates, front_dtes):
        if pd.isna(td) or np.isnan(front_dte) or front_dte <= 0:
            mask.append(True)
            continue

        tkr_earnings = earnings_by_ticker.get(ticker, set())
        if not tkr_earnings:
            mask.append(True)
            continue

        est_front_expiry = td + timedelta(days=int(front_dte))
        mask.append(not any(td <= ed <= est_front_expiry for ed in tkr_earnings))
    return pd.Series(mask, index=features.index)


def add_earnings_ok_column(features: pd.DataFrame, earnings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach a cached earnings gate column.

    The earnings check is independent of signal thresholds, so grid and
    walk-forward runs can compute it once per feature window instead of once
    per scenario.
    """
    if features.empty or EARNINGS_OK_COL in features.columns:
        return features
    out = features.copy()
    out[EARNINGS_OK_COL] = _earnings_ok_mask(out, _build_earnings_lookup(earnings_df))
    return out


# ── Vectorised entry screener (used by backtest engine) ─────────────────

def screen_entries(features: pd.DataFrame, earnings_df: pd.DataFrame,
                   cfg: dict) -> pd.DataFrame:
    """
    Apply all entry filters using vectorised pandas operations for speed.

    This is the hot path — called on every date during the backtest.
    Mirrors the row-level filter logic in passes_* functions exactly.
    Returns only passing rows, sorted by signal rank descending.

    Signal rank blends z-score strength and percentile position, giving
    higher priority to entries that are both statistically extreme and
    historically unusual.
    """
    if features.empty:
        return features

    sig   = cfg["signals"]
    liq   = cfg["liquidity"]
    expiry = cfg.get("expiry", {})
    entry = cfg["entry"]

    # ── Extract series ────────────────────────────────────────────────
    spot          = pd.to_numeric(features.get("stock_price", np.nan), errors="coerce")
    total_opt_vol = pd.to_numeric(features.get("total_opt_volume", np.nan), errors="coerce")
    avg_opt_vol   = pd.to_numeric(features.get("avgOptVolu20d", np.nan), errors="coerce")
    front_dte     = pd.to_numeric(features.get("front_dte", np.nan), errors="coerce")
    back_dte      = pd.to_numeric(features.get("back_dte", np.nan), errors="coerce")
    spread_z      = pd.to_numeric(features.get("spread_zscore", np.nan), errors="coerce")
    spread_pc     = pd.to_numeric(features.get("spread_pctile", np.nan), errors="coerce")
    back_iv_pc    = pd.to_numeric(features.get("back_iv_pctile", np.nan), errors="coerce")
    rv_iv         = pd.to_numeric(features.get("rv_iv_ratio", np.nan), errors="coerce")
    gamma_drag    = pd.to_numeric(features.get("daily_gamma_drag", np.nan), errors="coerce")
    # Prefer BS debit; fall back to straddle proxy
    debit = pd.to_numeric(
        features.get("calendar_debit_bs", features.get("calendar_debit_proxy", np.nan)),
        errors="coerce",
    )

    # ── Liquidity gate ────────────────────────────────────────────────
    avg_floor   = liq.get("min_avg_opt_volume_20d", liq["min_option_volume"])
    liquidity_ok = (
        spot.ge(liq["min_stock_price"])
        & (total_opt_vol.isna() | total_opt_vol.ge(liq["min_option_volume"]))
        & (avg_opt_vol.isna()   | avg_opt_vol.ge(avg_floor))
    )

    # ── Expiry structure gate ─────────────────────────────────────────
    front_min  = max(int(expiry.get("front_dte_min", 10)), 10)
    expiry_ok  = front_dte.notna() & front_dte.ge(front_min)
    front_max  = expiry.get("front_dte_max")
    if front_max is not None:
        expiry_ok &= front_dte.le(front_max)
    back_valid = back_dte.notna()
    expiry_ok &= (
        ~back_valid
        | (
            back_dte.gt(front_dte)
            & (True if expiry.get("back_dte_min") is None else back_dte.ge(expiry["back_dte_min"]))
            & (True if expiry.get("back_dte_max") is None else back_dte.le(expiry["back_dte_max"]))
            & (True if expiry.get("min_gap_days")  is None else (back_dte - front_dte).ge(expiry["min_gap_days"]))
        )
    )

    # ── Signal gate ───────────────────────────────────────────────────
    signal_ok = (
        spread_z.notna()
        & spread_pc.notna()
        & spread_z.ge(sig["zscore_entry_min"])
        & spread_z.le(sig["zscore_entry_max"])
        & spread_pc.ge(sig["percentile_entry_min"])
    )

    # Back-IV absolute level: reject elevated vol regimes
    back_iv_pctile_max = sig.get("back_iv_percentile_max")
    if back_iv_pctile_max is not None:
        signal_ok &= (back_iv_pc.isna() | back_iv_pc.le(back_iv_pctile_max))

    # RV/IV ratio: front must be genuinely rich vs. realized
    rv_iv_ratio_max = sig.get("rv_iv_ratio_max")
    if rv_iv_ratio_max is not None:
        signal_ok &= (rv_iv.isna() | rv_iv.le(rv_iv_ratio_max))

    # Gamma drag: reject if daily cost too high relative to debit
    max_drag_frac = sig.get("max_gamma_drag_pct_of_debit")
    if max_drag_frac is not None:
        # Only apply filter where both gamma_drag and debit are valid
        has_drag_data = gamma_drag.notna() & debit.notna() & debit.gt(0)
        drag_too_high = has_drag_data & ((gamma_drag / debit) > max_drag_frac)
        signal_ok &= ~drag_too_high

    # ── Debit cap gate ────────────────────────────────────────────────
    debit_ok = (
        debit.isna() | spot.isna() | spot.le(0)
        | (debit.gt(0) & ((debit / spot) <= entry["max_debit_pct_of_spot"]))
    )

    # ── Earnings gate ─────────────────────────────────────────────────
    earnings_by_ticker = {} if EARNINGS_OK_COL in features.columns else _build_earnings_lookup(earnings_df)
    earnings_ok        = _earnings_ok_mask(features, earnings_by_ticker)

    candidates = features[
        liquidity_ok & expiry_ok & signal_ok & debit_ok & earnings_ok
    ].copy()

    if candidates.empty:
        return candidates

    # Rank: 60% weight on z-score extremity, 40% on percentile position
    # Clip z-score at 4 to prevent extreme events from dominating ranking
    candidates["signal_rank"] = (
        0.6 * candidates["spread_zscore"].clip(upper=4) / 4.0
        + 0.4 * candidates["spread_pctile"] / 100.0
    )
    candidates.sort_values(
        ["tradeDate", "signal_rank"], ascending=[True, False], inplace=True
    )
    return candidates


# ── Diagnostic audit (slow row-by-row, only run for reporting) ──────────

def audit_entry_filters(features: pd.DataFrame, earnings_df: pd.DataFrame,
                        cfg: dict) -> pd.DataFrame:
    """
    Per-ticker count of rows passing/failing each entry gate.

    This is intentionally slow (row-by-row apply) because it runs only
    once for reporting — not inside the hot backtest loop.
    Use the output to diagnose which filter is most restrictive per ticker.
    """
    if features.empty:
        return pd.DataFrame()

    earnings_by_ticker = _build_earnings_lookup(earnings_df)
    sig = cfg["signals"]
    rows = []

    for ticker, grp in features.groupby("ticker"):
        liquidity_ok = grp.apply(lambda r: passes_liquidity(r, cfg), axis=1)
        expiry_ok    = grp.apply(lambda r: passes_expiry_structure(r, cfg), axis=1)
        signal_ok    = grp.apply(lambda r: passes_signal(r, cfg), axis=1)
        debit_ok     = grp.apply(lambda r: passes_debit_cap(r, cfg), axis=1)

        tkr_earnings = earnings_by_ticker.get(ticker, set())
        earnings_ok_list = [
            no_earnings_before_front(row, tkr_earnings) if tkr_earnings else True
            for _, row in grp.iterrows()
        ]
        earnings_ok = pd.Series(earnings_ok_list, index=grp.index)

        # Sub-components of signal filter for granular debugging
        z_ok = (
            grp["spread_zscore"].notna()
            & grp["spread_zscore"].ge(sig["zscore_entry_min"])
            & grp["spread_zscore"].le(sig["zscore_entry_max"])
        )
        pct_ok = (
            grp["spread_pctile"].notna()
            & grp["spread_pctile"].ge(sig["percentile_entry_min"])
        )

        # back_iv filter sub-count
        biv_max = sig.get("back_iv_percentile_max")
        back_iv_ok = (
            pd.Series(True, index=grp.index) if biv_max is None
            else grp["back_iv_pctile"].isna() | grp["back_iv_pctile"].le(biv_max)
        )

        # rv_iv filter sub-count
        rv_max = sig.get("rv_iv_ratio_max")
        rv_iv_ok = (
            pd.Series(True, index=grp.index) if rv_max is None
            else grp["rv_iv_ratio"].isna() | grp["rv_iv_ratio"].le(rv_max)
        )

        # Gamma drag filter sub-count
        drag_max = sig.get("max_gamma_drag_pct_of_debit")
        if drag_max is None or "daily_gamma_drag" not in grp.columns:
            gamma_ok = pd.Series(True, index=grp.index)
        else:
            debit_col = "calendar_debit_bs" if "calendar_debit_bs" in grp.columns else "calendar_debit_proxy"
            debit = grp.get(debit_col, pd.Series(np.nan, index=grp.index))
            drag  = grp["daily_gamma_drag"]
            has_data = drag.notna() & debit.notna() & (debit > 0)
            gamma_ok = ~(has_data & ((drag / debit) > drag_max))

        final_ok = liquidity_ok & expiry_ok & signal_ok & debit_ok & earnings_ok

        rows.append({
            "ticker":           ticker,
            "rows":             int(len(grp)),
            "with_history":     int(grp["spread_zscore"].notna().sum()),
            "liquidity_ok":     int(liquidity_ok.sum()),
            "expiry_ok":        int(expiry_ok.sum()),
            "signal_ok":        int(signal_ok.sum()),
            "debit_ok":         int(debit_ok.sum()),
            "earnings_ok":      int(earnings_ok.sum()),
            "z_ok":             int(z_ok.sum()),
            "pctile_ok":        int(pct_ok.sum()),
            "back_iv_ok":       int(back_iv_ok.sum()),
            "rv_iv_ok":         int(rv_iv_ok.sum()),
            "gamma_drag_ok":    int(gamma_ok.sum()),
            "final_candidates": int(final_ok.sum()),
        })

    audit = pd.DataFrame(rows)
    if audit.empty:
        return audit
    return audit.sort_values("ticker").reset_index(drop=True)


# ── Exit rules ───────────────────────────────────────────────────────────

class ExitReason:
    NORMALIZATION = "normalization"   # spread mean-reverted — primary profit path
    PROFIT_TARGET = "profit_target"   # hard profit cap hit
    STOP_LOSS     = "stop_loss"       # hard loss cap hit
    TIME_STOP     = "time_stop"       # front expiry approaching — avoid pin risk
    MAX_HOLD      = "max_hold"        # max calendar days held


def _is_nan(value) -> bool:
    """True for missing scalar values without raising on None."""
    try:
        return bool(pd.isna(value))
    except Exception:
        return True


def check_exit(position: dict, current_date, current_pnl_pct: float,
               current_zscore: float, current_front_dte: float,
               cfg: dict, current_front_iv: float = np.nan,
               current_back_iv: float = np.nan) -> str | None:
    """
    Evaluate whether an open position should be closed today.

    Exit priority:
    1. Stop-loss and profit-target: checked immediately (no min_hold_days gate)
    2. Normalization and time-stop: gated behind min_hold_days to prevent
       whipsaw exits on day-1 noise
    3. Max hold: always fires if the calendar days limit is reached

    Why min_hold_days matters:
        In high-vol regimes, the z-score can cross below zscore_exit within
        1-2 days purely from intraday oscillation in ORATS interpolated IVs —
        not from genuine term-structure normalization. A min_hold_days of 2-5
        prevents these "false exits" that realise a small loss before the
        actual trade thesis has had time to play out.
    """
    exit_cfg = cfg["exit"]
    sig_cfg  = cfg["signals"]

    entry_date   = position["entry_date"]
    if isinstance(entry_date, pd.Timestamp):
        entry_date = entry_date.date()
    if isinstance(current_date, pd.Timestamp):
        current_date = current_date.date()

    hold_days = (current_date - entry_date).days

    # Risk exits: fire immediately regardless of hold period
    if current_pnl_pct <= exit_cfg["stop_loss_pct"]:
        return ExitReason.STOP_LOSS
    if current_pnl_pct >= exit_cfg["profit_target_pct"]:
        return ExitReason.PROFIT_TARGET

    min_hold = int(exit_cfg.get("min_hold_days", 0) or 0)
    if hold_days >= min_hold:
        # Normalization: the spread has reverted — primary profit exit
        global_z_ok = (
            not _is_nan(current_zscore)
            and current_zscore <= sig_cfg["zscore_exit"]
        )

        entry_spread = position.get("entry_iv_spread", np.nan)
        current_spread = (
            current_front_iv - current_back_iv
            if not _is_nan(current_front_iv) and not _is_nan(current_back_iv)
            else np.nan
        )
        compression_ratio = exit_cfg.get("spread_compression_ratio", 0.4)
        if compression_ratio is None:
            spread_compressed = True
        else:
            spread_compressed = (
                not _is_nan(entry_spread)
                and not _is_nan(current_spread)
                and current_spread <= entry_spread * compression_ratio
            )

        min_norm_pnl = exit_cfg.get("normalization_min_pnl_pct", 0.0)
        if global_z_ok and spread_compressed and current_pnl_pct >= min_norm_pnl:
            return ExitReason.NORMALIZATION

        # Time stop: front leg expiry approaching — exit to avoid pin risk
        time_stop_dte = exit_cfg.get("time_stop_dte", 5)
        if not _is_nan(current_front_dte) and current_front_dte < time_stop_dte:
            return ExitReason.TIME_STOP

    # Max hold: hard cap regardless of P&L or z-score
    if hold_days >= exit_cfg["max_hold_days"]:
        return ExitReason.MAX_HOLD

    return None
