"""
Walk-forward optimisation for the calendar spread scanner.

What this does
--------------
Divides the full feature history into sequential train/test folds.
In each fold:
  1. Run the scenario grid on the TRAINING window to find the best parameter set
     (highest gross profit factor, subject to consistency/min-trades filters).
  2. Apply that best parameter set to the TEST window — pure out-of-sample.
  3. Record OOS P&L, Sharpe, drawdown, and trade statistics.
  4. Capital compounds: the ending equity of each test window becomes the
     starting capital of the next.

Performance optimisations
--------------------------
1. slim_features(): only the columns the backtest engine needs are passed
   into ProcessPoolExecutor — reduces pickle/unpickle overhead by 60-70%.
2. save_artifacts=False in training grid: skips audit_entry_filters()
   (slow row-by-row apply) during training passes where it's not needed.
3. Shared ProcessPoolExecutor across folds: avoids repeated process
   creation overhead.

Interpreting results
--------------------
- If OOS Sharpe consistently degrades vs. in-sample, overfitting is present.
- If the same scenario is selected in most folds, the strategy is stable.
- If OOS win rate is below 55%, reconsider the signal filters.
- stitched_max_drawdown_pct is the key live-trading risk metric: the worst
  drawdown the strategy would have experienced in real time.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .backtest import PortfolioState, Position, run_backtest
from .grid_backtest import _deep_update, run_backtest_grid, slim_features
from .llm_report import save_llm_walk_forward_report
from .metrics import compute_metrics


def build_walk_forward_folds(
    features:     pd.DataFrame,
    train_months: int = 24,
    test_months:  int = 6,
) -> list[dict[str, pd.Timestamp]]:
    """
    Create sequential rolling train/test windows from available dates.

    Each fold:
      [train_start ... train_end] | [test_start ... test_end]

    Folds are non-overlapping in the test window. The train window always
    starts from the beginning of the data — this is expanding-window WFO,
    not rolling-window, which generally gives more stable parameter selection.
    """
    if features.empty:
        return []

    dates  = pd.to_datetime(features["tradeDate"])
    start  = dates.min().normalize()
    end    = dates.max().normalize()
    folds: list[dict[str, pd.Timestamp]] = []

    cursor = start + pd.DateOffset(months=train_months)
    while cursor <= end:
        train_end = cursor - pd.Timedelta(days=1)
        test_end  = min(
            cursor + pd.DateOffset(months=test_months) - pd.Timedelta(days=1),
            end
        )
        folds.append({
            "train_start": start,
            "train_end":   train_end,
            "test_start":  cursor,
            "test_end":    test_end,
        })
        cursor = cursor + pd.DateOffset(months=test_months)

    return folds


def _slice_by_date(
    df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Slice a feature DataFrame to a date window. Dates are inclusive."""
    if df.empty:
        return df.copy()
    dates = pd.to_datetime(df["tradeDate"])
    mask  = dates.between(start, end)
    return df.loc[mask].copy()


def _select_best_training_scenario(
    train_summary: pd.DataFrame, min_trades: int
) -> pd.Series | None:
    """
    Select the best scenario from the training grid summary.

    min_trades filter: scenarios with too few training trades have
    unreliable Sharpe estimates. Filter them out before ranking.
    A reasonable minimum is 20-30 trades for a 12-18 month training window.

    Consistency filter: require at least 2 of 3 training sub-windows to
    have positive realized P&L. This penalizes scenarios whose training
    performance came from one lucky cluster.

    Ranking:
      1. Gross profit factor
      2. Expectancy per trade as % of capital deployed
      3. Total trades
    """
    eligible = train_summary.copy()
    if min_trades and min_trades > 0:
        eligible = eligible[eligible["total_trades"].fillna(0) >= min_trades].copy()

    if "positive_train_subwindows" in eligible.columns:
        eligible = eligible[eligible["positive_train_subwindows"].fillna(0) >= 2].copy()

    if eligible.empty:
        return None

    profit_factor_col = (
        "gross_profit_factor"
        if "gross_profit_factor" in eligible.columns
        else "profit_factor"
    )
    expectancy_col = (
        "expectancy_return_on_capital_pct"
        if "expectancy_return_on_capital_pct" in eligible.columns
        else "expectancy_usd"
    )

    eligible[profit_factor_col] = pd.to_numeric(
        eligible[profit_factor_col], errors="coerce"
    ).fillna(-np.inf)
    eligible[expectancy_col] = pd.to_numeric(
        eligible[expectancy_col], errors="coerce"
    ).fillna(-np.inf)
    eligible["total_trades"] = pd.to_numeric(
        eligible["total_trades"], errors="coerce"
    ).fillna(0)

    eligible = eligible.sort_values(
        [profit_factor_col, expectancy_col, "total_trades"],
        ascending=[False, False, False],
    )
    return eligible.iloc[0]


def _portfolio_to_frame(
    portfolio: list[PortfolioState], fold: int | None = None
) -> pd.DataFrame:
    """Convert a list of PortfolioState objects to a DataFrame."""
    if not portfolio:
        cols = ["date", "cash", "positions_value", "equity", "n_positions", "trades_today"]
        if fold is not None:
            cols.append("fold")
        return pd.DataFrame(columns=cols)

    pdf = pd.DataFrame([{
        "date":            p.date,
        "cash":            p.cash,
        "positions_value": p.positions_value,
        "equity":          p.equity,
        "n_positions":     p.n_positions,
        "trades_today":    p.trades_today,
    } for p in portfolio])
    pdf["date"] = pd.to_datetime(pdf["date"])
    if fold is not None:
        pdf["fold"] = fold
    return pdf


def _drawdown_series(equity: pd.Series) -> pd.Series:
    """Compute running drawdown as a percentage of the rolling peak."""
    peak = equity.cummax()
    return (equity / peak - 1.0) * 100.0


def _compute_window_metrics(
    trades:          list[dict[str, Any]],
    portfolio:       list[PortfolioState],
    starting_capital: float,
    fold_no:         int,
    scenario_name:   str,
    fold:            dict[str, pd.Timestamp],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build per-window OOS metrics row and tagged equity DataFrame."""
    metrics      = compute_metrics(trades, portfolio)
    tdf          = pd.DataFrame(trades) if trades else pd.DataFrame()
    portfolio_df = _portfolio_to_frame(portfolio, fold=fold_no)

    if portfolio_df.empty:
        oos_total_return = 0.0
        drawdown_pct     = 0.0
    else:
        ending_capital   = float(portfolio_df["equity"].iloc[-1])
        oos_total_return = (
            (ending_capital / starting_capital - 1.0) * 100.0
            if starting_capital else 0.0
        )
        portfolio_df["drawdown_pct"] = _drawdown_series(portfolio_df["equity"])
        drawdown_pct = float(portfolio_df["drawdown_pct"].min())

    row = {
        "fold":                               fold_no,
        "train_start":                        fold["train_start"].date(),
        "train_end":                          fold["train_end"].date(),
        "test_start":                         fold["test_start"].date(),
        "test_end":                           fold["test_end"].date(),
        "selected_scenario":                  scenario_name,
        "oos_total_return_pct":               round(oos_total_return, 2),
        "oos_sharpe":                         float(metrics.get("sharpe_ratio", 0) or 0),
        "oos_sortino":                        float(metrics.get("sortino_ratio", 0) or 0),
        "oos_max_drawdown_pct":               round(drawdown_pct, 2),
        "oos_profit_factor":                  float(metrics.get("profit_factor", 0) or 0),
        "oos_win_rate_pct":                   float(metrics.get("win_rate_pct", 0) or 0),
        "oos_expectancy_per_trade_usd":       float(metrics.get("expectancy_usd", 0) or 0),
        "number_of_trades":                   int(metrics.get("total_trades", 0) or 0),
        "average_hold_time_days":             float(metrics.get("avg_hold_days", 0) or 0),
        "average_return_on_capital_pct":      (
            float(tdf["return_on_capital_pct"].mean()) if not tdf.empty else 0.0
        ),
        "best_trade_usd":                     float(metrics.get("best_trade_usd", 0) or 0),
        "worst_trade_usd":                    float(metrics.get("worst_trade_usd", 0) or 0),
        "net_total_pnl_usd":                  float(metrics.get("total_pnl_usd", 0) or 0),
        "gross_total_pnl_usd":                (
            float(tdf["gross_pnl"].sum())
            if not tdf.empty and "gross_pnl" in tdf.columns else 0.0
        ),
        "transaction_costs_usd":              (
            float(tdf["transaction_costs"].sum())
            if not tdf.empty and "transaction_costs" in tdf.columns else 0.0
        ),
        "performance_after_transaction_costs_usd": float(metrics.get("total_pnl_usd", 0) or 0),
        "starting_capital":                   round(starting_capital, 2),
        "ending_capital":                     round(
            float(portfolio_df["equity"].iloc[-1])
            if not portfolio_df.empty else starting_capital, 2
        ),
    }
    return row, portfolio_df


def summarize_walk_forward_results(
    folds_df:          pd.DataFrame,
    stitched_portfolio: pd.DataFrame,
    stitched_trades:   pd.DataFrame,
) -> dict[str, Any]:
    """
    Aggregate OOS metrics across all test windows.

    The stitched equity curve concatenates all test-window equity curves in
    chronological order. Its max drawdown is the key live-trading risk metric:
    it represents the worst experience a real trader would have had following
    this strategy with WFO parameter selection.
    """
    if folds_df.empty:
        return {}

    summary = {
        "average_oos_sharpe":             round(float(folds_df["oos_sharpe"].mean()), 2),
        "median_oos_sharpe":              round(float(folds_df["oos_sharpe"].median()), 2),
        "percentage_positive_oos_windows": round(
            float((folds_df["net_total_pnl_usd"] > 0).mean() * 100.0), 2
        ),
        "percentage_positive_oos_trades": (
            round(float((stitched_trades["pnl"] > 0).mean() * 100.0), 2)
            if not stitched_trades.empty else 0.0
        ),
        "worst_oos_window_usd":           round(float(folds_df["net_total_pnl_usd"].min()), 2),
    }

    if not stitched_portfolio.empty:
        stitched = stitched_portfolio.sort_values("date").copy()
        stitched["drawdown_pct"] = _drawdown_series(stitched["equity"])
        summary["stitched_max_drawdown_pct"]  = round(float(stitched["drawdown_pct"].min()), 2)
        summary["stitched_total_return_pct"]  = round(
            float((stitched["equity"].iloc[-1] / stitched["equity"].iloc[0] - 1.0) * 100.0), 2
        )
    else:
        summary["stitched_max_drawdown_pct"] = 0.0
        summary["stitched_total_return_pct"] = 0.0

    return summary


def _finalize_open_positions(
    results:   dict[str, Any],
    fold_end:  pd.Timestamp,
    cfg:       dict[str, Any],
) -> tuple[list[dict[str, Any]], list[PortfolioState], float]:
    """
    Force-close any remaining open positions at the fold boundary.

    Positions that are still open at the end of a test window are closed at
    their last MTM value with slippage applied. This prevents positions from
    carrying over into the next fold where a different parameter set applies.
    """
    trades          = list(results["trades"])
    portfolio       = list(results["portfolio"])
    open_positions: list[Position] = list(results["open"])

    if not portfolio:
        return trades, portfolio, float(cfg["backtest"]["initial_capital"])

    last_equity  = portfolio[-1].equity
    capital_after = last_equity
    slip = cfg["costs"]["slippage_pct"]
    comm = cfg["costs"]["commission_per_contract"] * 2

    for pos in open_positions:
        gross_exit_value     = max(pos.current_value, 0)
        exit_value           = max(gross_exit_value * (1 - slip) - comm / 100.0, 0.0)
        gross_pnl_per_spread = (gross_exit_value - pos.entry_value) * 100
        pnl_per_spread       = (exit_value - pos.entry_cost) * 100
        pnl                  = pnl_per_spread * pos.contracts
        gross_pnl            = gross_pnl_per_spread * pos.contracts
        transaction_costs    = gross_pnl - pnl
        capital_deployed     = pos.entry_cost  * 100 * pos.contracts
        gross_capital        = pos.entry_value * 100 * pos.contracts

        trades.append({
            "ticker":                      pos.ticker,
            "entry_date":                  pos.entry_date,
            "exit_date":                   fold_end.date(),
            "entry_value":                 pos.entry_value,
            "entry_cost":                  pos.entry_cost,
            "gross_exit_value":            gross_exit_value,
            "exit_value":                  exit_value,
            "gross_pnl":                   gross_pnl,
            "pnl":                         pnl,
            "pnl_pct":                     pos.current_pnl_pct,
            "transaction_costs":           transaction_costs,
            "capital_deployed":            capital_deployed,
            "gross_capital_deployed":      gross_capital,
            "return_on_capital_pct":       (pnl / capital_deployed * 100) if capital_deployed > 0 else 0.0,
            "gross_return_on_capital_pct": (gross_pnl / gross_capital * 100) if gross_capital > 0 else 0.0,
            "contracts":                   pos.contracts,
            "hold_days":                   (fold_end.date() - pos.entry_date).days,
            "exit_reason":                 "walk_forward_fold_end",
            "entry_zscore":                pos.entry_spread_zscore,
            "entry_pctile":                pos.entry_spread_pctile,
            "entry_front_iv":              pos.entry_front_iv,
            "entry_back_iv":               pos.entry_back_iv,
            "entry_iv_spread":             pos.entry_iv_spread,
        })
        capital_after += (
            exit_value * 100 * pos.contracts
            - pos.current_value * 100 * pos.contracts
        )

    portfolio[-1] = PortfolioState(
        date            = portfolio[-1].date,
        cash            = capital_after,
        positions_value = 0.0,
        equity          = capital_after,
        n_positions     = 0,
        trades_today    = portfolio[-1].trades_today,
    )
    return trades, portfolio, capital_after


def run_walk_forward_optimisation(
    features:                    pd.DataFrame,
    earnings_df:                 pd.DataFrame,
    base_cfg:                    dict[str, Any],
    scenarios:                   list[dict[str, Any]],
    output_dir:                  Path,
    train_months:                int = 18,
    test_months:                 int = 6,
    min_train_trades:            int = 0,
    max_workers:                 int | None = None,
    progress_callback:           Any | None = None,
    scenario_progress_callback:  Any | None = None,
) -> dict[str, Any]:
    """
    Run rolling walk-forward optimisation and persist fold summaries.

    Parameters
    ----------
    progress_callback : called as (fold_no, total_folds, message_str) after
                        each fold completes. Use for tqdm updates in notebooks.
    scenario_progress_callback : called as (fold_no, completed, total, name)
                                 after each scenario within a training fold.

    Key speed improvements vs. original
    ------------------------------------
    - slim_features() strips the feature DataFrame to only the columns the
      backtest engine needs before passing it to the process pool. This
      reduces IPC serialisation from ~150MB to ~40MB per worker per fold.
    - save_artifacts=False during training grid skips audit_entry_filters().
    - A single ProcessPoolExecutor is created once and reused across all folds,
      avoiding the overhead of spawning new processes for each fold.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-slim features once before the fold loop to reduce pickling cost.
    # slim_features keeps only the columns actually used by backtest + signals.
    features_slim = slim_features(features)

    folds = build_walk_forward_folds(
        features, train_months=train_months, test_months=test_months
    )
    current_capital         = float(base_cfg["backtest"]["initial_capital"])
    combined_trades:        list[dict[str, Any]] = []
    combined_portfolio_frames: list[pd.DataFrame] = []
    fold_rows:              list[dict[str, Any]] = []

    # Create the pool once and reuse across all folds — avoids repeated
    # process spawn overhead which is significant on Windows/Mac
    shared_executor = (
        ProcessPoolExecutor(max_workers=max_workers)
        if (max_workers and max_workers > 1) else None
    )
    run_start = time.perf_counter()

    try:
        for fold_no, fold in enumerate(folds, start=1):
            fold_start = time.perf_counter()

            # Slice slimmed features to the training and test windows
            train_features = _slice_by_date(features_slim, fold["train_start"], fold["train_end"])
            test_features  = _slice_by_date(features_slim, fold["test_start"],  fold["test_end"])

            if train_features.empty or test_features.empty:
                continue

            # ── Training phase: find best parameter set ────────────────
            # save_artifacts=False: skip audit (slow) and file writes (unnecessary)
            train_dir   = output_dir / f"fold_{fold_no:02d}" / "train"
            train_start_t = time.perf_counter()
            train_results = run_backtest_grid(
                train_features,
                earnings_df,
                base_cfg,
                scenarios,
                train_dir,
                save_artifacts=False,   # skip audit + file writes during training
                max_workers=max_workers,
                progress_callback=(
                    None if scenario_progress_callback is None
                    else lambda completed, total, name, fn=fold_no:
                        scenario_progress_callback(fn, completed, total, name)
                ),
                executor=shared_executor,
            )
            train_runtime_sec = time.perf_counter() - train_start_t
            train_summary     = train_results["summary"].copy()

            if train_summary.empty:
                if progress_callback is not None:
                    progress_callback(
                        fold_no, len(folds),
                        f"fold_{fold_no:02d}: no training summary | "
                        f"train {train_runtime_sec:.1f}s | "
                        f"total {time.perf_counter() - run_start:.1f}s",
                    )
                continue

            chosen = _select_best_training_scenario(train_summary, min_train_trades)
            if chosen is None:
                # No scenario met the minimum trade count — record empty fold
                fold_rows.append({
                    "fold":                 fold_no,
                    "train_start":          fold["train_start"].date(),
                    "train_end":            fold["train_end"].date(),
                    "test_start":           fold["test_start"].date(),
                    "test_end":             fold["test_end"].date(),
                    "selected_scenario":    "no_eligible_scenario",
                    "oos_total_return_pct": 0.0,
                    "oos_sharpe":           0.0,
                    "oos_sortino":          0.0,
                    "oos_max_drawdown_pct": 0.0,
                    "oos_profit_factor":    0.0,
                    "oos_win_rate_pct":     0.0,
                    "oos_expectancy_per_trade_usd": 0.0,
                    "number_of_trades":     0,
                    "average_hold_time_days": 0.0,
                    "average_return_on_capital_pct": 0.0,
                    "best_trade_usd":       0.0,
                    "worst_trade_usd":      0.0,
                    "net_total_pnl_usd":    0.0,
                    "gross_total_pnl_usd":  0.0,
                    "transaction_costs_usd": 0.0,
                    "performance_after_transaction_costs_usd": 0.0,
                    "starting_capital":     round(current_capital, 2),
                    "ending_capital":       round(current_capital, 2),
                    "train_trades":         int(train_summary["total_trades"].max() if not train_summary.empty else 0),
                    "train_win_rate_pct":   float(train_summary["win_rate_pct"].max() if not train_summary.empty else 0),
                    "train_positive_subwindows": int(
                        train_summary["positive_train_subwindows"].max()
                        if "positive_train_subwindows" in train_summary.columns and not train_summary.empty
                        else 0
                    ),
                    "train_runtime_sec":    round(train_runtime_sec, 2),
                    "test_runtime_sec":     0.0,
                    "fold_runtime_sec":     round(time.perf_counter() - fold_start, 2),
                    "cumulative_runtime_sec": round(time.perf_counter() - run_start, 2),
                })
                if progress_callback is not None:
                    progress_callback(
                        fold_no, len(folds),
                        f"fold_{fold_no:02d}: no eligible scenario | "
                        f"train {train_runtime_sec:.1f}s",
                    )
                continue

            # ── OOS test phase: apply best scenario to test window ─────
            scenario_name = chosen["scenario"]
            scenario      = next(s for s in scenarios if s["name"] == scenario_name)
            test_cfg      = _deep_update(base_cfg, scenario.get("overrides", {}))
            test_cfg["backtest"]["initial_capital"] = current_capital
            starting_capital = current_capital

            test_start_t = time.perf_counter()
            # Use earnings_df from the full dataset (not slimmed)
            test_results  = run_backtest(test_features, earnings_df, test_cfg)
            oos_trades, oos_portfolio, current_capital = _finalize_open_positions(
                test_results, fold["test_end"], test_cfg
            )
            test_runtime_sec = time.perf_counter() - test_start_t

            window_row, window_portfolio = _compute_window_metrics(
                oos_trades, oos_portfolio, starting_capital,
                fold_no, scenario_name, fold,
            )
            # Append training-phase diagnostics for comparison
            window_row["train_trades"]         = int(chosen.get("total_trades", 0) or 0)
            window_row["train_win_rate_pct"]   = float(chosen.get("win_rate_pct", 0) or 0)
            window_row["train_profit_factor"]  = float(chosen.get("profit_factor", 0) or 0)
            window_row["train_gross_profit_factor"] = float(chosen.get("gross_profit_factor", 0) or 0)
            window_row["train_expectancy_return_on_capital_pct"] = float(
                chosen.get("expectancy_return_on_capital_pct", 0) or 0
            )
            window_row["train_positive_subwindows"] = int(
                chosen.get("positive_train_subwindows", 0) or 0
            )
            window_row["train_positive_subwindow_fraction"] = float(
                chosen.get("positive_train_subwindow_fraction", 0) or 0
            )
            window_row["train_subwindow_1_pnl"] = float(chosen.get("train_subwindow_1_pnl", 0) or 0)
            window_row["train_subwindow_2_pnl"] = float(chosen.get("train_subwindow_2_pnl", 0) or 0)
            window_row["train_subwindow_3_pnl"] = float(chosen.get("train_subwindow_3_pnl", 0) or 0)
            window_row["train_sharpe"]         = float(chosen.get("sharpe_ratio", 0) or 0)
            for param_col in [
                "zscore_entry_min",
                "percentile_entry_min",
                "back_iv_percentile_max",
                "rv_iv_ratio_max",
                "max_debit_pct_of_spot",
                "profit_target_pct",
                "stop_loss_pct",
                "max_hold_days_param",
                "slippage_pct",
            ]:
                if param_col in chosen:
                    window_row[f"selected_{param_col}"] = chosen.get(param_col)
            window_row["train_runtime_sec"]    = round(train_runtime_sec, 2)
            window_row["test_runtime_sec"]     = round(test_runtime_sec, 2)
            window_row["fold_runtime_sec"]     = round(time.perf_counter() - fold_start, 2)
            window_row["cumulative_runtime_sec"] = round(time.perf_counter() - run_start, 2)

            combined_trades.extend(oos_trades)
            combined_portfolio_frames.append(window_portfolio)
            fold_rows.append(window_row)

            if progress_callback is not None:
                progress_callback(
                    fold_no, len(folds),
                    f"{scenario_name} | "
                    f"train {train_runtime_sec:.1f}s | "
                    f"test {test_runtime_sec:.1f}s | "
                    f"total {time.perf_counter() - run_start:.1f}s",
                )

    finally:
        if shared_executor is not None:
            shared_executor.shutdown()

    # ── Assemble combined OOS results ─────────────────────────────────
    folds_df         = pd.DataFrame(fold_rows)
    stitched_trades  = pd.DataFrame(combined_trades) if combined_trades else pd.DataFrame()
    stitched_portfolio = (
        pd.concat(combined_portfolio_frames, ignore_index=True)
        if combined_portfolio_frames
        else pd.DataFrame(columns=[
            "date", "cash", "positions_value", "equity", "n_positions", "trades_today", "fold"
        ])
    )

    combined_metrics = compute_metrics(
        combined_trades,
        [
            PortfolioState(
                date            = row["date"].date() if hasattr(row["date"], "date") else row["date"],
                cash            = float(row["cash"]),
                positions_value = float(row["positions_value"]),
                equity          = float(row["equity"]),
                n_positions     = int(row["n_positions"]),
                trades_today    = int(row["trades_today"]),
            )
            for _, row in stitched_portfolio.iterrows()
        ],
    ) if not stitched_portfolio.empty else {"total_trades": 0, "error": "No trades generated"}

    summary_stats = summarize_walk_forward_results(folds_df, stitched_portfolio, stitched_trades)
    summary_stats["total_runtime_sec"] = round(time.perf_counter() - run_start, 2)

    # ── Persist outputs ───────────────────────────────────────────────
    folds_df.to_csv(output_dir / "walk_forward_folds.csv", index=False)
    if combined_trades:
        stitched_trades.to_csv(output_dir / "walk_forward_trades.csv", index=False)
    if not stitched_portfolio.empty:
        stitched_portfolio.to_csv(output_dir / "walk_forward_portfolio.csv", index=False)
    with (output_dir / "walk_forward_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2, default=str)
    with (output_dir / "walk_forward_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, default=str)
    save_llm_walk_forward_report(
        output_dir,
        base_cfg,
        combined_metrics,
        summary_stats,
        folds_df,
        stitched_trades,
        stitched_portfolio,
    )

    return {
        "folds":          folds_df,
        "metrics":        combined_metrics,
        "summary":        summary_stats,
        "trades":         combined_trades,
        "trades_df":      stitched_trades,
        "portfolio":      stitched_portfolio,
        "ending_capital": current_capital,
    }
