"""LLM-friendly backtest report artifacts."""
from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_default(value: Any):
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        if np.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return float(value)
    if pd.isna(value):
        return None
    return str(value)


def _clean_json(value: Any):
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [_clean_json(v) for v in value]
    if isinstance(value, (date, datetime, pd.Timestamp, np.integer, np.floating)):
        return _json_default(value)
    if isinstance(value, float):
        if np.isnan(value):
            return None
        if np.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _portfolio_frame(portfolio: list) -> pd.DataFrame:
    if not portfolio:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "date": p.date,
        "cash": p.cash,
        "positions_value": p.positions_value,
        "equity": p.equity,
        "n_positions": p.n_positions,
        "trades_today": getattr(p, "trades_today", 0),
    } for p in portfolio])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date")


def _trade_summaries(trades: list[dict]) -> dict[str, Any]:
    if not trades:
        return {
            "by_ticker": [],
            "by_exit_reason": [],
            "by_year": [],
            "best_trades": [],
            "worst_trades": [],
        }

    tdf = pd.DataFrame(trades).copy()
    for col in ["entry_date", "exit_date"]:
        if col in tdf.columns:
            tdf[col] = pd.to_datetime(tdf[col], errors="coerce")
    if "pnl_pct" in tdf.columns:
        tdf["pnl_pct"] = tdf["pnl_pct"] * 100
    if "entry_date" in tdf.columns:
        tdf["entry_year"] = tdf["entry_date"].dt.year

    by_ticker = (
        tdf.groupby("ticker", dropna=False)
        .agg(
            trades=("pnl", "size"),
            total_pnl_usd=("pnl", "sum"),
            avg_pnl_usd=("pnl", "mean"),
            win_rate_pct=("pnl", lambda s: (s > 0).mean() * 100),
            avg_pnl_pct=("pnl_pct", "mean"),
            avg_hold_days=("hold_days", "mean"),
            avg_entry_zscore=("entry_zscore", "mean"),
            avg_entry_pctile=("entry_pctile", "mean"),
        )
        .reset_index()
        .sort_values("total_pnl_usd", ascending=False)
    )

    by_exit_reason = (
        tdf.groupby("exit_reason", dropna=False)
        .agg(
            trades=("pnl", "size"),
            total_pnl_usd=("pnl", "sum"),
            avg_pnl_usd=("pnl", "mean"),
            win_rate_pct=("pnl", lambda s: (s > 0).mean() * 100),
            avg_hold_days=("hold_days", "mean"),
        )
        .reset_index()
        .sort_values("trades", ascending=False)
    )

    by_year = (
        tdf.groupby("entry_year", dropna=False)
        .agg(
            trades=("pnl", "size"),
            total_pnl_usd=("pnl", "sum"),
            avg_pnl_usd=("pnl", "mean"),
            win_rate_pct=("pnl", lambda s: (s > 0).mean() * 100),
        )
        .reset_index()
        .sort_values("entry_year")
    )

    trade_cols = [
        "ticker", "entry_date", "exit_date", "pnl", "pnl_pct", "hold_days",
        "exit_reason", "entry_zscore", "entry_pctile", "entry_front_iv", "entry_back_iv",
    ]
    trade_cols = [c for c in trade_cols if c in tdf.columns]

    return {
        "by_ticker": by_ticker.round(4).to_dict("records"),
        "by_exit_reason": by_exit_reason.round(4).to_dict("records"),
        "by_year": by_year.round(4).to_dict("records"),
        "best_trades": tdf.sort_values("pnl", ascending=False).head(10)[trade_cols].to_dict("records"),
        "worst_trades": tdf.sort_values("pnl", ascending=True).head(10)[trade_cols].to_dict("records"),
    }


def _portfolio_summary(portfolio: list) -> dict[str, Any]:
    pdf = _portfolio_frame(portfolio)
    if pdf.empty:
        return {}
    equity = pdf["equity"]
    drawdown_pct = (equity - equity.cummax()) / equity.cummax() * 100
    return {
        "start_date": pdf["date"].iloc[0],
        "end_date": pdf["date"].iloc[-1],
        "starting_equity": float(equity.iloc[0]),
        "ending_equity": float(equity.iloc[-1]),
        "peak_equity": float(equity.max()),
        "min_equity": float(equity.min()),
        "max_drawdown_pct_from_curve": float(drawdown_pct.min()),
        "avg_open_positions": float(pdf["n_positions"].mean()),
        "max_open_positions": int(pdf["n_positions"].max()),
        "days_with_trades": int((pdf.get("trades_today", 0) > 0).sum()),
    }


def _feature_summary(features: pd.DataFrame) -> dict[str, Any]:
    if features.empty:
        return {}
    out: dict[str, Any] = {
        "rows": int(len(features)),
        "tickers": sorted(features["ticker"].dropna().astype(str).str.upper().unique().tolist())
        if "ticker" in features.columns else [],
    }
    if "tradeDate" in features.columns:
        dates = pd.to_datetime(features["tradeDate"], errors="coerce").dropna()
        if not dates.empty:
            out["date_range"] = {"start": dates.min(), "end": dates.max()}
    if {"ticker", "tradeDate"}.issubset(features.columns):
        counts = (
            features.groupby("ticker")["tradeDate"]
            .count()
            .sort_values(ascending=False)
            .rename("rows")
            .reset_index()
        )
        out["rows_by_ticker"] = counts.to_dict("records")
    return out


def _audit_summary(audit: pd.DataFrame) -> list[dict[str, Any]]:
    if audit is None or audit.empty:
        return []
    return audit.to_dict("records")


def save_llm_backtest_report(
    output_dir: Path,
    cfg: dict,
    metrics: dict,
    regimes: dict,
    trades: list[dict],
    portfolio: list,
    features: pd.DataFrame,
    audit: pd.DataFrame,
) -> None:
    """Save compact JSON plus a prompt text file for LLM analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "report_type": "calendar_spread_backtest_overview",
        "generated_from": "python run_backtest.py",
        "config_snapshot": {
            "universe": cfg.get("universe", []),
            "backtest": cfg.get("backtest", {}),
            "expiry": cfg.get("expiry", {}),
            "features": cfg.get("features", {}),
            "signals": cfg.get("signals", {}),
            "liquidity": cfg.get("liquidity", {}),
            "entry": cfg.get("entry", {}),
            "exit": cfg.get("exit", {}),
            "costs": cfg.get("costs", {}),
            "sizing": cfg.get("sizing", {}),
        },
        "metrics": metrics,
        "portfolio_summary": _portfolio_summary(portfolio),
        "feature_summary": _feature_summary(features),
        "trade_summaries": _trade_summaries(trades),
        "regime_analysis": regimes,
        "entry_filter_audit_by_ticker": _audit_summary(audit),
        "artifact_files": {
            "trade_log_csv": "trade_log.csv",
            "metrics_json": "metrics.json",
            "regimes_json": "regimes.json",
            "features_parquet": "features.parquet",
            "equity_curve_png": "equity_curve.png",
            "pnl_distribution_png": "pnl_distribution.png",
            "monthly_heatmap_png": "monthly_heatmap.png",
        },
        "analysis_questions": [
            "What worked well in this run?",
            "Which tickers, years, exit reasons, and signal buckets drove profits and losses?",
            "Are there signs of overfitting, under-trading, over-trading, or weak robustness?",
            "Which config parameters should be tweaked first, and why?",
            "What follow-up backtests or grid ranges would be most informative?",
        ],
    }

    json_path = output_dir / "llm_backtest_overview.json"
    json_path.write_text(
        json.dumps(_clean_json(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    prompt = f"""You are analyzing a systematic long ATM calendar spread backtest.

Use the attached/available JSON file `llm_backtest_overview.json` as the primary source. It summarizes the config, metrics, portfolio curve, feature coverage, trade outcomes, regime breakdowns, and entry-filter audit.

Please produce:
1. Executive summary of the run.
2. What went well, with evidence from the metrics and breakdowns.
3. What looks weak or risky, including ticker/year/exit reason concentrations.
4. Which parameters should be tweaked first, with concrete suggested ranges.
5. Follow-up tests to run next.

Important context:
- Strategy: long ATM calendar spreads targeting term-structure mean reversion.
- Backtest date range: {cfg.get('backtest', {}).get('start_date')} to {cfg.get('backtest', {}).get('end_date')}.
- Universe size: {len(cfg.get('universe', []))}.
- Main result files are in this same results folder.
"""
    (output_dir / "llm_analysis_prompt.txt").write_text(prompt, encoding="utf-8")


def _walk_forward_fold_summary(folds: pd.DataFrame) -> dict[str, Any]:
    if folds.empty:
        return {}
    f = folds.copy()
    for col in ["train_start", "train_end", "test_start", "test_end"]:
        if col in f.columns:
            f[col] = pd.to_datetime(f[col], errors="coerce")

    out: dict[str, Any] = {
        "fold_count": int(len(f)),
        "folds": f.to_dict("records"),
    }
    if "net_total_pnl_usd" in f.columns:
        out["positive_oos_folds"] = int((f["net_total_pnl_usd"] > 0).sum())
        out["negative_oos_folds"] = int((f["net_total_pnl_usd"] <= 0).sum())
        out["best_folds"] = (
            f.sort_values("net_total_pnl_usd", ascending=False)
            .head(5)
            .to_dict("records")
        )
        out["worst_folds"] = (
            f.sort_values("net_total_pnl_usd", ascending=True)
            .head(5)
            .to_dict("records")
        )
    if "selected_scenario" in f.columns:
        scenario = (
            f.groupby("selected_scenario", dropna=False)
            .agg(
                folds_selected=("fold", "count"),
                total_oos_trades=("number_of_trades", "sum")
                if "number_of_trades" in f.columns else ("fold", "count"),
                total_oos_pnl_usd=("net_total_pnl_usd", "sum")
                if "net_total_pnl_usd" in f.columns else ("fold", "count"),
                avg_oos_sharpe=("oos_sharpe", "mean")
                if "oos_sharpe" in f.columns else ("fold", "count"),
            )
            .reset_index()
            .sort_values(["folds_selected", "total_oos_pnl_usd"], ascending=[False, False])
        )
        out["scenario_selection"] = scenario.to_dict("records")

    param_cols = [
        c for c in f.columns
        if c.startswith("selected_") and c != "selected_scenario"
    ]
    if param_cols:
        out["selected_parameter_values_by_fold"] = f[
            ["fold", "selected_scenario", *param_cols]
            if "selected_scenario" in f.columns else ["fold", *param_cols]
        ].to_dict("records")
        out["selected_parameter_frequency"] = {}
        for col in param_cols:
            out["selected_parameter_frequency"][col] = (
                f[col].value_counts(dropna=False).sort_index().to_dict()
            )
    return out


def save_llm_walk_forward_report(
    output_dir: Path,
    cfg: dict,
    metrics: dict,
    summary: dict,
    folds: pd.DataFrame,
    trades: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> None:
    """Save compact JSON plus a prompt text file for walk-forward analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    portfolio_summary = {}
    if portfolio is not None and not portfolio.empty:
        pdf = portfolio.copy()
        pdf["date"] = pd.to_datetime(pdf["date"], errors="coerce")
        equity = pd.to_numeric(pdf["equity"], errors="coerce")
        drawdown_pct = (equity - equity.cummax()) / equity.cummax() * 100
        portfolio_summary = {
            "start_date": pdf["date"].min(),
            "end_date": pdf["date"].max(),
            "starting_equity": float(equity.iloc[0]),
            "ending_equity": float(equity.iloc[-1]),
            "peak_equity": float(equity.max()),
            "min_equity": float(equity.min()),
            "stitched_max_drawdown_pct": float(drawdown_pct.min()),
            "avg_open_positions": float(pdf["n_positions"].mean())
            if "n_positions" in pdf.columns else None,
            "max_open_positions": int(pdf["n_positions"].max())
            if "n_positions" in pdf.columns else None,
        }

    trade_records = trades.to_dict("records") if trades is not None and not trades.empty else []
    report = {
        "report_type": "calendar_spread_walk_forward_overview",
        "generated_from": "python run_walk_forward.py",
        "config_snapshot": {
            "universe": cfg.get("universe", []),
            "backtest": cfg.get("backtest", {}),
            "grid_ranges": cfg.get("grid_ranges", {}),
            "expiry": cfg.get("expiry", {}),
            "features": cfg.get("features", {}),
            "signals": cfg.get("signals", {}),
            "liquidity": cfg.get("liquidity", {}),
            "entry": cfg.get("entry", {}),
            "exit": cfg.get("exit", {}),
            "costs": cfg.get("costs", {}),
            "sizing": cfg.get("sizing", {}),
        },
        "walk_forward_summary": summary,
        "stitched_oos_metrics": metrics,
        "stitched_portfolio_summary": portfolio_summary,
        "fold_analysis": _walk_forward_fold_summary(folds),
        "trade_summaries": _trade_summaries(trade_records),
        "artifact_files": {
            "folds_csv": "walk_forward_folds.csv",
            "trades_csv": "walk_forward_trades.csv",
            "portfolio_csv": "walk_forward_portfolio.csv",
            "metrics_json": "walk_forward_metrics.json",
            "summary_json": "walk_forward_summary.json",
        },
        "analysis_questions": [
            "Do the selected parameters remain stable across folds?",
            "Is OOS performance concentrated in only a few folds or broadly distributed?",
            "Which folds/tickers/exit reasons drove gains and losses?",
            "Are the current config parameters robust or fragile around the tested grid?",
            "What should be adjusted before testing harsher slippage or longer history?",
        ],
    }

    json_path = output_dir / "llm_walk_forward_overview.json"
    json_path.write_text(
        json.dumps(_clean_json(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    prompt = f"""You are analyzing a walk-forward validation of a systematic long ATM calendar spread strategy.

Use `llm_walk_forward_overview.json` as the primary source. It contains the config snapshot, tested grid ranges, stitched out-of-sample metrics, fold-by-fold performance, selected scenarios/parameters, and trade breakdowns.

Please produce:
1. Executive summary of whether the current parameters appear robust or fragile.
2. Fold-by-fold diagnosis: which windows worked, which failed, and whether failures cluster by market period.
3. Parameter stability analysis: selected z-score, percentile, profit target, stop loss, max hold, slippage, and other selected values.
4. What went well, with evidence.
5. What should be tweaked next, with concrete candidate ranges.
6. Whether it is reasonable to proceed to harsher slippage tests or longer history tests.

Important context:
- Strategy: long ATM calendar spreads targeting term-structure mean reversion.
- Walk-forward source range: {cfg.get('backtest', {}).get('start_date')} to {cfg.get('backtest', {}).get('end_date')}.
- Universe size: {len(cfg.get('universe', []))}.
- The results are out-of-sample by fold; prefer OOS evidence over in-sample training metrics.
"""
    (output_dir / "llm_walk_forward_prompt.txt").write_text(prompt, encoding="utf-8")
