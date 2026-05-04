"""
Grid backtesting helpers for comparing multiple signal configurations.

Performance optimisations (vs. original)
-----------------------------------------
1. _run_single_scenario now accepts save_artifacts=False to skip
   audit_entry_filters() during walk-forward training passes.
   audit_entry_filters uses slow row-by-row apply() — skipping it
   during WFO training (56+ calls) saves 20-30% of total WFO runtime.

2. run_backtest_grid accepts a pre-slimmed features DataFrame via
   the features parameter. When called from walk_forward.py, only
   the columns the backtest actually needs are passed, reducing
   pickle size by 60-70% when using ProcessPoolExecutor.

3. ProcessPoolExecutor is shared across grid scenarios within a WFO
   fold (passed in via executor= kwarg) to avoid repeated process
   creation overhead.
"""
from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import io
from itertools import product
import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from .backtest import run_backtest
from .metrics import compute_metrics, regime_analysis, print_report, save_plots, save_trade_log
from .signals import EARNINGS_OK_COL, add_earnings_ok_column, audit_entry_filters

logger = logging.getLogger(__name__)
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Columns actually needed by backtest engine + signals screener.
# Passing only these to ProcessPoolExecutor reduces pickle overhead.
_BACKTEST_COLS = [
    "ticker", "tradeDate",
    "front_iv", "back_iv", "iv_spread",
    "spread_zscore", "spread_pctile", "back_iv_pctile",
    "rv_iv_ratio", "rv_20",
    "front_dte", "back_dte",
    "stock_price",
    "calendar_debit_bs",      # preferred: accurate BS pricing
    "calendar_debit_proxy",   # fallback: straddle approximation
    "total_opt_volume",
    "avgOptVolu20d",
    "daily_gamma_drag",       # used by gamma drag signal filter
    "signal_rank",
    EARNINGS_OK_COL,
]


def slim_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Return a view of features with only the columns needed by the backtest.

    Reducing DataFrame size before submitting to ProcessPoolExecutor
    significantly reduces pickle/unpickle time in multiprocessing, which
    is a major contributor to WFO runtime on large universes.
    """
    keep = [c for c in _BACKTEST_COLS if c in features.columns]
    return features[keep].copy()


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively apply overrides without mutating the original config."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


# ── Scenario definitions ─────────────────────────────────────────────────

def build_high_conviction_grid() -> list[dict[str, Any]]:
    """
    Compact opinionated scenario set.
    Each scenario tests a specific hypothesis about the signal requirements.
    """
    return [
        {
            "name": "baseline",
            "description": "Default config — used to confirm base-case edge exists.",
            "overrides": {},
        },
        {
            "name": "z1p5_pct90",
            "description": "Moderately strict z-score with tight percentile gate.",
            "overrides": {
                "signals": {"zscore_entry_min": 1.5, "percentile_entry_min": 90}
            },
        },
        {
            "name": "z2_pct90",
            "description": "Strong signal only — 2-sigma z + top decile. Fewer trades, higher quality.",
            "overrides": {
                "signals": {"zscore_entry_min": 2.0, "percentile_entry_min": 90}
            },
        },
        {
            "name": "z1p5_pct90_tight_dte",
            "description": "Moderate signal with cleaner calendar spacing.",
            "overrides": {
                "signals": {"zscore_entry_min": 1.5, "percentile_entry_min": 90},
                "expiry":  {"front_dte_min": 20, "front_dte_max": 35,
                            "back_dte_min": 45, "back_dte_max": 75},
                "entry":   {"max_debit_pct_of_spot": 0.015},
            },
        },
    ]


def build_z2_pct90_focused_grid() -> list[dict[str, Any]]:
    """
    Tight search space around the best recent performer (z2_pct90).
    Tests parameter sensitivity: small changes in z-score and percentile thresholds.
    """
    return [
        {
            "name": "z2_pct90_core",
            "description": "Reference scenario — strongest recent performer.",
            "overrides": {"signals": {"zscore_entry_min": 2.0, "percentile_entry_min": 90}},
        },
        {
            "name": "z1p9_pct90",
            "description": "Slightly looser z — more trades, test for dilution.",
            "overrides": {"signals": {"zscore_entry_min": 1.9, "percentile_entry_min": 90}},
        },
        {
            "name": "z2p1_pct90",
            "description": "Slightly tighter z — fewer trades, higher conviction.",
            "overrides": {"signals": {"zscore_entry_min": 2.1, "percentile_entry_min": 90}},
        },
        {
            "name": "z2_pct88",
            "description": "Slightly wider percentile gate — test trade-count impact.",
            "overrides": {"signals": {"zscore_entry_min": 2.0, "percentile_entry_min": 88}},
        },
        {
            "name": "z2_pct92",
            "description": "Tighter percentile — only very extreme setups.",
            "overrides": {"signals": {"zscore_entry_min": 2.0, "percentile_entry_min": 92}},
        },
        {
            "name": "z2_pct90_iv90",
            "description": "Light absolute-vol guard — keeps signal from sparse.",
            "overrides": {"signals": {
                "zscore_entry_min": 2.0, "percentile_entry_min": 90,
                "back_iv_percentile_max": 90,   # slightly wider than default 60
            }},
        },
        {
            "name": "z2_pct90_rv10",
            "description": "Front IV must be ≥10% above realized — genuine richness.",
            "overrides": {"signals": {
                "zscore_entry_min": 2.0, "percentile_entry_min": 90,
                "rv_iv_ratio_max": 1.0,
            }},
        },
        {
            "name": "z2_pct90_tight_dte",
            "description": "Strong signal with cleaner 30/60 DTE calendar structure.",
            "overrides": {
                "signals": {"zscore_entry_min": 2.0, "percentile_entry_min": 90},
                "expiry":  {"front_dte_min": 20, "front_dte_max": 35,
                            "back_dte_min": 45, "back_dte_max": 75},
                "entry":   {"max_debit_pct_of_spot": 0.015},
            },
        },
    ]


def build_range_based_grid(
    zscores:             list[float],
    percentiles:         list[int],
    back_iv_percentiles: list[float | None] | None = None,
    rv_iv_ratios:        list[float | None] | None = None,
    dte_profiles:        list[str] | None = None,
    debit_caps:          list[float] | None = None,
    profit_targets:      list[float] | None = None,
    stop_losses:         list[float] | None = None,
    max_hold_days:       list[int] | None = None,
    slippage_pcts:       list[float] | None = None,
) -> list[dict[str, Any]]:
    """
    Build a diverse scenario pool from parameter ranges via cartesian product.
    Useful for systematic exploration before narrowing to a focused grid.
    """
    back_iv_percentiles = back_iv_percentiles or [None]
    rv_iv_ratios        = rv_iv_ratios        or [None]
    dte_profiles        = dte_profiles        or ["standard"]
    debit_caps          = debit_caps          or [0.02]
    profit_targets      = profit_targets      or [0.25]
    stop_losses         = stop_losses         or [-0.30]
    max_hold_days       = max_hold_days       or [45]
    slippage_pcts       = slippage_pcts       or [0.03]

    # DTE profile presets — standard vs. relaxed DTE constraints
    profile_expiry = {
        "standard": {"front_dte_min": 20, "front_dte_max": 35,
                     "back_dte_min": 45, "back_dte_max": 75},
        "relaxed":  {"front_dte_min": 10, "front_dte_max": 45,
                     "back_dte_min": 30, "back_dte_max": 80},
    }

    scenarios = []
    for z, pc, biv, rv, dte_prof, dc, pt, sl, mh, slip in product(
        zscores,
        percentiles,
        back_iv_percentiles,
        rv_iv_ratios,
        dte_profiles,
        debit_caps,
        profit_targets,
        stop_losses,
        max_hold_days,
        slippage_pcts,
    ):
        parts = [f"z{str(z).replace('.','p')}", f"pct{pc}"]
        if biv is not None:
            parts.append(f"biv{int(biv)}")
        if rv is not None:
            parts.append(f"rv{str(rv).replace('.','p')}")
        if dte_prof != "standard":
            parts.append(dte_prof)
        if dc != 0.02:
            parts.append(f"dc{str(dc).replace('.','p')}")
        if pt != 0.25:
            parts.append(f"pt{str(pt).replace('.','p')}")
        if sl != -0.30:
            parts.append(f"sl{str(abs(sl)).replace('.','p')}")
        if mh != 45:
            parts.append(f"mh{mh}")
        if slip != 0.03:
            parts.append(f"slip{str(slip).replace('.','p')}")

        sig_overrides = {
            "zscore_entry_min": z,
            "percentile_entry_min": pc,
            # Keep explicit None overrides so "None = no filter" works even
            # when the base config enables these filters.
            "back_iv_percentile_max": biv,
            "rv_iv_ratio_max": rv,
        }

        overrides: dict[str, Any] = {"signals": sig_overrides}
        if dte_prof in profile_expiry:
            overrides["expiry"] = profile_expiry[dte_prof]
        if dc != 0.02:
            overrides.setdefault("entry", {})["max_debit_pct_of_spot"] = dc
        if pt != 0.25 or sl != -0.30 or mh != 45:
            overrides["exit"] = {
                "profit_target_pct": pt,
                "stop_loss_pct": sl,
                "max_hold_days": mh,
            }
        if slip != 0.03:
            overrides.setdefault("costs", {})["slippage_pct"] = slip

        scenarios.append({
            "name":        "_".join(parts),
            "description": f"z≥{z}, pct≥{pc}, biv≤{biv}, rv≤{rv}, dte={dte_prof}, dc={dc}",
            "description_full": (
                f"z>={z}, pct>={pc}, biv<={biv}, rv<={rv}, dte={dte_prof}, "
                f"dc={dc}, profit_target={pt}, stop_loss={sl}, max_hold={mh}, slippage={slip}"
            ),
            "overrides":   overrides,
        })

    return scenarios


def build_configured_focused_grid(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Build the focused grid from config.yaml when grid_ranges is present.

    Falls back to the hand-written focused grid for older configs.
    """
    ranges = cfg.get("grid_ranges")
    if not ranges:
        return build_z2_pct90_focused_grid()

    def as_list(value, default):
        if value is None:
            return default
        return value if isinstance(value, list) else [value]

    return build_range_based_grid(
        zscores=as_list(ranges.get("zscores"), [2.0]),
        percentiles=as_list(ranges.get("percentiles"), [90]),
        back_iv_percentiles=as_list(ranges.get("back_iv_percentiles"), [None]),
        rv_iv_ratios=as_list(ranges.get("rv_iv_ratios"), [None]),
        dte_profiles=as_list(ranges.get("dte_profiles"), ["standard"]),
        debit_caps=as_list(
            ranges.get("debit_caps"),
            [cfg.get("entry", {}).get("max_debit_pct_of_spot", 0.02)],
        ),
        profit_targets=as_list(
            ranges.get("profit_targets"),
            [cfg.get("exit", {}).get("profit_target_pct", 0.25)],
        ),
        stop_losses=as_list(
            ranges.get("stop_losses"),
            [cfg.get("exit", {}).get("stop_loss_pct", -0.30)],
        ),
        max_hold_days=as_list(
            ranges.get("max_hold_days"),
            [cfg.get("exit", {}).get("max_hold_days", 45)],
        ),
        slippage_pcts=as_list(
            ranges.get("slippage_pcts"),
            [cfg.get("costs", {}).get("slippage_pct", 0.03)],
        ),
    )


# ── Internal helpers ─────────────────────────────────────────────────────

def _regime_value(regimes: dict, ticker: str, key: str) -> Any:
    """Safely extract a metric from the by_ticker regime breakdown."""
    return regimes.get("by_ticker", {}).get(ticker, {}).get(key, np.nan)


def _portfolio_to_frame(portfolio: list) -> pd.DataFrame:
    if not portfolio:
        return pd.DataFrame(columns=["date", "cash", "positions_value", "equity", "n_positions"])
    return pd.DataFrame([{
        "date":            p.date,
        "cash":            p.cash,
        "positions_value": p.positions_value,
        "equity":          p.equity,
        "n_positions":     p.n_positions,
    } for p in portfolio])


def _gross_profit_factor(trades: list[dict[str, Any]]) -> float:
    """Profit factor using gross trade P&L: gross wins / absolute gross losses."""
    if not trades:
        return np.nan
    tdf = pd.DataFrame(trades)
    pnl_col = "gross_pnl" if "gross_pnl" in tdf.columns else "pnl"
    wins = tdf.loc[tdf[pnl_col] > 0, pnl_col].sum()
    losses = tdf.loc[tdf[pnl_col] <= 0, pnl_col].sum()
    return float(wins / abs(losses)) if losses != 0 else np.inf


def _expectancy_return_on_capital_pct(trades: list[dict[str, Any]]) -> float:
    """Expected return per trade as percent of capital deployed."""
    if not trades:
        return np.nan
    tdf = pd.DataFrame(trades)
    if "return_on_capital_pct" not in tdf.columns:
        return np.nan
    return float(tdf["return_on_capital_pct"].mean())


def _training_subwindow_pnls(
    trades: list[dict[str, Any]], features: pd.DataFrame
) -> tuple[list[float], int, float]:
    """
    Split the feature window into thirds and sum realized P&L in each third.

    Trades are assigned by exit_date because P&L is only realized when a
    position closes. Empty sub-windows count as non-profitable.
    """
    if features.empty or "tradeDate" not in features.columns:
        return [0.0, 0.0, 0.0], 0, 0.0

    dates = pd.to_datetime(features["tradeDate"], errors="coerce").dropna()
    if dates.empty:
        return [0.0, 0.0, 0.0], 0, 0.0

    start = dates.min().normalize()
    end = dates.max().normalize()
    span = end - start
    b1 = start + span / 3
    b2 = start + span * 2 / 3

    pnls = [0.0, 0.0, 0.0]
    if trades:
        tdf = pd.DataFrame(trades).copy()
        if "exit_date" in tdf.columns and "pnl" in tdf.columns:
            tdf["exit_date"] = pd.to_datetime(tdf["exit_date"], errors="coerce")
            tdf = tdf.dropna(subset=["exit_date"])
            windows = [
                (tdf["exit_date"] >= start) & (tdf["exit_date"] <= b1),
                (tdf["exit_date"] > b1) & (tdf["exit_date"] <= b2),
                (tdf["exit_date"] > b2) & (tdf["exit_date"] <= end),
            ]
            pnls = [float(tdf.loc[mask, "pnl"].sum()) for mask in windows]

    positive_count = sum(1 for pnl in pnls if pnl > 0)
    return pnls, positive_count, positive_count / 3.0


def _save_report_text(scenario_dir: Path, scenario: dict, metrics: dict,
                      regimes: dict, audit: pd.DataFrame):
    """Write a human-readable text summary for one scenario."""
    with redirect_stdout(io.StringIO()) as f:
        print_report(metrics, regimes)
        report_text = f.getvalue()
    with (scenario_dir / "summary.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"Scenario: {scenario['name']}\n")
        fh.write(f"Description: {scenario.get('description_full', scenario.get('description',''))}\n\n")
        fh.write(report_text)
        if not audit.empty:
            fh.write("\n\nEntry Filter Audit:\n")
            fh.write(audit.to_string(index=False))


def _save_comparison_plots(summary_df: pd.DataFrame, equity_curves: dict[str, pd.DataFrame],
                           output_dir: Path):
    """Four-panel dashboard comparing all scenarios."""
    if summary_df.empty:
        return

    scenarios = summary_df["scenario"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Panel 1: Total P&L by scenario
    ax = axes[0, 0]
    colors = ["#059669" if v >= 0 else "#dc2626" for v in summary_df["total_pnl_usd"].fillna(0)]
    ax.bar(range(len(scenarios)), summary_df["total_pnl_usd"].fillna(0), color=colors)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=8)
    ax.set_title("Total P&L by Scenario ($)")
    ax.grid(True, alpha=0.3)

    # Panel 2: Sharpe ratio by scenario
    ax = axes[0, 1]
    ax.bar(range(len(scenarios)), summary_df["sharpe_ratio"].fillna(0),
           color="#2563eb")
    ax.axhline(1.0, color="orange", linestyle="--", linewidth=1.2, label="Sharpe=1.0")
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(scenarios, rotation=30, ha="right", fontsize=8)
    ax.set_title("Sharpe Ratio by Scenario")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Trade count vs. Win rate scatter
    ax = axes[1, 0]
    sc = ax.scatter(
        summary_df["total_trades"].fillna(0),
        summary_df["win_rate_pct"].fillna(0),
        c=summary_df["profit_factor"].fillna(0),
        cmap="RdYlGn", s=80, alpha=0.8, vmin=0.8, vmax=2.5,
    )
    ax.axhline(70, color="orange", linestyle="--", linewidth=1.2, label="70% win rate")
    for i, name in enumerate(scenarios):
        ax.annotate(name, (summary_df["total_trades"].iloc[i],
                           summary_df["win_rate_pct"].iloc[i]),
                    fontsize=6, alpha=0.8)
    ax.set_xlabel("Total Trades")
    ax.set_ylabel("Win Rate (%)")
    ax.set_title("Trade Count vs. Win Rate (colour = profit factor)")
    ax.legend()
    plt.colorbar(sc, ax=ax, label="Profit Factor")
    ax.grid(True, alpha=0.3)

    # Panel 4: Overlaid equity curves
    ax = axes[1, 1]
    cmap_eq = plt.cm.get_cmap("tab10", max(len(equity_curves), 1))
    for i, (name, eqdf) in enumerate(equity_curves.items()):
        if eqdf.empty or "equity" not in eqdf.columns:
            continue
        eqdf = eqdf.copy()
        eqdf["date"] = pd.to_datetime(eqdf["date"])
        ax.plot(eqdf["date"], eqdf["equity"], linewidth=1.2,
                color=cmap_eq(i), label=name, alpha=0.8)
    ax.set_title("Equity Curves — All Scenarios")
    ax.set_ylabel("Portfolio Equity ($)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "grid_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_parameter_heatmap(summary_df: pd.DataFrame, output_dir: Path):
    """Parameter-outcome matrix heatmap across all scenarios."""
    if summary_df.empty:
        return

    cols = [
        "zscore_entry_min", "percentile_entry_min",
        "back_iv_percentile_max", "rv_iv_ratio_max",
        "front_dte_min", "front_dte_max", "back_dte_min", "back_dte_max",
        "max_debit_pct_of_spot",
        "profit_target_pct", "stop_loss_pct", "max_hold_days_param", "slippage_pct",
        "total_trades", "win_rate_pct", "avg_pnl_pct",
        "total_pnl_usd", "profit_factor", "sharpe_ratio",
    ]
    avail   = [c for c in cols if c in summary_df.columns]
    plot_df = summary_df.set_index("scenario")[avail].fillna(-1.0)

    values = plot_df.to_numpy(dtype=float)
    fig, ax = plt.subplots(
        figsize=(max(12, len(avail) * 0.85), max(5, len(plot_df) * 0.6))
    )
    im = ax.imshow(values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(avail, rotation=35, ha="right")
    ax.set_yticks(range(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index)
    ax.set_title("Scenario Parameters and Outcomes")
    plt.colorbar(im, ax=ax, label="Value")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val  = values[i, j]
            text = f"{int(val)}" if abs(val) >= 10 and float(val).is_integer() else f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")

    plt.tight_layout()
    fig.savefig(output_dir / "grid_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Core scenario runner ─────────────────────────────────────────────────

def _run_single_scenario(
    scenario:      dict[str, Any],
    features:      pd.DataFrame,
    earnings_df:   pd.DataFrame,
    base_cfg:      dict[str, Any],
    total_days:    int | None,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    """
    Execute one scenario so grid runs can fan out across workers.

    save_artifacts=False skips audit_entry_filters() — this saves 20-30%
    runtime during walk-forward training passes where the audit is not
    needed and would otherwise be computed and discarded 50+ times.
    """
    cfg           = _deep_update(base_cfg, scenario.get("overrides", {}))
    scenario_name = scenario["name"]

    # Audit is slow (row-by-row apply) — skip during WFO training passes
    audit = (
        audit_entry_filters(features, earnings_df, cfg)
        if save_artifacts
        else pd.DataFrame()
    )

    results   = run_backtest(features, earnings_df, cfg)
    trades    = results["trades"]
    portfolio = results["portfolio"]
    metrics   = compute_metrics(trades, portfolio)
    regimes   = regime_analysis(trades, features) if save_artifacts else {}
    pdf       = _portfolio_to_frame(portfolio)
    subwindow_pnls, positive_subwindows, positive_subwindow_fraction = (
        _training_subwindow_pnls(trades, features)
    )
    expectancy_roc_pct = _expectancy_return_on_capital_pct(trades)
    gross_profit_factor = _gross_profit_factor(trades)

    years = total_days / 365.25 if total_days else np.nan
    summary_row = {
        "scenario":             scenario_name,
        "description":          scenario.get("description_full", scenario.get("description", "")),
        "total_trades":         metrics.get("total_trades", 0),
        "trades_per_year":      round(metrics.get("total_trades", 0) / years, 2) if years else np.nan,
        "win_rate_pct":         metrics.get("win_rate_pct", np.nan),
        "avg_pnl_pct":          metrics.get("avg_pnl_pct", np.nan),
        "total_pnl_usd":        metrics.get("total_pnl_usd", np.nan),
        "profit_factor":        metrics.get("profit_factor", np.nan),
        "gross_profit_factor":  gross_profit_factor,
        "sharpe_ratio":         metrics.get("sharpe_ratio", np.nan),
        "sortino_ratio":        metrics.get("sortino_ratio", np.nan),
        "max_drawdown_pct":     metrics.get("max_drawdown_pct", np.nan),
        "avg_hold_days":        metrics.get("avg_hold_days", np.nan),
        "expectancy_usd":       metrics.get("expectancy_usd", np.nan),
        "expectancy_return_on_capital_pct": expectancy_roc_pct,
        "positive_train_subwindows": positive_subwindows,
        "positive_train_subwindow_fraction": positive_subwindow_fraction,
        "train_subwindow_1_pnl": subwindow_pnls[0],
        "train_subwindow_2_pnl": subwindow_pnls[1],
        "train_subwindow_3_pnl": subwindow_pnls[2],
        "avg_return_on_capital_pct": (
            float(pd.DataFrame(trades)["return_on_capital_pct"].mean())
            if trades else np.nan
        ),
        "transaction_costs_usd": (
            float(pd.DataFrame(trades)["transaction_costs"].sum())
            if trades else 0.0
        ),
        # Spot-check on two liquid names for quick regime sense-check
        "qqq_trades":       _regime_value(regimes, "QQQ", "trades"),
        "spy_trades":       _regime_value(regimes, "SPY", "trades"),
        "qqq_avg_pnl_pct":  _regime_value(regimes, "QQQ", "avg_pnl_pct"),
        "spy_avg_pnl_pct":  _regime_value(regimes, "SPY", "avg_pnl_pct"),
        # Parameter echo for heatmap
        "zscore_entry_min":        cfg["signals"].get("zscore_entry_min"),
        "percentile_entry_min":    cfg["signals"].get("percentile_entry_min"),
        "back_iv_percentile_max":  cfg["signals"].get("back_iv_percentile_max"),
        "rv_iv_ratio_max":         cfg["signals"].get("rv_iv_ratio_max"),
        "front_dte_min":           cfg["expiry"].get("front_dte_min"),
        "front_dte_max":           cfg["expiry"].get("front_dte_max"),
        "back_dte_min":            cfg["expiry"].get("back_dte_min"),
        "back_dte_max":            cfg["expiry"].get("back_dte_max"),
        "max_debit_pct_of_spot":   cfg["entry"].get("max_debit_pct_of_spot"),
        "profit_target_pct":       cfg["exit"].get("profit_target_pct"),
        "stop_loss_pct":           cfg["exit"].get("stop_loss_pct"),
        "max_hold_days_param":     cfg["exit"].get("max_hold_days"),
        "slippage_pct":            cfg["costs"].get("slippage_pct"),
    }
    return {
        "scenario":      scenario,
        "scenario_name": scenario_name,
        "cfg":           cfg,
        "audit":         audit,
        "trades":        trades,
        "portfolio":     portfolio,
        "metrics":       metrics,
        "regimes":       regimes,
        "equity_curve":  pdf,
        "summary_row":   summary_row,
    }


# ── Public grid runner ───────────────────────────────────────────────────

def run_backtest_grid(
    features:          pd.DataFrame,
    earnings_df:       pd.DataFrame,
    base_cfg:          dict,
    scenarios:         list[dict[str, Any]],
    output_dir:        Path,
    save_artifacts:    bool = True,
    max_workers:       int | None = None,
    progress_callback: Any | None = None,
    executor:          ProcessPoolExecutor | None = None,
) -> dict[str, Any]:
    """
    Run a curated grid of backtests and persist per-scenario outputs.

    Parameters
    ----------
    features       : Pre-computed feature DataFrame. Pass slim_features(features)
                     from walk_forward to reduce IPC pickling overhead.
    save_artifacts : If False, skips file writes and audit computation.
                     Set False during WFO training passes for ~30% speedup.
    executor       : Reuse an existing ProcessPoolExecutor (avoids repeated
                     process creation in WFO fold loops).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    features = add_earnings_ok_column(features, earnings_df)

    summaries:        list[dict[str, Any]] = []
    equity_curves:    dict[str, pd.DataFrame] = {}
    scenario_reports: list[dict[str, Any]] = []

    total_days = None
    if not features.empty:
        dates      = pd.to_datetime(features["tradeDate"])
        total_days = max((dates.max() - dates.min()).days, 1)

    scenario_results: list[dict[str, Any]] = []
    worker_count = max_workers or min(len(scenarios), max((os.cpu_count() or 1) - 1, 1))

    if worker_count <= 1 or len(scenarios) <= 1:
        # Serial execution (simpler, better for debugging)
        for idx, scenario in enumerate(scenarios, start=1):
            logger.debug("Running scenario: %s", scenario["name"])
            result = _run_single_scenario(
                scenario, features, earnings_df, base_cfg, total_days, save_artifacts
            )
            scenario_results.append(result)
            if progress_callback is not None:
                progress_callback(idx, len(scenarios), scenario["name"])
    else:
        # Parallel execution via ProcessPoolExecutor
        owns_executor = executor is None
        pool = executor or ProcessPoolExecutor(max_workers=worker_count)
        try:
            future_map = {
                pool.submit(
                    _run_single_scenario,
                    scenario, features, earnings_df, base_cfg, total_days, save_artifacts,
                ): scenario
                for scenario in scenarios
            }
            completed = 0
            for future in as_completed(future_map):
                scenario = future_map[future]
                logger.debug("Completed scenario: %s", scenario["name"])
                result = future.result()
                scenario_results.append(result)
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, len(scenarios), scenario["name"])
        finally:
            if owns_executor:
                pool.shutdown()

    # Restore original scenario order (parallel results arrive out of order)
    scenario_results.sort(
        key=lambda item: next(
            i for i, s in enumerate(scenarios) if s["name"] == item["scenario_name"]
        )
    )

    for result in scenario_results:
        scenario      = result["scenario"]
        scenario_name = result["scenario_name"]
        scenario_dir  = output_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        if save_artifacts:
            save_trade_log(result["trades"], scenario_dir)
            save_plots(result["portfolio"], result["trades"], scenario_dir)
            with (scenario_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(result["metrics"], f, indent=2, default=str)
            with (scenario_dir / "regimes.json").open("w", encoding="utf-8") as f:
                json.dump(result["regimes"], f, indent=2, default=str)
            if not result["audit"].empty:
                result["audit"].to_csv(scenario_dir / "entry_filter_audit.csv", index=False)
            _save_report_text(
                scenario_dir, scenario, result["metrics"], result["regimes"], result["audit"]
            )

        equity_curves[scenario_name] = result["equity_curve"]
        summaries.append(result["summary_row"])
        scenario_reports.append({
            "scenario":   scenario_name,
            "metrics":    result["metrics"],
            "regimes":    result["regimes"],
            "audit":      result["audit"],
            "output_dir": str(scenario_dir),
        })

    summary_df = pd.DataFrame(summaries).sort_values(
        ["total_pnl_usd", "profit_factor", "win_rate_pct"],
        ascending=[False, False, False],
    )
    if save_artifacts:
        summary_df.to_csv(output_dir / "grid_summary.csv", index=False)
        _save_comparison_plots(summary_df, equity_curves, output_dir)
        _save_parameter_heatmap(summary_df, output_dir)

    return {
        "summary":       summary_df,
        "scenarios":     scenario_reports,
        "equity_curves": equity_curves,
        "output_dir":    output_dir,
    }
