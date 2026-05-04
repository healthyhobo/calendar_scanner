"""
Performance metrics and reporting for backtest results.
"""
import logging

import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib

logger = logging.getLogger(__name__)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pathlib import Path


def compute_metrics(trades: list[dict], portfolio: list) -> dict:
    """
    Compute summary performance metrics from backtest results.

    Returns dict of metric_name → value.
    """
    if not trades:
        return {"total_trades": 0, "error": "No trades generated"}

    tdf = pd.DataFrame(trades)

    total      = len(tdf)
    winners    = tdf[tdf["pnl"] > 0]
    losers     = tdf[tdf["pnl"] <= 0]
    win_rate   = len(winners) / total * 100

    avg_pnl       = tdf["pnl"].mean()
    avg_pnl_pct   = tdf["pnl_pct"].mean() * 100
    median_pnl    = tdf["pnl"].median()
    total_pnl     = tdf["pnl"].sum()

    avg_win       = winners["pnl"].mean() if len(winners) > 0 else 0
    avg_loss      = losers["pnl"].mean()  if len(losers)  > 0 else 0
    profit_factor = (winners["pnl"].sum() / abs(losers["pnl"].sum())
                     if len(losers) > 0 and losers["pnl"].sum() != 0 else np.inf)

    expectancy    = win_rate / 100 * avg_win + (1 - win_rate / 100) * avg_loss

    avg_hold      = tdf["hold_days"].mean()
    median_hold   = tdf["hold_days"].median()
    max_hold      = tdf["hold_days"].max()

    best_trade    = tdf["pnl"].max()
    worst_trade   = tdf["pnl"].min()

    # Equity curve metrics
    if portfolio:
        pdf = pd.DataFrame([{
            "date": p.date, "equity": p.equity
        } for p in portfolio])
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf.sort_values("date", inplace=True)

        peak = pdf["equity"].expanding().max()
        dd   = (pdf["equity"] - peak) / peak
        max_dd = dd.min() * 100

        # Annualized return
        days = (pdf["date"].iloc[-1] - pdf["date"].iloc[0]).days
        total_return = pdf["equity"].iloc[-1] / pdf["equity"].iloc[0] - 1
        ann_return = (1 + total_return) ** (365.25 / max(days, 1)) - 1

        # Sharpe (daily returns)
        daily_ret = pdf["equity"].pct_change().dropna()
        sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 0

        # Sortino
        downside = daily_ret[daily_ret < 0]
        sortino = (daily_ret.mean() / downside.std() * np.sqrt(252)
                   if len(downside) > 0 and downside.std() > 0 else 0)

        # Calmar
        calmar = ann_return / abs(max_dd / 100) if max_dd != 0 else 0
    else:
        max_dd = ann_return = sharpe = sortino = calmar = 0

    # Exit reason breakdown
    exit_counts = tdf["exit_reason"].value_counts().to_dict()

    return {
        "total_trades":        total,
        "win_rate_pct":        round(win_rate, 1),
        "avg_pnl_usd":        round(avg_pnl, 2),
        "avg_pnl_pct":        round(avg_pnl_pct, 2),
        "median_pnl_usd":     round(median_pnl, 2),
        "total_pnl_usd":      round(total_pnl, 2),
        "avg_win_usd":        round(avg_win, 2),
        "avg_loss_usd":       round(avg_loss, 2),
        "profit_factor":      round(profit_factor, 2),
        "expectancy_usd":     round(expectancy, 2),
        "best_trade_usd":     round(best_trade, 2),
        "worst_trade_usd":    round(worst_trade, 2),
        "avg_hold_days":      round(avg_hold, 1),
        "median_hold_days":   round(median_hold, 1),
        "max_hold_days":      int(max_hold),
        "annualized_return":  round(ann_return * 100, 2),
        "max_drawdown_pct":   round(max_dd, 2),
        "sharpe_ratio":       round(sharpe, 2),
        "sortino_ratio":      round(sortino, 2),
        "calmar_ratio":       round(calmar, 2),
        "exit_reasons":       exit_counts,
    }


def regime_analysis(trades: list[dict], features: pd.DataFrame) -> dict:
    """
    Break down performance by regime buckets.
    Returns dict of regime_name → sub-metrics.
    """
    if not trades:
        return {}

    tdf = pd.DataFrame(trades)
    results = {}

    # ── By entry z-score bucket ──
    bins = [0, 1.5, 2.0, 2.5, 3.0, 4.0]
    labels = ["<1.5", "1.5-2.0", "2.0-2.5", "2.5-3.0", "3.0+"]
    tdf["zscore_bucket"] = pd.cut(tdf["entry_zscore"], bins=bins, labels=labels)
    zscore_regime = {}
    for bucket, grp in tdf.groupby("zscore_bucket", observed=True):
        if len(grp) == 0:
            continue
        zscore_regime[str(bucket)] = {
            "trades": len(grp),
            "win_rate": round(len(grp[grp["pnl"] > 0]) / len(grp) * 100, 1),
            "avg_pnl_pct": round(grp["pnl_pct"].mean() * 100, 2),
        }
    results["by_zscore_bucket"] = zscore_regime

    # ── By ticker ──
    ticker_regime = {}
    for tkr, grp in tdf.groupby("ticker"):
        if len(grp) == 0:
            continue
        ticker_regime[tkr] = {
            "trades": len(grp),
            "win_rate": round(len(grp[grp["pnl"] > 0]) / len(grp) * 100, 1),
            "avg_pnl_pct": round(grp["pnl_pct"].mean() * 100, 2),
            "total_pnl": round(grp["pnl"].sum(), 2),
        }
    results["by_ticker"] = ticker_regime

    # ── By exit reason ──
    exit_regime = {}
    for reason, grp in tdf.groupby("exit_reason"):
        exit_regime[reason] = {
            "trades": len(grp),
            "win_rate": round(len(grp[grp["pnl"] > 0]) / len(grp) * 100, 1),
            "avg_pnl_pct": round(grp["pnl_pct"].mean() * 100, 2),
        }
    results["by_exit_reason"] = exit_regime

    # ── By year ──
    tdf["year"] = pd.to_datetime(tdf["entry_date"]).dt.year
    year_regime = {}
    for yr, grp in tdf.groupby("year"):
        year_regime[int(yr)] = {
            "trades": len(grp),
            "win_rate": round(len(grp[grp["pnl"] > 0]) / len(grp) * 100, 1),
            "avg_pnl_pct": round(grp["pnl_pct"].mean() * 100, 2),
            "total_pnl": round(grp["pnl"].sum(), 2),
        }
    results["by_year"] = year_regime

    # ── By hold period bucket ──
    hold_bins = [0, 7, 14, 21, 30, 100]
    hold_labels = ["1-7d", "8-14d", "15-21d", "22-30d", "30d+"]
    tdf["hold_bucket"] = pd.cut(tdf["hold_days"], bins=hold_bins, labels=hold_labels)
    hold_regime = {}
    for bucket, grp in tdf.groupby("hold_bucket", observed=True):
        if len(grp) == 0:
            continue
        hold_regime[str(bucket)] = {
            "trades": len(grp),
            "win_rate": round(len(grp[grp["pnl"] > 0]) / len(grp) * 100, 1),
            "avg_pnl_pct": round(grp["pnl_pct"].mean() * 100, 2),
        }
    results["by_hold_period"] = hold_regime

    return results


# ── Reporting ───────────────────────────────────────────────────

def print_report(metrics: dict, regimes: dict):
    """Print a formatted summary to stdout."""
    print("\n" + "=" * 64)
    print("  CALENDAR SPREAD BACKTEST - SUMMARY")
    print("=" * 64)

    # Core metrics
    rows = [
        ("Total trades",       metrics.get("total_trades")),
        ("Win rate",           f"{metrics.get('win_rate_pct', 0)}%"),
        ("Avg P&L (USD)",      f"${metrics.get('avg_pnl_usd', 0):,.2f}"),
        ("Avg P&L (%)",        f"{metrics.get('avg_pnl_pct', 0):.2f}%"),
        ("Median P&L (USD)",   f"${metrics.get('median_pnl_usd', 0):,.2f}"),
        ("Total P&L (USD)",    f"${metrics.get('total_pnl_usd', 0):,.2f}"),
        ("Profit factor",      metrics.get("profit_factor", 0)),
        ("Expectancy (USD)",   f"${metrics.get('expectancy_usd', 0):,.2f}"),
        ("Best trade",         f"${metrics.get('best_trade_usd', 0):,.2f}"),
        ("Worst trade",        f"${metrics.get('worst_trade_usd', 0):,.2f}"),
        ("Avg hold (days)",    metrics.get("avg_hold_days", 0)),
        ("Max hold (days)",    metrics.get("max_hold_days", 0)),
        ("Ann. return",        f"{metrics.get('annualized_return', 0):.2f}%"),
        ("Max drawdown",       f"{metrics.get('max_drawdown_pct', 0):.2f}%"),
        ("Sharpe ratio",       metrics.get("sharpe_ratio", 0)),
        ("Sortino ratio",      metrics.get("sortino_ratio", 0)),
        ("Calmar ratio",       metrics.get("calmar_ratio", 0)),
    ]
    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))

    # Exit reasons
    exit_reasons = metrics.get("exit_reasons", {})
    if exit_reasons:
        print("\n-- Exit Reasons ------------------------------------")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:20s}  {count:4d}")

    # Regime breakdowns
    for regime_name, regime_data in regimes.items():
        print(f"\n-- {regime_name} ----------------------------------")
        if isinstance(regime_data, dict):
            rows = []
            for bucket, stats in regime_data.items():
                rows.append([
                    bucket,
                    stats.get("trades", 0),
                    f"{stats.get('win_rate', 0):.1f}%",
                    f"{stats.get('avg_pnl_pct', 0):.2f}%",
                    f"${stats.get('total_pnl', 0):,.0f}" if "total_pnl" in stats else "",
                ])
            headers = ["Bucket", "Trades", "Win Rate", "Avg P&L%", "Total P&L"]
            print(tabulate(rows, headers=headers, tablefmt="simple"))

    print("\n" + "=" * 64 + "\n")


def save_plots(portfolio: list, trades: list[dict], output_dir: Path):
    """Generate and save backtest charts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not portfolio:
        return

    pdf = pd.DataFrame([{
        "date": p.date, "equity": p.equity, "n_positions": p.n_positions
    } for p in portfolio])
    pdf["date"] = pd.to_datetime(pdf["date"])

    # ── Equity curve ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})

    axes[0].plot(pdf["date"], pdf["equity"], color="#1a73e8", linewidth=1.2)
    axes[0].set_ylabel("Portfolio equity ($)")
    axes[0].set_title("Calendar Spread Backtest — Equity Curve")
    axes[0].grid(True, alpha=0.3)

    # Drawdown
    peak = pdf["equity"].expanding().max()
    dd = (pdf["equity"] - peak) / peak * 100
    axes[1].fill_between(pdf["date"], dd, 0, color="#e8453c", alpha=0.5)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(True, alpha=0.3)

    # Number of open positions
    axes[2].fill_between(pdf["date"], pdf["n_positions"], 0, color="#34a853", alpha=0.4)
    axes[2].set_ylabel("Open positions")
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_formatter(DateFormatter("%Y-%m"))

    plt.tight_layout()
    fig.savefig(output_dir / "equity_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if not trades:
        return

    tdf = pd.DataFrame(trades)

    # ── P&L distribution ──
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#34a853" if x > 0 else "#e8453c" for x in tdf["pnl"]]
    ax.hist(tdf["pnl_pct"] * 100, bins=40, color="#1a73e8", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Trade P&L (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("P&L Distribution — Calendar Spreads")
    ax.grid(True, alpha=0.3)
    fig.savefig(output_dir / "pnl_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Monthly returns heatmap ──
    tdf["exit_month"] = pd.to_datetime(tdf["exit_date"])
    tdf["year"]  = tdf["exit_month"].dt.year
    tdf["month"] = tdf["exit_month"].dt.month
    monthly = tdf.groupby(["year", "month"])["pnl"].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(monthly.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_yticks(range(len(monthly.index)))
    ax.set_yticklabels(monthly.index)
    ax.set_title("Monthly P&L Heatmap ($)")
    plt.colorbar(im, ax=ax, label="P&L ($)")
    fig.savefig(output_dir / "monthly_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Charts saved to %s", output_dir)


def save_trade_log(trades: list[dict], output_dir: Path):
    """Save detailed trade log as CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if trades:
        pd.DataFrame(trades).to_csv(output_dir / "trade_log.csv", index=False)
