# Calendar Spread Scanner — MVP

A systematic Python scanner and backtester for long ATM calendar spreads
on US stocks and ETFs, targeting term-structure mean-reversion.

## Strategy

Sell the front-month ATM option, buy the next-month ATM option at the same
strike. Enter when the front-month IV is statistically rich relative to the
back month (z-score + percentile rank). Exit when the term structure
normalizes — not a theta-harvest trade, but a relative-value volatility trade.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Put your ORATS API token in a `.env` file (preferred)
#    Create a file named `.env` at the project root with the line:
#
#      ORATS_TOKEN=your-orats-token-here
#
#    The loader will read `.env` automatically if `python-dotenv` is
#    installed. As a fallback you can still set `orats.token` in `config.yaml`.

# 3. Fetch 5 years of historical data (~15-30 min depending on universe size)
python run_fetch.py

# 4. Run the backtest
python run_backtest.py

# 5. Compare a curated grid of high-conviction parameter sets
python run_grid_backtest.py

# 6. (Optional) Live scan via IBKR — requires TWS/Gateway running
python run_scanner.py --with-history
```

## Project structure

```
calendar_scanner/
├── config.yaml          # All tunable parameters
├── requirements.txt
├── run_fetch.py         # Download ORATS data → data/raw/
├── run_backtest.py      # Run backtest → data/results/
├── run_grid_backtest.py # Run curated parameter grid → data/results/grid/
├── run_scanner.py       # Live IBKR scanner
├── src/
│   ├── config.py        # Config loader
│   ├── orats_data.py    # ORATS API client + Parquet caching
│   ├── features.py      # ATM IV extraction, rolling stats, RV
│   ├── signals.py       # Entry/exit rules
│   ├── backtest.py      # Backtest engine + position tracking
│   ├── metrics.py       # Performance metrics + charts
│   └── ibkr_scanner.py  # Live scanner via ib_insync
└── data/
    ├── raw/             # Cached ORATS API responses (Parquet)
    ├── processed/       # Combined data files
    └── results/         # Backtest outputs (CSV, PNG, JSON)
```

## Configuration

All parameters are in `config.yaml`. Key groups:

- **universe**: list of tickers to scan (default: 15 liquid ETFs)
- **expiry**: DTE ranges for front/back month selection
- **signals**: z-score and percentile thresholds for entry/exit
- **liquidity**: minimum OI, volume, and spread width filters
- **exit**: profit target, stop loss, time stop, max hold days
- **costs**: slippage model and commissions
- **sizing**: risk per trade and position limits

## Data flow

1. `run_fetch.py` calls ORATS `/datav2/hist/strikes` per ticker per
   6-month chunk. Each chunk is cached as Parquet in `data/raw/{ticker}/`.
   Combined files are saved in `data/processed/`.

2. `run_backtest.py` loads the combined Parquet, runs the feature pipeline
   (ATM IV extraction → rolling z-score/percentile → RV → regime tags),
   then feeds features into the backtest engine.

3. The engine iterates day-by-day: MTM existing positions → check exits →
   screen new entries → open positions → record portfolio state.

4. Outputs: trade log CSV, equity curve PNG, P&L distribution, monthly
   heatmap, metrics JSON, and regime breakdown printed to stdout.

## Backtest outputs

After running `run_backtest.py`:

| File | Description |
|---|---|
| `data/results/trade_log.csv` | Every trade with entry/exit dates, P&L, exit reason |
| `data/results/equity_curve.png` | Equity + drawdown + position count |
| `data/results/pnl_distribution.png` | Histogram of trade P&L% |
| `data/results/monthly_heatmap.png` | Monthly P&L by year |
| `data/results/metrics.json` | Summary metrics (Sharpe, win rate, etc.) |
| `data/results/regimes.json` | Performance by regime (z-score bucket, ticker, year) |
| `data/results/features.parquet` | Full feature table for ad-hoc analysis |

After running `run_grid_backtest.py`:

| File | Description |
|---|---|
| `data/results/grid/grid_summary.csv` | Scenario-by-scenario comparison table |
| `data/results/grid/grid_comparison.png` | Dashboard with P&L, trade count, return scatter, and equity curves |
| `data/results/grid/grid_heatmap.png` | Parameter-and-outcome matrix across scenarios |
| `data/results/grid/<scenario>/summary.txt` | Human-readable summary for one scenario |
| `data/results/grid/<scenario>/entry_filter_audit.csv` | Candidate attrition by ticker |
| `data/results/grid/<scenario>/trade_log.csv` | Trade log for that scenario |
| `data/results/grid/<scenario>/*.png` | Per-scenario equity and P&L charts |

## IBKR live scanner

The scanner uses the same feature and signal code as the backtest,
but pulls live option chains from IBKR via `ib_insync`.

```bash
# Without historical z-scores (ranks by raw IV spread)
python run_scanner.py

# With z-scores (requires prior backtest to produce features.parquet)
python run_scanner.py --with-history
```

TWS/Gateway must be running with API connections enabled on the configured
port (default 7497 for paper trading).

## ORATS API notes

- The minimum subscription tier that provides `hist/strikes` is required.
- Data is fetched in 6-month chunks to stay within row limits.
- API calls are rate-limited to ~1/second (configurable in config.yaml).
- Once fetched, data is cached as Parquet — re-running `run_fetch.py`
  skips already-cached chunks.
- If ORATS column names change across API versions, update the
  `STRIKES_RENAME` dict in `src/orats_data.py`.

## Tuning the strategy

Start with the default config and examine the regime breakdown. Key
levers to adjust based on backtest results:

1. **z-score thresholds** (`signals.zscore_entry_min`): lower = more trades
   but weaker signal; higher = fewer but higher-conviction.
2. **DTE ranges** (`expiry.*`): shorter front DTE = faster theta decay
   but more gamma risk near expiration.
3. **Exit normalization** (`signals.zscore_exit`): lower = hold longer
   for full normalization; higher = take profits earlier.
4. **Stop loss** (`exit.stop_loss_pct`): tighter stops reduce tail risk
   but increase whipsaw rate.
5. **Slippage** (`costs.slippage_pct`): run at 3% (baseline) and 5%
   (conservative) — if edge disappears at 5%, it's not real.

## Interpreting results

Key things to check:

- **Win rate > 55%** and **profit factor > 1.3** as minimum bar for edge.
- **Exit reason mix**: majority should be `normalization` or `profit_target`,
  not `stop_loss` or `time_stop`. Heavy stop-loss exits suggest the signal
  is not predictive.
- **z-score bucket monotonicity**: higher z-score buckets should have
  higher win rates. If flat or inverted, the signal isn't working.
- **Consistency across years**: if edge exists only in 2020 or 2022,
  it's likely regime-specific rather than structural.
- **Sharpe > 1.0 after slippage**: the go/no-go criterion.
