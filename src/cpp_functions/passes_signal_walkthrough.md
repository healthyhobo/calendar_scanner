# Exercise: `passes_signal` in C++

This exercise translates `passes_signal` from `src/signals.py`.

## Why This Rule Exists

This is the core entry-signal filter. It decides whether the current volatility term structure is unusual enough to consider a trade.

Your project looks for front-month implied volatility that is rich relative to back-month implied volatility:

```text
iv_spread = front_iv - back_iv
```

Then it asks whether that spread is unusual for the same ticker:

- `spread_zscore`: how many standard deviations above normal the spread is.
- `spread_pctile`: where today's spread ranks compared with recent history.

The signal filter also rejects setups where the apparent opportunity may be dangerous:

- `back_iv_pctile` too high means the whole volatility regime is elevated.
- `rv_iv_ratio` too high means realized volatility is already close to or above implied volatility.
- `daily_gamma_drag / debit` too high means expected spot-move cost can eat the trade.

## Rule Summary

The Python function applies these gates:

```text
1. z-score and percentile must exist
2. z-score must be in [zscore_entry_min, zscore_entry_max]
3. percentile must be >= percentile_entry_min
4. optional back IV percentile cap
5. optional RV/IV ratio cap
6. optional gamma-drag-to-debit cap
```

## Trading Intuition

A high z-score says the front/back IV spread is dislocated. That is good for a mean-reversion strategy.

But extremely high volatility can be a warning. A spread may look cheap or rich because the market is pricing a real event. The extra filters try to avoid trades where:

- volatility is elevated everywhere,
- realized stock movement is already high,
- the short front gamma is likely to hurt too much.

## Translation Map

| Python | C++ |
|---|---|
| `row.get("spread_zscore", np.nan)` | `row.spread_zscore` |
| `cfg["signals"]` | `SignalConfig` struct |
| `np.isnan(x)` | `std::isnan(x)` |
| `is not None` | not `NaN` in this exercise |
| `and` | `&&` |
| `or` | `||` |

## Step-by-Step Plan

1. Open `passes_signal_exercise.cpp`.
2. Reject missing z-score or percentile.
3. Apply the z-score min and max.
4. Apply the percentile minimum.
5. Add the optional back-IV percentile cap.
6. Add the optional RV/IV ratio cap.
7. Add the optional gamma-drag-to-debit cap.
8. Return `true` if every check passes.

## Compile and Run

```powershell
g++ -std=c++17 src/cpp_functions/passes_signal_exercise.cpp -o passes_signal_exercise
.\passes_signal_exercise.exe
```

Or with Microsoft `cl`:

```powershell
cl /EHsc /std:c++17 src\cpp_functions\passes_signal_exercise.cpp
.\passes_signal_exercise.exe
```

## Check Yourself

Expected behavior:

```text
valid signal -> true
zscore too low -> false
percentile too low -> false
back iv too elevated -> false
rv iv too high -> false
gamma drag too high -> false
missing optional filters are allowed -> true
```

## References

- pandas rolling standard deviation background for z-score calculations: https://pandas.pydata.org/docs/reference/api/pandas.core.window.rolling.Rolling.std.html
- Cboe Options Institute glossary for implied volatility, realized volatility, and Greeks: https://www.cboe.com/optionsinstitute/glossary/
- FINRA options Greeks overview: https://www.finra.org/investors/insights/options-z-basics-greeks
