# Exercise: `passes_liquidity` in C++

This exercise translates `passes_liquidity` from `src/signals.py`.

```python
def passes_liquidity(row: pd.Series, cfg: dict) -> bool:
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
```

## Why This Rule Exists

A calendar spread can look attractive on paper but be hard to trade if the underlying or options are illiquid. Thinly traded options can have wide bid/ask spreads, stale marks, and poor fills. That matters because your backtest assumes an estimated slippage model, but real trading costs can easily overwhelm a small volatility edge.

This rule checks three things:

- The stock price is high enough.
- Current option volume is not too low when available.
- 20-day average option volume is not too low when available.

The Python function allows missing volume fields because some data rows may not include them. It only rejects low volume when the volume value is present.

## Translation Map

| Python | C++ |
|---|---|
| `cfg["liquidity"]` | `LiquidityConfig` struct |
| `row.get("stock_price", np.nan)` | `row.stock_price` |
| `np.isnan(x)` | `std::isnan(x)` |
| `or` | `||` |
| `and` | `&&` |
| `False` / `True` | `false` / `true` |

## Step-by-Step Plan

1. Open `passes_liquidity_exercise.cpp`.
2. Read the `LiquidityConfig` and `LiquidityRow` structs.
3. Complete the stock-price rejection rule.
4. Complete the current option-volume rejection rule.
5. Complete the average option-volume rejection rule.
6. Return `true` if all checks pass.
7. Compile and run.

## Compile and Run

With `g++`:

```powershell
g++ -std=c++17 src/cpp_functions/passes_liquidity_exercise.cpp -o passes_liquidity_exercise
.\passes_liquidity_exercise.exe
```

With Microsoft `cl`:

```powershell
cl /EHsc /std:c++17 src\cpp_functions\passes_liquidity_exercise.cpp
.\passes_liquidity_exercise.exe
```

## Check Yourself

Expected behavior:

```text
liquid row -> true
stock price too low -> false
current option volume too low -> false
avg option volume too low -> false
missing volume fields are allowed -> true
```

## References

- Cboe Options Institute glossary for bid, ask, spread, and liquidity-related options terms: https://www.cboe.com/optionsinstitute/glossary/
- FINRA options basics, including practical risks around options trading: https://www.finra.org/investors/investing/investment-products/options
