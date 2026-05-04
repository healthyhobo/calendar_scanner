# Exercise: `_atm_gamma_approx` in C++

This exercise translates the Python helper from `src/features.py`:

```python
def _atm_gamma_approx(spot: float, iv_dec: float, dte: float) -> float:
    if dte <= 0 or iv_dec <= 0 or spot <= 0:
        return np.nan
    T = max(dte, 1) / 365.25
    return 0.3989 / (spot * iv_dec * np.sqrt(T))
```

## Why This Function Exists

Your strategy trades long calendar spreads:

```text
short front-month ATM option
long back-month ATM option
```

Gamma measures how quickly an option's delta changes when the underlying stock price moves. Nearer-term ATM options usually have higher gamma than farther-term ATM options. Because this calendar spread is short the front option and long the back option, the position is usually net short gamma.

That matters because short gamma loses from large spot moves in either direction. The Python project estimates this cost with:

```text
daily_gamma_drag = 0.5 * abs(net_gamma) * estimated_daily_move^2
```

The `_atm_gamma_approx` function gives a fast estimate of each leg's gamma so the scanner can avoid trades where expected daily gamma drag is too large relative to the calendar debit.

## Math Background

In Black-Scholes, option gamma for a non-dividend-paying stock is commonly written as:

```text
Gamma = N'(d1) / (S * sigma * sqrt(T))
```

where:

- `N'(d1)` is the standard normal probability density at `d1`
- `S` is the stock price
- `sigma` is implied volatility as a decimal
- `T` is time to expiry in years

For an at-the-money option, `d1` is often close to zero. The standard normal density at zero is:

```text
N'(0) = 1 / sqrt(2 * pi) ~= 0.3989
```

So the project uses this approximation:

```text
Gamma ~= 0.3989 / (S * sigma * sqrt(T))
```

This is not perfect, but it is fast and good enough for a screening feature. It also teaches an important trading idea: as expiry gets closer, `sqrt(T)` gets smaller, so gamma gets larger. That is why short-dated ATM options can become very sensitive to stock moves.

## Translation Map

| Python | C++ |
|---|---|
| `float` | `double` |
| `or` | `||` |
| `np.nan` | `std::numeric_limits<double>::quiet_NaN()` |
| `max(dte, 1)` | `std::max(dte, 1.0)` |
| `np.sqrt(T)` | `std::sqrt(T)` |

## Step-by-Step Plan

1. Open `atm_gamma_approx_exercise.cpp`.
2. Add the invalid-input guard.
3. Create `T` using `std::max(dte, 1.0) / 365.25`.
4. Return `0.3989 / (spot * iv_dec * std::sqrt(T))`.
5. Compile and run the file.
6. Compare your output to the expected values printed by `main`.

## Compile and Run

From the project root:

```powershell
g++ -std=c++17 src/cpp_functions/atm_gamma_approx_exercise.cpp -o atm_gamma_approx_exercise
.\atm_gamma_approx_exercise.exe
```

If `g++` is not installed, install a C++ compiler such as MSYS2/MinGW, Visual Studio Build Tools, or LLVM/Clang.

## Check Yourself

When finished, your output should be close to:

```text
spot=100, iv_dec=0.20, dte=30 -> gamma=0.06959348
spot=50, iv_dec=0.35, dte=45 -> gamma=0.06494038
spot=425, iv_dec=0.18, dte=60 -> gamma=0.01286536
```

Small differences in the last few decimals are fine.

## Common Beginner Mistakes

- Writing `or` instead of `||`.
- Writing `np.sqrt` instead of `std::sqrt`.
- Forgetting to include `<cmath>`.
- Using `1` instead of `1.0` inside `std::max`, which can confuse C++ type matching.
- Passing volatility as `20.0` instead of `0.20`. This function expects decimal volatility.

## References

- FINRA overview of options Greeks, including gamma as the change in delta for a change in the underlying: https://www.finra.org/investors/insights/options-z-basics-greeks
- Cboe Options Institute glossary for Greeks and implied volatility terms: https://www.cboe.com/optionsinstitute/glossary/
- Charles Schwab explanation of gamma as a measure of how delta changes with stock movement: https://www.schwab.com/learn/story/options-greeks-gamma-explained
