# Exercise: `_bs_atm_call` in C++

This exercise translates the Python helper from `src/features.py`:

```python
def _bs_atm_call(spot: float, iv_dec: float, dte: float, r: float = 0.04) -> float:
    if dte <= 0 or iv_dec <= 0 or spot <= 0:
        return np.nan
    T = dte / 365.25
    sqrt_T = np.sqrt(T)
    d1 = (r + 0.5 * iv_dec ** 2) * T / (iv_dec * sqrt_T)
    d2 = d1 - iv_dec * sqrt_T
    return spot * norm.cdf(d1) - spot * np.exp(-r * T) * norm.cdf(d2)
```

## Why This Function Exists

Your strategy trades long calendar spreads: sell a nearer-term ATM option and buy a farther-term ATM option. To estimate the spread's value, the Python code prices each ATM call with Black-Scholes and subtracts:

```text
calendar debit ~= back_month_call - front_month_call
```

This is more realistic than simply using half the difference between two straddle prices because option value does not grow linearly with time or volatility. The Black-Scholes model gives you a clean way to turn spot price, implied volatility, days to expiry, and interest rate into a theoretical option price.

In this project, the function is deliberately simplified:

- It prices an at-the-money call, so strike `K` is treated as equal to spot `S`.
- It assumes no dividends.
- It uses implied volatility as a decimal, such as `0.20` for 20%.
- It converts calendar days to years with `dte / 365.25`.

## Math Background

The standard Black-Scholes European call formula is:

```text
C = S * N(d1) - K * exp(-rT) * N(d2)
```

where:

```text
d1 = [ln(S / K) + (r + 0.5 * sigma^2)T] / [sigma * sqrt(T)]
d2 = d1 - sigma * sqrt(T)
```

For an ATM option, `S = K`, so:

```text
ln(S / K) = ln(1) = 0
```

That leaves the simpler formula used in your Python code:

```text
d1 = (r + 0.5 * sigma^2)T / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
```

The normal CDF `N(x)` gives the probability that a standard normal random variable is less than `x`. In Python, SciPy provides `norm.cdf`. In C++, the standard library does not directly provide `normal_cdf`, but you can build it from `std::erfc`:

```text
N(x) = 0.5 * erfc(-x / sqrt(2))
```

## Translation Map

| Python | C++ |
|---|---|
| `float` | `double` |
| `np.sqrt(T)` | `std::sqrt(T)` |
| `np.exp(x)` | `std::exp(x)` |
| `iv_dec ** 2` | `iv_dec * iv_dec` |
| `or` | `||` |
| `np.nan` | `std::numeric_limits<double>::quiet_NaN()` |
| `norm.cdf(x)` | your `normal_cdf(x)` helper |

## Step-by-Step Plan

1. Open `bs_atm_call_exercise.cpp`.
2. Fill in `normal_cdf`.
3. Add the invalid-input guard inside `bs_atm_call`.
4. Create `T` and `sqrt_T`.
5. Create `d1` and `d2`.
6. Return the final call price.
7. Compile and run the file.
8. Compare your output to the expected values printed by `main`.

## Compile and Run

From the project root:

```powershell
g++ -std=c++17 src/cpp_functions/bs_atm_call_exercise.cpp -o bs_atm_call_exercise
.\bs_atm_call_exercise.exe
```

If `g++` is not installed, install a C++ compiler such as MSYS2/MinGW, Visual Studio Build Tools, or LLVM/Clang.

## Check Yourself

When finished, your output should be close to:

```text
spot=100, iv_dec=0.20, dte=30, r=0.04 -> price=2.450367
spot=50, iv_dec=0.35, dte=45, r=0.04 -> price=2.567829
spot=425, iv_dec=0.18, dte=60, r=0.04 -> price=13.767923
```

Small differences in the last few decimals are fine.

## References

- Fischer Black and Myron Scholes, "The Pricing of Options and Corporate Liabilities", Journal of Political Economy, 1973: https://www.jstor.org/stable/1831029
- Robert C. Merton, "Theory of Rational Option Pricing", Bell Journal of Economics and Management Science, 1973: https://www.jstor.org/stable/3003143
- Cboe Options Institute glossary for implied volatility, Greeks, and related options terms: https://www.cboe.com/optionsinstitute/glossary/
