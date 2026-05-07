# Exercise: `_to_percent_points` in C++

This exercise translates the Python helper from `src/features.py`:

```python
def _to_percent_points(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    finite = s[np.isfinite(s)]
    if finite.empty:
        return s
    return s.where(s.abs() > 2.0, s * 100.0)
```

## Why This Function Exists

Volatility data is often represented in two different ways:

```text
0.20  means 20 percent as a decimal
20.0  means 20 percentage points
```

Those two values mean the same thing financially, but they are very different numbers. If one row says `0.20` and another says `20.0`, a backtest can produce nonsense unless the values are normalized first.

Your scanner wants implied volatility columns such as `iv30d` and `iv60d` in percentage-point form:

```text
front_iv = 20.0
back_iv  = 18.5
spread   = 1.5 percentage points
```

That makes the spread easier to read, rank, and compare across tickers.

## The Rule

The Python function uses a simple per-value heuristic:

```text
if abs(value) > 2.0:
    keep it unchanged
else:
    multiply by 100
```

Why `2.0`? A normal implied volatility decimal is usually something like `0.15`, `0.25`, or `0.80`. Even a very high volatility decimal like `1.50` means 150 percent. So values below or equal to `2.0` are treated as decimal volatility and converted.

Values above `2.0` are assumed to already be percentage points:

```text
15.0 stays 15.0
0.15 becomes 15.0
1.50 becomes 150.0
```

The function applies this per value, not once per whole column. That matters because older data can sometimes mix formats in the same column.

## Translation Map

| Python | C++ |
|---|---|
| `pd.Series` | `std::vector<double>` |
| `np.isfinite(x)` | `std::isfinite(x)` |
| `s.abs()` | `std::abs(value)` |
| `s.where(condition, fallback)` | `if (...) return ...; else return ...;` |
| `np.nan` | `std::numeric_limits<double>::quiet_NaN()` |

## Step-by-Step Plan

1. Open `to_percent_points_exercise.cpp`.
2. Complete the single-value helper `to_percent_points`.
3. Add the finite-value guard using `std::isfinite`.
4. Add the threshold rule using `std::abs(value) > 2.0`.
5. Complete the vector wrapper `to_percent_points_vector`.
6. Compile and run the file.
7. Compare your output to the expected values printed by `main`.

## Compile and Run

With `g++`:

```powershell
g++ -std=c++17 src/cpp_functions/to_percent_points_exercise.cpp -o to_percent_points_exercise
.\to_percent_points_exercise.exe
```

With Microsoft `cl` from a Developer PowerShell:

```powershell
cl /EHsc /std:c++17 src\cpp_functions\to_percent_points_exercise.cpp
.\to_percent_points_exercise.exe
```

## Check Yourself

Input:

```text
0.20 15.0 1.50 -0.30 -25.0 nan 2.00 2.01
```

Expected output:

```text
20.0000 15.0000 150.0000 -30.0000 -25.0000 nan 200.0000 2.0100
```

The surprising one is `2.00`. Because the Python rule is `abs(value) > 2.0`, exactly `2.0` does not pass the threshold, so it becomes `200.0`.

## Common Beginner Mistakes

- Forgetting that volatility can be decimal or percentage-point form.
- Checking `value > 2.0` instead of `std::abs(value) > 2.0`.
- Accidentally converting `15.0` into `1500.0`.
- Trying to compare directly to `NaN`. Use `std::isnan` or `std::isfinite`.

## References

- Cboe Options Institute glossary, including implied volatility terminology: https://www.cboe.com/optionsinstitute/glossary/
- NumPy documentation for `isfinite`, the Python concept this exercise mirrors in C++: https://numpy.org/doc/stable/reference/generated/numpy.isfinite.html
- C++ reference for `std::isfinite`: https://en.cppreference.com/w/cpp/numeric/math/isfinite
